"""Gene-expression autoencoder with separable encode / decode.

- :meth:`~_ReconModule.create_train_state` builds a Flax
  :class:`~flax.training.train_state.TrainState` (params + optimizer state), the standard
  JAX/Flax training bundle, mirroring ``ConditionalVelocityField.create_train_state``.
- :meth:`~_ReconModule.train` runs a minibatch training loop and returns the final state and
  loss history. Flax modules are frozen/stateless, so ``train`` returns the new state rather
  than mutating ``self``::

      ae = Autoencoder(gene_dim=2000, latent_dim=128)
      state, losses = ae.train(X)                # learned-latent AE
      dec = Decoder(output_dim=2000)
      state, losses = dec.train(X_state, X)      # predefined-latent decoder
"""

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state

from scaleflow.networks._utils import MLPBlock

__all__ = [
    "Encoder",
    "Decoder",
    "Autoencoder",
    "ReconDecoder",
    "reconstruction_loss",
]


def reconstruction_loss(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    kind: Literal["mse", "nb"] = "mse",
    log_theta: jnp.ndarray | None = None,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Mean reconstruction loss.

    Parameters
    ----------
    pred
        Decoder output of shape ``(batch, genes)``. For ``kind="mse"`` this is the
        reconstructed log-normalized expression; for ``kind="nb"`` it is the positive NB
        mean (use a positive ``output_activation`` on the decoder).
    target
        Ground-truth of shape ``(batch, genes)`` — log-normalized expression for
        ``kind="mse"``, raw counts for ``kind="nb"``.
    kind
        ``"mse"`` (default) or ``"nb"`` (negative-binomial NLL).
    log_theta
        Per-gene log-dispersion of shape ``(genes,)``; required for ``kind="nb"``.
    eps
        Numerical stabiliser.
    """
    if kind == "mse":
        return jnp.mean((pred - target) ** 2)
    if kind == "nb":
        if log_theta is None:
            raise ValueError("kind='nb' requires log_theta (per-gene log-dispersion).")
        theta = jnp.exp(log_theta)  # inverse-dispersion, shape (genes,)
        mu = pred
        log_theta_mu_eps = jnp.log(theta + mu + eps)
        log_prob = (
            theta * (jnp.log(theta + eps) - log_theta_mu_eps)
            + target * (jnp.log(mu + eps) - log_theta_mu_eps)
            + jax.scipy.special.gammaln(target + theta)
            - jax.scipy.special.gammaln(theta)
            - jax.scipy.special.gammaln(target + 1.0)
        )
        return -jnp.mean(log_prob)
    raise ValueError(f"Unknown loss kind: {kind!r}")


class _ReconModule(nn.Module):
    """Mixin adding the Flax train-state factory and a minibatch ``train`` loop.

    Shared by :class:`Encoder`, :class:`Decoder` and :class:`Autoencoder`. The methods are
    stateless: they call ``self.init`` / ``self.apply`` and return a new
    :class:`~flax.training.train_state.TrainState`, never mutating the (frozen) module.
    """

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.GradientTransformation,
        input_dim: int,
    ) -> train_state.TrainState:
        """Initialise the module and wrap it in a ``TrainState``.

        ``input_dim`` is the module's *input* width: ``gene_dim`` for an
        :class:`Autoencoder`, ``latent_dim`` for a standalone :class:`Decoder`. Init goes
        through ``__call__`` so that every submodule's parameters are created.
        """
        params = self.init(rng, jnp.ones((1, input_dim)), training=False)["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)

    def train(
        self,
        input_array: np.ndarray,
        target_array: np.ndarray | None = None,
        *,
        n_iters: int = 2000,
        batch_size: int = 256,
        lr: float = 1e-3,
        seed: int = 0,
        loss_kind: str = "mse",
        optimizer: optax.GradientTransformation | None = None,
        state: train_state.TrainState | None = None,
        log_every: int = 0,
    ) -> tuple[train_state.TrainState, list[float]]:
        """Minibatch training loop over numpy arrays.

        - **Learned-latent AE:** ``ae.train(X)`` — input and target are both ``X``.
        - **Decoder-only (predefined latent):** ``decoder.train(Z, X)`` — input is the latent
          ``Z`` (e.g. ``X_state``), target is gene expression ``X``.

        Batches are fixed-size (sampling with replacement), so the jitted step compiles once.

        Parameters
        ----------
        input_array
            Model input of shape ``(n, input_dim)``.
        target_array
            Reconstruction target of shape ``(n, output_dim)``. ``None`` defaults to
            ``input_array`` (autoencoder mode).
        n_iters, batch_size, lr, seed
            Training-loop hyperparameters.
        loss_kind
            Passed to :func:`reconstruction_loss`.
        optimizer
            Optional optax optimizer (defaults to ``optax.adam(lr)``). Ignored if ``state``
            is provided.
        state
            Optional existing train state to continue from.
        log_every
            If ``> 0``, print the loss every ``log_every`` iterations.

        Returns
        -------
        The final train state and the per-iteration loss history.
        """
        X_in = np.asarray(input_array, dtype=np.float32)
        X_tgt = X_in if target_array is None else np.asarray(target_array, dtype=np.float32)
        if X_in.shape[0] != X_tgt.shape[0]:
            raise ValueError(
                f"input and target must share the first axis; got {X_in.shape[0]} vs {X_tgt.shape[0]}."
            )

        n, in_dim = X_in.shape
        rng = jax.random.PRNGKey(seed)

        if state is None:
            optimizer = optimizer if optimizer is not None else optax.adam(lr)
            rng, init_rng = jax.random.split(rng)
            state = self.create_train_state(init_rng, optimizer, input_dim=in_dim)

        @jax.jit
        def train_step(state, batch_in, batch_tgt, step_rng):
            def loss_fn(params):
                out = state.apply_fn(
                    {"params": params},
                    batch_in,
                    training=True,
                    rngs={"dropout": step_rng},
                )
                # Autoencoder.__call__ returns (x_recon, z); Encoder/Decoder return an array.
                pred = out[0] if isinstance(out, tuple) else out
                return reconstruction_loss(pred, batch_tgt, kind=loss_kind)

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            return state.apply_gradients(grads=grads), loss

        np_rng = np.random.default_rng(seed)
        losses: list[float] = []
        for it in range(n_iters):
            idx = np_rng.integers(0, n, size=batch_size)  # fixed size -> no jit recompile
            rng, step_rng = jax.random.split(rng)
            state, loss = train_step(
                state, jnp.asarray(X_in[idx]), jnp.asarray(X_tgt[idx]), step_rng
            )
            loss = float(loss)
            losses.append(loss)
            if log_every and (it % log_every == 0 or it == n_iters - 1):
                print(f"  iter {it:>6d}  loss={loss:.4f}")

        return state, losses


class Encoder(_ReconModule):
    """Map gene expression to a latent space.

    The hidden trunk is an :class:`~scaleflow.networks._utils.MLPBlock`; the final latent
    projection is a plain :class:`~flax.linen.Dense` with **no** activation, so the latent
    is unconstrained.

    Parameters
    ----------
    latent_dim
        Dimensionality of the latent space.
    hidden_dims
        Widths of the hidden layers (empty ``()`` gives a single linear map).
    dropout_rate
        Dropout rate applied inside the hidden trunk.
    act_fn
        Activation function for the hidden layers.
    """

    latent_dim: int
    hidden_dims: Sequence[int] = (512, 256)
    dropout_rate: float = 0.0
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        if len(self.hidden_dims) > 0:
            x = MLPBlock(
                dims=tuple(self.hidden_dims),
                dropout_rate=self.dropout_rate,
                act_last_layer=True,
                act_fn=self.act_fn,
            )(x, training=training)
        z = nn.Dense(self.latent_dim)(x)
        return z


class Decoder(_ReconModule):
    """Map a latent space back to gene expression.

    Usable standalone on a *predefined* latent (e.g. ``X_state``) or as the decoding half
    of an :class:`Autoencoder`.

    Parameters
    ----------
    output_dim
        Number of output features (genes).
    hidden_dims
        Widths of the hidden layers (empty ``()`` gives a single linear map).
    dropout_rate
        Dropout rate applied inside the hidden trunk.
    act_fn
        Activation function for the hidden layers.
    output_activation
        Optional activation on the final output. ``None`` (default) leaves the output
        unconstrained for MSE on log-normalized expression; e.g. :func:`flax.linen.softplus`
        or :func:`jax.numpy.exp` gives a positive mean for count-based losses.
    """

    output_dim: int
    hidden_dims: Sequence[int] = (256, 512)
    dropout_rate: float = 0.0
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    output_activation: Callable[[jnp.ndarray], jnp.ndarray] | None = None

    @nn.compact
    def __call__(self, z: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        if len(self.hidden_dims) > 0:
            z = MLPBlock(
                dims=tuple(self.hidden_dims),
                dropout_rate=self.dropout_rate,
                act_last_layer=True,
                act_fn=self.act_fn,
            )(z, training=training)
        x_hat = nn.Dense(self.output_dim)(z)
        if self.output_activation is not None:
            x_hat = self.output_activation(x_hat)
        return x_hat


class Autoencoder(_ReconModule):
    """Compose an :class:`Encoder` and a :class:`Decoder`.

    Exposes ``encode`` and ``decode`` as named methods so they can be applied separately::

        ae = Autoencoder(gene_dim=2000, latent_dim=128)
        params = ae.init(rng, jnp.ones((1, 2000)), training=False)["params"]
        z     = ae.apply({"params": params}, x, training=False, method="encode")
        x_hat = ae.apply({"params": params}, z, training=False, method="decode")

    ``init`` must go through ``__call__`` (the default) so that **both** submodules'
    parameters are created; initialising through ``encode`` alone would leave the decoder
    without parameters.

    Parameters
    ----------
    gene_dim
        Number of genes (input and reconstruction width).
    latent_dim
        Dimensionality of the learned latent space.
    encoder_hidden, decoder_hidden
        Hidden-layer widths of the encoder / decoder trunks.
    dropout_rate
        Dropout rate shared by both trunks.
    act_fn
        Hidden-layer activation.
    output_activation
        Optional activation on the decoder output (see :class:`Decoder`).
    """

    gene_dim: int
    latent_dim: int
    encoder_hidden: Sequence[int] = (512, 256)
    decoder_hidden: Sequence[int] = (256, 512)
    dropout_rate: float = 0.0
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    output_activation: Callable[[jnp.ndarray], jnp.ndarray] | None = None

    def setup(self) -> None:
        self.encoder = Encoder(
            latent_dim=self.latent_dim,
            hidden_dims=self.encoder_hidden,
            dropout_rate=self.dropout_rate,
            act_fn=self.act_fn,
        )
        self.decoder = Decoder(
            output_dim=self.gene_dim,
            hidden_dims=self.decoder_hidden,
            dropout_rate=self.dropout_rate,
            act_fn=self.act_fn,
            output_activation=self.output_activation,
        )

    def encode(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Map gene expression ``x`` to the latent space."""
        return self.encoder(x, training=training)

    def decode(self, z: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Map latent ``z`` back to gene expression."""
        return self.decoder(z, training=training)

    def __call__(self, x: jnp.ndarray, training: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
        z = self.encode(x, training=training)
        x_recon = self.decode(z, training=training)
        return x_recon, z


@dataclass
class ReconDecoder:
    """Self-contained, picklable reconstruction decoder: maps a latent representation to gene
    expression.

    Bundles the flax ``module`` + trained ``params`` + ``metadata`` into one object so it can be
    saved standalone (cloudpickle ``.pkl``, mirroring :meth:`scaleflow.model.ScaleFlow.save`) and
    later attached to a flow model: decode its latent-space predictions back to gene expression.

    ``module`` is either a :class:`Decoder` (input = a predefined embedding such as ``X_state``)
    or an :class:`Autoencoder` (decode is applied via its ``decode`` method on the learned
    latent). ``decode`` dispatches on the module type.

    Parameters
    ----------
    module
        The trained flax module (:class:`Decoder` or :class:`Autoencoder`).
    params
        The trained parameter pytree. Stored as host (numpy) arrays for device-portability.
    metadata
        Free-form dict; conventionally holds ``input_key`` (the obsm key of the expected input,
        e.g. ``"X_state"``), ``input_dim``, ``var_names`` (the reconstructed genes),
        ``preprocess`` and the training ``metrics``.
    """

    module: nn.Module
    params: Any
    metadata: dict = field(default_factory=dict)

    @property
    def input_dim(self) -> int | None:
        """Expected width of the input latent (e.g. 2058 for ``X_state``)."""
        return self.metadata.get("input_dim")

    @property
    def input_key(self) -> str | None:
        """obsm key of the latent this decoder expects (e.g. ``"X_state"``)."""
        return self.metadata.get("input_key")

    @property
    def var_names(self) -> list | None:
        """Names of the reconstructed genes (the decoder's output space)."""
        return self.metadata.get("var_names")

    def decode(self, latent: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        """Map ``latent`` ``(n, input_dim)`` -> reconstructed gene expression ``(n, n_genes)``.

        Deterministic (no dropout). Applied in batches to bound memory.
        """
        method = "decode" if isinstance(self.module, Autoencoder) else None
        arr = np.asarray(latent, dtype=np.float32)
        outs = []
        for i in range(0, len(arr), batch_size):
            kwargs = {"method": method} if method is not None else {}
            out = self.module.apply(
                {"params": self.params}, jnp.asarray(arr[i:i + batch_size]), training=False, **kwargs
            )
            out = out[0] if isinstance(out, tuple) else out
            outs.append(np.asarray(out))
        return np.concatenate(outs, axis=0)

    def save(self, dir_path: str, file_prefix: str | None = None, overwrite: bool = False) -> str:
        """Pickle the decoder to ``<dir_path>/[<file_prefix>_]ReconDecoder.pkl``.

        Mirrors :meth:`scaleflow.model.ScaleFlow.save` (cloudpickle, ``file_prefix``/``overwrite``).
        """
        file_name = (
            f"{file_prefix}_{type(self).__name__}.pkl"
            if file_prefix is not None
            else f"{type(self).__name__}.pkl"
        )
        file_dir = os.path.join(dir_path, file_name) if dir_path is not None else file_name
        if not overwrite and os.path.exists(file_dir):
            raise RuntimeError(f"Unable to save to an existing file `{file_dir}` use `overwrite=True` to overwrite it.")
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_dir, "wb") as f:
            cloudpickle.dump(self, f)
        return file_dir

    @classmethod
    def load(cls, filename: str) -> "ReconDecoder":
        """Load a :class:`ReconDecoder` from a ``.pkl`` file or a directory containing one."""
        file_name = os.path.join(filename, f"{cls.__name__}.pkl") if os.path.isdir(filename) else filename
        with open(file_name, "rb") as f:
            obj = cloudpickle.load(f)
        if type(obj) is not cls:
            raise TypeError(f"Expected the object to be type of `{cls}`, found `{type(obj)}`.")
        return obj
