# /home/icb/alejandro.tejada/CellFlow2/src/scaleflow/solvers/_eqm.py

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict
from flax.training import train_state
from ott.solvers import utils as solver_utils

from scaleflow import utils
from scaleflow._types import ArrayLike
from scaleflow.networks._velocity_field import ConditionalVelocityField
from scaleflow.solvers.utils import ema_update

__all__ = ["EquilibriumMatching"]


class EquilibriumMatching:
    """Equilibrium Matching for generative modeling.

    Based on "Equilibrium Matching" (Wang & Du, 2024).
    Learns a time-invariant equilibrium gradient field instead of
    time-conditional velocities.

    Parameters
    ----------
        vf
            Vector field parameterized by a neural network (without time conditioning).
        match_fn
            Function to match samples from the source and the target
            distributions. It has a ``(src, tgt) -> matching`` signature,
            see e.g. :func:`scaleflow.utils.match_linear`. If :obj:`None`, no
            matching is performed.
        gamma_sampler
            Noise level sampler with a ``(rng, n_samples) -> gamma`` signature.
            Defaults to uniform sampling on [0, 1].
        c_fn
            Weighting function c(gamma). Defaults to c(gamma) = 1 - gamma.
        kwargs
            Keyword arguments for :meth:`scaleflow.networks.ConditionalVelocityField.create_train_state`.
    """

    def __init__(
        self,
        vf: ConditionalVelocityField,
        match_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
        gamma_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
        c_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        **kwargs: Any,
    ):
        self._is_trained: bool = False
        self.vf = vf
        self.condition_encoder_mode = self.vf.condition_mode
        self.condition_encoder_regularization = self.vf.regularization
        self.gamma_sampler = gamma_sampler
        self.c_fn = c_fn if c_fn is not None else lambda gamma: 1.0 - gamma
        self.match_fn = jax.jit(match_fn) if match_fn is not None else None
        self.ema = kwargs.pop("ema", 1.0)

        self.vf_state = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_state_inference = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_step_fn = self._get_vf_step_fn()

    def _get_vf_step_fn(self) -> Callable:
        @jax.jit
        def vf_step_fn(
            rng: jax.Array,
            vf_state: train_state.TrainState,
            gamma: jnp.ndarray,
            source: jnp.ndarray,
            target: jnp.ndarray,
            conditions: dict[str, jnp.ndarray],
            encoder_noise: jnp.ndarray,
        ):
            def loss_fn(
                params: jnp.ndarray,
                gamma: jnp.ndarray,
                source: jnp.ndarray,
                target: jnp.ndarray,
                conditions: dict[str, jnp.ndarray],
                encoder_noise: jnp.ndarray,
                rng: jax.Array,
            ) -> jnp.ndarray:
                rng_encoder, rng_dropout = jax.random.split(rng, 2)

                # Interpolate between source (noise) and target (data)
                gamma_expanded = gamma[:, jnp.newaxis]
                x_gamma = gamma_expanded * target + (1.0 - gamma_expanded) * source

                # Predict gradient field (no time input)
                f_pred, mean_cond, logvar_cond = vf_state.apply_fn(
                    {"params": params},
                    x_gamma,
                    conditions,
                    encoder_noise=encoder_noise,
                    rngs={"dropout": rng_dropout, "condition_encoder": rng_encoder},
                )

                # Target gradient: (source - target) * c(gamma)
                c_gamma = self.c_fn(gamma)[:, jnp.newaxis]
                target_gradient = (source - target) * c_gamma

                # EqM loss
                eqm_loss = jnp.mean((f_pred - target_gradient) ** 2)

                # Condition encoder regularization (same as flow matching)
                condition_mean_regularization = 0.5 * jnp.mean(mean_cond**2)
                condition_var_regularization = -0.5 * jnp.mean(1 + logvar_cond - jnp.exp(logvar_cond))

                if self.condition_encoder_mode == "stochastic":
                    encoder_loss = condition_mean_regularization + condition_var_regularization
                elif (self.condition_encoder_mode == "deterministic") and (self.condition_encoder_regularization > 0):
                    encoder_loss = condition_mean_regularization
                else:
                    encoder_loss = 0.0

                return eqm_loss + encoder_loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(vf_state.params, gamma, source, target, conditions, encoder_noise, rng)
            return vf_state.apply_gradients(grads=grads), loss

        return vf_step_fn

    def step_fn(
        self,
        rng: jnp.ndarray,
        batch: dict[str, ArrayLike],
    ) -> float:
        """Single step function of the solver.

        Parameters
        ----------
        rng
            Random number generator.
        batch
            Data batch with keys ``src_cell_data``, ``tgt_cell_data``, and
            optionally ``condition``.

        Returns
        -------
        Loss value.
        """
        src, tgt = batch["src_cell_data"], batch["tgt_cell_data"]
        condition = batch.get("condition")
        rng_resample, rng_gamma, rng_step_fn, rng_encoder_noise = jax.random.split(rng, 4)
        n = src.shape[0]
        gamma = self.gamma_sampler(rng_gamma, n).squeeze()
        encoder_noise = jax.random.normal(rng_encoder_noise, (n, self.vf.condition_embedding_dim))

        if self.match_fn is not None:
            tmat = self.match_fn(src, tgt)
            src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
            src, tgt = src[src_ixs], tgt[tgt_ixs]

        self.vf_state, loss = self.vf_step_fn(
            rng_step_fn,
            self.vf_state,
            gamma,
            src,
            tgt,
            condition,
            encoder_noise,
        )

        if self.ema == 1.0:
            self.vf_state_inference = self.vf_state
        else:
            self.vf_state_inference = self.vf_state_inference.replace(
                params=ema_update(self.vf_state_inference.params, self.vf_state.params, self.ema)
            )
        return loss

    def get_condition_embedding(self, condition: dict[str, ArrayLike], return_as_numpy=True) -> ArrayLike:
        """Get learnt embeddings of the conditions.

        Parameters
        ----------
        condition
            Conditions to encode
        return_as_numpy
            Whether to return the embeddings as numpy arrays.

        Returns
        -------
        Mean and log-variance of encoded conditions.
        """
        cond_mean, cond_logvar = self.vf.apply(
            {"params": self.vf_state_inference.params},
            condition,
            method="get_condition_embedding",
        )
        if return_as_numpy:
            return np.asarray(cond_mean), np.asarray(cond_logvar)
        return cond_mean, cond_logvar

    def _predict_jit(
        self,
        x: ArrayLike,
        condition: dict[str, ArrayLike],
        rng: jax.Array | None = None,
        eta: float = 0.003,
        max_steps: int = 250,
        use_nesterov: bool = True,
        mu: float = 0.35,
        **kwargs: Any,
    ) -> ArrayLike:
        """Predict using gradient descent sampling.

        Parameters
        ----------
        x
            Initial samples (typically noise).
        condition
            Conditioning information.
        rng
            Random number generator for stochastic conditioning.
        eta
            Step size for gradient descent.
        max_steps
            Maximum number of gradient descent steps.
        use_nesterov
            Whether to use Nesterov accelerated gradient.
        mu
            Momentum parameter for Nesterov.

        Returns
        -------
        Generated samples.
        """
        noise_dim = (1, self.vf.condition_embedding_dim)
        use_mean = rng is None or self.condition_encoder_mode == "deterministic"
        rng = utils.default_prng_key(rng)
        encoder_noise = jnp.zeros(noise_dim) if use_mean else jax.random.normal(rng, noise_dim)

        def gradient_field(x: jnp.ndarray, condition: dict[str, jnp.ndarray], encoder_noise: jnp.ndarray) -> jnp.ndarray:
            params = self.vf_state_inference.params
            return self.vf_state_inference.apply_fn({"params": params}, x, condition, encoder_noise, train=False)[0]

        def sample_gd(x: jnp.ndarray, condition: dict[str, jnp.ndarray], encoder_noise: jnp.ndarray) -> jnp.ndarray:
            """Basic gradient descent sampler."""
            for _ in range(max_steps):
                f = gradient_field(x, condition, encoder_noise)
                x = x - eta * f
            return x

        def sample_nag(x: jnp.ndarray, condition: dict[str, jnp.ndarray], encoder_noise: jnp.ndarray) -> jnp.ndarray:
            """Nesterov accelerated gradient descent sampler."""
            velocity = jnp.zeros_like(x)
            for _ in range(max_steps):
                x_lookahead = x - mu * velocity
                f = gradient_field(x_lookahead, condition, encoder_noise)
                velocity = mu * velocity + eta * f
                x = x - velocity
            return x

        sampler = sample_nag if use_nesterov else sample_gd
        x_pred = jax.jit(jax.vmap(sampler, in_axes=[0, None, None]))(x, condition, encoder_noise)
        return x_pred

    def predict(
        self,
        x: ArrayLike | dict[str, ArrayLike],
        condition: dict[str, ArrayLike] | dict[str, dict[str, ArrayLike]],
        rng: jax.Array | None = None,
        batched: bool = False,
        eta: float = 0.003,
        max_steps: int = 250,
        use_nesterov: bool = True,
        mu: float = 0.35,
        **kwargs: Any,
    ) -> ArrayLike | dict[str, ArrayLike]:
        """Predict the translated source ``x`` under condition ``condition``.

        This function performs gradient descent on the learned equilibrium landscape.

        Parameters
        ----------
        x
            A dictionary with keys indicating the name of the condition and values containing
            the input data as arrays. If ``batched=False`` provide an array of shape [batch_size, ...].
        condition
            A dictionary with keys indicating the name of the condition and values containing
            the condition of input data as arrays. If ``batched=False`` provide an array of shape
            [batch_size, ...].
        rng
            Random number generator to sample from the latent distribution,
            only used if ``condition_mode='stochastic'``. If :obj:`None`, the
            mean embedding is used.
        batched
            Whether to use batched prediction.
        eta
            Step size for gradient descent (default: 0.003 as in paper).
        max_steps
            Number of gradient descent steps (default: 250 as in paper).
        use_nesterov
            Whether to use Nesterov accelerated gradient (recommended).
        mu
            Momentum parameter for Nesterov (default: 0.35 as in paper).
        kwargs
            Additional keyword arguments (for compatibility).

        Returns
        -------
        The push-forward distribution of ``x`` under condition ``condition``.
        """
        if batched and not x:
            return {}

        predict_fn = partial(
            self._predict_jit,
            rng=rng,
            eta=eta,
            max_steps=max_steps,
            use_nesterov=use_nesterov,
            mu=mu,
            **kwargs,
        )

        if batched:
            keys = sorted(x.keys())
            condition_keys = sorted(set().union(*(condition[k].keys() for k in keys)))
            _predict_jit = jax.jit(lambda x, condition: predict_fn(x, condition))
            batched_predict = jax.vmap(_predict_jit, in_axes=(0, dict.fromkeys(condition_keys, 0)))
            n_cells = x[keys[0]].shape[0]
            for k in keys:
                assert x[k].shape[0] == n_cells, "The number of cells must be the same for each condition"
            src_inputs = jnp.stack([x[k] for k in keys], axis=0)
            batched_conditions = {}
            for cond_key in condition_keys:
                batched_conditions[cond_key] = jnp.stack([condition[k][cond_key] for k in keys])

            pred_targets = batched_predict(src_inputs, batched_conditions)
            return {k: pred_targets[i] for i, k in enumerate(keys)}
        elif isinstance(x, dict):
            return jax.tree.map(
                predict_fn,
                x,
                condition,
            )
        else:
            x_pred = predict_fn(x, condition)
            return np.array(x_pred)

    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value
