# /home/icb/alejandro.tejada/CellFlow2/src/scaleflow/solvers/_eqm.py

import warnings
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from cellflow.solvers._base import BaseSolver
from cellflow.solvers.utils import ema_update
from flax.core import frozen_dict
from flax.training import train_state
from ott.solvers import utils as solver_utils

from cellflow import utils
from cellflow._types import ArrayLike
from scaleflow.networks._velocity_field import ConditionalVelocityField

__all__ = ["EquilibriumMatching"]


class EquilibriumMatching(BaseSolver):
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
            see e.g. :func:`cellflow.utils.match_linear`. If :obj:`None`, no
            matching is performed.
        gamma_sampler
            Noise level sampler with a ``(rng, n_samples) -> gamma`` signature.
            Defaults to uniform sampling on [0, 1].
        c_fn
            Weighting function c(gamma). Defaults to c(gamma) = 1 - gamma.
        kwargs
            Keyword arguments for :meth:`scaleflow.networks.ConditionalVelocityField.create_train_state`.
    """

    @staticmethod
    def _match_kwargs(*, match_fn: Callable, data_dim: int) -> dict[str, Any]:
        """EqM matches source/target on ``match_fn`` and needs no explicit dimensions."""
        return {"match_fn": match_fn}

    def __init__(
        self,
        vf: ConditionalVelocityField,
        match_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
        gamma_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
        c_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        **kwargs: Any,
    ):
        # EqM has no probability path or time sampler (it interpolates via gamma), so pass
        # ``None`` for the base's generic slots; the rest of the shared scaffolding
        # (is_trained flag, vf, condition-encoder settings, predict-fn cache) comes from BaseSolver.
        super().__init__(vf, probability_path=None, time_sampler=None)
        self.gamma_sampler = gamma_sampler
        self.c_fn = c_fn if c_fn is not None else lambda gamma: 1.0 - gamma
        self.match_fn = jax.jit(match_fn) if match_fn is not None else None
        self.ema = kwargs.pop("ema", 1.0)

        self.vf_state = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_state_inference = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_step_fn = self._get_vf_step_fn()

    @property
    def _inference_state(self) -> train_state.TrainState:
        """EqM reads condition embeddings from the EMA inference state."""
        return self.vf_state_inference

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

        config_frozen = frozen_dict.freeze({"eta": eta, "max_steps": max_steps, "use_nesterov": use_nesterov, "mu": mu})
        predict_fn = self._get_predict_fn(config_frozen)
        return predict_fn(self.vf_state_inference.params, x, condition, encoder_noise)

    def _get_predict_fn(self, config_frozen: frozen_dict.FrozenDict) -> Callable:  # type: ignore[type-arg]
        """Build (and cache) the jitted gradient-descent sampler for a sampler config.

        ``params`` are threaded as an argument rather than closed over, so the compiled
        function is reused across parameter updates instead of recompiling each call.
        """
        if config_frozen in self._predict_fn_cache:
            return self._predict_fn_cache[config_frozen]

        cfg = dict(config_frozen)
        eta = cfg["eta"]
        max_steps = cfg["max_steps"]
        use_nesterov = cfg["use_nesterov"]
        mu = cfg["mu"]

        def gradient_field(
            params: Any, x: jnp.ndarray, condition: dict[str, jnp.ndarray], encoder_noise: jnp.ndarray
        ) -> jnp.ndarray:
            return self.vf_state_inference.apply_fn({"params": params}, x, condition, encoder_noise, train=False)[0]

        def sample_gd(
            params: Any, x: jnp.ndarray, condition: dict[str, jnp.ndarray], encoder_noise: jnp.ndarray
        ) -> jnp.ndarray:
            """Basic gradient descent sampler."""

            def gd_step(i, x_val):
                f = gradient_field(params, x_val, condition, encoder_noise)
                return x_val - eta * f

            return jax.lax.fori_loop(0, max_steps, gd_step, x)

        def sample_nag(
            params: Any, x: jnp.ndarray, condition: dict[str, jnp.ndarray], encoder_noise: jnp.ndarray
        ) -> jnp.ndarray:
            """Nesterov accelerated gradient descent sampler."""

            def nag_step(i, state):
                x_val, velocity = state
                x_lookahead = x_val - mu * velocity
                f = gradient_field(params, x_lookahead, condition, encoder_noise)
                new_velocity = mu * velocity + eta * f
                new_x = x_val - new_velocity
                return (new_x, new_velocity)

            init_state = (x, jnp.zeros_like(x))
            final_x, _ = jax.lax.fori_loop(0, max_steps, nag_step, init_state)
            return final_x

        sampler = sample_nag if use_nesterov else sample_gd
        fn = jax.jit(jax.vmap(sampler, in_axes=[None, 0, None, None]))
        self._predict_fn_cache[config_frozen] = fn
        return fn

    def predict(
        self,
        x: ArrayLike | dict[str, ArrayLike],
        condition: dict[str, ArrayLike] | dict[str, dict[str, ArrayLike]],
        rng: jax.Array | None = None,
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
            Either a single array of shape ``[batch_size, ...]`` or a dictionary mapping
            condition names to such arrays (predicted per condition).
        condition
            The condition(s) corresponding to ``x``: a dict of arrays for a single input,
            or a dict mapping condition names to such dicts when ``x`` is a dict.
        rng
            Random number generator to sample from the latent distribution,
            only used if ``condition_mode='stochastic'``. If :obj:`None`, the
            mean embedding is used.
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
        if "batched" in kwargs:
            warnings.warn(
                "The `batched` argument is deprecated and ignored. Dictionary input is "
                "predicted per condition; the lazy per-condition path provides the same "
                "parallelism without eagerly materializing arrays.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.pop("batched")

        if isinstance(x, dict) and not x:
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

        if isinstance(x, dict):
            return jax.tree.map(
                predict_fn,
                x,
                condition,
            )
        else:
            x_pred = predict_fn(x, condition)
            return np.array(x_pred)
