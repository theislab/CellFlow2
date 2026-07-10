import warnings
from collections.abc import Callable
from functools import partial
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from cellflow.solvers.utils import ema_update
from flax.core import frozen_dict
from flax.training import train_state
from ott.solvers import utils as solver_utils

from scaleflow import utils
from scaleflow._compat import BaseFlow
from scaleflow._types import ArrayLike
from scaleflow.networks._velocity_field import ConditionalVelocityField

__all__ = ["OTFlowMatching"]


class OTFlowMatching:
    """(OT) flow matching :cite:`lipman:22` extended to the conditional setting.

    With an extension to OT-CFM :cite:`tong:23,pooladian:23`, and its
    unbalanced version :cite:`eyring:24`.

    Parameters
    ----------
        vf
            Vector field parameterized by a neural network.
        probability_path
            Probability path between the source and the target distributions.
        match_fn
            Function to match samples from the source and the target
            distributions. It has a ``(src, tgt) -> matching`` signature,
            see e.g. :func:`scaleflow.utils.match_linear`. If :obj:`None`, no
            matching is performed, and pure probability_path matching :cite:`lipman:22`
            is applied.
        time_sampler
            Time sampler with a ``(rng, n_samples) -> time`` signature, see e.g.
            :func:`ott.solvers.utils.uniform_sampler`.
        kwargs
            Keyword arguments for :meth:`scaleflow.networks.ConditionalVelocityField.create_train_state`.
    """

    def __init__(
        self,
        vf: ConditionalVelocityField,
        probability_path: BaseFlow,
        match_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
        time_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
        **kwargs: Any,
    ):
        self._is_trained: bool = False
        self.vf = vf
        self.condition_encoder_mode = self.vf.condition_mode
        self.condition_encoder_regularization = self.vf.regularization
        self.probability_path = probability_path
        self.time_sampler = time_sampler
        self.match_fn = jax.jit(match_fn) if match_fn is not None else None
        self.ema = kwargs.pop("ema", 1.0)

        self.vf_state = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_state_inference = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_step_fn = self._get_vf_step_fn()
        # Cache of jitted predict fns keyed on the frozen diffeqsolve kwargs. Params are
        # passed as an argument (not closed over), so the compiled fn is reused across
        # parameter updates instead of recompiling on every predict call.
        self._predict_fn_cache: dict[frozen_dict.FrozenDict, Any] = {}

    def _get_vf_step_fn(self) -> Callable:  # type: ignore[type-arg]
        @jax.jit
        def vf_step_fn(
            rng: jax.Array,
            vf_state: train_state.TrainState,
            time: jnp.ndarray,
            source: jnp.ndarray,
            target: jnp.ndarray,
            conditions: dict[str, jnp.ndarray],
            encoder_noise: jnp.ndarray,
        ):
            def loss_fn(
                params: jnp.ndarray,
                t: jnp.ndarray,
                source: jnp.ndarray,
                target: jnp.ndarray,
                conditions: dict[str, jnp.ndarray],
                encoder_noise: jnp.ndarray,
                rng: jax.Array,
            ) -> jnp.ndarray:
                rng_flow, rng_encoder, rng_dropout = jax.random.split(rng, 3)
                x_t = self.probability_path.compute_xt(rng_flow, t, source, target)
                v_t, mean_cond, logvar_cond = vf_state.apply_fn(
                    {"params": params},
                    t,
                    x_t,
                    conditions,
                    encoder_noise=encoder_noise,
                    rngs={"dropout": rng_dropout, "condition_encoder": rng_encoder},
                )
                u_t = self.probability_path.compute_ut(t, x_t, source, target)
                flow_matching_loss = jnp.mean((v_t - u_t) ** 2)
                condition_mean_regularization = 0.5 * jnp.mean(mean_cond**2)
                condition_var_regularization = -0.5 * jnp.mean(1 + logvar_cond - jnp.exp(logvar_cond))
                if self.condition_encoder_mode == "stochastic":
                    encoder_loss = condition_mean_regularization + condition_var_regularization
                elif (self.condition_encoder_mode == "deterministic") and (self.condition_encoder_regularization > 0):
                    encoder_loss = condition_mean_regularization
                else:
                    encoder_loss = 0.0
                return flow_matching_loss + encoder_loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(vf_state.params, time, source, target, conditions, encoder_noise, rng)
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
        rng_resample, rng_time, rng_step_fn, rng_encoder_noise = jax.random.split(rng, 4)
        n = src.shape[0]
        time = self.time_sampler(rng_time, n)
        encoder_noise = jax.random.normal(rng_encoder_noise, (n, self.vf.condition_embedding_dim))

        if self.match_fn is not None:
            tmat = self.match_fn(src, tgt)
            src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
            src, tgt = src[src_ixs], tgt[tgt_ixs]

        self.vf_state, loss = self.vf_step_fn(
            rng_step_fn,
            self.vf_state,
            time,
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

    @property
    def cfg_enabled(self) -> bool:
        """Classifier-free guidance is active only when the model was TRAINED with condition
        dropout (``condition_dropout_prob > 0``); otherwise the unconditional field ``v_null``
        was never learned and guidance is undefined. Defaults off, so a model trained without
        CFG runs the original conditional-only path (no ``v_null`` computation)."""
        return float(getattr(self.vf, "condition_dropout_prob", 0.0)) > 0.0

    def _get_predict_fn(self, kwargs_frozen: frozen_dict.FrozenDict) -> Callable:  # type: ignore[type-arg]
        """Build (and cache) the jitted predict fn for a given set of diffeqsolve kwargs.

        ``params`` are threaded through as an argument rather than closed over, so the
        compiled function can be reused as the inference parameters change.
        """
        if kwargs_frozen in self._predict_fn_cache:
            return self._predict_fn_cache[kwargs_frozen]

        kwargs = dict(kwargs_frozen)
        # classifier-free guidance scale (not a diffrax arg → pop it). w=1 → plain conditional.
        guidance_scale = float(kwargs.pop("guidance_scale", 1.0))
        # Guidance (v_null path) only runs when CFG is enabled AND a non-trivial w is requested.
        # When CFG is disabled we keep the ORIGINAL conditional-only code — v_null is never
        # computed — regardless of any guidance_scale passed in.
        apply_guidance = self.cfg_enabled and guidance_scale != 1.0
        if guidance_scale != 1.0 and not self.cfg_enabled:
            print(f"[predict] guidance_scale (w) = {guidance_scale} IGNORED — model was not trained "
                  f"with CFG (condition_dropout_prob = 0); using plain conditional v_cond.", flush=True)
        elif apply_guidance:
            # fires once per unique predict-config (this fn is cached), not per predict call
            print(f"[predict] classifier-free guidance ON — guidance_scale (w) = {guidance_scale}", flush=True)

        if not apply_guidance:
            # original path: conditional velocity only, no v_null.
            def vf(t: jnp.ndarray, x: jnp.ndarray, args: tuple[Any, dict[str, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
                params, condition, encoder_noise = args
                return self.vf_state_inference.apply_fn(
                    {"params": params}, t, x, condition, encoder_noise, train=False
                )[0]
        else:
            def vf(t: jnp.ndarray, x: jnp.ndarray, args: tuple[Any, dict[str, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
                params, condition, encoder_noise = args
                v_cond = self.vf_state_inference.apply_fn(
                    {"params": params}, t, x, condition, encoder_noise, train=False
                )[0]
                # v = v_null + w·(v_cond − v_null): amplify the condition-specific velocity.
                v_null = self.vf_state_inference.apply_fn(
                    {"params": params}, t, x, condition, encoder_noise, train=False, force_uncond=True
                )[0]
                return v_null + guidance_scale * (v_cond - v_null)

        def solve_ode(
            params: Any, x: jnp.ndarray, condition: dict[str, jnp.ndarray], encoder_noise: jnp.ndarray
        ) -> jnp.ndarray:
            ode_term = diffrax.ODETerm(vf)
            result = diffrax.diffeqsolve(
                ode_term,
                t0=0.0,
                t1=1.0,
                y0=x,
                args=(params, condition, encoder_noise),
                **kwargs,
            )
            return result.ys[0]

        fn = jax.jit(jax.vmap(solve_ode, in_axes=[None, 0, None, None]))
        self._predict_fn_cache[kwargs_frozen] = fn
        return fn

    def _predict_jit(
        self, x: ArrayLike, condition: dict[str, ArrayLike], rng: jax.Array | None = None, **kwargs: Any
    ) -> ArrayLike:
        """See :meth:`OTFlowMatching.predict`."""
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault("stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5))
        kwargs_frozen = frozen_dict.freeze(kwargs)

        noise_dim = (1, self.vf.condition_embedding_dim)
        use_mean = rng is None or self.condition_encoder_mode == "deterministic"
        rng = utils.default_prng_key(rng)
        encoder_noise = jnp.zeros(noise_dim) if use_mean else jax.random.normal(rng, noise_dim)

        predict_fn = self._get_predict_fn(kwargs_frozen)
        return predict_fn(self.vf_state_inference.params, x, condition, encoder_noise)

    def predict(
        self,
        x: ArrayLike | dict[str, ArrayLike],
        condition: dict[str, ArrayLike] | dict[str, dict[str, ArrayLike]],
        rng: jax.Array | None = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> ArrayLike | dict[str, ArrayLike]:
        """Predict the translated source ``x`` under condition ``condition``.

        This function solves the ODE learnt with
        the :class:`~scaleflow.networks.ConditionalVelocityField`.

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
        show_progress
            Whether to show a progress bar when predicting over multiple conditions.
        kwargs
            Keyword arguments for :func:`diffrax.diffeqsolve`.

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

        if isinstance(x, dict):
            if show_progress:
                from tqdm import tqdm

                predict_fn = partial(self._predict_jit, rng=rng, **kwargs)
                results = {}
                keys = sorted(x.keys())
                for key in tqdm(keys, desc="Predicting conditions", leave=False):
                    results[key] = predict_fn(x[key], condition[key])
                return results
            else:
                return jax.tree.map(
                    partial(self._predict_jit, rng=rng, **kwargs),
                    x,
                    condition,  # type: ignore[attr-defined]
                )
        else:
            x_pred = self._predict_jit(x, condition, rng, **kwargs)
            return np.array(x_pred)

    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value
