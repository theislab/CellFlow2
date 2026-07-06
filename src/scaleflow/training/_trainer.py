import gc
from collections.abc import Sequence
from typing import Any, Literal

import jax
import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from scaleflow.data import SamplerABC
from scaleflow.solvers import _eqm, _genot, _otfm
from scaleflow.training._callbacks import BaseCallback, CallbackRunner


class CellFlowTrainer:
    """Trainer for the OTFM/GENOT/EqM solver with a conditional velocity field.

    Parameters
    ----------
        dataloader
            Data sampler.
        solver
            :class:`~scaleflow.solvers._otfm.OTFlowMatching`,
            :class:`~scaleflow.solvers._genot.GENOT`, or
            :class:`~scaleflow.solvers._eqm.EquilibriumMatching` solver with a conditional velocity field.
        predict_kwargs
            Keyword arguments for the prediction functions
            :func:`scaleflow.solvers._otfm.OTFlowMatching.predict`,
            :func:`scaleflow.solvers._genot.GENOT.predict`, or
            :func:`scaleflow.solvers._eqm.EquilibriumMatching.predict` used during validation.
        seed
            Random seed for subsampling validation data.

    Returns
    -------
        :obj:`None`
    """

    def __init__(
        self,
        solver: _otfm.OTFlowMatching | _genot.GENOT | _eqm.EquilibriumMatching,
        predict_kwargs: dict[str, Any] | None = None,
        seed: int = 0,
    ):
        if not isinstance(solver, (_otfm.OTFlowMatching | _genot.GENOT | _eqm.EquilibriumMatching)):
            raise NotImplementedError(
                f"Solver must be an instance of OTFlowMatching, GENOT, or EquilibriumMatching, got {type(solver)}"
            )

        self.solver = solver
        self.predict_kwargs = predict_kwargs or {}
        # Classifier-free guidance: optional list of guidance scales to evaluate at each
        # validation. When set (len > 1), the SAME sampled val batch is predicted once per w
        # so metrics are comparable across w. Popped out of predict_kwargs so it never reaches
        # solver.predict/diffeqsolve. The scalar predict_kwargs["guidance_scale"] (default 1.0)
        # is the baseline/positional w handed to the non-w-aware callbacks.
        gs = self.predict_kwargs.pop("guidance_scales", None)
        self.guidance_scales: list[float] = [float(w) for w in gs] if gs else []
        self.rng_subsampling = np.random.default_rng(seed)
        self.training_logs: dict[str, Any] = {}

    def _validation_step(
        self,
        val_data: dict[str, SamplerABC],
        mode: Literal["on_log_iteration", "on_train_end"] = "on_log_iteration",
    ) -> tuple[
        dict[str, dict[str, ArrayLike]],
        dict[str, dict[str, ArrayLike]],
        dict[str, dict[str, ArrayLike]],
        dict[float, dict[str, dict[str, ArrayLike]]],
    ]:
        """Compute predictions for validation data.

        Handles ValidationSampler format: {"source": dict, "condition": dict, "target": dict}
        where each dict maps condition_key -> data.

        Returns ``(valid_source_data, valid_true_data, valid_pred_data, pred_data_by_w)``.
        ``valid_pred_data`` are the predictions at the baseline guidance scale (positional,
        for w-agnostic callbacks). ``pred_data_by_w`` maps each guidance scale w →
        ``{val_key: {cond_key: pred}}`` from the SAME sampled batch, so metrics are comparable
        across w. In the default (no ``guidance_scales``) case this holds a single entry.
        """
        from functools import partial

        import jax

        base_w = float(self.predict_kwargs.get("guidance_scale", 1.0))
        # Only sweep guidance scales when the model is CFG-enabled; otherwise every w returns
        # the same conditional velocity, so collapse to a single (baseline) pass.
        cfg_on = getattr(self.solver, "cfg_enabled", False)
        ws = self.guidance_scales if (self.guidance_scales and cfg_on) else [base_w]

        valid_source_data: dict[str, dict[str, ArrayLike]] = {}
        valid_true_data: dict[str, dict[str, ArrayLike]] = {}
        pred_data_by_w: dict[float, dict[str, dict[str, ArrayLike]]] = {w: {} for w in ws}

        def _predict_kwargs_for(w: float) -> dict:
            kw = dict(self.predict_kwargs)
            kw["guidance_scale"] = w
            return kw

        # Add progress bar for validation
        print(f"\nStarting validation on {len(val_data)} dataset(s)...")
        if len(ws) > 1:
            print(f"  classifier-free guidance sweep over w = {ws}")
        val_pbar = tqdm(val_data.items(), desc="Validation", leave=True, total=len(val_data))
        for val_key, vdl in val_pbar:
            val_pbar.set_description(f"Validation ({val_key}) - sampling")
            # Initialize sampler if not already initialized
            if hasattr(vdl, "_initialized") and not vdl._initialized:
                vdl.init_sampler()
            batch = vdl.sample()  # Samplers use internal rng

            val_pbar.set_description(f"Validation ({val_key}) - extracting data")

            # Handle ValidationSampler format: {"source": dict, "condition": dict, "target": dict}
            if "source" in batch and "condition" in batch:
                src = batch["source"]  # dict mapping cond_key -> source cells
                condition = batch["condition"]  # dict mapping cond_key -> conditions
                true_tgt = batch.get("target", {})  # dict mapping cond_key -> target cells
                valid_source_data[val_key] = src
                valid_true_data[val_key] = true_tgt

                for w in ws:
                    val_pbar.set_description(
                        f"Validation ({val_key}) - predicting ({len(src)} conditions)"
                        + (f" w={w}" if len(ws) > 1 else "")
                    )
                    # Use jax.tree.map for efficient per-condition prediction
                    pred_data_by_w[w][val_key] = jax.tree.map(
                        partial(self.solver.predict, **_predict_kwargs_for(w)),
                        src,
                        condition,
                    )
            else:
                # Handle old format (single batch): {"src_cell_data", "tgt_cell_data", "condition"}
                src = batch["src_cell_data"]
                condition = batch.get("condition", None)
                true_tgt = batch["tgt_cell_data"]
                valid_source_data[val_key] = src
                valid_true_data[val_key] = true_tgt

                for w in ws:
                    val_pbar.set_description(
                        f"Validation ({val_key}) - predicting" + (f" w={w}" if len(ws) > 1 else "")
                    )
                    pred_data_by_w[w][val_key] = self.solver.predict(
                        src, condition=condition, **_predict_kwargs_for(w)
                    )

            val_pbar.set_description(f"Validation ({val_key}) - done")

        print("Validation complete!")
        valid_pred_data = pred_data_by_w.get(base_w, pred_data_by_w[ws[0]])
        return valid_source_data, valid_true_data, valid_pred_data, pred_data_by_w

    def _update_logs(self, logs: dict[str, Any]) -> None:
        """Update training logs."""
        for k, v in logs.items():
            if k not in self.training_logs:
                self.training_logs[k] = []
            self.training_logs[k].append(v)

    def train(
        self,
        dataloader: SamplerABC,
        num_iterations: int,
        valid_freq: int,
        valid_loaders: dict[str, SamplerABC] | None = None,
        monitor_metrics: Sequence[str] = [],
        callbacks: Sequence[BaseCallback] = [],
        log_every: int = 1000,
    ) -> _otfm.OTFlowMatching | _genot.GENOT | _eqm.EquilibriumMatching:
        """Trains the model.

        Parameters
        ----------
            dataloader
                Dataloader used. The dataloader is responsible for returning batches
                with appropriate 'task' field ('gex' or 'functional').
            num_iterations
                Number of iterations to train the model.
            valid_freq
                Frequency of validation.
            valid_loaders
                Valid loaders.
            callbacks
                Callback functions.
            monitor_metrics
                Metrics to monitor.

        Returns
        -------
            The trained model.
        """
        self.training_logs = {"loss": [], "loss_gex": [], "loss_functional": []}
        rng_jax = jax.random.PRNGKey(0)

        # Initiate callbacks
        valid_loaders = valid_loaders or {}
        crun = CallbackRunner(
            callbacks=callbacks,
        )
        crun.on_train_begin()

        pbar = tqdm(range(num_iterations))
        sampler = dataloader
        for it in pbar:
            rng_jax, rng_step_fn = jax.random.split(rng_jax, 2)

            # Sample batch (dataloader controls which task)
            batch = sampler.sample()
            loss = self.solver.step_fn(rng_step_fn, batch)

            # Track losses
            task = batch.get("task", "gex")
            self.training_logs["loss"].append(float(loss))
            self.training_logs[f"loss_{task}"].append(float(loss))

            if it % log_every == 0:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"train_loss": float(loss)})
                except ImportError:
                    pass

            if ((it - 1) % valid_freq == 0) and (it > 1):
                # Get predictions from validation data
                valid_source_data, valid_true_data, valid_pred_data, pred_data_by_w = self._validation_step(
                    valid_loaders, mode="on_log_iteration"
                )

                # Calculate mean losses
                mean_loss = np.mean(self.training_logs["loss"][-valid_freq:])
                additional_metrics = {"train_loss": mean_loss}

                # Add task-specific losses if available
                if self.training_logs["loss_gex"]:
                    mean_loss_gex = np.mean([l for l in self.training_logs["loss_gex"][-valid_freq:] if l is not None])
                    additional_metrics["train_loss_gex"] = mean_loss_gex
                if self.training_logs["loss_functional"]:
                    mean_loss_func = np.mean(
                        [l for l in self.training_logs["loss_functional"][-valid_freq:] if l is not None]
                    )
                    additional_metrics["train_loss_functional"] = mean_loss_func

                # Run callbacks with loss as additional metric
                metrics = crun.on_log_iteration(
                    valid_source_data,
                    valid_true_data,
                    valid_pred_data,
                    self.solver,
                    additional_metrics=additional_metrics,
                    iteration=it,
                    pred_data_by_w=pred_data_by_w,
                )
                self._update_logs(metrics)
                # Update progress bar
                postfix_dict = {metric: round(self.training_logs[metric][-1], 3) for metric in monitor_metrics}
                postfix_dict["train_loss"] = round(mean_loss, 3)
                if "train_loss_gex" in additional_metrics:
                    postfix_dict["loss_gex"] = round(additional_metrics["train_loss_gex"], 3)
                if "train_loss_functional" in additional_metrics:
                    postfix_dict["loss_func"] = round(additional_metrics["train_loss_functional"], 3)
                pbar.set_postfix(postfix_dict)

                # Free the validation predictions NOW. Otherwise these locals stay referenced
                # through the next valid_freq training steps and, crucially, while the *next*
                # _validation_step builds its own predictions — transiently holding
                # prev + new in RAM. With a guidance-scale sweep that is (N×prev)+(N×new),
                # which is what OOM-kills the second validation. The callbacks already returned
                # the scalar metrics, so the arrays are no longer needed.
                del valid_source_data, valid_true_data, valid_pred_data, pred_data_by_w
                gc.collect()

        if num_iterations > 0:
            valid_source_data, valid_true_data, valid_pred_data, pred_data_by_w = self._validation_step(
                valid_loaders, mode="on_train_end"
            )
            metrics = crun.on_train_end(
                valid_source_data, valid_true_data, valid_pred_data, self.solver,
                pred_data_by_w=pred_data_by_w,
            )
            self._update_logs(metrics)

        self.solver.is_trained = True
        return self.solver
