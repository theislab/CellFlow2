from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from cellflow.training._callbacks import BaseCallback, ComputationCallback, LoggingCallback

from cellflow._types import ArrayLike

if TYPE_CHECKING:
    from scaleflow.solvers import GENOT, OTFlowMatching


__all__ = [
    "LearningRateMonitor",
    "CallbackRunner",
]


class LearningRateMonitor(LoggingCallback):
    """Callback to monitor and log learning rate during training

    Parameters
    ----------
    schedule
        The learning rate schedule function (e.g., from optax.warmup_cosine_decay_schedule).
        Should be a callable that takes a step count and returns a learning rate.

    Returns
    -------
        :obj:`None`
    """

    def __init__(self, schedule: Callable[[int], float]):
        self.schedule = schedule
        self.step_count = 0

    def on_train_begin(self) -> Any:
        """Called at the beginning of training"""
        self.step_count = 0

    def on_log_iteration(self, dict_to_log: dict[str, float], iteration: int = None, **_: Any) -> Any:
        """Called at each validation/log iteration to add learning rate to logs"""
        if iteration is not None:
            self.step_count = iteration
        lr = float(self.schedule(self.step_count))
        dict_to_log["learning_rate"] = lr
        return dict_to_log

    def on_train_end(self, dict_to_log: dict[str, float]) -> Any:
        """Called at the end of training"""
        pass


def _guidance_kwarg(fn: Callable, pred_data_by_w: Any) -> dict[str, Any]:
    """Return ``{"pred_data_by_w": ...}`` only if ``fn`` accepts that kwarg.

    Lets w-aware callbacks (classifier-free guidance sweep) receive the per-w
    predictions while leaving built-in callbacks (no such parameter) untouched.
    """
    if pred_data_by_w is None:
        return {}
    try:
        params = inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return {}
    accepts = "pred_data_by_w" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    return {"pred_data_by_w": pred_data_by_w} if accepts else {}


class CallbackRunner:
    """Runs a set of computational and logging callbacks in the :class:`~scaleflow.training.CellFlowTrainer`

    Parameters
    ----------
    callbacks
        List of callbacks to run. Callbacks should be of type
        :class:`~scaleflow.training.ComputationCallback` or
        :class:`~scaleflow.training.LoggingCallback`

    Returns
    -------
        :obj:`None`
    """

    def __init__(
        self,
        callbacks: Sequence[BaseCallback],
    ) -> None:
        self.computation_callbacks: list[ComputationCallback] = [
            c for c in callbacks if isinstance(c, ComputationCallback)
        ]
        self.logging_callbacks: list[LoggingCallback] = [c for c in callbacks if isinstance(c, LoggingCallback)]

        if len(self.computation_callbacks) == 0 & len(self.logging_callbacks) != 0:
            raise ValueError("No computation callbacks defined to compute metrics to log")

    def on_train_begin(self) -> Any:
        """Called at the beginning of training to initiate callbacks"""
        for callback in self.computation_callbacks:
            callback.on_train_begin()

        for callback in self.logging_callbacks:
            callback.on_train_begin()

    def on_log_iteration(
        self,
        valid_source_data: dict[str, dict[str, ArrayLike]],
        valid_data: dict[str, dict[str, ArrayLike]],
        pred_data: dict[str, dict[str, ArrayLike]],
        solver: OTFlowMatching | GENOT,
        additional_metrics: dict[str, Any] | None = None,
        iteration: int | None = None,
        pred_data_by_w: dict[float, dict[str, dict[str, ArrayLike]]] | None = None,
    ) -> dict[str, Any]:
        """Called at each validation/log iteration to run callbacks. First computes metrics with computation callbacks and then logs data with logging callbacks.

        Parameters
        ----------
        valid_source_data
            Source data in nested dictionary format with same keys as ``valid_true_data``
        valid_true_data
            Validation data in nested dictionary format with same keys as ``valid_pred_data``
        valid_pred_data
            Predicted data in nested dictionary format with same keys as ``valid_true_data``
        solver
            :class:`~scaleflow.solvers.OTFlowMatching` solver or :class:`~scaleflow.solvers.GENOT`
            solver with a conditional velocity field.
        additional_metrics
            Optional dictionary of metrics to include before computing validation metrics (e.g., train_loss)
        iteration
            Current training iteration number

        Returns
        -------
            ``dict_to_log``: Dictionary containing data to log
        """
        dict_to_log: dict[str, Any] = {}

        # Add additional metrics first
        if additional_metrics is not None:
            dict_to_log.update(additional_metrics)

        for callback in self.computation_callbacks:
            extra = _guidance_kwarg(callback.on_log_iteration, pred_data_by_w)
            results = callback.on_log_iteration(valid_source_data, valid_data, pred_data, solver, **extra)
            dict_to_log.update(results)

        for callback in self.logging_callbacks:
            callback.on_log_iteration(dict_to_log, iteration=iteration)  # type: ignore[call-arg]

        return dict_to_log

    def on_train_end(
        self,
        valid_source_data: dict[str, dict[str, ArrayLike]],
        valid_data: dict[str, dict[str, ArrayLike]],
        pred_data: dict[str, dict[str, ArrayLike]],
        solver: OTFlowMatching | GENOT,
        pred_data_by_w: dict[float, dict[str, dict[str, ArrayLike]]] | None = None,
    ) -> dict[str, Any]:
        """Called at the end of training to run callbacks. First computes metrics with computation callbacks and then logs data with logging callbacks.

        Parameters
        ----------
        valid_source_data
            Source data in nested dictionary format with same keys as ``valid_true_data``
        valid_true_data
            Validation data in nested dictionary format with same keys as ``valid_pred_data``
        valid_pred_data
            Predicted data in nested dictionary format with same keys as ``valid_true_data``
        solver
            :class:`~scaleflow.solvers.OTFlowMatching` solver or :class:`~scaleflow.solvers.GENOT`
            solver with a conditional velocity field.

        Returns
        -------
            ``dict_to_log``: Dictionary containing data to log
        """
        dict_to_log: dict[str, Any] = {}

        for callback in self.computation_callbacks:
            extra = _guidance_kwarg(callback.on_train_end, pred_data_by_w)
            results = callback.on_train_end(valid_source_data, valid_data, pred_data, solver, **extra)
            dict_to_log.update(results)

        for callback in self.logging_callbacks:
            callback.on_train_end(dict_to_log)  # type: ignore[call-arg]

        return dict_to_log
