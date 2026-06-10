"""Validation/test callbacks and the r_squared_delta metric."""
from __future__ import annotations

import json
import os
from pathlib import Path

import cloudpickle
import jax
import numpy as np
from tqdm import tqdm

from scaleflow.training._callbacks import ComputationCallback
from scaleflow.metrics._metrics import (
    compute_r_squared,
    compute_e_distance_fast,
    compute_scalar_mmd,
)


def r_squared_delta(y_true, y_pred, source) -> float:
    """R² of the perturbation delta: R²(pred - control, gt - control)."""
    ctrl = np.asarray(source).mean(axis=0)
    return float(compute_r_squared(np.asarray(y_true) - ctrl, np.asarray(y_pred) - ctrl))


def _condition_metrics(y_true, y_pred, source) -> dict:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    return {
        "r_squared":  float(compute_r_squared(yt, yp)),
        "e_distance": float(compute_e_distance_fast(yt, yp)),
        "mmd":        float(compute_scalar_mmd(yt, yp)),
        "r_squared_delta": r_squared_delta(yt, yp, source) if source is not None else float("nan"),
    }


class ValMetricsLogger(ComputationCallback):
    """Logs pooled val metrics to JSON + wandb; returns per-dataset r²Δ for monitoring."""

    METRICS = ("r_squared", "e_distance", "mmd", "r_squared_delta")

    def __init__(self, save_path: str, valid_freq: int, wandb_run=None):
        self.save_path   = save_path
        self._valid_freq = valid_freq
        self._step       = 0
        self._wandb_run  = wandb_run

    def on_train_begin(self, *args, **kwargs) -> None:
        self._step = 0

    def _gather(self, valid_source_data, valid_true_data, valid_pred_data):
        per_ds: dict = {}
        for ds in valid_true_data:
            for cond_key, true_arr in valid_true_data[ds].items():
                pred_arr = valid_pred_data[ds].get(cond_key)
                if pred_arr is None:
                    continue
                src_arr = valid_source_data.get(ds, {}).get(cond_key)
                per_ds.setdefault(ds, []).append(_condition_metrics(true_arr, pred_arr, src_arr))
        return per_ds

    def _compute_and_save(self, valid_source_data, valid_true_data, valid_pred_data) -> dict:
        per_ds = self._gather(valid_source_data, valid_true_data, valid_pred_data)
        if not per_ds:
            return {}

        flat = [m for ms in per_ds.values() for m in ms]
        entry = {"step": self._step, "n_conditions": len(flat)}
        for k in self.METRICS:
            entry[k] = float(np.nanmean([m[k] for m in flat]))

        entries = []
        if os.path.exists(self.save_path):
            with open(self.save_path) as f:
                entries = json.load(f)
        entries.append(entry)
        with open(self.save_path, "w") as f:
            json.dump(entries, f, indent=2)

        print(f"    val  R²={entry['r_squared']:.4f}  ΔR²={entry['r_squared_delta']:.4f}  "
              f"E-dist={entry['e_distance']:.4f}  MMD={entry['mmd']:.4f}  (step {self._step})")
        if self._wandb_run is not None:
            self._wandb_run.log({f"val_{k}": entry[k] for k in self.METRICS})

        return {
            f"{ds}_r_squared_delta_mean": float(np.nanmean([m["r_squared_delta"] for m in ms]))
            for ds, ms in per_ds.items()
        }

    def on_log_iteration(self, valid_source_data, valid_true_data,
                         valid_pred_data, solver, **kwargs) -> dict:
        self._step += self._valid_freq
        return self._compute_and_save(valid_source_data, valid_true_data, valid_pred_data)

    def on_train_end(self, valid_source_data, valid_true_data,
                     valid_pred_data, solver, **kwargs) -> dict:
        return self._compute_and_save(valid_source_data, valid_true_data, valid_pred_data)


class BestModelCheckpoint(ComputationCallback):
    """Cloudpickle the solver whenever mean val R² improves."""

    def __init__(self, save_path: str, wandb_run=None):
        self.save_path  = save_path
        self.best_r2    = -np.inf
        self._wandb_run = wandb_run

    def on_train_begin(self, *args, **kwargs) -> None:
        self.best_r2 = -np.inf

    def on_log_iteration(self, valid_source_data, valid_true_data,
                         valid_pred_data, solver, **kwargs) -> dict:
        scores = []
        for ds in valid_true_data:
            for cond_key, true_arr in valid_true_data[ds].items():
                pred_arr = valid_pred_data[ds].get(cond_key)
                if pred_arr is None:
                    continue
                scores.append(compute_r_squared(np.asarray(true_arr), np.asarray(pred_arr)))
        if not scores:
            return {}
        mean_r2 = float(np.mean(scores))
        if mean_r2 > self.best_r2:
            self.best_r2 = mean_r2
            with open(self.save_path, "wb") as f:
                cloudpickle.dump(solver, f)
            print(f"    ✓ checkpoint saved  (val R²={mean_r2:.4f})")
        if self._wandb_run is not None:
            self._wandb_run.log({"best_val_r2": self.best_r2})
        return {"best_val_r2": self.best_r2}

    def on_train_end(self, valid_source_data, valid_true_data,
                     valid_pred_data, solver, **kwargs) -> dict:
        return self.on_log_iteration(valid_source_data, valid_true_data,
                                     valid_pred_data, solver)


def evaluate_test(solver, test_samplers: dict) -> dict:
    """Per-condition and aggregated test metrics for each dataset."""
    keys = list(ValMetricsLogger.METRICS)
    per_dataset: dict = {}
    all_per_condition: dict = {}

    for name, sampler in test_samplers.items():
        batch = sampler.sample(mode="on_train_end")
        src, cond, true = batch["source"], batch["condition"], batch["target"]

        print(f"  [{name}] predicting {len(src)} test conditions …")
        pred = jax.tree.map(solver.predict, src, cond)

        per_condition = {}
        for cond_key in tqdm(sorted(true.keys(), key=str), desc=f"  test metrics [{name}]"):
            src_arr = src.get(cond_key) if isinstance(src, dict) else None
            per_condition[str(cond_key)] = _condition_metrics(true[cond_key], pred[cond_key], src_arr)
            all_per_condition[f"{name}/{cond_key}"] = per_condition[str(cond_key)]

        per_dataset[name] = {
            "per_condition": per_condition,
            "aggregated": {k: float(np.nanmean([v[k] for v in per_condition.values()])) for k in keys},
        }

    aggregated = {k: float(np.nanmean([v[k] for v in all_per_condition.values()])) for k in keys}
    return {"per_dataset": per_dataset, "per_condition": all_per_condition, "aggregated": aggregated}


def save_logs(name: str, logs: dict, output_dir: Path) -> None:
    path = output_dir / f"{name}_training_logs.json"
    serialisable = {k: [float(v) for v in vals] for k, vals in logs.items() if vals}
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  logs saved  → {path}")
