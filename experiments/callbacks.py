"""Validation/test callbacks and metrics."""
from __future__ import annotations

import json
import os
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from scipy.stats import pearsonr
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


def nn_displacement_corr(y_true, y_pred, source, debug: bool = False) -> float:
    """Pearson r between per-cell displacement vectors (NN-matched true vs exact pred).

    For each src[i], finds nearest true cell, computes disp_true[i] = true[nn_i] - src[i]
    and disp_pred[i] = pred[i] - src[i], then returns Pearson r of all flattened vectors.
    """
    true = np.asarray(y_true)   # (n_tgt, d)
    pred = np.asarray(y_pred)   # (n_src, d)
    src  = np.asarray(source)   # (n_src, d)

    src_sq  = (src  ** 2).sum(axis=1)
    true_sq = (true ** 2).sum(axis=1)
    cross   = src @ true.T
    sq_dists = src_sq[:, None] + true_sq[None, :] - 2.0 * cross
    nn_idx  = sq_dists.argmin(axis=1)

    disp_true = true[nn_idx] - src
    disp_pred = pred          - src

    r, _ = pearsonr(disp_true.flatten(), disp_pred.flatten())

    if debug:
        unique_matched = len(np.unique(nn_idx))
        disp_true_norm = np.linalg.norm(disp_true, axis=1)
        disp_pred_norm = np.linalg.norm(disp_pred, axis=1)
        print(f"[nn_disp debug] n_src={len(src)}  n_tgt={len(true)}  "
              f"unique_true_matched={unique_matched}/{len(true)}  "
              f"mean|disp_true|={disp_true_norm.mean():.4f}  "
              f"mean|disp_pred|={disp_pred_norm.mean():.4f}  "
              f"ratio={disp_pred_norm.mean()/disp_true_norm.mean():.4f}  "
              f"r={r:.4f}", flush=True)

    return float(r)


def mean_nn_displacement_corr(valid_source_data, valid_true_data, valid_pred_data) -> float:
    scores = []
    for ds in valid_true_data:
        for cond_key, true_arr in valid_true_data[ds].items():
            pred_arr = valid_pred_data[ds].get(cond_key)
            src_arr  = valid_source_data.get(ds, {}).get(cond_key)
            if pred_arr is None or src_arr is None:
                continue
            scores.append(nn_displacement_corr(true_arr, pred_arr, src_arr))
    if not scores:
        return float("nan")
    return float(np.nanmean(scores))


def _condition_metrics(y_true, y_pred, source, debug: bool = False) -> dict:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    return {
        "r_squared":  float(compute_r_squared(yt, yp)),
        "e_distance": float(compute_e_distance_fast(yt, yp)),
        "mmd":        float(compute_scalar_mmd(yt, yp)),
        "r_squared_delta": r_squared_delta(yt, yp, source) if source is not None else float("nan"),
        "nn_displacement_corr": nn_displacement_corr(yt, yp, source, debug=debug) if source is not None else float("nan"),
    }


class ValMetricsLogger(ComputationCallback):
    """Logs pooled val metrics to JSON + wandb; returns per-dataset nn_displacement_corr for monitoring."""

    METRICS = ("r_squared", "e_distance", "mmd", "r_squared_delta", "nn_displacement_corr")

    def __init__(self, save_path: str, valid_freq: int, wandb_run=None, debug: bool = False):
        self.save_path   = save_path
        self._valid_freq = valid_freq
        self._step       = 0
        self._wandb_run  = wandb_run
        self._debug      = debug

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
                per_ds.setdefault(ds, []).append(
                    _condition_metrics(true_arr, pred_arr, src_arr, debug=self._debug)
                )
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
              f"nn_disp_corr={entry['nn_displacement_corr']:.4f}  "
              f"E-dist={entry['e_distance']:.4f}  MMD={entry['mmd']:.4f}  (step {self._step})")
        if self._wandb_run is not None:
            self._wandb_run.log({f"val_{k}": entry[k] for k in self.METRICS})

        return {
            f"{ds}_nn_displacement_corr": float(np.nanmean([m["nn_displacement_corr"] for m in ms]))
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
    """Save solver params with orbax whenever mean val nn_displacement_corr improves."""

    def __init__(self, save_path: str, wandb_run=None):
        # save_path is used as a directory for orbax (e.g. .../model_X_best_ckpt)
        self.save_path    = Path(save_path)
        self.best_score   = -np.inf
        self._wandb_run   = wandb_run
        self._ckptr       = ocp.PyTreeCheckpointer()

    def on_train_begin(self, *args, **kwargs) -> None:
        self.best_score = -np.inf

    def on_log_iteration(self, valid_source_data, valid_true_data,
                         valid_pred_data, solver, **kwargs) -> dict:
        score = mean_nn_displacement_corr(valid_source_data, valid_true_data, valid_pred_data)
        if np.isnan(score):
            return {}
        if score > self.best_score:
            self.best_score = score
            import shutil
            if self.save_path.exists():
                shutil.rmtree(self.save_path)
            self._ckptr.save(self.save_path, solver.vf_state.params)
            print(f"    ✓ checkpoint saved  (val nn_disp_corr={score:.4f})")
        if self._wandb_run is not None:
            self._wandb_run.log({"best_val_nn_disp_corr": self.best_score})
        return {"best_val_nn_disp_corr": self.best_score}

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
