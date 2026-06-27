"""Validation/test callbacks and metrics."""
from __future__ import annotations

import json
import os
from pathlib import Path

import shutil

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


def pearson_r_delta(y_true, y_pred, source) -> float:
    """Pearson r between mean perturbation deltas: corr(mean(pred)-ctrl, mean(true)-ctrl)."""
    ctrl = np.asarray(source).mean(axis=0)
    delta_true = np.asarray(y_true).mean(axis=0) - ctrl
    delta_pred = np.asarray(y_pred).mean(axis=0) - ctrl
    r, _ = pearsonr(delta_true, delta_pred)
    return float(r)


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
        "r_squared_delta":    r_squared_delta(yt, yp, source)    if source is not None else float("nan"),
        "pearson_r_delta":    pearson_r_delta(yt, yp, source)    if source is not None else float("nan"),
        "nn_displacement_corr": nn_displacement_corr(yt, yp, source, debug=debug) if source is not None else float("nan"),
    }


class ValMetricsLogger(ComputationCallback):
    """Logs pooled val metrics to JSON + wandb; returns per-dataset nn_displacement_corr for monitoring."""

    METRICS = ("r_squared", "e_distance", "mmd", "r_squared_delta", "pearson_r_delta", "nn_displacement_corr")

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
              f"Δr={entry['pearson_r_delta']:.4f}  "
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


# Metrics where higher = better. All others (e_distance, mmd) → lower = better.
_MAXIMIZE_METRICS = {"r_squared", "r_squared_delta", "pearson_r_delta", "nn_displacement_corr"}


def _solver_params(solver) -> dict:
    """Extract all inference-relevant params from a solver into a pytree dict."""
    p = {
        "vf_params":           solver.vf_state.params,
        "vf_inference_params": solver.vf_state_inference.params,
    }
    if hasattr(solver, "phenotype_state") and solver.phenotype_state is not None:
        p["phenotype_params"] = solver.phenotype_state.params
    return p


def restore_solver_params(solver, params: dict) -> None:
    """Restore orbax-loaded params dict back into a solver in-place."""
    solver.vf_state           = solver.vf_state.replace(params=params["vf_params"])
    solver.vf_state_inference = solver.vf_state_inference.replace(params=params["vf_inference_params"])
    if "phenotype_params" in params and hasattr(solver, "phenotype_state") and solver.phenotype_state is not None:
        solver.phenotype_state = solver.phenotype_state.replace(params=params["phenotype_params"])


class BestModelCheckpoint(ComputationCallback):
    """Save solver params with orbax whenever the chosen val metric improves."""

    def __init__(self, save_path: str, wandb_run=None, metric: str = "nn_displacement_corr"):
        self.save_path  = Path(save_path)  # orbax writes a directory here
        self._metric    = metric
        self._maximize  = metric in _MAXIMIZE_METRICS
        self.best_score = -np.inf if self._maximize else np.inf
        self._wandb_run = wandb_run
        self._ckptr     = ocp.PyTreeCheckpointer()

    def on_train_begin(self, *args, **kwargs) -> None:
        self.best_score = -np.inf if self._maximize else np.inf

    def _compute_score(self, valid_source_data, valid_true_data, valid_pred_data) -> float:
        scores = []
        for ds in valid_true_data:
            for cond_key, true_arr in valid_true_data[ds].items():
                pred_arr = valid_pred_data[ds].get(cond_key)
                if pred_arr is None:
                    continue
                src_arr = valid_source_data.get(ds, {}).get(cond_key)
                m = _condition_metrics(true_arr, pred_arr, src_arr)
                scores.append(m[self._metric])
        return float(np.nanmean(scores)) if scores else float("nan")

    def on_log_iteration(self, valid_source_data, valid_true_data,
                         valid_pred_data, solver, **kwargs) -> dict:
        score = self._compute_score(valid_source_data, valid_true_data, valid_pred_data)
        if np.isnan(score):
            return {}
        is_better = score > self.best_score if self._maximize else score < self.best_score
        if is_better:
            self.best_score = score
            if self.save_path.exists():
                shutil.rmtree(self.save_path)
            self._ckptr.save(str(self.save_path), _solver_params(solver))
            print(f"    ✓ checkpoint saved  (val {self._metric}={score:.4f})")
        wandb_key = f"best_val_{self._metric}"
        if self._wandb_run is not None:
            self._wandb_run.log({wandb_key: self.best_score})
        return {wandb_key: self.best_score}

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


class ReconMetricsLogger(ComputationCallback):
    """Gene-space R²δ and Pearson-rδ via a ReconDecoder.

    Decodes predicted latent → genes, compares mean perturbation delta
    (perturbed − ctrl) against ground truth from the raw h5ad.
    Condition keys are tuples (*src_dist_keys, *tgt_dist_keys) as strings.
    """

    def __init__(
        self,
        decoder,
        adata,
        condition_obs_keys: list[str],
        cell_line_obs_key: str,
        control_obs_key: str = "control",
        log_dose_obs_key: str | None = None,
        valid_freq: int = 1,
        wandb_run=None,
    ):
        self._decoder = decoder
        self._adata = adata
        self._cond_keys = condition_obs_keys
        self._cl_key = cell_line_obs_key
        self._ctrl_key = control_obs_key
        # obs column whose condition-key value is log1p(raw): match numerically, not by string
        self._log_dose_key = log_dose_obs_key
        self._valid_freq = valid_freq
        self._wandb_run = wandb_run
        self._step = 0
        self._ctrl_cache: dict[str, np.ndarray] = {}
        # precompute column indices for decoder's var_names
        var_names = decoder.var_names
        if var_names is not None:
            adata_vars = list(adata.var_names)
            self._var_idx = np.array([adata_vars.index(v) for v in var_names], dtype=np.intp)
        else:
            self._var_idx = None

    def on_train_begin(self, *args, **kwargs) -> None:
        self._step = 0
        self._ctrl_cache = {}

    @staticmethod
    def _to_dense(X) -> np.ndarray:
        return np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)

    def _get_true_genes(self, cond_key: tuple) -> np.ndarray | None:
        obs = self._adata.obs
        mask = np.ones(len(obs), dtype=bool)
        for col, val in zip(self._cond_keys, cond_key):
            if col == self._log_dose_key:
                # condition key holds log1p(dose); h5ad stores raw dose → match numerically
                mask &= np.isclose(
                    np.log1p(obs[col].astype(float).values), float(val), atol=1e-4
                )
            else:
                mask &= obs[col].astype(str) == str(val)
        if mask.sum() == 0:
            return None
        X = self._to_dense(self._adata[mask].X)
        return X[:, self._var_idx] if self._var_idx is not None else X

    def _get_ctrl_genes(self, cond_key: tuple) -> np.ndarray | None:
        cl_idx = self._cond_keys.index(self._cl_key)
        cell_line = str(cond_key[cl_idx])
        if cell_line not in self._ctrl_cache:
            obs = self._adata.obs
            mask = obs[self._ctrl_key].astype(bool) & (obs[self._cl_key].astype(str) == cell_line)
            if mask.sum() == 0:
                return None
            X = self._to_dense(self._adata[mask].X)
            self._ctrl_cache[cell_line] = X[:, self._var_idx] if self._var_idx is not None else X
        return self._ctrl_cache[cell_line]

    def _compute(self, valid_source_data, valid_true_data, valid_pred_data) -> dict:
        r2_deltas, pearson_deltas = [], []
        for ds in valid_pred_data:
            for cond_key, pred_latent in valid_pred_data[ds].items():
                true_genes = self._get_true_genes(cond_key)
                ctrl_genes = self._get_ctrl_genes(cond_key)
                if true_genes is None or ctrl_genes is None:
                    continue
                pred_genes = self._decoder.decode(np.asarray(pred_latent, dtype=np.float32))
                ctrl_mean = ctrl_genes.mean(axis=0)
                delta_true = true_genes.mean(axis=0) - ctrl_mean
                delta_pred = pred_genes.mean(axis=0) - ctrl_mean
                r, _ = pearsonr(delta_true, delta_pred)
                r2_deltas.append(float(r ** 2))
                pearson_deltas.append(float(r))

        if not r2_deltas:
            return {}

        out = {
            "val_recon_r2_delta":       float(np.nanmean(r2_deltas)),
            "val_recon_pearson_r_delta": float(np.nanmean(pearson_deltas)),
        }
        print(f"    recon  R²δ={out['val_recon_r2_delta']:.4f}  "
              f"rδ={out['val_recon_pearson_r_delta']:.4f}  (step {self._step})")
        if self._wandb_run is not None:
            self._wandb_run.log(out)
        return out

    def on_log_iteration(self, valid_source_data, valid_true_data,
                         valid_pred_data, solver, **kwargs) -> dict:
        self._step += self._valid_freq
        return self._compute(valid_source_data, valid_true_data, valid_pred_data)

    def on_train_end(self, valid_source_data, valid_true_data,
                     valid_pred_data, solver, **kwargs) -> dict:
        return self._compute(valid_source_data, valid_true_data, valid_pred_data)


def load_recon_decoder(dir_path: str):
    """Load a ReconDecoder from orbax params + metadata.json (JAX-version-independent)."""
    from scaleflow.model._recon import Autoencoder, Decoder, ReconDecoder

    dir_path = Path(dir_path)
    with open(dir_path / "metadata.json") as f:
        meta = json.load(f)

    mode = meta.get("mode", "pretrained")
    if mode == "ae":
        module = Autoencoder(
            gene_dim=int(meta["gene_dim"]),
            latent_dim=int(meta["latent_dim"]),
            encoder_hidden=tuple(int(x) for x in meta["encoder_hidden"]),
            decoder_hidden=tuple(int(x) for x in meta["decoder_hidden"]),
            dropout_rate=float(meta.get("dropout_rate", 0.0)),
        )
        input_dim = int(meta["gene_dim"])
    else:
        module = Decoder(
            output_dim=int(meta["gene_dim"]),
            hidden_dims=tuple(int(x) for x in meta["decoder_hidden"]),
            dropout_rate=float(meta.get("dropout_rate", 0.0)),
        )
        input_dim = int(
            meta.get("input_dim") or meta.get("pretrained_dim") or meta.get("latent_dim")
        )

    dummy = np.ones((1, input_dim), dtype=np.float32)
    params_struct = module.init(jax.random.PRNGKey(0), dummy, training=False)["params"]
    params = ocp.PyTreeCheckpointer().restore(str(dir_path / "params"), item=params_struct)
    meta["input_dim"] = input_dim
    return ReconDecoder(module=module, params=params, metadata=meta)


def save_logs(name: str, logs: dict, output_dir: Path) -> None:
    path = output_dir / f"{name}_training_logs.json"
    serialisable = {k: [float(v) for v in vals] for k, vals in logs.items() if vals}
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  logs saved  → {path}")
