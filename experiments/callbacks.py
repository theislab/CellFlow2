"""Validation/test callbacks and metrics."""
from __future__ import annotations

import ast
import json
import os
import shutil
from functools import partial
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from cellflow.metrics import compute_e_distance_fast, compute_scalar_mmd
from cellflow.training import ComputationCallback
from scipy.stats import pearsonr, ttest_ind
from tqdm import tqdm


def pearson_r_delta(y_true, y_pred, source) -> float:
    """Pearson r between mean perturbation deltas: corr(mean(pred)-ctrl, mean(true)-ctrl)."""
    ctrl = np.asarray(source).mean(axis=0)
    delta_true = np.asarray(y_true).mean(axis=0) - ctrl
    delta_pred = np.asarray(y_pred).mean(axis=0) - ctrl
    r, _ = pearsonr(delta_true, delta_pred)
    return float(r)


def pearson_r(y_true, y_pred) -> float:
    """Pearson r between mean profiles (no control): corr(mean(true), mean(pred))."""
    r, _ = pearsonr(np.asarray(y_true).mean(axis=0), np.asarray(y_pred).mean(axis=0))
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


# ── differential-expression (DE) metrics ─────────────────────────────────────
def _bh_fdr_reject(pvals, alpha: float) -> np.ndarray:
    """Benjamini–Hochberg: boolean 'is significant' mask at FDR ≤ alpha."""
    p = np.nan_to_num(np.asarray(pvals, dtype=float), nan=1.0)
    n = p.size
    reject = np.zeros(n, dtype=bool)
    if n == 0:
        return reject
    order = np.argsort(p)
    ranked = p[order]
    passed = ranked <= alpha * (np.arange(1, n + 1) / n)
    if passed.any():
        cutoff = ranked[np.nonzero(passed)[0].max()]   # largest p below its BH threshold
        reject = p <= cutoff
    return reject


def _de_vs_control(perturbed, control, fdr: float):
    """Per-gene (logFC, BH-significant mask) for `perturbed` vs `control`.

    Expression is log1p-normalized, so logFC = mean(perturbed) − mean(control).
    Significance is a per-gene Welch t-test (perturbed vs control), BH-corrected.
    """
    perturbed = np.asarray(perturbed)
    control = np.asarray(control)
    logfc = perturbed.mean(axis=0) - control.mean(axis=0)
    if perturbed.shape[0] < 2 or control.shape[0] < 2:
        return logfc, np.zeros(logfc.shape, dtype=bool)   # too few cells to test
    _, pvals = ttest_ind(perturbed, control, axis=0, equal_var=False)
    return logfc, _bh_fdr_reject(pvals, fdr)


def de_metrics(y_true, y_pred, source, source_pred=None, fdr: float = 0.05) -> dict:
    """DE-based condition metrics (all in [0, 1], higher = better). ALWAYS gene-space:
    callers must pass gene-expression arrays (decode a latent to genes first).

    Significant DE genes are called (FDR ≤ `fdr`, default 0.05) separately for the
    true perturbed cells and the predicted perturbed cells, each vs the control
    distribution; genes are ranked by |logFC|. ``source_pred`` (optional) is the control
    used for the PREDICTED side — pass decoded control cells so the decoder offset cancels
    (decode(pred) − decode(ctrl)); defaults to ``source`` (the observed control).

    DEOver (DE overlap)     — |predTop-N ∩ trueSig| / N, where N = #true-significant
                              genes and predTop-N = the N predicted-significant genes
                              with the largest |logFC|. Recovery of the top true DE genes.
    DEPrec (DE precision)   — |predSig ∩ trueSig| / |predSig|: of predicted-significant
                              genes, the fraction that are truly significant.
    DirAgr (direction agr.) — over predSig ∩ trueSig, the fraction whose logFC sign
                              matches the true logFC sign (up/down-regulation agreement).
    """
    ctrl_t = np.asarray(source)
    ctrl_p = np.asarray(source_pred) if source_pred is not None else ctrl_t
    logfc_t, sig_t = _de_vs_control(y_true, ctrl_t, fdr)
    logfc_p, sig_p = _de_vs_control(y_pred, ctrl_p, fdr)

    true_genes = np.nonzero(sig_t)[0]
    pred_genes = np.nonzero(sig_p)[0]
    N, nP = int(true_genes.size), int(pred_genes.size)
    inter = sig_t & sig_p

    if N == 0:                                    # no true DE genes → overlap undefined
        de_over = float("nan")
    else:                                         # top-N predicted (by |logFC|) ∩ true set
        pred_topN = pred_genes[np.argsort(-np.abs(logfc_p[pred_genes]))][:N]
        de_over = float(np.isin(pred_topN, true_genes).sum() / N)

    de_prec = float(inter.sum() / nP) if nP > 0 else float("nan")
    dir_agr = (float((np.sign(logfc_t[inter]) == np.sign(logfc_p[inter])).mean())
               if inter.any() else float("nan"))

    return {"de_overlap": de_over, "de_precision": de_prec, "dir_agreement": dir_agr}


_DE_KEYS = ("de_overlap", "de_precision", "dir_agreement")
_DE_NAN = {k: float("nan") for k in _DE_KEYS}


def _condition_metrics(y_true, y_pred, source, debug: bool = False, compute_de: bool = True) -> dict:
    """Per-condition metrics on the model's OUTPUT space. DE metrics are only meaningful
    in gene space, so `compute_de` must be False for latent-output runs (the gene-space DE
    is computed instead in ReconMetricsLogger on decoded genes).
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    m = {
        "pearson_r":  pearson_r(yt, yp),
        "e_distance": float(compute_e_distance_fast(yt, yp)),
        "mmd":        float(compute_scalar_mmd(yt, yp)),
        "pearson_r_delta":    pearson_r_delta(yt, yp, source)    if source is not None else float("nan"),
        "nn_displacement_corr": nn_displacement_corr(yt, yp, source, debug=debug) if source is not None else float("nan"),
    }
    if compute_de:
        m.update(de_metrics(yt, yp, source) if source is not None else _DE_NAN)
    return m


class ValMetricsLogger(ComputationCallback):
    """Logs pooled val metrics to JSON + wandb; returns per-dataset nn_displacement_corr for monitoring."""

    METRICS = ("pearson_r", "e_distance", "mmd", "pearson_r_delta", "nn_displacement_corr",
               "de_overlap", "de_precision", "dir_agreement")

    def __init__(self, save_path: str, valid_freq: int, wandb_run=None, debug: bool = False,
                 compute_de: bool = True):
        self.save_path   = save_path
        self._valid_freq = valid_freq
        self._step       = 0
        self._wandb_run  = wandb_run
        self._debug      = debug
        # DE metrics are gene-space only; for latent-output runs pass compute_de=False
        # (the gene-space DE is logged by ReconMetricsLogger on decoded genes instead).
        self._compute_de = compute_de
        self.METRICS = ValMetricsLogger.METRICS if compute_de else tuple(
            m for m in ValMetricsLogger.METRICS if m not in _DE_KEYS
        )

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
                    _condition_metrics(true_arr, pred_arr, src_arr, debug=self._debug,
                                       compute_de=self._compute_de)
                )
        return per_ds

    # guidance scale picked (by this primary metric) when a CFG w-sweep is active
    _PRIMARY = "pearson_r_delta"

    def _entry(self, per_ds: dict) -> dict:
        """Mean+median of each metric across all conditions."""
        flat = [m for ms in per_ds.values() for m in ms]
        entry = {"n_conditions": len(flat)}
        for k in self.METRICS:
            vals = [m[k] for m in flat]
            entry[k] = float(np.nanmean(vals))
            entry[f"{k}_median"] = float(np.nanmedian(vals))
        return entry

    def _monitor(self, per_ds: dict) -> dict:
        return {
            f"{ds}_nn_displacement_corr": float(np.nanmean([m["nn_displacement_corr"] for m in ms]))
            for ds, ms in per_ds.items()
        }

    def _print_entry(self, entry: dict, tag: str = "") -> None:
        print(f"    val{tag}  r={entry['pearson_r']:.4f}  "
              f"Δr={entry['pearson_r_delta']:.4f}  "
              f"nn_disp_corr={entry['nn_displacement_corr']:.4f}  "
              f"E-dist={entry['e_distance']:.4f}  MMD={entry['mmd']:.4f}  (step {self._step})")

    def _save_entry(self, entry: dict) -> None:
        entries = []
        if os.path.exists(self.save_path):
            with open(self.save_path) as f:
                entries = json.load(f)
        entries.append(entry)
        with open(self.save_path, "w") as f:
            json.dump(entries, f, indent=2)

    @staticmethod
    def _score(entry: dict, key: str) -> float:
        s = entry.get(key)
        return -np.inf if (s is None or np.isnan(s)) else float(s)

    def _compute_and_save(self, valid_source_data, valid_true_data, valid_pred_data,
                          pred_data_by_w=None) -> dict:
        multi = pred_data_by_w is not None and len(pred_data_by_w) > 1

        # ── single guidance scale (default behaviour) ──
        if not multi:
            per_ds = self._gather(valid_source_data, valid_true_data, valid_pred_data)
            if not per_ds:
                return {}
            entry = self._entry(per_ds)
            entry["step"] = self._step
            self._save_entry(entry)
            self._print_entry(entry)
            if self._wandb_run is not None:
                log = {f"val_{k}": entry[k] for k in self.METRICS}
                log.update({f"val_{k}_median": entry[f"{k}_median"] for k in self.METRICS})
                self._wandb_run.log(log)
            return self._monitor(per_ds)

        # ── classifier-free guidance sweep: evaluate every w, log all, graph the best ──
        per_w: dict = {}  # w -> (entry, per_ds)
        for w, pred in pred_data_by_w.items():
            per_ds = self._gather(valid_source_data, valid_true_data, pred)
            if per_ds:
                per_w[w] = (self._entry(per_ds), per_ds)
        if not per_w:
            return {}

        best_w = max(per_w, key=lambda w: self._score(per_w[w][0], self._PRIMARY))
        best_entry, best_per_ds = per_w[best_w]

        wandb_log: dict = {}
        for w in sorted(per_w):
            e = per_w[w][0]
            self._print_entry(e, tag=f" [w={w}]")
            for k in self.METRICS:                       # all-w curves
                wandb_log[f"val_{k}__w{w}"] = e[k]
                wandb_log[f"val_{k}_median__w{w}"] = e[f"{k}_median"]
        for k in self.METRICS:                           # best-w → the default (graphed) val_* keys
            wandb_log[f"val_{k}"] = best_entry[k]
            wandb_log[f"val_{k}_median"] = best_entry[f"{k}_median"]
        wandb_log["val_best_w"] = float(best_w)
        print(f"    val  → best w={best_w} by {self._PRIMARY} "
              f"(Δr={best_entry[self._PRIMARY]:.4f})  (step {self._step})")
        if self._wandb_run is not None:
            self._wandb_run.log(wandb_log)

        save_entry = dict(best_entry)
        save_entry["step"] = self._step
        save_entry["best_w"] = float(best_w)
        save_entry["per_w"] = {str(w): per_w[w][0] for w in sorted(per_w)}
        self._save_entry(save_entry)
        return self._monitor(best_per_ds)

    def on_log_iteration(self, valid_source_data, valid_true_data,
                         valid_pred_data, solver, pred_data_by_w=None, **kwargs) -> dict:
        self._step += self._valid_freq
        return self._compute_and_save(valid_source_data, valid_true_data, valid_pred_data, pred_data_by_w)

    def on_train_end(self, valid_source_data, valid_true_data,
                     valid_pred_data, solver, pred_data_by_w=None, **kwargs) -> dict:
        return self._compute_and_save(valid_source_data, valid_true_data, valid_pred_data, pred_data_by_w)


# Metrics where higher = better. All others (e_distance, mmd) → lower = better.
_MAXIMIZE_METRICS = {"pearson_r", "pearson_r_delta", "nn_displacement_corr",
                     "de_overlap", "de_precision", "dir_agreement"}


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
        self.best_w     = None   # guidance scale that achieved best_score (CFG sweep only)
        self._wandb_run = wandb_run
        self._ckptr     = ocp.PyTreeCheckpointer()

    def on_train_begin(self, *args, **kwargs) -> None:
        self.best_score = -np.inf if self._maximize else np.inf
        self.best_w     = None

    def _score_pred(self, valid_source_data, valid_true_data, valid_pred_data) -> float:
        scores = []
        for ds in valid_true_data:
            for cond_key, true_arr in valid_true_data[ds].items():
                pred_arr = valid_pred_data[ds].get(cond_key)
                if pred_arr is None:
                    continue
                src_arr = valid_source_data.get(ds, {}).get(cond_key)
                m = _condition_metrics(true_arr, pred_arr, src_arr, compute_de=self._metric in _DE_KEYS)
                scores.append(m[self._metric])
        return float(np.nanmean(scores)) if scores else float("nan")

    def on_log_iteration(self, valid_source_data, valid_true_data,
                         valid_pred_data, solver, pred_data_by_w=None, **kwargs) -> dict:
        # Guidance is inference-only, so the saved params are identical for every w; the w
        # sweep only changes the SCORE used to decide "is this iterate the best". We pick the
        # iterate that scores best at its own optimal guidance and record that w.
        eval_w = None
        if pred_data_by_w is not None and len(pred_data_by_w) > 1:
            w_scores = {w: self._score_pred(valid_source_data, valid_true_data, pred)
                        for w, pred in pred_data_by_w.items()}
            w_scores = {w: s for w, s in w_scores.items() if not np.isnan(s)}
            if not w_scores:
                return {}
            eval_w = (max if self._maximize else min)(w_scores, key=w_scores.get)
            score = w_scores[eval_w]
        else:
            score = self._score_pred(valid_source_data, valid_true_data, valid_pred_data)
            if np.isnan(score):
                return {}

        is_better = score > self.best_score if self._maximize else score < self.best_score
        if is_better:
            self.best_score = score
            self.best_w = eval_w
            if self.save_path.exists():
                shutil.rmtree(self.save_path)
            self._ckptr.save(str(self.save_path), _solver_params(solver))
            w_msg = f" @ w={eval_w}" if eval_w is not None else ""
            print(f"    ✓ checkpoint saved  (val {self._metric}={score:.4f}{w_msg})")
        wandb_key = f"best_val_{self._metric}"
        out = {wandb_key: self.best_score}
        if self.best_w is not None:
            out["best_val_w"] = float(self.best_w)
        if self._wandb_run is not None:
            self._wandb_run.log(out)
        return out

    def on_train_end(self, valid_source_data, valid_true_data,
                     valid_pred_data, solver, pred_data_by_w=None, **kwargs) -> dict:
        return self.on_log_iteration(valid_source_data, valid_true_data,
                                     valid_pred_data, solver, pred_data_by_w=pred_data_by_w)


def _test_guidance_plan(predict_kwargs: dict | None) -> tuple[dict, list[float], float]:
    """Resolve test predict kwargs + the list of guidance scales to evaluate.

    Returns ``(base_pk, ws, base_w)`` where ``base_pk`` is the diffrax kwargs with the CFG
    w-sweep list stripped (so ``guidance_scale`` is set per w), ``ws`` is the list of scales
    to evaluate (the same ``guidance_scales`` used at validation, else the single configured
    ``guidance_scale``), and ``base_w`` is the scale whose metrics fill the default keys.
    """
    pk_all = dict(predict_kwargs or {})
    gs = pk_all.pop("guidance_scales", None)
    base_pk = dict(pk_all)
    base_w = float(base_pk.get("guidance_scale", 1.0))
    ws = [float(w) for w in gs] if gs else [base_w]
    return base_pk, ws, base_w


def evaluate_test(solver, test_samplers: dict, predict_kwargs: dict | None = None,
                  compute_de: bool = True) -> dict:
    """Per-condition and aggregated test metrics for each dataset.

    When ``predict_kwargs`` carries a ``guidance_scales`` list (the CFG w-sweep used at
    validation), the SAME sampled test batch is predicted once per w. The returned dict holds
    the default-w (config ``guidance_scale``) results plus ``per_w_aggregated`` = {w: aggregated}
    so every w can be plotted; no best-w is selected. ``compute_de`` False for latent-output
    runs (gene-space DE is computed by ReconMetricsLogger on decoded genes instead).
    """
    base_pk, ws, base_w = _test_guidance_plan(predict_kwargs)
    # skip the w-sweep for non-CFG models: every w gives the same conditional prediction.
    if not getattr(solver, "cfg_enabled", False):
        ws = [base_w]
    keys = [k for k in ValMetricsLogger.METRICS if compute_de or k not in _DE_KEYS]

    # sample each dataset ONCE so all w are compared on the same cells/conditions
    batches = {}
    for name, sampler in test_samplers.items():
        b = sampler.sample(mode="on_train_end")
        batches[name] = (b["source"], b["condition"], b["target"])

    if len(ws) > 1:
        print(f"  test classifier-free guidance sweep over w = {ws}")

    per_w_result: dict = {}
    for w in ws:
        pkw = dict(base_pk)
        pkw["guidance_scale"] = w
        per_dataset: dict = {}
        all_per_condition: dict = {}
        for name, (src, cond, true) in batches.items():
            print(f"  [{name}] predicting {len(src)} test conditions …"
                  + (f" (w={w})" if len(ws) > 1 or w != 1.0 else ""))
            pred = jax.tree.map(partial(solver.predict, **pkw), src, cond)
            per_condition = {}
            for cond_key in tqdm(sorted(true.keys(), key=str), desc=f"  test metrics [{name}] w={w}"):
                src_arr = src.get(cond_key) if isinstance(src, dict) else None
                per_condition[str(cond_key)] = _condition_metrics(true[cond_key], pred[cond_key], src_arr, compute_de=compute_de)
                all_per_condition[f"{name}/{cond_key}"] = per_condition[str(cond_key)]
            per_dataset[name] = {
                "per_condition": per_condition,
                "aggregated": {k: float(np.nanmean([v[k] for v in per_condition.values()])) for k in keys},
            }
        aggregated = {k: float(np.nanmean([v[k] for v in all_per_condition.values()])) for k in keys}
        per_w_result[w] = {
            "per_dataset": per_dataset,
            "per_condition": all_per_condition,
            "aggregated": aggregated,
        }

    default = per_w_result.get(base_w, per_w_result[ws[0]])
    if len(ws) > 1:
        default["per_w_aggregated"] = {w: per_w_result[w]["aggregated"] for w in ws}
    return default


class ReconMetricsLogger(ComputationCallback):
    """Gene-space Pearson-rδ via a ReconDecoder.

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
        emb_obsm_key: str | None = None,
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
        # obsm key of the latent the model predicts (e.g. X_state). When set, the PRED delta
        # subtracts decode(control latent) instead of the observed control genes, so the
        # decoder's offset cancels in the predicted perturbation.
        self._emb_key = emb_obsm_key
        self._valid_freq = valid_freq
        self._wandb_run = wandb_run
        self._step = 0
        self._ctrl_cache: dict[str, np.ndarray] = {}
        self._ctrl_decoded_cache: dict[str, np.ndarray] = {}
        self._ctrl_decoded_cells_cache: dict[str, np.ndarray] = {}
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
        self._ctrl_decoded_cache = {}
        self._ctrl_decoded_cells_cache = {}

    @staticmethod
    def _to_dense(X) -> np.ndarray:
        return np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)

    @staticmethod
    def _normalize_key(cond_key: tuple) -> tuple:
        """ValidationSampler yields keys as a 1-tuple holding the str(tuple), e.g.
        ``("('A549', 'A-366', 2.39)",)`` — parse it back into ``('A549', 'A-366', 2.39)``.
        Already-clean tuples pass through unchanged.
        """
        if len(cond_key) == 1 and isinstance(cond_key[0], str) and cond_key[0].lstrip().startswith("("):
            try:
                return tuple(ast.literal_eval(cond_key[0]))
            except (ValueError, SyntaxError):
                pass
        return cond_key

    def _get_true_genes(self, cond_key: tuple) -> np.ndarray | None:
        cond_key = self._normalize_key(cond_key)
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
        cond_key = self._normalize_key(cond_key)
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

    def _get_ctrl_decoded(self, cond_key: tuple) -> np.ndarray | None:
        """Mean of decode(control-cell latents): the decoder's own control gene profile.

        Uses the same control cells as :meth:`_get_ctrl_genes` but their ``emb_obsm_key``
        latent, decoded — so the pred delta becomes decode(pred) − decode(ctrl_latent),
        cancelling the decoder offset. Cached per cell line.
        """
        if self._emb_key is None or self._emb_key not in self._adata.obsm:
            return None
        cond_key = self._normalize_key(cond_key)
        cl_idx = self._cond_keys.index(self._cl_key)
        cell_line = str(cond_key[cl_idx])
        if cell_line not in self._ctrl_decoded_cache:
            obs = self._adata.obs
            mask = obs[self._ctrl_key].astype(bool) & (obs[self._cl_key].astype(str) == cell_line)
            if mask.sum() == 0:
                return None
            Z = np.asarray(self._adata[mask].obsm[self._emb_key], dtype=np.float32)
            self._ctrl_decoded_cache[cell_line] = self._decoder.decode(Z).mean(axis=0)
        return self._ctrl_decoded_cache[cell_line]

    def _get_ctrl_decoded_cells(self, cond_key: tuple) -> np.ndarray | None:
        """Full decode(control-cell latents) matrix (cells × genes), cached per cell line.

        Used as the pred-side control for gene-space DE so the decoder offset cancels
        (DE-pred = decode(pred) vs decode(ctrl_latent)). Falls back to None when no latent
        is available, in which case DE uses the observed control cells instead.
        """
        if self._emb_key is None or self._emb_key not in self._adata.obsm:
            return None
        cond_key = self._normalize_key(cond_key)
        cl_idx = self._cond_keys.index(self._cl_key)
        cell_line = str(cond_key[cl_idx])
        if cell_line not in self._ctrl_decoded_cells_cache:
            obs = self._adata.obs
            mask = obs[self._ctrl_key].astype(bool) & (obs[self._cl_key].astype(str) == cell_line)
            if mask.sum() == 0:
                return None
            Z = np.asarray(self._adata[mask].obsm[self._emb_key], dtype=np.float32)
            self._ctrl_decoded_cells_cache[cell_line] = np.asarray(self._decoder.decode(Z), dtype=np.float32)
        return self._ctrl_decoded_cells_cache[cell_line]

    def _compute_recon(self, pred_data: dict, prefix: str, step_label: str, emit: bool = True) -> dict:
        """Gene-space delta metrics over ``pred_data`` ({ds: {cond_key: pred_latent}}).

        ``prefix`` selects the metric namespace (``"val"`` or ``"test"``). When ``emit`` is
        False the result is returned without logging to wandb (used by the CFG w-sweep, which
        logs all w at once afterwards).
        """
        pearson_deltas = []
        pearson_fulls = []   # non-delta: decode(pred) vs true genes (no control)
        de_over, de_prec, de_dir = [], [], []   # gene-space DE on decoded genes
        n_total = n_unmatched = 0
        first_unmatched = None
        pred_sigs, predgene_sigs = [], []  # diagnostic: do recon's inputs/outputs vary?
        for ds in pred_data:
            for cond_key, pred_latent in pred_data[ds].items():
                n_total += 1
                true_genes = self._get_true_genes(cond_key)
                ctrl_genes = self._get_ctrl_genes(cond_key)
                if true_genes is None or ctrl_genes is None:
                    n_unmatched += 1
                    if first_unmatched is None:
                        first_unmatched = (cond_key, true_genes is None, ctrl_genes is None)
                    continue
                pred_arr = np.asarray(pred_latent, dtype=np.float32)
                pred_genes = self._decoder.decode(pred_arr)
                pred_sigs.append(float(pred_arr.mean()))
                predgene_sigs.append(float(pred_genes.mean()))
                true_mean = true_genes.mean(axis=0)
                pred_mean = pred_genes.mean(axis=0)
                ctrl_mean = ctrl_genes.mean(axis=0)               # observed control genes
                # pred delta uses decode(control latent) when available, so the decoder
                # offset cancels (decode(pred) − decode(ctrl)); else fall back to observed.
                ctrl_pred = self._get_ctrl_decoded(cond_key)
                if ctrl_pred is None:
                    ctrl_pred = ctrl_mean
                # delta metric (perturbation effect)
                r, _ = pearsonr(true_mean - ctrl_mean, pred_mean - ctrl_pred)
                pearson_deltas.append(float(r))
                # non-delta metric (absolute reconstruction): decode(pred) vs true genes
                rf, _ = pearsonr(true_mean, pred_mean)
                pearson_fulls.append(float(rf))
                # gene-space DE (decoded cells): DE-true = perturbed vs observed control,
                # DE-pred = decode(pred) vs decode(control latent) so the decoder offset cancels.
                ctrl_pred_cells = self._get_ctrl_decoded_cells(cond_key)
                de = de_metrics(true_genes, pred_genes, ctrl_genes, source_pred=ctrl_pred_cells)
                de_over.append(de["de_overlap"])
                de_prec.append(de["de_precision"])
                de_dir.append(de["dir_agreement"])
        if pred_sigs:
            print(f"    {prefix} recon  [diag] pred_latent mean={np.mean(pred_sigs):.6f}  "
                  f"decoded mean={np.mean(predgene_sigs):.6f}  ({step_label})")

        # Always emit the keys (NaN when nothing matched) so monitor_metrics never KeyErrors.
        keys = ["pearson_r_delta", "pearson_r", *_DE_KEYS]
        if not pearson_deltas:
            ck, no_true, no_ctrl = (first_unmatched or (None, None, None))
            print(f"    {prefix} recon  WARNING: 0/{n_total} conditions matched the h5ad "
                  f"(cond_keys={list(self._cond_keys)}, log_dose_key={self._log_dose_key}). "
                  f"First unmatched cond_key={ck!r}  no_true={no_true} no_ctrl={no_ctrl}  ({step_label})")
            out = {}
            for k in keys:
                out[f"{prefix}_recon_{k}"] = float("nan")
                out[f"{prefix}_recon_{k}_median"] = float("nan")
            if emit and self._wandb_run is not None:
                self._wandb_run.log(out)
            return out
        if n_unmatched:
            print(f"    {prefix} recon  note: {n_unmatched}/{n_total} conditions unmatched "
                  f"(e.g. {first_unmatched[0]!r})")

        vals = {"pearson_r_delta": pearson_deltas, "pearson_r": pearson_fulls,
                "de_overlap": de_over, "de_precision": de_prec, "dir_agreement": de_dir}
        out = {}
        for k, v in vals.items():
            out[f"{prefix}_recon_{k}"]        = float(np.nanmean(v))
            out[f"{prefix}_recon_{k}_median"] = float(np.nanmedian(v))
        print(f"    {prefix} recon  "
              f"rδ={out[f'{prefix}_recon_pearson_r_delta']:.4f}  r={out[f'{prefix}_recon_pearson_r']:.4f}  "
              f"(med rδ={out[f'{prefix}_recon_pearson_r_delta_median']:.4f})  "
              f"DEover={out[f'{prefix}_recon_de_overlap']:.3f} DEprec={out[f'{prefix}_recon_de_precision']:.3f} "
              f"DirAgr={out[f'{prefix}_recon_dir_agreement']:.3f} "
              f"({step_label})")
        if emit and self._wandb_run is not None:
            self._wandb_run.log(out)
        return out

    def _compute_recon_multi_w(self, pred_data_by_w: dict) -> dict:
        """CFG sweep: decode+score at each w, log all w curves, graph the best w."""
        primary = "val_recon_pearson_r_delta"
        per_w = {}
        for w, pred in pred_data_by_w.items():
            per_w[w] = self._compute_recon(pred, "val", f"step {self._step} w={w}", emit=False)

        def score(out: dict) -> float:
            s = out.get(primary)
            return -np.inf if (s is None or np.isnan(s)) else float(s)

        best_w = max(per_w, key=lambda w: score(per_w[w]))
        best = per_w[best_w]

        wandb_log: dict = {}
        for w in sorted(per_w):
            for k, v in per_w[w].items():
                wandb_log[f"{k}__w{w}"] = v          # all-w curves
        wandb_log.update(best)                        # best-w → default val_recon_* keys
        wandb_log["val_recon_best_w"] = float(best_w)
        print(f"    val recon  → best w={best_w} by {primary} "
              f"(rδ={score(best):.4f})  (step {self._step})")
        if self._wandb_run is not None:
            self._wandb_run.log(wandb_log)
        return best

    def _compute(self, valid_source_data, valid_true_data, valid_pred_data, pred_data_by_w=None) -> dict:
        if pred_data_by_w is not None and len(pred_data_by_w) > 1:
            return self._compute_recon_multi_w(pred_data_by_w)
        return self._compute_recon(valid_pred_data, "val", f"step {self._step}")

    def evaluate_test(self, solver, test_samplers: dict, predict_kwargs: dict | None = None) -> dict:
        """Gene-space recon metrics on the held-out test set (logged as ``test_recon_*``).

        When ``predict_kwargs`` carries a ``guidance_scales`` list (the CFG w-sweep used at
        validation), the same sampled test batch is decoded/scored at every w and each is
        returned as ``test_recon_<k>__w<w>``; the default ``test_recon_<k>`` keys hold the
        config ``guidance_scale`` (base w). No best-w is selected.
        """
        base_pk, ws, base_w = _test_guidance_plan(predict_kwargs)
        # skip the w-sweep for non-CFG models: every w gives the same conditional prediction
        # (mirrors evaluate_test). Avoids duplicate recon-test passes when cfg_enabled is False.
        if not getattr(solver, "cfg_enabled", False):
            ws = [base_w]

        # sample once so all w share the same cells/conditions
        batches = {name: sampler.sample(mode="on_train_end") for name, sampler in test_samplers.items()}

        per_w = {}
        for w in ws:
            pkw = dict(base_pk)
            pkw["guidance_scale"] = w
            pred_data = {
                name: jax.tree.map(partial(solver.predict, **pkw), b["source"], b["condition"])
                for name, b in batches.items()
            }
            per_w[w] = self._compute_recon(pred_data, "test", f"test w={w}", emit=False)

        default = per_w.get(base_w, per_w[ws[0]])
        if len(ws) == 1:
            return default
        out = dict(default)  # default test_recon_* at base w
        for w in sorted(per_w):
            for k, v in per_w[w].items():
                out[f"{k}__w{w}"] = v            # all-w curves
        return out

    def on_log_iteration(self, valid_source_data, valid_true_data,
                         valid_pred_data, solver, pred_data_by_w=None, **kwargs) -> dict:
        self._step += self._valid_freq
        return self._compute(valid_source_data, valid_true_data, valid_pred_data, pred_data_by_w)

    def on_train_end(self, valid_source_data, valid_true_data,
                     valid_pred_data, solver, pred_data_by_w=None, **kwargs) -> dict:
        return self._compute(valid_source_data, valid_true_data, valid_pred_data, pred_data_by_w)


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
