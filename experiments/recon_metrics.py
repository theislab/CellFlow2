"""Reconstruction-quality metrics (mean-profile R² / perturbation-delta).

Shared by ``recon_emb.py`` (held-out test evaluation) and the reconstruction notebook. All
R² follow ``scaleflow``'s convention of scoring **mean profiles** rather than per-cell values.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def per_gene_pearson(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-gene Pearson r across cells (finite genes only)."""
    t = true - true.mean(0)
    p = pred - pred.mean(0)
    den = np.sqrt((t ** 2).sum(0) * (p ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (t * p).sum(0) / den
    return r[np.isfinite(r)]


def pseudobulk_r2(true: np.ndarray, pred: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    """Per group (e.g. cell line): ``r2_score(mean_true, mean_pred)`` across genes."""
    scores = {str(g): float(r2_score(true[labels == g].mean(0), pred[labels == g].mean(0)))
              for g in np.unique(labels)}
    return float(np.mean(list(scores.values()))), scores


def control_means(X: np.ndarray, labels: np.ndarray, is_ctrl: np.ndarray) -> dict:
    """Per group control baseline = mean expression of that group's control cells."""
    return {g: X[(labels == g) & is_ctrl].mean(0) for g in np.unique(labels)}


def delta_metrics(true, pred, labels, drugs, treated, ctrl_mean, min_cells: int = 20) -> dict:
    """Per (group, drug): R² and Pearson r of the perturbation delta.

    The delta is ``mean_treated - control_mean`` (subtracting the group's control baseline),
    scored true-vs-reconstructed across genes. Only conditions with ``>= min_cells`` treated
    cells are kept.
    """
    r2s, prs = [], []
    for g in np.unique(labels):
        g_treated = treated & (labels == g)
        base = ctrl_mean[g]
        for dg in np.unique(drugs[g_treated]):
            m = g_treated & (drugs == dg)
            if int(m.sum()) < min_cells:
                continue
            td, rd = true[m].mean(0) - base, pred[m].mean(0) - base
            r2s.append(float(r2_score(td, rd)))
            prs.append(float(np.corrcoef(td, rd)[0, 1]) if td.std() > 0 and rd.std() > 0 else np.nan)
    if not r2s:
        return dict(r2_mean=float("nan"), r2_median=float("nan"),
                    pearson_mean=float("nan"), pearson_median=float("nan"), n=0)
    return dict(r2_mean=float(np.mean(r2s)), r2_median=float(np.median(r2s)),
                pearson_mean=float(np.nanmean(prs)), pearson_median=float(np.nanmedian(prs)), n=len(r2s))


def pure_reconstruction(true: np.ndarray, pred: np.ndarray) -> dict:
    """Direct reconstruction fidelity (decode(encode(x)) ≈ x), no perturbation-delta.

    - ``mse``                : mean squared error.
    - ``reconstruction_r2``  : variance-weighted R² across genes (overall variance explained).
    - ``reconstruction_r2_per_gene`` : unweighted mean of per-gene R² (treats genes equally).
    - ``median_per_gene_r``  : median per-gene Pearson r.
    """
    return {
        "mse": float(((true - pred) ** 2).mean()),
        "reconstruction_r2": float(r2_score(true, pred, multioutput="variance_weighted")),
        "reconstruction_r2_per_gene": float(r2_score(true, pred, multioutput="uniform_average")),
        "median_per_gene_r": float(np.median(per_gene_pearson(true, pred))),
    }


def reconstruction_report(true, pred, labels, drugs, treated, ctrl_mean, min_cells: int = 20) -> dict:
    """One flat dict of reconstruction metrics on a held-out set."""
    pb, _ = pseudobulk_r2(true, pred, labels)
    d = delta_metrics(true, pred, labels, drugs, treated, ctrl_mean, min_cells)
    return {
        "mse": float(((true - pred) ** 2).mean()),
        "pseudobulk_r2": pb,
        "delta_r2_mean": d["r2_mean"], "delta_r2_median": d["r2_median"],
        "delta_pearson_mean": d["pearson_mean"], "delta_pearson_median": d["pearson_median"],
        "median_per_gene_r": float(np.median(per_gene_pearson(true, pred))),
        "n_conditions": d["n"],
    }
