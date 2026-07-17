"""
temp_edit.py — STAGING module for effect-size / perturbation diagnostics.

Not yet integrated into callbacks.py or src/. Wire it in from train_zarr.py by import.

Two pieces:
  (1) EffectSizeMonitor — a callback that logs effect-size scalars each val step
      (reuses the trainer's already-computed predictions; no extra inference).
  (2) full_diagnostics — a FINAL step: one subsampled predict pass over train / val /
      test, a per-condition table with a `split` column, and plots coloured by split
      (so you can see train↔val↔test gaps in one view).

Usage in train_zarr.py (imports only — src/callbacks untouched):

    from temp_edit import EffectSizeMonitor, make_diagnostic_samplers, full_diagnostics

    cbs.append(EffectSizeMonitor(valid_freq=int(cfg.training.valid_freq), wandb_run=wandb_run))
    # optional: monitor_metrics += [f"{ds}_gap_closure_mean" for ds in val_samplers]

    # after training (uses `data` = split_datasets output, and the prophet `transform`):
    diag = make_diagnostic_samplers(data, n_conditions=100, transform=transform, seed=int(cfg.seed))
    full_diagnostics(best_solver, diag, output_dir, name, wandb_run=wandb_run)

Definitions (analyze_checkpoints_cf_tahoe_X_state.ipynb):
  effect size = ‖mean(X) − mean(ctrl)‖₂  ·  gap_closure = 1 − MMD(pred,true)/MMD(ctrl,true)
  effect_ratio = ‖pred−ctrl‖ / ‖true−ctrl‖  (<1 ⇒ under-shoot)
Each condition is subsampled to `max_cells` (equal N) before MMD/means.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from cellflow.metrics import compute_e_distance_fast, compute_scalar_mmd
from cellflow.training import ComputationCallback

SPLIT_COLORS = {"train": "#9e9e9e", "val": "#2196F3", "test": "#e74c3c"}


# ── primitives ───────────────────────────────────────────────────────────────
def _subsample(a, n, rng):
    a = np.asarray(a)
    return a if a.shape[0] <= n else a[rng.choice(a.shape[0], n, replace=False)]


def _pearson(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _pearson_r(true, pred):
    """Pearson r of mean profiles (over features): corr(mean(true), mean(pred))."""
    return _pearson(np.asarray(true).mean(0), np.asarray(pred).mean(0))


def _pearson_r_delta(true, pred, ctrl_mean):
    """Pearson r of mean perturbation deltas: corr(mean(true)-ctrl, mean(pred)-ctrl)."""
    return _pearson(np.asarray(true).mean(0) - ctrl_mean, np.asarray(pred).mean(0) - ctrl_mean)


# ── per-condition diagnostics (operates on PREDICTIONS — never infers) ────────
def condition_diagnostics(source, true, pred, max_cells: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per-condition effect sizes + metrics for ONE dataset's {cond_key: array} dicts."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in true:
        if k not in pred:
            continue
        t = _subsample(true[k], max_cells, rng)
        p = _subsample(pred[k], max_cells, rng)
        rec = {"condition": str(k), "n_true": int(np.asarray(true[k]).shape[0])}

        s_arr = source.get(k) if isinstance(source, dict) else None
        if s_arr is not None:
            s = _subsample(s_arr, max_cells, rng)
            cm = s.mean(0)
            true_eff = float(np.linalg.norm(t.mean(0) - cm))
            pred_eff = float(np.linalg.norm(p.mean(0) - cm))
            mmd_ct = float(compute_scalar_mmd(s, t))
            mmd_pt = float(compute_scalar_mmd(p, t))
            rec.update(
                true_effect=true_eff, pred_effect=pred_eff,
                effect_ratio=pred_eff / (true_eff + 1e-8),
                mmd_ctrl_true=mmd_ct, mmd_pred_true=mmd_pt,
                gap_closure=1.0 - mmd_pt / (mmd_ct + 1e-8),
                pearson_r_delta=_pearson_r_delta(t, p, cm),
            )
        rec.update(
            pearson_r=_pearson_r(t, p),
            e_distance=float(compute_e_distance_fast(t, p)),
            mmd=float(compute_scalar_mmd(p, t)),
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def diagnostics_over_datasets(vsrc, vtrue, vpred, max_cells: int = 2000) -> pd.DataFrame:
    """condition_diagnostics across nested {ds: {cond: array}} dicts (e.g. val callback)."""
    frames = []
    for ds in vtrue:
        df = condition_diagnostics(vsrc.get(ds, {}), vtrue[ds], vpred[ds], max_cells=max_cells)
        if not df.empty:
            df.insert(0, "dataset", ds)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def scalar_diagnostics(df: pd.DataFrame) -> dict:
    out: dict = {}
    if "gap_closure" in df:
        gc = df["gap_closure"].to_numpy()
        out["gap_closure_mean"] = float(np.nanmean(gc))
        out["gap_closure_frac_pos"] = float(np.nanmean(gc > 0))
    if {"true_effect", "pred_effect"} <= set(df.columns):
        x, y = df["true_effect"].to_numpy(), df["pred_effect"].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2:
            out["effect_calib_slope"] = float(np.polyfit(x[m], y[m], 1)[0])
            out["effect_corr"] = float(np.corrcoef(x[m], y[m])[0, 1])
        out["effect_ratio_mean"] = float(np.nanmean(df["effect_ratio"]))
    if "pearson_r" in df:
        out["pearson_r_mean"] = float(np.nanmean(df["pearson_r"]))
    if "pearson_r_delta" in df:
        out["pearson_r_delta_mean"] = float(np.nanmean(df["pearson_r_delta"]))
    return out


# ── (1) training-time callback (scalars → wandb; reuses trainer predictions) ──
class EffectSizeMonitor(ComputationCallback):
    """Logs effect-size / gap-closure scalars each val step; returns per-dataset
    {ds}_gap_closure_mean for monitor_metrics. No inference — uses valid_pred_data.
    """

    def __init__(self, valid_freq: int, wandb_run=None, max_cells: int = 2000, prefix: str = "val"):
        self._valid_freq, self._wandb_run = valid_freq, wandb_run
        self._max_cells, self._prefix, self._step = max_cells, prefix, 0

    def on_train_begin(self, *args, **kwargs) -> None:
        self._step = 0

    def _run(self, vsrc, vtrue, vpred) -> dict:
        df = diagnostics_over_datasets(vsrc, vtrue, vpred, max_cells=self._max_cells)
        if df.empty:
            return {}
        out: dict = {}
        for ds, g in df.groupby("dataset"):                # per-dataset scalars (in-loop = val)
            sc = scalar_diagnostics(g)
            out.update({f"{ds}_{k}": v for k, v in sc.items()})
            print(f"    [effect {ds}] " + "  ".join(f"{k}={v:.3f}" for k, v in sc.items()))
        # Two logging paths, so neither train_zarr (scaleflow) nor train_comparison (cellflow) regresses:
        #   • if a wandb_run was passed (train_zarr), self-log — the scaleflow trainer does NOT log returns;
        #   • ALWAYS return the scalars — cellflow's WandbLogger (a LoggingCallback) logs callback returns.
        # On the cellflow path we construct this WITHOUT a wandb_run, so there's no double-logging.
        if self._wandb_run is not None:
            self._wandb_run.log(out)
        return out

    def on_log_iteration(self, valid_source_data, valid_true_data, valid_pred_data, solver, **kwargs):
        self._step += self._valid_freq
        return self._run(valid_source_data, valid_true_data, valid_pred_data)

    def on_train_end(self, valid_source_data, valid_true_data, valid_pred_data, solver, **kwargs):
        return self._run(valid_source_data, valid_true_data, valid_pred_data)


# ── (2) final step: one subsampled predict pass over train / val / test ───────
def make_diagnostic_samplers(data: dict, n_conditions: int = 100, transform=None, seed: int = 0) -> dict:
    """data: {ds: {"train": gd, "val": gd, "test": gd}} (split_datasets output).
    → {split: {ds: ValidationSampler}} with conditions capped at n_conditions (so the
    three splits are comparable and the predict pass stays bounded).
    """
    from scaleflow.data._dataloader import ValidationSampler
    out: dict = {}
    for split in ("train", "val", "test"):
        smp = {}
        for ds, gds in data.items():
            if gds.get(split) is None:          # all-train datasets (holdout=false) have val/test=None
                continue
            s = ValidationSampler(
                gds[split],
                n_conditions_on_log_iteration=n_conditions,
                n_conditions_on_train_end=n_conditions,
                seed=seed,
                condition_transform=transform,
            )
            s.init_sampler()
            smp[ds] = s
        if smp:
            out[split] = smp
    return out


def full_diagnostics(solver, split_samplers: dict, output_dir, name: str = "model",
                     wandb_run=None, max_cells: int = 2000, seed: int = 0,
                     guidance_scales=None):
    """Predict each (split, dataset) subsample ONCE PER guidance weight → combined
    per-condition table (w + split + dataset columns) + plots coloured by split, one
    set per w, plus a guidance-sweep summary curve (metric vs w, line per split).

    ``guidance_scales`` is the list of CFG weights to evaluate (defaults to ``[1.0]`` =
    plain conditional). Every (split, dataset) is sampled ONCE and reused across all w so
    the curves are comparable. split_samplers from make_diagnostic_samplers.
    """
    from functools import partial

    import jax

    output_dir = Path(output_dir)
    ws = [float(w) for w in guidance_scales] if guidance_scales else [1.0]
    # non-CFG model → every w is identical; collapse to a single conditional pass.
    if not getattr(solver, "cfg_enabled", False):
        ws = [1.0]

    # sample each (split, dataset) ONCE so every w is compared on the same cells/conditions
    sampled: dict = {}
    for split, samplers in split_samplers.items():
        for ds, sampler in samplers.items():
            batch = sampler.sample(mode="on_train_end")
            sampled[(split, ds)] = (batch["source"], batch["condition"], batch["target"])

    if len(ws) > 1:
        print(f"  [diagnostics] classifier-free guidance sweep over w = {ws}")

    frames = []
    for w in ws:
        for (split, ds), (src, cond, true) in sampled.items():
            pred = jax.tree.map(partial(solver.predict, guidance_scale=w), src, cond)
            df = condition_diagnostics(src, true, pred, max_cells=max_cells, seed=seed)
            if not df.empty:
                df.insert(0, "dataset", ds)
                df.insert(0, "split", split)
                df.insert(0, "w", w)
                frames.append(df)

    alldf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if alldf.empty:
        print("  [diagnostics] no conditions to evaluate")
        return alldf, {}, {}

    csv = output_dir / f"{name}_diagnostics.csv"
    alldf.to_csv(csv, index=False)

    # per-w scatter plots (coloured by split) + per-(w, split) scalar summary
    all_figs: dict = {}
    summary: dict = {}
    for w in ws:
        wdf = alldf[alldf["w"] == w]
        if wdf.empty:
            continue
        w_tag = f"w{w:g}"
        all_figs[w] = plot_diagnostics(wdf, output_dir, tag=f"{name}_{w_tag}")
        summary[w] = {sp: scalar_diagnostics(g) for sp, g in wdf.groupby("split")}

    # guidance-sweep summary: metric vs w, one line per split (only meaningful for >1 w)
    sweep_figs = plot_guidance_sweep(summary, output_dir, tag=name) if len(ws) > 1 else {}

    print(f"  diagnostics table → {csv}")
    for w in ws:
        for sp, sc in summary.get(w, {}).items():
            print(f"  [w={w:g} {sp:5}] " + "  ".join(f"{k}={v:.3f}" for k, v in sc.items()))

    if wandb_run is not None:
        import wandb
        base_w = 1.0 if 1.0 in summary else ws[0]
        log: dict = {}
        for w in ws:
            for sp, sc in summary.get(w, {}).items():
                log.update({f"{sp}_{k}__w{w:g}": v for k, v in sc.items()})   # per-split, per-w
                if w == base_w:
                    log.update({f"{sp}_{k}": v for k, v in sc.items()})       # baseline default keys
            for nm, fig in all_figs.get(w, {}).items():
                log[f"diagnostics/w{w:g}/{nm}"] = wandb.Image(fig)             # per-w scatter plots
        for nm, fig in sweep_figs.items():
            log[f"diagnostics/guidance_sweep/{nm}"] = wandb.Image(fig)         # metric-vs-w curves
        log["diagnostics/table"] = wandb.Table(dataframe=alldf)
        wandb_run.log(log)

    # close figures to bound matplotlib memory (one set per w + the sweep curves)
    import matplotlib.pyplot as plt
    for figs in list(all_figs.values()) + [sweep_figs]:
        for fig in figs.values():
            plt.close(fig)

    return alldf, summary, all_figs


# ── plots (coloured by split) ────────────────────────────────────────────────
def plot_diagnostics(df: pd.DataFrame, output_dir, tag: str = "diag", metric: str = "pearson_r_delta") -> dict:
    """Three scatters coloured by `split`: effect-vs-metric, calibration, strength-vs-error.

    `metric` defaults to pearson_r_delta — plain pearson_r is inflated by the control
    baseline, so effect-size-vs-rΔ is what reveals weak-effect conditions carrying the score.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    groups = list(df.groupby("split")) if "split" in df else [("all", df)]
    figs: dict = {}

    def _scatter(ax, xcol, ycol):
        for sp, g in groups:
            ax.scatter(g[xcol], g[ycol], s=16, alpha=0.55, edgecolors="white", lw=0.3,
                       color=SPLIT_COLORS.get(sp, None), label=sp)

    # (1) effect size vs a quality metric
    if "true_effect" in df and metric in df:
        fig, ax = plt.subplots(figsize=(5, 4))
        _scatter(ax, "true_effect", metric)
        ax.set_xlabel("true effect  ‖mean(true) − mean(ctrl)‖"); ax.set_ylabel(metric)
        ax.set_title(f"effect size vs {metric}"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(output_dir / f"{tag}_effect_vs_{metric}.png", dpi=130)
        figs["effect_vs_metric"] = fig

    # (2) effect-size calibration (true vs pred effect; y=x = right magnitude)
    if {"true_effect", "pred_effect"} <= set(df.columns):
        fig, ax = plt.subplots(figsize=(5, 4))
        lim = float(np.nanmax([df["true_effect"].max(), df["pred_effect"].max()])) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="y=x")
        _scatter(ax, "true_effect", "pred_effect")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("true effect  ‖true − ctrl‖"); ax.set_ylabel("pred effect  ‖pred − ctrl‖")
        ax.set_title("effect-size calibration"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(output_dir / f"{tag}_effect_calibration.png", dpi=130)
        figs["calibration"] = fig

    # (3) strength vs error (below y=x ⇒ beats 'return control')
    if {"mmd_ctrl_true", "mmd_pred_true"} <= set(df.columns):
        fig, ax = plt.subplots(figsize=(5, 4))
        lim = float(np.nanmax([df["mmd_ctrl_true"].max(), df["mmd_pred_true"].max()])) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="y=x (return control)")
        _scatter(ax, "mmd_ctrl_true", "mmd_pred_true")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("MMD(ctrl, true) — strength"); ax.set_ylabel("MMD(pred, true) — error")
        ax.set_title("strength vs error"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(output_dir / f"{tag}_strength_vs_error.png", dpi=130)
        figs["strength_vs_error"] = fig

    return figs


# ── guidance-sweep summary (metric vs w, one line per split) ──────────────────
def plot_guidance_sweep(summary: dict, output_dir, tag: str = "diag",
                        metrics=("pearson_r_delta_mean", "gap_closure_mean", "effect_ratio_mean")) -> dict:
    """One line plot per metric: x = guidance weight w, y = mean metric, a line per split.

    ``summary`` is ``{w: {split: {metric: value}}}`` (from full_diagnostics). Shows how each
    quality metric responds to the CFG weight so the best w is readable at a glance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    ws = sorted(summary.keys())
    splits = sorted({sp for w in ws for sp in summary[w]})
    figs: dict = {}

    for metric in metrics:
        # only plot if at least one (w, split) reported this metric
        if not any(metric in summary[w].get(sp, {}) for w in ws for sp in splits):
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        for sp in splits:
            ys = [summary[w].get(sp, {}).get(metric, np.nan) for w in ws]
            ax.plot(ws, ys, marker="o", color=SPLIT_COLORS.get(sp, None), label=sp)
        ax.set_xlabel("guidance weight  w"); ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs guidance weight"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(output_dir / f"{tag}_sweep_{metric}.png", dpi=130)
        figs[metric] = fig

    return figs


# ── (3) plot diagnostics as a callback on the validation prediction dicts ─────
class DiagnosticsPlots(ComputationCallback):
    """Build the per-condition effect-size table from the validation predictions and log the
    scatter panels (effect calibration, strength-vs-error, effect-vs-rΔ) + the table to wandb.

    Runs at train-end by default; ``every_val=True`` also emits each validation. Optional UMAP
    effect panels for the top-``umap_top_k`` conditions per dataset (needs the ``umap`` package).

    Uses the SAME nested {ds:{cond:array}} dicts cellflow hands the callback — no extra inference and
    no scaleflow samplers, which is why the sampler-coupled full_diagnostics isn't used on this branch.
    """

    def __init__(self, output_dir, wandb_run=None, max_cells: int = 2000, seed: int = 0,
                 every_val: bool = False, umap_top_k: int = 0):
        self._out = Path(output_dir)
        self._wandb_run = wandb_run
        self._max_cells, self._seed = max_cells, seed
        self._every_val, self._umap_k = every_val, int(umap_top_k)

    def on_train_begin(self, *args, **kwargs) -> None:
        return None

    def _resolve_run(self):
        import wandb
        return self._wandb_run or getattr(wandb, "run", None)

    def _emit(self, vsrc, vtrue, vpred, tag: str) -> dict:
        df = diagnostics_over_datasets(vsrc, vtrue, vpred, max_cells=self._max_cells)
        if df.empty:
            return {}
        # color panels by split: dataset names are 'train'/'val'/'test' → plot_diagnostics groups by
        # 'split' and maps them onto SPLIT_COLORS (grey / blue / red).
        if "dataset" in df.columns:
            df["split"] = df["dataset"]
        self._out.mkdir(parents=True, exist_ok=True)
        figs = plot_diagnostics(df, self._out, tag=tag)
        run = self._resolve_run()
        log: dict = {}
        if self._umap_k > 0:                                   # optional qualitative UMAP panels
            try:
                import wandb
                from utils import umap_effect_panels
                for ds in vtrue:
                    if ds not in vpred:
                        continue
                    out_png = self._out / f"{tag}_{ds}_umap.png"
                    umap_effect_panels(vsrc.get(ds, {}), vtrue[ds], {"model": vpred[ds]},
                                       k=self._umap_k, seed=self._seed, out_path=str(out_png),
                                       title=f"{ds} effect panels ({tag})")
                    if run is not None:
                        log[f"diagnostics/{tag}_{ds}_umap"] = wandb.Image(str(out_png))
            except Exception as e:                             # umap missing / too few conditions
                print(f"    [diagnostics] UMAP panels skipped: {e}")
        if run is not None:
            import wandb
            log.update({f"diagnostics/{tag}_{nm}": wandb.Image(fig) for nm, fig in figs.items()})
            log[f"diagnostics/{tag}_table"] = wandb.Table(dataframe=df)
            run.log(log)
        print(f"    [diagnostics] {tag}: {len(figs)} panels"
              + (" + UMAP" if self._umap_k else "")
              + (" → wandb" if run is not None else f" → {self._out}"))
        import matplotlib.pyplot as plt
        for fig in figs.values():
            plt.close(fig)
        return {}

    def on_log_iteration(self, valid_source_data, valid_true_data, valid_pred_data, solver, **kwargs):
        return self._emit(valid_source_data, valid_true_data, valid_pred_data, "val") if self._every_val else {}

    def on_train_end(self, valid_source_data, valid_true_data, valid_pred_data, solver, **kwargs):
        return self._emit(valid_source_data, valid_true_data, valid_pred_data, "final")


# ── train_zarr-style FINAL diagnostics, cellflow-native ───────────────────────
def cellflow_split_diagnostics(solver, split_samplers: dict, output_dir, name: str = "model",
                               wandb_run=None, max_cells: int = 2000, seed: int = 0,
                               predict_kwargs: dict | None = None):
    """One predict pass per split (train/val/test) → combined per-condition table with a ``split``
    column + panels COLOURED BY SPLIT. cellflow port of ``full_diagnostics``: ``split_samplers`` maps
    a split name to a cellflow ``ValidationSampler``; each is sampled once (mode='on_train_end') and
    the trained ``solver`` predicts it (exactly like the trainer's validation step). Reuses
    condition_diagnostics / plot_diagnostics / scalar_diagnostics — no scaleflow samplers, no CFG sweep.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predict_kwargs = predict_kwargs or {}
    frames = []
    for split, sampler in split_samplers.items():
        batch = sampler.sample(mode="on_train_end")
        src, cond, true = batch["source"], batch.get("condition", None), batch["target"]
        pred = solver.predict(src, condition=cond, **predict_kwargs)
        df = condition_diagnostics(src, true, pred, max_cells=max_cells, seed=seed)
        if not df.empty:
            df.insert(0, "split", split)
            frames.append(df)
    alldf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if alldf.empty:
        print("  [diagnostics] no conditions to evaluate")
        return alldf, {}, {}
    csv = output_dir / f"{name}_diagnostics.csv"
    alldf.to_csv(csv, index=False)
    figs = plot_diagnostics(alldf, output_dir, tag=name)          # grouped/coloured by `split`
    summary = {sp: scalar_diagnostics(g) for sp, g in alldf.groupby("split")}
    for sp, sc in summary.items():
        print(f"  [diag {sp:5}] " + "  ".join(f"{k}={v:.3f}" for k, v in sc.items()))
    if wandb_run is not None:
        import wandb
        log: dict = {}
        for sp, sc in summary.items():
            log.update({f"diag_{sp}_{k}": v for k, v in sc.items()})
        log.update({f"diagnostics/{nm}": wandb.Image(fig) for nm, fig in figs.items()})
        log["diagnostics/table"] = wandb.Table(dataframe=alldf)
        wandb_run.log(log)
    import matplotlib.pyplot as plt
    for fig in figs.values():
        plt.close(fig)
    print(f"  diagnostics table → {csv}  (panels coloured by split: {list(summary)})")
    return alldf, summary, figs
