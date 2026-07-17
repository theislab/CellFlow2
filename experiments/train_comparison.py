"""Hydra entrypoint: train ScaleFlow from a raw .h5ad on the cellflow stack, using the SAME config
groups as train_zarr (datasets / data / split / model / condition_encoder / conditioning / match_fn /
training / ablation / wandb / solver). Select a dataset via an experiment, e.g.:

    python experiments/train_comparison.py +experiment=combosciplex_h5ad
    python experiments/train_comparison.py +experiment=sciplex3_h5ad wandb.enabled=true
    python experiments/train_comparison.py +experiment=combosciplex_h5ad ablation=prophet training.num_iterations=300

The scaleflow `DataManager` schema in ``cfg.data`` (src_dist_keys / tgt_dist_keys / rep_keys /
extra_rep_keys / cell_embedding.key / log1p_dose_from) is TRANSLATED into cellflow's
``ScaleFlow.prepare_data`` — the deleted zarr/DataManager layer isn't used here:

  • src_dist_keys            → split_covariates (one source/control pool per, e.g. cell_line)
  • tgt_dist_keys            → SEPARATE perturbation-covariate tokens: {k: [k]} per key
                               (drug_0 and drug_1 are two independent tokens, max_combination_length=1)
  • rep_keys                 → perturbation_covariate_reps (a key with no rep = numeric scalar, e.g. dose)
  • log1p_dose_from          → obs['dose'] = log1p(<raw>) before grouping
  • extra_rep_keys (prophet) → extra token(s), kept only when ablation.mode == 'prophet'
  • cell_embedding.key       → sample_rep ('X' = gene space, else an obsm latent)

Metrics live in gene space. Decode only if ``cfg.recon.decoder_path`` is set (ReconMetricsLogger +
DEGMetrics(recon=...)); else the model output is assumed gene space. Logs, best-model checkpoint and
the train/val/test split diagnostics mirror train_zarr.
"""
import os
import logging as _logging

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/storage/jax_cache")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "5")
os.environ.setdefault("WANDB_SILENT", "true")
# orbax's checkpoint save emits ~25 absl INFO lines each save, burying the metric prints in stdout;
# quiet absl/jax INFO so the val metrics / effect / checkpoint lines stay readable.
for _n in ("absl", "jax", "orbax", "orbax.checkpoint"):
    _logging.getLogger(_n).setLevel(_logging.WARNING)
try:
    from absl import logging as _absl_logging   # absl uses its own logger, not always the stdlib one
    _absl_logging.set_verbosity(_absl_logging.WARNING)
except Exception:
    pass

import time
from functools import partial
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from scaleflow.model import ScaleFlow
from cellflow.data import TrainSampler, ValidationSampler
from cellflow.utils import match_linear

import utils
import temp_edit
from callbacks import (PearsonDeltaMetrics, DEGMetrics, ReconMetricsLogger, ValMetricsLogger,
                       BestModelCheckpoint, save_logs, _solver_params, restore_solver_params)


def split_by_group(adata, by_cols, control_key, ratios, seed):
    """Train/val/test held out by the tuple of ``by_cols`` (e.g. (drug_0, drug_1) or (drug,)).
    Control cells (the source pool) are in EVERY split; only perturbed groups are partitioned."""
    obs = adata.obs
    is_ctrl = obs[control_key].to_numpy().astype(bool)
    grp = obs[list(by_cols)].astype(str).agg(tuple, axis=1)
    uniq = sorted(set(grp[~is_ctrl]))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n = len(uniq); n_tr = int(round(ratios[0] * n)); n_va = int(round(ratios[1] * n))
    sets = {"train": {uniq[i] for i in perm[:n_tr]},
            "val":   {uniq[i] for i in perm[n_tr:n_tr + n_va]},
            "test":  {uniq[i] for i in perm[n_tr + n_va:]}}

    def sub(s):
        return adata[is_ctrl | grp.isin(s).to_numpy()].copy()

    parts = {k: sub(s) for k, s in sets.items()}
    print(f"split by {list(by_cols)} (seed={seed}, ratios={ratios}): {n} groups -> "
          f"train={len(sets['train'])} val={len(sets['val'])} test={len(sets['test'])} "
          f"(+{int(is_ctrl.sum())} control cells each)")
    for k, v in parts.items():
        n_c = int(v.obs[control_key].sum())
        print(f"  {k}: {v.n_obs:,} cells ({n_c:,} control + {v.n_obs - n_c:,} perturbed)")
    return parts["train"], parts["val"], parts["test"]


def build_cellflow_covariates(adata, data_cfg: DictConfig, ablation_mode: str):
    """Translate the scaleflow DataManager `data:` schema → cellflow prepare_data args (separate tokens).

    Returns (perturbation_covariates, perturbation_covariate_reps, split_covariates, sample_rep,
    control_key, cond_cols) where cond_cols are the obs columns that define a condition (for splitting
    / diagnostics)."""
    control_key = str(data_cfg.dist_flag_key)
    split_covariates = list(data_cfg.src_dist_keys)
    rep_keys = OmegaConf.to_container(data_cfg.rep_keys, resolve=True) or {}

    # dose (or any log1p'd scalar): obs['dose'] = log1p(<raw>), used as a numeric covariate
    dose_src = data_cfg.get("log1p_dose_from")
    if dose_src and dose_src in adata.obs:
        raw = np.nan_to_num(adata.obs[dose_src].astype("float32").to_numpy(), nan=0.0)
        adata.obs["dose"] = np.log1p(raw)

    pert_covs, pert_reps = {}, {}
    for k in list(data_cfg.tgt_dist_keys):               # one SEPARATE token per target key
        if k not in rep_keys and k in adata.obs and not np.issubdtype(adata.obs[k].dtype, np.number):
            raise TypeError(f"covariate '{k}' has no rep_key and is non-numeric — give it a rep or make it numeric")
        pert_covs[k] = [k]
        if k in rep_keys:
            pert_reps[k] = rep_keys[k]

    # extra tokens (e.g. prophet) kept only under ablation=prophet; duplicate the source column so
    # cellflow doesn't see two covariate groups on the same obs column.
    extra = data_cfg.get("extra_rep_keys")
    if ablation_mode == "prophet" and extra:
        for group, spec in OmegaConf.to_container(extra, resolve=True).items():
            col, uns_key = spec[0], spec[1]
            dup = f"{col}__{group}"
            adata.obs[dup] = adata.obs[col].astype("category")
            pert_covs[group] = [dup]
            pert_reps[group] = uns_key

    sample_rep = str(data_cfg.cell_embedding.key)         # 'X' (gene space) or an obsm latent key
    cond_cols = list(split_covariates) + list(data_cfg.tgt_dist_keys)
    return pert_covs, pert_reps, split_covariates, sample_rep, control_key, cond_cols


@hydra.main(config_path="config", config_name="train_comparison", version_base=None)
def main(cfg: DictConfig) -> None:
    if not cfg.get("selected_datasets"):
        raise ValueError("no selected_datasets — pass an experiment, e.g. +experiment=combosciplex_h5ad")
    name = list(cfg.selected_datasets)[0]
    h5ad = str(cfg.datasets[name].path)
    run_name = cfg.wandb.get("run_name") or f"{name}_{str(cfg.data.cell_embedding.key).lower()}"
    out_dir = os.path.join(str(cfg.output_dir), run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"dataset={name}  h5ad={h5ad}  ablation={cfg.ablation.mode}  conditioning={cfg.conditioning.conditioning_key}  run={run_name}")
    m_cfg = cfg.model
    print(f"  ablation.mode              = {cfg.ablation.mode}")
    print(f"  model.hidden_dims          = {list(m_cfg.hidden_dims)}")
    print(f"  model.decoder_dims         = {list(m_cfg.decoder_dims)}")
    print(f"  model.conditioning_key     = {m_cfg.conditioning_key}")
    print(f"  cond_output_dropout        = {m_cfg.condition_encoder.cond_output_dropout}")
    print(f"  condition_dropout_prob     = {m_cfg.get('condition_dropout_prob', 0.0)}  (CFG null-drop)")
    print(f"  training.peak_lr           = {cfg.training.peak_lr}")
    print(f"  training.num_iterations    = {cfg.training.num_iterations}")
    print(f"  match_fn.epsilon           = {cfg.match_fn.epsilon}")

    adata = ad.read_h5ad(h5ad)
    print(f"loaded {adata.n_obs:,} x {adata.n_vars:,}")
    pert_covs, pert_reps, split_cov, sample_rep, control_key, cond_cols = build_cellflow_covariates(
        adata, cfg.data, str(cfg.ablation.mode))
    for rep in pert_reps.values():
        if rep not in adata.uns:
            raise KeyError(f"rep '{rep}' not in uns. Available: {[k for k in adata.uns if 'emb' in k.lower()]}")
    if sample_rep != "X" and sample_rep not in adata.obsm:
        raise KeyError(f"sample_rep '{sample_rep}' not in obsm. Available: {list(adata.obsm)}")
    max_comb = max(len(v) for v in pert_covs.values())   # separate tokens → 1
    print(f"condition tokens: { {k: v for k, v in pert_covs.items()} }  reps={list(pert_reps)}  "
          f"sample_rep={sample_rep}  max_combination_length={max_comb}")

    train_adata, val_adata, test_adata = split_by_group(
        adata, cfg.split.by, control_key, list(cfg.split.ratios), int(cfg.split.get("random_state", 42)))

    common = dict(sample_rep=sample_rep, control_key=control_key,
                  perturbation_covariates=pert_covs, perturbation_covariate_reps=pert_reps,
                  split_covariates=split_cov)
    sf = ScaleFlow(solver=str(cfg.solver.solver_key))
    print("prepare_data (train) …")
    sf.prepare_data(adata=train_adata, **common)
    sf.prepare_validation_data(adata=val_adata, name="val")
    if bool(cfg.get("eval_test", False)):
        sf.prepare_validation_data(adata=test_adata, name="test")
    print(f"in-loop validation: val{' + test' if cfg.get('eval_test', False) else ''}")

    sample_batch = TrainSampler(sf.train_data, batch_size=min(int(cfg.training.batch_size), 64)).sample(
        np.random.default_rng(0))
    print("condition modalities in sample_batch:", list(sample_batch["condition"].keys()))

    # ── model: translate cfg.model / condition_encoder / conditioning / match_fn → prepare_model ──
    m = cfg.model
    ce = m.condition_encoder
    encoder_arch = OmegaConf.to_container(ce.encoder_arch, resolve=True)
    layers_before_pool = {k: encoder_arch for k in sample_batch["condition"]}   # one MLP per token
    layers_after_pool = OmegaConf.to_container(ce.layers_after_pool, resolve=True)
    optimizer, _ = utils.build_optimizer(cfg)
    cond_key = str(m.conditioning_key)
    print(f"prepare_model … (conditioning={cond_key})")
    sf.prepare_model(
        sample_batch=sample_batch,
        max_combination_length=max_comb,
        conditioning=cond_key,
        conditioning_kwargs=OmegaConf.to_container(m.conditioning_kwargs, resolve=True),
        pooling=str(ce.pooling),
        pooling_kwargs=OmegaConf.to_container(ce.pooling_kwargs, resolve=True),
        layers_before_pool=layers_before_pool,
        layers_after_pool=layers_after_pool,
        cond_output_dropout=float(ce.cond_output_dropout),
        condition_embedding_dim=int(m.condition_embedding_dim),
        condition_dropout_prob=float(m.get("condition_dropout_prob", 0.0)),
        hidden_dims=tuple(int(x) for x in m.hidden_dims),
        decoder_dims=tuple(int(x) for x in m.decoder_dims),
        probability_path=OmegaConf.to_container(m.probability_path_kwargs, resolve=True),
        match_fn=partial(match_linear, epsilon=float(cfg.match_fn.epsilon)),
        optimizer=optimizer,
    )
    print(f"  model ready ({run_name})")

    # ── callbacks: metrics + logs + best-model checkpoint + diagnostics (+ recon if a decoder) ──
    from cellflow.training import Metrics
    ckpt_path = os.path.join(out_dir, f"{run_name}_best_ckpt")
    metrics_cb = Metrics(metrics=["e_distance", "mmd", "r_squared"], metric_aggregations=["mean", "median"])
    cbs = [metrics_cb,
           ValMetricsLogger(save_path=os.path.join(out_dir, f"{run_name}_val_metrics.json"),
                            valid_freq=int(cfg.training.valid_freq), compute_de=False),
           BestModelCheckpoint(save_path=ckpt_path,
                               metric=str(cfg.training.get("checkpoint_metric", "pearson_r_delta")))]
    # ds-keyed metric callbacks (return {ds}_metric) reused verbatim for the held-out TEST eval below,
    # so test gets the SAME metrics as val. (ValMetricsLogger hard-codes a 'val_' prefix → val only.)
    metric_cbs = [metrics_cb]

    dec_path = cfg.get("recon", {}).get("decoder_path")
    if dec_path:                                          # decode only if a decoder is provided
        from scaleflow.model._recon import ReconDecoder
        rc = cfg.recon
        decoder = ReconDecoder.load(str(dec_path))
        recon_adata = adata if not rc.get("h5ad_path") else ad.read_h5ad(str(rc.h5ad_path))
        cond_obs_keys = list(rc.get("condition_obs_keys", cond_cols))
        recon_cb = ReconMetricsLogger(
            decoder=decoder, adata=recon_adata, condition_obs_keys=cond_obs_keys,
            cell_line_obs_key=str(rc.get("cell_line_obs_key", split_cov[0])),
            control_obs_key=str(rc.get("control_obs_key", control_key)),
            log_dose_obs_key=rc.get("log_dose_obs_key"),
            emb_obsm_key=(sample_rep if sample_rep != "X" else None),
            valid_freq=int(cfg.training.valid_freq), wandb_run=None)
        deg_cb = DEGMetrics(recon=recon_cb)
        cbs += [recon_cb, deg_cb]
        metric_cbs.append(deg_cb)                         # DEGMetrics(recon=…) decodes + keys by ds
        print(f"  gene-space recon metrics via decoder → {len(decoder.var_names or [])} genes")
    else:
        pdm, deg_cb = PearsonDeltaMetrics(), DEGMetrics()
        cbs += [pdm, deg_cb]
        metric_cbs += [pdm, deg_cb]
        print("  no decoder → model output assumed gene space; metrics computed directly")

    diag = cfg.get("diagnostics", {}) or {}
    if diag.get("enabled", True):
        cbs.append(temp_edit.EffectSizeMonitor(valid_freq=int(cfg.training.valid_freq),
                                               max_cells=int(diag.get("max_cells", 2000))))
    if bool(cfg.wandb.get("enabled", False)):
        from cellflow.training import WandbLogger
        cbs.append(WandbLogger(project=str(cfg.wandb.project), out_dir=out_dir,
                               config=OmegaConf.to_container(cfg, resolve=True), name=run_name))

    n_iter = int(cfg.training.num_iterations)
    vf = int(cfg.training.valid_freq)
    bs = int(cfg.training.batch_size)
    print(f"train: num_iterations={n_iter} batch_size={bs} valid_freq={vf}")
    t0 = time.perf_counter()
    sf.train(num_iterations=n_iter, batch_size=bs, valid_freq=vf, callbacks=cbs, monitor_metrics=[])
    print(f"training done in {(time.perf_counter() - t0) / 60:.1f} min; loss len:",
          len(sf.trainer.training_logs.get("loss", [])))

    # ── logs JSON + restore best checkpoint (so diagnostics run on the best model) ──
    save_logs(run_name, sf.trainer.training_logs, Path(out_dir))
    if os.path.isdir(ckpt_path):
        import orbax.checkpoint as ocp
        print(f"restoring best checkpoint from {ckpt_path} …")
        best = ocp.PyTreeCheckpointer().restore(ckpt_path, item=_solver_params(sf.solver))
        restore_solver_params(sf.solver, best)
    else:
        print("no checkpoint saved — using final iterate")

    # ── held-out TEST metrics on the BEST model, over ALL test conditions (no cell/condition cap),
    #    using the SAME metric callbacks as val. Done once post-training so test never enters the loop.
    #    (If eval_test=true it was already scored in-loop, so skip.) ──
    if (not bool(cfg.get("eval_test", False)) and test_adata is not None
            and int((~test_adata.obs[control_key].astype(bool)).to_numpy().sum()) > 0):
        import json as _json
        sf.prepare_validation_data(adata=test_adata, name="test_final", n_conditions_on_train_end=None)
        b = ValidationSampler(sf.validation_data["test_final"]).sample(mode="on_train_end")
        pk = sf.validation_data.get("predict_kwargs", {})
        src_t = {"test": b["source"]}
        true_t = {"test": b["target"]}
        pred_t = {"test": sf.solver.predict(b["source"], condition=b.get("condition", None), **pk)}
        test_metrics: dict = {}
        for cb in metric_cbs:
            try:
                r = cb.on_train_end(src_t, true_t, pred_t, sf.solver)
                if isinstance(r, dict):
                    test_metrics.update(r)
            except Exception as e:                        # noqa: BLE001 — a metric shouldn't sink the run
                print(f"  test metric {type(cb).__name__} skipped: {e}")
        with open(os.path.join(out_dir, f"{run_name}_test_metrics.json"), "w") as f:
            _json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)
        if bool(cfg.wandb.get("enabled", False)):
            import wandb
            if wandb.run is not None:
                wandb.run.log(test_metrics)
        print(f"held-out TEST metrics on {len(true_t['test'])} conditions (all cells):")
        for k in sorted(test_metrics):
            print(f"  {k}: {test_metrics[k]:.4f}")

    # ── train/val/test split diagnostics (registered post-training → not evaluated in-loop) ──
    if diag.get("enabled", True):
        n_diag = int(diag.get("n_conditions", 100))
        splits_adata = {"train": train_adata, "val": val_adata, "test": test_adata}
        split_samplers = {}
        for sname in list(diag.get("splits", ["train", "val", "test"])):
            ad_ = splits_adata.get(sname)
            if ad_ is None:
                continue
            pert = (~ad_.obs[control_key].astype(bool)).to_numpy()
            if pert.sum() == 0:
                print(f"  diagnostics: skip '{sname}' — no perturbed conditions")
                continue
            n_cond = ad_.obs.loc[pert, cond_cols].astype(str).drop_duplicates().shape[0]
            sf.prepare_validation_data(adata=ad_, name=f"diag_{sname}",
                                       n_conditions_on_train_end=min(n_diag, n_cond))
            split_samplers[sname] = ValidationSampler(sf.validation_data[f"diag_{sname}"])
        if split_samplers:
            wbr = None
            if bool(cfg.wandb.get("enabled", False)):
                import wandb
                wbr = wandb.run
            print(f"running split diagnostics on {list(split_samplers)} (≤{n_diag} conditions/split) …")
            temp_edit.cellflow_split_diagnostics(
                sf.solver, split_samplers, out_dir, run_name, wandb_run=wbr,
                max_cells=int(diag.get("max_cells", 2000)), seed=int(cfg.split.get("random_state", 42)),
                predict_kwargs=sf.validation_data.get("predict_kwargs", {}))

    for k in sorted(sf.trainer.training_logs):
        if any(t in k for t in ("pearson_delta", "deg_dice", "mean_genediff", "recon")):
            print(f"  {k}: {sf.trainer.training_logs[k][-1]:.4f}")


if __name__ == "__main__":
    main()
