"""Train a CellFlow2 model from raw .h5ad files (in-memory GroupedDistributions via DataManager).

Concise Hydra entrypoint that MIRRORS train_zarr.py and reuses callbacks.py / utils.py / temp_edit.py.
The only difference vs train_zarr.py: a dataset whose path ends in ``.h5ad`` is grouped in memory via
the DataManager (config in ``cfg.data``; ``data_location = obsm[cfg.data.cell_embedding.key]`` = the
flow latent, e.g. ``AE_128_opt``) instead of read from a prebuilt zarr. Everything after loading —
per-dataset samplers/metrics, gene-space ReconMetricsLogger, EffectSizeMonitor, wandb — is shared.

    python experiments/train_comparison.py                          # sciplex3_with_emb + AE_128 latent
    python experiments/train_comparison.py wandb.enabled=true
    python experiments/train_comparison.py training.num_iterations=300 training.valid_freq=150
"""
import os

# set JAX env before importing jax (pulled in by scaleflow / utils)
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/storage/jax_cache")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "5")

# allow wandb up to 2 hours to upload at the end of a long training run
os.environ.setdefault("WANDB_HTTP_TIMEOUT", "7200")
os.environ.setdefault("WANDB_INIT_TIMEOUT", "7200")
# retry failed uploads instead of dropping data
os.environ.setdefault("WANDB_RETRY_MAX", "10")
# write logs offline if network drops mid-run; sync manually later
os.environ.setdefault("WANDB_SILENT", "true")  # suppress wandb spam to stdout

import time
from functools import partial
from pathlib import Path

import cloudpickle
import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

from scaleflow.data import GroupedDistribution, split_datasets
from scaleflow.data._dataloader import CombinedSampler, ReservoirSampler, ValidationSampler
from scaleflow.model import ScaleFlow
from cellflow.training import Metrics
from cellflow.utils import match_linear

import utils
import callbacks
import temp_edit


def mark_control(adata, flag_key: str):
    """Ensure a boolean control column. No-op if ``flag_key`` already present (e.g. sciplex3_with_emb),
    else derive control = every drug slot == 'control' (unipert/tahoe: drug_0[/drug_1])."""
    if flag_key in adata.obs:
        adata.obs[flag_key] = adata.obs[flag_key].astype(bool)
        return adata
    drug_col = "drug_0" if "drug_0" in adata.obs else "drug"
    ctrl = adata.obs[drug_col].astype(str) == "control"
    if "drug_1" in adata.obs:
        ctrl &= adata.obs["drug_1"].astype(str) == "control"
    adata.obs[flag_key] = ctrl
    return adata


def prep_adata(adata, prep, data_cfg: DictConfig):
    """Per-dataset harmonization (config-driven, for cross-dataset runs). Reusable for the recon adata.
      - ``cell_line_map`` {old: new}  → canonicalize obs['cell_line'] (e.g. CVCL_0023→A549) AND copy the
        CCLE embedding to the new key so rep_keys['cell_line'] resolves across datasets.
      - ``cell_line_keep`` <name>     → filter to one cell line (e.g. sciplex → A549).
      - ``batch`` <label>             → obs['batch']=label + inject uns['batch_embeddings'] = one-hot over
        data.batch_labels, so both datasets share one batch space (batch is a src_dist_key → a covariate)."""
    prep = OmegaConf.to_container(prep, resolve=True) if prep is not None else {}
    cmap = prep.get("cell_line_map")
    if cmap:
        adata.obs["cell_line"] = adata.obs["cell_line"].astype(str).replace(cmap).astype("category")
        cl_uns = (OmegaConf.to_container(data_cfg.rep_keys, resolve=True) or {}).get("cell_line")
        if cl_uns and cl_uns in adata.uns:
            d = dict(adata.uns[cl_uns])
            for old, new in cmap.items():
                if old in d and new not in d:
                    d[new] = d[old]
            adata.uns[cl_uns] = d
    keep = prep.get("cell_line_keep")
    if keep:
        adata = adata[adata.obs["cell_line"].astype(str) == str(keep)].copy()
    batch = prep.get("batch")
    if batch:
        adata.obs["batch"] = str(batch)
        labels = list(data_cfg.get("batch_labels", [batch]))
        eye = np.eye(len(labels), dtype=np.float32)
        adata.uns["batch_embeddings"] = {lab: eye[i] for i, lab in enumerate(labels)}
    return adata


def build_gd_from_h5ad(path: str, data_cfg: DictConfig, ds_cfg: DictConfig | None = None) -> GroupedDistribution:
    """Group a raw .h5ad into an in-memory GroupedDistribution via the DataManager, mirroring
    scripts/prepare_sciplex_prophet.py. ``data_location = obsm[data_cfg.cell_embedding.key]`` is the
    flow latent (e.g. AE_128_opt / X_state). ``ds_cfg.prep`` applies per-dataset harmonization
    (cell_line canonicalization/filter, batch covariate). Optional ``log1p_dose_from`` (per-dataset via
    prep, else data_cfg) adds a log1p 'dose' scalar condition; ``extra_rep_keys`` (e.g. prophet) adds
    condition tokens looked up from uns — kept or dropped at sample time by the ablation's ConditionTransform."""
    import scanpy as sc
    from scaleflow.data import AnnDataLocation, DataManager
    prep = ds_cfg.get("prep", None) if ds_cfg is not None else None
    adata = prep_adata(sc.read_h5ad(path), prep, data_cfg)
    flag_key = str(data_cfg.dist_flag_key)
    adata = mark_control(adata, flag_key)
    # dose as a log1p scalar condition (obs['dose'] = log1p(<dose col>)); NOT given a rep_key, so
    # DataManager stores it as a raw scalar. Per-dataset dose column (tahoe 'dosage' / sciplex 'dose_value').
    dose_src = (OmegaConf.to_container(prep, resolve=True) or {}).get("log1p_dose_from") if prep else None
    dose_src = dose_src or data_cfg.get("log1p_dose_from", None)
    if dose_src and dose_src in adata.obs:
        adata.obs["dose"] = np.log1p(adata.obs[dose_src].astype("float32"))
    extra = data_cfg.get("extra_rep_keys", None)
    extra = {k: tuple(v) for k, v in OmegaConf.to_container(extra, resolve=True).items()} if extra else None
    dm = DataManager(
        dist_flag_key=flag_key,
        src_dist_keys=list(data_cfg.src_dist_keys),
        tgt_dist_keys=list(data_cfg.tgt_dist_keys),
        rep_keys=OmegaConf.to_container(data_cfg.rep_keys, resolve=True),
        data_location=AnnDataLocation().obsm[str(data_cfg.cell_embedding.key)],
        extra_rep_keys=extra,
    )
    return dm.prepare_data(adata, verbose=True)


def run(cfg: DictConfig, gds: dict | None = None) -> dict:
    # ── wandb sweep is dominant: init first, then overlay its params onto cfg ──
    # In sweep mode, project is set by the sweep yaml; pass None to avoid overriding.
    wandb_run = None
    if cfg.wandb.enabled or os.environ.get("WANDB_SWEEP_ID"):
        try:
            import wandb
            project = cfg.wandb.project if not os.environ.get("WANDB_SWEEP_ID") else None
            wandb_run = wandb.init(
                project=project,
                entity=cfg.wandb.get("entity"),
                name=cfg.wandb.get("run_name"),
                config=OmegaConf.to_container(cfg, resolve=True),
                settings=wandb.Settings(init_timeout=7200),
            )
            OmegaConf.set_struct(cfg, False)
            for k, v in dict(wandb_run.config).items():
                if "." in k or not isinstance(v, dict):
                    OmegaConf.update(cfg, k, v)
            OmegaConf.set_struct(cfg, True)
            print(f"  wandb run: {wandb_run.url}")
        except ImportError:
            print("  wandb not installed — skipping")

    # ── log resolved config so sweep overrides are visible in the run output ──
    m_cfg = cfg.model
    for ds_name in cfg.selected_datasets:
        ds = cfg.datasets[ds_name]
        print(f"  datasets.{ds_name}.path       = {ds.path}")
    print(f"  data.cell_embedding.key    = {cfg.data.cell_embedding.key}")
    print(f"  ablation.mode              = {cfg.ablation.mode}")
    print(f"  model.hidden_dims          = {list(m_cfg.hidden_dims)}")
    print(f"  model.decoder_dims         = {list(m_cfg.decoder_dims)}")
    print(f"  model.conditioning_key     = {m_cfg.conditioning_key}")
    print(f"  training.peak_lr           = {cfg.training.peak_lr}")
    print(f"  training.num_iterations    = {cfg.training.num_iterations}")
    print(f"  match_fn.epsilon           = {cfg.match_fn.epsilon}")

    # ── loading: must happen AFTER sweep overlay so path/latent is correct.
    #    .h5ad → in-memory GroupedDistribution via DataManager; .zarr → read_zarr. ──
    if gds is None:
        gds = {}
        for ds_name in cfg.selected_datasets:
            path = str(cfg.datasets[ds_name].path)
            print(f"Reading [{ds_name}] ← {path}")
            if path.endswith(".h5ad"):
                gds[ds_name] = build_gd_from_h5ad(path, cfg.data, cfg.datasets[ds_name])
            else:
                gds[ds_name] = GroupedDistribution.read_zarr(Path(path))

    mode       = cfg.ablation.mode
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag    = wandb_run.id if wandb_run is not None else "local"
    name       = f"model_{mode}_{run_tag}"
    ckpt_path  = output_dir / f"{name}_best_ckpt"
    transform  = utils.ConditionTransform(mode, seed=int(cfg.seed)) if mode != "prophet" else None

    print(f"\n{'='*64}")
    print(f"  {name}  |  mode={mode}  split_by={list(cfg.split.by)}  "
          f"solver={cfg.solver.solver_key}  conditioning={cfg.model.conditioning_key}")
    print(f"{'='*64}")

    # ── per-dataset split: `holdout: false` (e.g. tahoe) → all conditions to train (no val/test);
    #    `holdout: true` (default, e.g. sciplex) → split by drug → validation/recon land on it only. ──
    print("Splitting datasets …")
    data = {}
    for ds in gds:
        if bool(cfg.datasets[ds].get("holdout", True)):
            data[ds] = split_datasets(
                {ds: gds[ds]},
                holdout_combinations=bool(cfg.split.holdout_combinations),
                split_by=list(cfg.split.by),
                split_key="split",
                ratios=list(cfg.split.ratios),
                random_state=int(cfg.split.random_state),
            )[ds]
        else:
            data[ds] = {"train": gds[ds], "val": None, "test": None}
        print(f"  [{ds}] holdout={bool(cfg.datasets[ds].get('holdout', True))}")

    print("Creating samplers …")
    bs    = int(cfg.training.batch_size)
    n_val = cfg.training.get("n_val_conditions", None)
    n_val = int(n_val) if n_val is not None else None

    train_samplers, val_samplers, test_samplers = {}, {}, {}
    for ds in gds:
        seed = int(cfg.datasets[ds].get("seed", cfg.seed))
        train_samplers[ds] = ReservoirSampler(
            data[ds]["train"], np.random.default_rng(seed),
            batch_size=bs,
            pool_fraction=float(cfg.training.pool_fraction),
            replacement_prob=float(cfg.training.replacement_prob),
            condition_transform=transform,
        )
        if data[ds].get("val") is not None:
            val_samplers[ds] = ValidationSampler(
                data[ds]["val"],
                n_conditions_on_log_iteration=n_val,
                n_conditions_on_train_end=n_val,
                seed=seed,
                condition_transform=transform,
            )
        if data[ds].get("test") is not None:
            test_samplers[ds] = ValidationSampler(
                data[ds]["test"],
                n_conditions_on_log_iteration=None,
                n_conditions_on_train_end=None,
                seed=seed,
                condition_transform=transform,
            )

    train_sampler = CombinedSampler(
        samplers=train_samplers,
        rng=np.random.default_rng(int(cfg.split.random_state)),
    )
    train_sampler.init_sampler()
    for s in val_samplers.values():
        s.init_sampler()
    for s in test_samplers.values():
        s.init_sampler()

    sample_batch = train_sampler.sample()
    print(f"  latent dim (src_cell_data): {np.asarray(sample_batch['src_cell_data']).shape[-1]}")
    print(f"  condition keys: {[(k, tuple(np.asarray(v).shape)) for k, v in sample_batch['condition'].items()]}")

    print("Building model …")
    m            = cfg.model
    cond_key     = m.conditioning_key
    hidden_dims  = tuple(int(x) for x in m.hidden_dims)
    decoder_dims = tuple(int(x) for x in m.decoder_dims)
    if cond_key == "adaln_zero" and decoder_dims[-1] != hidden_dims[-1]:
        raise ValueError(
            f"adaln_zero requires decoder_dims[-1] == hidden_dims[-1]; "
            f"got {decoder_dims[-1]} vs {hidden_dims[-1]}."
        )

    # set-encoder: one pre-pool MLP per condition key (incl. cell_line)
    ce           = m.condition_encoder
    encoder_arch = OmegaConf.to_container(ce.encoder_arch, resolve=True)
    layers_before_pool = {k: encoder_arch for k in sample_batch["condition"]}
    layers_after_pool  = OmegaConf.to_container(ce.layers_after_pool, resolve=True)
    print(f"  set-encoder layers per modality: {list(layers_before_pool)}")

    optimizer, _ = utils.build_optimizer(cfg)
    predict_kwargs = OmegaConf.to_container(cfg.solver.get("predict_kwargs", {}), resolve=True)
    sf = ScaleFlow(solver=cfg.solver.solver_key)
    sf._validation_data["predict_kwargs"] = predict_kwargs
    sf.prepare_model(
        sample_batch=sample_batch,
        max_combination_length=int(m.max_combination_length),
        conditioning=cond_key,
        conditioning_kwargs=OmegaConf.to_container(m.conditioning_kwargs, resolve=True),
        pooling=ce.pooling,
        pooling_kwargs=OmegaConf.to_container(ce.pooling_kwargs, resolve=True),
        layers_before_pool=layers_before_pool,
        layers_after_pool=layers_after_pool,
        cond_output_dropout=float(ce.cond_output_dropout),
        hidden_dims=hidden_dims,
        decoder_dims=decoder_dims,
        condition_embedding_dim=int(m.condition_embedding_dim),
        match_fn=partial(match_linear, epsilon=float(cfg.match_fn.epsilon)),
        probability_path=OmegaConf.to_container(m.probability_path_kwargs, resolve=True),
        optimizer=optimizer,
    )
    n_params = sum(x.size for x in jax.tree.leaves(sf.solver.vf_state.params))
    print(f"  total parameters: {n_params:,}")

    val_log_path = str(output_dir / f"{name}_val_metrics.json")
    cbs = [
        Metrics(
            metrics=["r_squared", "e_distance", "mmd"],
            metric_aggregations=["mean"],
        ),
        callbacks.ValMetricsLogger(save_path=val_log_path, valid_freq=int(cfg.training.valid_freq), wandb_run=wandb_run, debug=bool(cfg.match_fn.get("debug", False))),
        callbacks.BestModelCheckpoint(save_path=ckpt_path, wandb_run=wandb_run, metric=cfg.training.checkpoint_metric),
        temp_edit.EffectSizeMonitor(valid_freq=int(cfg.training.valid_freq), wandb_run=wandb_run),
    ]

    # ── optional gene-space reconstruction metrics (decode latent → genes) ──
    recon_cb  = None
    recon_cfg = cfg.get("recon", {})
    dec_path  = recon_cfg.get("decoder_path")
    h5ad_path = recon_cfg.get("h5ad_path")
    if dec_path and h5ad_path:
        import scanpy as sc
        from scaleflow.model._recon import ReconDecoder
        print(f"Loading ReconDecoder from {dec_path} …")
        # decoders under recon_weights/ are saved in the cloudpickle .pkl format (ReconDecoder.save),
        # not the orbax params/ dir that callbacks.load_recon_decoder expects → use ReconDecoder.load.
        recon_dec = ReconDecoder.load(str(dec_path))
        # apply the same per-dataset prep (cell_line filter + batch inject) so obs matches the cond_key
        # (cell_line, [batch], drug, dose) for the evaluated (holdout) dataset.
        adata_recon = prep_adata(sc.read_h5ad(str(h5ad_path)), recon_cfg.get("prep", None), cfg.data)
        log_dose_key = recon_cfg.get("log_dose_obs_key", None)
        # obsm key of the latent the model predicts — used to decode the control latent so the pred
        # delta is decode(pred) − decode(ctrl_latent) (convention B; offset cancels). Resolve from GD.
        emb_obsm_key = recon_cfg.get("emb_obsm_key")
        if not emb_obsm_key:
            try:
                emb_obsm_key = gds[cfg.selected_datasets[0]].annotation.data_location.to_path()[-1][1]
            except Exception:  # noqa: BLE001
                emb_obsm_key = None
        recon_cb = callbacks.ReconMetricsLogger(
            decoder=recon_dec,
            adata=adata_recon,
            condition_obs_keys=list(recon_cfg.condition_obs_keys),
            cell_line_obs_key=str(recon_cfg.cell_line_obs_key),
            control_obs_key=str(recon_cfg.get("control_obs_key", "control")),
            log_dose_obs_key=str(log_dose_key) if log_dose_key else None,
            emb_obsm_key=str(emb_obsm_key) if emb_obsm_key else None,
            valid_freq=int(cfg.training.valid_freq),
            wandb_run=wandb_run,
        )
        cbs.append(recon_cb)
        print(f"  recon metrics enabled: {recon_dec.input_key} → {len(recon_dec.var_names or [])} genes "
              f"(pred delta vs decode(ctrl latent '{emb_obsm_key}'))")

    monitor_metrics = ["loss"]
    for ds in val_samplers:
        monitor_metrics += [
            f"{ds}_r_squared_mean",
            f"{ds}_e_distance_mean",
            f"{ds}_mmd_mean",
            f"{ds}_nn_displacement_corr",
            f"{ds}_gap_closure_mean",
        ]
    if dec_path and h5ad_path:
        monitor_metrics += ["val_recon_r2_delta", "val_recon_pearson_r_delta"]

    print(f"Training {int(cfg.training.num_iterations)} iterations "
          f"(val every {int(cfg.training.valid_freq)} steps) …")
    t0 = time.perf_counter()
    sf.train(
        train_dataloader=train_sampler,
        val_dataloader=val_samplers,
        num_iterations=int(cfg.training.num_iterations),
        valid_freq=int(cfg.training.valid_freq),
        log_every=int(cfg.training.get("log_every", 1000)),
        callbacks=cbs,
        monitor_metrics=monitor_metrics,
    )
    print(f"  training done in {(time.perf_counter() - t0) / 60:.1f} min")
    callbacks.save_logs(name, sf.trainer.training_logs, output_dir)

    if ckpt_path.exists():
        print(f"Loading best checkpoint from {ckpt_path} …")
        import orbax.checkpoint as ocp
        target      = callbacks._solver_params(sf.solver)
        best_params = ocp.PyTreeCheckpointer().restore(str(ckpt_path), item=target)
        callbacks.restore_solver_params(sf.solver, best_params)
    else:
        print("  no checkpoint found — using final iterate")
    best_solver = sf.solver

    print("Evaluating on test set …")
    test_metrics = callbacks.evaluate_test(best_solver, test_samplers)

    # ── gene-space recon metrics on the test set (test_recon_*) ──
    test_recon = {}
    if recon_cb is not None:
        print("Evaluating gene-space recon on test set …")
        test_recon = recon_cb.evaluate_test(best_solver, test_samplers)

    result_path = output_dir / f"{name}_results.pkl"
    with open(result_path, "wb") as f:
        cloudpickle.dump({**test_metrics, "recon": test_recon}, f)
    print(f"  test results saved → {result_path}")

    if wandb_run is not None:
        test_log = {f"test_{k}": v for k, v in test_metrics["aggregated"].items()}
        for dsname, dsres in test_metrics["per_dataset"].items():
            for k, v in dsres["aggregated"].items():
                test_log[f"test_{dsname}_{k}"] = v
        test_log.update(test_recon)  # test_recon_r2_delta / pearson (+ medians)
        wandb_run.log(test_log)
        for k, v in test_log.items():
            wandb_run.summary[k] = v

    # ── effect-size diagnostics over train / val / test (one predict pass each) ──
    diag_cfg  = cfg.get("diagnostics", {})
    n_diag    = int(diag_cfg.get("n_conditions", 100))
    max_cells = int(diag_cfg.get("max_cells", 2000))
    print(f"Running effect-size diagnostics ({n_diag} conditions/split) …")
    diag_samplers = temp_edit.make_diagnostic_samplers(
        data, n_conditions=n_diag, transform=transform, seed=int(cfg.seed)
    )
    temp_edit.full_diagnostics(
        best_solver, diag_samplers, output_dir, name,
        wandb_run=wandb_run, max_cells=max_cells, seed=int(cfg.seed),
    )

    print(f"\n{'='*64}")
    print(f"  Final test metrics — {name}")
    print(f"{'='*64}")
    for k, v in test_metrics["aggregated"].items():
        print(f"  {k:<18} {v:.4f}")

    if wandb_run is not None:
        wandb_run.finish()

    return {"solver": best_solver, "test_metrics": test_metrics}


@hydra.main(config_path="config", config_name="train_comparison", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        run(cfg)  # loading happens inside run() after wandb overlay
    except Exception:
        try:
            import wandb
            if wandb.run is not None:
                wandb.run.finish(exit_code=1)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
