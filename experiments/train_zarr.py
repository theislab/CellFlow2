"""
train_zarr.py — train a CellFlow2 model from prebuilt zarr GroupedDistributions.

train_comparison-style training: full Metrics callback + per-step JSON val logging
(incl. r_squared_delta) + best-checkpoint + test evaluation + manual wandb.

  python experiments/train_zarr.py
  python experiments/train_zarr.py ablation=prophet
  python experiments/train_zarr.py solver=eqm conditioning=adaln_zero
  python experiments/train_zarr.py selected_datasets=[tahoe] wandb.enabled=true
"""
import os

# JAX env tweaks must be set before anything imports jax (scaleflow / utils do).
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/storage/jax_cache")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "5")

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
from scaleflow.training import Metrics
from scaleflow.utils import match_linear

import utils       # ConditionTransform, build_optimizer
import callbacks   # ValMetricsLogger, BestModelCheckpoint, evaluate_test, save_logs


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────
def run(cfg: DictConfig, gds: dict) -> dict:
    """Split → samplers → model → train → test, for already-loaded GroupedDistributions."""
    mode       = cfg.ablation.mode
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name       = f"model_{mode}"
    ckpt_path  = str(output_dir / f"{name}_best.pkl")
    transform  = utils.ConditionTransform(mode, seed=int(cfg.seed)) if mode != "prophet" else None

    print(f"\n{'='*64}")
    print(f"  {name}  |  mode={mode}  split_by={list(cfg.split.by)}  "
          f"solver={cfg.solver.solver_key}  conditioning={cfg.model.conditioning_key}")
    print(f"{'='*64}")

    # ── Optional wandb (manual, train_comparison style) ───────────────────────
    wandb_run = None
    if cfg.wandb.enabled:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.get("entity"),
                name=cfg.wandb.get("run_name"),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            print(f"  wandb run: {wandb_run.url}")
        except ImportError:
            print("  wandb not installed — skipping")

    # ── Split ─────────────────────────────────────────────────────────────────
    print("Splitting datasets …")
    data = split_datasets(
        gds,
        holdout_combinations=bool(cfg.split.holdout_combinations),
        split_by=list(cfg.split.by),
        split_key="split",
        ratios=list(cfg.split.ratios),
        random_state=int(cfg.split.random_state),
    )

    # ── Samplers (train / val / test) ─────────────────────────────────────────
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
        val_samplers[ds] = ValidationSampler(
            data[ds]["val"],
            n_conditions_on_log_iteration=n_val,
            n_conditions_on_train_end=n_val,
            seed=seed,
            condition_transform=transform,
        )
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
    print(f"  condition keys: {[(k, tuple(v.shape)) for k, v in sample_batch['condition'].items()]}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Building model …")
    m            = cfg.model
    cond_key     = m.conditioning_key
    cond_kwargs  = OmegaConf.to_container(m.conditioning_kwargs, resolve=True)
    hidden_dims  = tuple(int(x) for x in m.hidden_dims)
    decoder_dims = tuple(int(x) for x in m.decoder_dims)

    if cond_key == "adaln_zero" and decoder_dims[-1] != hidden_dims[-1]:
        raise ValueError(
            f"For 'adaln_zero' conditioning, decoder_dims[-1] must equal hidden_dims[-1]. "
            f"Got {decoder_dims[-1]} vs {hidden_dims[-1]}."
        )

    optimizer, _lr_schedule = utils.build_optimizer(cfg)
    sf = ScaleFlow(solver=cfg.solver.solver_key)
    sf.prepare_model(
        sample_batch=sample_batch,
        max_combination_length=int(m.max_combination_length),
        conditioning=cond_key,
        conditioning_kwargs=cond_kwargs,
        hidden_dims=hidden_dims,
        decoder_dims=decoder_dims,
        condition_embedding_dim=int(m.condition_embedding_dim),
        match_fn=partial(match_linear, epsilon=float(cfg.match_fn.epsilon)),
        probability_path=OmegaConf.to_container(m.probability_path_kwargs, resolve=True),
        optimizer=optimizer,
    )
    n_params = sum(x.size for x in jax.tree.leaves(sf.solver.vf_state.params))
    print(f"  total parameters: {n_params:,}")

    # ── Callbacks (train_comparison style + r_squared_delta) ──────────────────
    val_log_path = str(output_dir / f"{name}_val_metrics.json")
    cbs = [
        Metrics(
            metrics=["r_squared", "e_distance", "mmd"],
            metric_aggregations=["mean"],
            use_gpu_optimized=True,
            precision="bfloat16",
        ),
        callbacks.ValMetricsLogger(save_path=val_log_path, valid_freq=int(cfg.training.valid_freq), wandb_run=wandb_run),
        callbacks.BestModelCheckpoint(save_path=ckpt_path, wandb_run=wandb_run),
    ]

    # Metrics callback keys metrics as "{ds}_{metric}_mean"; ValMetricsLogger adds
    # "{ds}_r_squared_delta_mean". Monitor all of them per dataset.
    monitor_metrics = ["loss"]
    for ds in val_samplers:
        monitor_metrics += [
            f"{ds}_r_squared_mean",
            f"{ds}_e_distance_mean",
            f"{ds}_mmd_mean",
            f"{ds}_r_squared_delta_mean",
        ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"Training {int(cfg.training.num_iterations)} iterations "
          f"(val every {int(cfg.training.valid_freq)} steps) …")
    t0 = time.perf_counter()
    sf.train(
        train_dataloader=train_sampler,
        val_dataloader=val_samplers,
        num_iterations=int(cfg.training.num_iterations),
        valid_freq=int(cfg.training.valid_freq),
        callbacks=cbs,
        monitor_metrics=monitor_metrics,
    )
    print(f"  training done in {(time.perf_counter() - t0) / 60:.1f} min")
    callbacks.save_logs(name, sf.trainer.training_logs, output_dir)

    # ── Load best checkpoint, evaluate on test ────────────────────────────────
    if os.path.exists(ckpt_path):
        print(f"Loading best checkpoint from {ckpt_path} …")
        with open(ckpt_path, "rb") as f:
            best_solver = cloudpickle.load(f)
    else:
        print("  no checkpoint found — using final iterate")
        best_solver = sf.solver

    print("Evaluating on test set …")
    test_metrics = callbacks.evaluate_test(best_solver, test_samplers)

    result_path = output_dir / f"{name}_results.pkl"
    with open(result_path, "wb") as f:
        cloudpickle.dump(test_metrics, f)
    print(f"  test results saved → {result_path}")

    if wandb_run is not None:
        test_log = {f"test_{k}": v for k, v in test_metrics["aggregated"].items()}
        for dsname, dsres in test_metrics["per_dataset"].items():
            for k, v in dsres["aggregated"].items():
                test_log[f"test_{dsname}_{k}"] = v
        wandb_run.log(test_log)
        for k, v in test_log.items():
            wandb_run.summary[k] = v

    print(f"\n{'='*64}")
    print(f"  Final test metrics — {name}")
    print(f"{'='*64}")
    for k, v in test_metrics["aggregated"].items():
        print(f"  {k:<18} {v:.4f}")

    if wandb_run is not None:
        wandb_run.finish()

    return {"solver": best_solver, "test_metrics": test_metrics}


@hydra.main(config_path="config", config_name="train_zarr", version_base=None)
def main(cfg: DictConfig) -> None:
    gds = {}
    for name in cfg.selected_datasets:
        path = str(cfg.datasets[name].path)
        print(f"Reading [{name}] ← {path}")
        gds[name] = GroupedDistribution.read_zarr(Path(path))
    run(cfg, gds)


if __name__ == "__main__":
    main()
