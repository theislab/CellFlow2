"""
train_comparison.py

Trains a CellFlow2 model on Tahoe data.

Usage
─────
  # Use default config:
  python train_comparison.py

  # Override specific keys:
  python train_comparison.py --model prophet
  python train_comparison.py --model random --split.by cell_line

  # Custom config file:
  python train_comparison.py --config config/default.yaml --model prophet

  # Enable wandb logging:
  python train_comparison.py --wandb

  # Run as wandb sweep agent (agent passes params automatically):
  wandb sweep config/sweep_model.yaml
  wandb agent <sweep_id>
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

# ── JAX persistent compilation cache ─────────────────────────────────────────
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/storage/jax_cache")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "5")
# ─────────────────────────────────────────────────────────────────────────────

import cloudpickle
import jax
import numpy as np
from tqdm import tqdm

from scaleflow.data import GroupedDistribution, split_datasets
from scaleflow.data._dataloader import CombinedSampler, ReservoirSampler, ValidationSampler
from scaleflow.model import ScaleFlow
from scaleflow.training import Metrics
from scaleflow.training._callbacks import ComputationCallback
from scaleflow.metrics._metrics import (
    compute_r_squared,
    compute_e_distance_fast,
    compute_scalar_mmd,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "zarr_path":  "/storage/pancellflow/tahoe.zarr",
    "output_dir": "/storage/pancellflow/outputs",
    "seed": 42,
    "split": {
        "by":     "drug",
        "ratios": [0.7, 0.2, 0.1],
    },
    "training": {
        "batch_size":       1024,
        "pool_fraction":    0.7,
        "replacement_prob": 0.5,
        "num_iterations":   200_000,
        "valid_freq":       20_000,
        "n_val_conditions": 20,
    },
    "model": "default",
    "wandb": {
        "enabled":  False,
        "project":  "pancellflow",
        "entity":   None,
        "run_name": None,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str | None, overrides: dict) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if config_path:
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f)
        cfg = deep_merge(cfg, file_cfg or {})
    cfg = deep_merge(cfg, overrides)
    return cfg


def set_nested(d: dict, dotkey: str, value: Any) -> None:
    """Set d['a']['b'] from dotkey='a.b'."""
    keys = dotkey.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


# ─────────────────────────────────────────────────────────────────────────────
# ConditionTransform — replaces wrapper sampler classes
# Passed directly into ReservoirSampler and ValidationSampler.
# ─────────────────────────────────────────────────────────────────────────────
class ConditionTransform:
    """Transforms condition dicts at sample time.

    mode="default" : drops the 'prophet' key entirely.
    mode="prophet" : no-op, returns condition unchanged.
    mode="random"  : replaces 'prophet' values with random vectors.
                     Training (cond_key=None)  → new random each call.
                     Validation (cond_key=str) → fixed random per condition.
    """

    def __init__(self, mode: str, seed: int = 42):
        self.mode            = mode
        self._rng            = np.random.default_rng(seed)
        self._seed           = seed
        self._cache: dict    = {}   # (cond_key, emb_key, shape) → fixed array

    def __call__(self, cond: dict, cond_key: str | None = None) -> dict:
        if self.mode == "prophet":
            return cond

        result = {}
        for k, v in cond.items():
            if k == "prophet":
                if self.mode == "default":
                    continue                          # drop key entirely
                elif self.mode == "random":
                    if cond_key is not None:
                        # Fixed per condition across val steps
                        cache_key = (cond_key, k, v.shape)
                        if cache_key not in self._cache:
                            int_seed = abs(hash(cond_key + k + str(self._seed))) % (2 ** 31)
                            self._cache[cache_key] = (
                                np.random.default_rng(int_seed)
                                .standard_normal(v.shape)
                                .astype(v.dtype)
                            )
                        result[k] = self._cache[cache_key]
                    else:
                        # New random each training step
                        result[k] = self._rng.standard_normal(v.shape).astype(v.dtype)
            else:
                result[k] = v
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Dataset statistics
# ─────────────────────────────────────────────────────────────────────────────
def print_dataset_stats(gd: GroupedDistribution, label: str = "Full dataset") -> None:
    ann  = gd.annotation
    data = gd.data

    n_src = len(data.src_data)
    n_tgt = len(data.tgt_data)

    src_labels = list(ann.src_dist_idx_to_labels.values())
    tgt_labels = list(ann.tgt_dist_idx_to_labels.values())

    cell_lines = sorted({
        str(lbl[0]) if isinstance(lbl, (list, tuple)) else str(lbl)
        for lbl in src_labels
    })
    drugs = sorted({
        str(lbl[1]) if isinstance(lbl, (list, tuple)) and len(lbl) > 1
        else str(lbl[0]) if isinstance(lbl, (list, tuple)) else str(lbl)
        for lbl in tgt_labels
    })

    src_sizes = [v.shape[0] for v in data.src_data.values()]
    tgt_sizes = [v.shape[0] for v in data.tgt_data.values()]
    cond_keys = list(next(iter(data.conditions.values())).keys()) if data.conditions else []

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Cell lines      : {len(cell_lines)}  →  {', '.join(cell_lines)}")
    print(f"  Drugs           : {len(drugs)}")
    print(f"  Conditions      : {n_tgt}  (cell_line × drug pairs)")
    print(f"  Src dists       : {n_src}  (one control pool per cell line)")
    print(f"  Control cells   : {sum(src_sizes):,}  "
          f"(min={min(src_sizes):,}  max={max(src_sizes):,}  "
          f"mean={int(np.mean(src_sizes)):,})")
    print(f"  Treated cells   : {sum(tgt_sizes):,}  "
          f"(min={min(tgt_sizes):,}  max={max(tgt_sizes):,}  "
          f"mean={int(np.mean(tgt_sizes)):,})")
    print(f"  Condition keys  : {cond_keys}")
    print(f"{'─'*60}")


def print_split_stats(train_gd, val_gd, test_gd, split_by: str) -> None:
    def label_sets(gd):
        tgt_labels = list(gd.annotation.tgt_dist_idx_to_labels.values())
        src_labels = list(gd.annotation.src_dist_idx_to_labels.values())
        drugs = {
            str(lbl[1]) if isinstance(lbl, (list, tuple)) and len(lbl) > 1
            else str(lbl[0]) if isinstance(lbl, (list, tuple)) else str(lbl)
            for lbl in tgt_labels
        }
        cell_lines = {
            str(lbl[0]) if isinstance(lbl, (list, tuple)) else str(lbl)
            for lbl in src_labels
        }
        return drugs, cell_lines

    tr_drugs, tr_cls = label_sets(train_gd)
    va_drugs, va_cls = label_sets(val_gd)
    te_drugs, te_cls = label_sets(test_gd)

    print(f"\n  Split (by {split_by})")
    print(f"    {'':8}  {'conditions':>10}  {'drugs':>6}  {'cell lines':>10}")
    print(f"    {'Train':8}  {len(train_gd.data.tgt_data):>10}  {len(tr_drugs):>6}  "
          f"{len(tr_cls):>10}  {sorted(tr_cls)}")
    print(f"    {'Val':8}  {len(val_gd.data.tgt_data):>10}  {len(va_drugs):>6}  "
          f"{len(va_cls):>10}  {sorted(va_cls)}")
    print(f"    {'Test':8}  {len(test_gd.data.tgt_data):>10}  {len(te_drugs):>6}  "
          f"{len(te_cls):>10}  {sorted(te_cls)}")


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_logs(name: str, logs: dict, output_dir: Path) -> None:
    path = output_dir / f"{name}_training_logs.json"
    serialisable = {k: [float(v) for v in vals] for k, vals in logs.items() if vals}
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  logs saved  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────
class ValMetricsLogger(ComputationCallback):
    """Appends val metrics to a JSON file immediately after each validation step."""

    def __init__(self, save_path: str, valid_freq: int, wandb_run=None):
        self.save_path   = save_path
        self._valid_freq = valid_freq
        self._step       = 0
        self._wandb_run  = wandb_run

    def on_train_begin(self, *args, **kwargs) -> None:
        self._step = 0

    def _compute_and_save(self, valid_true_data, valid_pred_data) -> dict:
        r2s, eds, mmds = [], [], []
        for dataset_key in valid_true_data:
            for cond_key, true_arr in valid_true_data[dataset_key].items():
                pred_arr = valid_pred_data[dataset_key].get(cond_key)
                if pred_arr is None:
                    continue
                y_true = np.array(true_arr)
                y_pred = np.array(pred_arr)
                r2s.append(float(compute_r_squared(y_true, y_pred)))
                eds.append(float(compute_e_distance_fast(y_true, y_pred)))
                mmds.append(float(compute_scalar_mmd(y_true, y_pred)))

        if not r2s:
            return {}

        entry = {
            "step":         self._step,
            "n_conditions": len(r2s),
            "r_squared":    float(np.mean(r2s)),
            "e_distance":   float(np.mean(eds)),
            "mmd":          float(np.mean(mmds)),
        }

        if os.path.exists(self.save_path):
            with open(self.save_path) as f:
                entries = json.load(f)
        else:
            entries = []
        entries.append(entry)
        with open(self.save_path, "w") as f:
            json.dump(entries, f, indent=2)

        print(f"    val metrics saved → step={self._step}  "
              f"R²={entry['r_squared']:.4f}  "
              f"E-dist={entry['e_distance']:.4f}  "
              f"MMD={entry['mmd']:.4f}")

        if self._wandb_run is not None:
            self._wandb_run.log({
                "val_r_squared":  entry["r_squared"],
                "val_e_distance": entry["e_distance"],
                "val_mmd":        entry["mmd"],
            })

        return {}

    def on_log_iteration(self, valid_source_data, valid_true_data,
                         valid_pred_data, solver, **kwargs) -> dict:
        self._step += self._valid_freq
        self._compute_and_save(valid_true_data, valid_pred_data)
        return {}

    def on_train_end(self, valid_source_data, valid_true_data,
                     valid_pred_data, solver, **kwargs) -> dict:
        self._compute_and_save(valid_true_data, valid_pred_data)
        return {}


class BestModelCheckpoint(ComputationCallback):
    def __init__(self, save_path: str, wandb_run=None):
        self.save_path  = save_path
        self.best_r2    = -np.inf
        self._wandb_run = wandb_run

    def on_train_begin(self, *args, **kwargs) -> None:
        self.best_r2 = -np.inf

    def on_log_iteration(self, valid_source_data, valid_true_data,
                         valid_pred_data, solver, **kwargs) -> dict:
        scores = []
        for dataset_key in valid_true_data:
            for cond_key, true_arr in valid_true_data[dataset_key].items():
                pred_arr = valid_pred_data[dataset_key].get(cond_key)
                if pred_arr is None:
                    continue
                scores.append(compute_r_squared(np.array(true_arr), np.array(pred_arr)))
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


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_split(cfg: dict):
    zarr_path = Path(cfg["zarr_path"])
    split_by  = cfg["split"]["by"]
    ratios    = cfg["split"]["ratios"]
    seed      = cfg["seed"]

    gd = GroupedDistribution.read_zarr(zarr_path)
    print_dataset_stats(gd, "Full dataset")

    splits = split_datasets(
        {"gd": gd},
        split_by=[split_by],
        split_key="split",
        ratios=ratios,
        random_state=seed,
        holdout_combinations=False,
    )
    train_gd = splits["gd"]["train"]
    val_gd   = splits["gd"]["val"]
    test_gd  = splits["gd"]["test"]
    print_split_stats(train_gd, val_gd, test_gd, split_by)
    return train_gd, val_gd, test_gd


def make_samplers(train_gd, val_gd, test_gd, cfg: dict, transform: ConditionTransform | None):
    seed      = cfg["seed"]
    tcfg      = cfg["training"]
    rng       = np.random.default_rng(seed)
    n_val     = tcfg["n_val_conditions"]

    train_sampler = CombinedSampler(
        samplers={"gd": ReservoirSampler(
            train_gd, rng,
            batch_size=tcfg["batch_size"],
            pool_fraction=tcfg["pool_fraction"],
            replacement_prob=tcfg["replacement_prob"],
            condition_transform=transform,
        )},
        rng=rng,
    )
    val_sampler  = ValidationSampler(
        val_gd,
        n_conditions_on_log_iteration=n_val,
        n_conditions_on_train_end=n_val,
        seed=seed,
        condition_transform=transform,
    )
    test_sampler = ValidationSampler(
        test_gd,
        n_conditions_on_log_iteration=None,
        n_conditions_on_train_end=None,
        seed=seed,
        condition_transform=transform,
    )

    train_sampler.init_sampler()
    val_sampler.init_sampler()
    test_sampler.init_sampler()
    return train_sampler, val_sampler, test_sampler


# ─────────────────────────────────────────────────────────────────────────────
# Test evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_test(solver, test_sampler) -> dict:
    batch = test_sampler.sample(mode="on_train_end")
    src   = batch["source"]
    cond  = batch["condition"]
    true  = batch["target"]

    print(f"  predicting {len(src)} test conditions …")
    pred = jax.tree.map(solver.predict, src, cond)

    per_condition = {}
    for cond_key in tqdm(sorted(true.keys(), key=str), desc="  test metrics"):
        y_true = np.array(true[cond_key])
        y_pred = np.array(pred[cond_key])
        per_condition[str(cond_key)] = {
            "r_squared":  float(compute_r_squared(y_true, y_pred)),
            "e_distance": float(compute_e_distance_fast(y_true, y_pred)),
            "mmd":        float(compute_scalar_mmd(y_true, y_pred)),
        }

    metrics    = ["r_squared", "e_distance", "mmd"]
    aggregated = {m: float(np.mean([v[m] for v in per_condition.values()])) for m in metrics}
    return {"per_condition": per_condition, "aggregated": aggregated}


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────
def train_model(cfg: dict, wandb_run=None) -> dict:
    mode       = cfg["model"]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    name_map = {"default": "model_default", "prophet": "model_prophet", "random": "model_random"}
    name     = name_map[mode]
    ckpt_path = str(output_dir / f"{name}_best.pkl")

    print(f"\n{'='*64}")
    print(f"  {name}  |  mode={mode}  split_by={cfg['split']['by']}")
    print(f"{'='*64}")

    # Build condition transform (None = no-op for prophet)
    transform = ConditionTransform(mode, seed=cfg["seed"]) if mode != "prophet" else None

    t0 = time.perf_counter()
    print("Loading & splitting data …")
    train_gd, val_gd, test_gd = make_split(cfg)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    print("Building samplers …")
    train_sampler, val_sampler, test_sampler = make_samplers(
        train_gd, val_gd, test_gd, cfg, transform
    )

    sample_batch    = train_sampler.sample()
    cond_keys       = list(sample_batch["condition"].keys())
    data_dim        = sample_batch["src_cell_data"].shape[-1]
    cond_input_dims = {k: sample_batch["condition"][k].shape[-1] for k in cond_keys}
    print(f"  data_dim={data_dim}  condition_keys={cond_keys}")

    print("Building model …")
    sf = ScaleFlow()
    sf.prepare_model(sample_batch=sample_batch, max_combination_length=1)

    vf       = sf.solver.vf
    n_params = sum(x.size for x in jax.tree.leaves(sf.solver.vf_state.params))

    prophet_status = {"default": "NO  ✗", "prophet": "YES ✓", "random": "RANDOM ✓"}[mode]
    print(f"\n{'─'*60}")
    print(f"  Model architecture")
    print(f"{'─'*60}")
    print(f"  Total parameters      : {n_params:,}")
    print(f"  Data dim              : {data_dim}")
    print(f"  Prophet embedding     : {prophet_status}")
    for k, d in cond_input_dims.items():
        print(f"  Input dim [{k:<10}]: {d}")
    print(f"{'─'*60}")

    tcfg         = cfg["training"]
    valid_freq   = tcfg["valid_freq"]
    val_log_path = str(output_dir / f"{name}_val_metrics.json")

    callbacks = [
        Metrics(
            metrics=["r_squared", "e_distance", "mmd"],
            metric_aggregations=["mean"],
            use_gpu_optimized=True,
            precision="bfloat16",
        ),
        ValMetricsLogger(save_path=val_log_path, valid_freq=valid_freq, wandb_run=wandb_run),
        BestModelCheckpoint(save_path=ckpt_path, wandb_run=wandb_run),
    ]

    print(f"Training {tcfg['num_iterations']} iterations "
          f"(val every {valid_freq} steps, {tcfg['n_val_conditions']} conditions) …")
    t0 = time.perf_counter()
    sf.train(
        train_dataloader=train_sampler,
        val_dataloader={"val": val_sampler},
        num_iterations=tcfg["num_iterations"],
        valid_freq=valid_freq,
        callbacks=callbacks,
        monitor_metrics=["loss", "val_r_squared_mean", "val_e_distance_mean", "val_mmd_mean"],
    )
    elapsed = (time.perf_counter() - t0) / 60
    print(f"  training done in {elapsed:.1f} min")

    save_logs(name, sf.trainer.training_logs, output_dir)


    if os.path.exists(ckpt_path):
        print(f"Loading best checkpoint from {ckpt_path} …")
        with open(ckpt_path, "rb") as f:
            best_solver = cloudpickle.load(f)
    else:
        print("  no checkpoint found – using final iterate")
        best_solver = sf.solver

    print("Evaluating on test set …")
    test_metrics = evaluate_test(best_solver, test_sampler)

    result_path = output_dir / f"{name}_results.pkl"
    with open(result_path, "wb") as f:
        cloudpickle.dump(test_metrics, f)
    print(f"  test results saved → {result_path}")

    if wandb_run is not None:
        test_log = {f"test_{metric}": val for metric, val in test_metrics["aggregated"].items()}
        wandb_run.log(test_log)                      # shows up as a chart point
        for k, v in test_log.items():
            wandb_run.summary[k] = v                 # also pinned in Summary panel

    return {"solver": best_solver, "test_metrics": test_metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--model", choices=["default", "prophet", "random"],
                        help="Override config model")
    parser.add_argument("--split.by", dest="split_by",
                        choices=["drug", "cell_line"],
                        help="Override config split.by")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    # Generic override: --set training.batch_size=2048
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE",
                        help="Override any config key using dot notation, e.g. training.batch_size=2048")
    args, _ = parser.parse_known_args()  # ignore extra args passed by wandb agent

    # ── Build config ─────────────────────────────────────────────────────────
    overrides: dict = {}
    if args.model:
        overrides["model"] = args.model
    if args.split_by:
        overrides.setdefault("split", {})["by"] = args.split_by
    if args.set:
        for kv in args.set:
            key, _, val = kv.partition("=")
            # Try to parse as int/float/bool, else keep as string
            for cast in (int, float):
                try:
                    val = cast(val)
                    break
                except ValueError:
                    pass
            if val == "true":  val = True
            if val == "false": val = False
            set_nested(overrides, key, val)
    if args.wandb:
        overrides.setdefault("wandb", {})["enabled"] = True

    cfg = load_config(args.config, overrides)

    # ── Optional wandb ────────────────────────────────────────────────────────
    import os
    wandb_run = None
    if cfg["wandb"]["enabled"] or os.environ.get("WANDB_SWEEP_ID"):
        try:
            import wandb
            wcfg = cfg["wandb"]
            # Flatten config for wandb (sweep may override values)
            flat_cfg = {
                "model":                    cfg["model"],
                "split.by":                 cfg["split"]["by"],
                "training.batch_size":      cfg["training"]["batch_size"],
                "training.pool_fraction":   cfg["training"]["pool_fraction"],
                "training.replacement_prob":cfg["training"]["replacement_prob"],
                "training.num_iterations":  cfg["training"]["num_iterations"],
                "training.valid_freq":      cfg["training"]["valid_freq"],
                "training.n_val_conditions":cfg["training"]["n_val_conditions"],
                "seed":                     cfg["seed"],
            }
            # If wandb agent already initialized a run (sweep mode), reuse it
            if wandb.run is not None:
                wandb_run = wandb.run
                wandb_run.config.update(flat_cfg, allow_val_change=True)
            else:
                wandb_run = wandb.init(
                    project=wcfg.get("project", "pancellflow"),
                    entity=wcfg.get("entity"),
                    name=wcfg.get("run_name"),
                    config=flat_cfg,
                )
            # Sweep may override config — read back
            sweep_cfg = dict(wandb_run.config)
            for flat_key, val in sweep_cfg.items():
                set_nested(cfg, flat_key, val)
            print(f"  wandb run: {wandb_run.url}")
        except ImportError:
            print("  wandb not installed — skipping")

    result = train_model(cfg, wandb_run=wandb_run)

    print(f"\n{'='*64}")
    print(f"  Final test metrics — {cfg['model']}")
    print(f"{'='*64}")
    for metric, val in result["test_metrics"]["aggregated"].items():
        print(f"  {metric:<20} {val:.4f}")

    if wandb_run is not None:
        wandb_run.finish()