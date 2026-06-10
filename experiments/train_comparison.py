"""
train_comparison.py

Trains a CellFlow2 model on Tahoe data.

All defaults live in experiments/base.yaml (the single source of truth); it is
always loaded first. An optional --config file overlays it, then --set CLI
overrides, then wandb sweep params.

Run from the repo root:
─────
  # Use base config as-is:
  python experiments/train_comparison.py

  # Override specific keys:
  python experiments/train_comparison.py --model prophet
  python experiments/train_comparison.py --model random --split.by cell_line
  python experiments/train_comparison.py --set arch.hidden_dims=[1024] optimizer.peak_lr=2e-4

  # Overlay another YAML on top of base.yaml (path resolved next to the script):
  python experiments/train_comparison.py --config multi_dataset.yaml

  # Enable wandb logging:
  python experiments/train_comparison.py --wandb

  # Run as wandb sweep agent (agent passes params automatically):
  wandb sweep experiments/sweep_single.yaml
  wandb agent <sweep_id>
"""

import argparse
import ast
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

from functools import partial

import cloudpickle
import jax
import numpy as np
import optax
from tqdm import tqdm

from scaleflow.data import GroupedDistribution, split_datasets
from scaleflow.data._dataloader import CombinedSampler, ReservoirSampler, ValidationSampler
from scaleflow.model import ScaleFlow
from scaleflow.training import Metrics
from scaleflow.training._callbacks import ComputationCallback
from scaleflow.utils import match_linear
from scaleflow.metrics._metrics import (
    compute_r_squared,
    compute_e_distance_fast,
    compute_scalar_mmd,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
#
# All defaults live in experiments/base.yaml — the single source of truth.
# There are no hardcoded config values in this file. base.yaml (resolved next to
# this script) is always loaded first, then an optional --config overlay, then
# CLI --set overrides.
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_DIR       = Path(__file__).resolve().parent          # experiments/
BASE_CONFIG_PATH = CONFIG_DIR / "base.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_config(config_path: str) -> Path:
    """Resolve an overlay path: as given, else relative to this script's dir."""
    p = Path(config_path)
    if p.exists():
        return p
    alt = CONFIG_DIR / config_path
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Config overlay not found: {config_path} (also tried {alt})")


def load_config(config_path: str | None, overrides: dict) -> dict:
    """Load config: base.yaml → optional --config overlay → CLI overrides."""
    if not BASE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Base config not found at {BASE_CONFIG_PATH}. "
            f"It is the single source of truth for all defaults — it must exist."
        )
    with open(BASE_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}

    if config_path:
        with open(_resolve_config(config_path)) as f:
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


def build_optimizer(cfg: dict) -> optax.GradientTransformation:
    """Warmup-cosine-decay Adam, optionally wrapped in MultiSteps grad accumulation.

    Schedule units: `num_iterations` / `warmup_iterations` are given in *training*
    steps; since MultiSteps applies one real update every `grad_accumulation`
    micro-steps, the schedule is expressed in optimizer-update units (divide by
    grad_accumulation) so the cosine completes over the actual number of updates.
    """
    ocfg     = cfg["optimizer"]
    num_iter = cfg["training"]["num_iterations"]
    accum    = int(ocfg.get("grad_accumulation", 1)) or 1

    opt_steps  = max(num_iter // accum, 1)
    warmup_opt = max(min(int(ocfg["warmup_iterations"]) // accum, opt_steps - 1), 1)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=float(ocfg.get("init_lr", 0.0)),
        peak_value=float(ocfg["peak_lr"]),
        warmup_steps=warmup_opt,
        decay_steps=opt_steps,
        end_value=float(ocfg["end_lr"]),
    )
    base_opt = optax.adam(learning_rate=lr_schedule)
    return optax.MultiSteps(base_opt, accum) if accum > 1 else base_opt


def _parse_cli_value(val: str) -> Any:
    """Parse a CLI --set value into int/float/bool/list/None, else keep as string.

    Handles: 256, 0.1, true/false, null/none, [1024], [1024,1024,1024].
    """
    low = val.strip().lower()
    if low in ("null", "none"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    # list / tuple literals, e.g. [1024] or [1024,1024]
    if val.strip().startswith(("[", "(")):
        try:
            return list(ast.literal_eval(val))
        except (ValueError, SyntaxError):
            return val
    for cast in (int, float):
        try:
            return cast(val)
        except ValueError:
            pass
    return val


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

        import wandb as _wandb
        print(f"    [val diag] self._wandb_run={self._wandb_run!r}  "
              f"global wandb.run={_wandb.run!r}  "
              f"same={self._wandb_run is _wandb.run}")
        if self._wandb_run is not None:
            self._wandb_run.log({
                "val_r_squared":  entry["r_squared"],
                "val_e_distance": entry["e_distance"],
                "val_mmd":        entry["mmd"],
            })
            print(f"    [val diag] logged val metrics, run._step={getattr(self._wandb_run, '_step', '?')}")

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
def _dataset_specs(cfg: dict) -> list[tuple[str, str, int, float]]:
    """Return [(name, zarr_path, seed, weight), ...].

    Datasets are declared in cfg['datasets'] = {name: {path, seed?, weight?}}
    and chosen by name via cfg['selected_datasets'] = [name, ...]. This holds
    for both single- and multi-dataset training (single = one entry).
    A legacy single 'zarr_path' is still honoured as a last-resort fallback.
    """
    sel = cfg.get("selected_datasets")
    if sel:
        specs = []
        for name in sel:
            if name not in (cfg.get("datasets") or {}):
                raise KeyError(
                    f"selected_datasets includes '{name}' but it is not in "
                    f"cfg['datasets'] (have: {list((cfg.get('datasets') or {}).keys())})"
                )
            info = cfg["datasets"][name]
            specs.append((
                str(name),
                str(info["path"]),
                int(info.get("seed", cfg["seed"])),
                float(info.get("weight", 1.0)),
            ))
        return specs
    # legacy fallback: a single bare zarr_path
    zarr_path = cfg.get("zarr_path")
    if not zarr_path:
        raise ValueError(
            "No datasets configured. Set 'selected_datasets' (+ 'datasets' registry) "
            "in experiments/base.yaml, or provide a single 'zarr_path'."
        )
    return [("gd", str(zarr_path), int(cfg["seed"]), 1.0)]


def make_split(cfg: dict) -> dict:
    """Read & split each selected dataset.

    Returns {name: {"train": gd, "val": gd, "test": gd, "seed": int, "weight": float}}.
    """
    split_by = cfg["split"]["by"]
    ratios   = cfg["split"]["ratios"]

    out: dict = {}
    for name, path, seed, weight in _dataset_specs(cfg):
        print(f"  reading [{name}] ← {path}")
        gd = GroupedDistribution.read_zarr(Path(path))
        print_dataset_stats(gd, f"Full dataset [{name}]")
        splits = split_datasets(
            {name: gd},
            split_by=[split_by],
            split_key="split",
            ratios=ratios,
            random_state=seed,
            holdout_combinations=False,
        )
        train_gd = splits[name]["train"]
        val_gd   = splits[name]["val"]
        test_gd  = splits[name]["test"]
        print_split_stats(train_gd, val_gd, test_gd, split_by)
        out[name] = {"train": train_gd, "val": val_gd, "test": test_gd,
                     "seed": seed, "weight": weight}
    return out


def make_samplers(splits: dict, cfg: dict, transform: ConditionTransform | None):
    """Build one combined train sampler + per-dataset val/test sampler dicts."""
    tcfg  = cfg["training"]
    rng   = np.random.default_rng(cfg["seed"])
    n_val = tcfg["n_val_conditions"]

    train_samplers: dict = {}
    weights: dict = {}
    for name, d in splits.items():
        train_samplers[name] = ReservoirSampler(
            d["train"], np.random.default_rng(d["seed"]),
            batch_size=tcfg["batch_size"],
            pool_fraction=tcfg["pool_fraction"],
            replacement_prob=tcfg["replacement_prob"],
            condition_transform=transform,
        )
        weights[name] = d["weight"]
    # weights all-equal → CombinedSampler normalizes to uniform (same as before)
    train_sampler = CombinedSampler(samplers=train_samplers, rng=rng, weights=weights)

    val_samplers = {
        name: ValidationSampler(
            d["val"],
            n_conditions_on_log_iteration=n_val,
            n_conditions_on_train_end=n_val,
            seed=d["seed"],
            condition_transform=transform,
        )
        for name, d in splits.items()
    }
    test_samplers = {
        name: ValidationSampler(
            d["test"],
            n_conditions_on_log_iteration=None,
            n_conditions_on_train_end=None,
            seed=d["seed"],
            condition_transform=transform,
        )
        for name, d in splits.items()
    }

    train_sampler.init_sampler()
    for s in val_samplers.values():
        s.init_sampler()
    for s in test_samplers.values():
        s.init_sampler()
    return train_sampler, val_samplers, test_samplers


# ─────────────────────────────────────────────────────────────────────────────
# Test evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_test(solver, test_samplers: dict) -> dict:
    """Evaluate the solver on each dataset's test split.

    Returns {"per_dataset": {name: {per_condition, aggregated}},
             "per_condition": {name/cond: metrics},   # flat, dataset-prefixed
             "aggregated": {metric: overall_mean}}.
    """
    metrics = ["r_squared", "e_distance", "mmd"]
    per_dataset: dict = {}
    all_per_condition: dict = {}

    for name, sampler in test_samplers.items():
        batch = sampler.sample(mode="on_train_end")
        src, cond, true = batch["source"], batch["condition"], batch["target"]

        print(f"  [{name}] predicting {len(src)} test conditions …")
        pred = jax.tree.map(solver.predict, src, cond)

        per_condition = {}
        for cond_key in tqdm(sorted(true.keys(), key=str), desc=f"  test metrics [{name}]"):
            y_true = np.array(true[cond_key])
            y_pred = np.array(pred[cond_key])
            m = {
                "r_squared":  float(compute_r_squared(y_true, y_pred)),
                "e_distance": float(compute_e_distance_fast(y_true, y_pred)),
                "mmd":        float(compute_scalar_mmd(y_true, y_pred)),
            }
            per_condition[str(cond_key)] = m
            all_per_condition[f"{name}/{cond_key}"] = m

        per_dataset[name] = {
            "per_condition": per_condition,
            "aggregated": {mm: float(np.mean([v[mm] for v in per_condition.values()]))
                           for mm in metrics},
        }

    aggregated = {mm: float(np.mean([v[mm] for v in all_per_condition.values()]))
                  for mm in metrics}
    return {"per_dataset": per_dataset, "per_condition": all_per_condition,
            "aggregated": aggregated}


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
    splits = make_split(cfg)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    print("Building samplers …")
    train_sampler, val_samplers, test_samplers = make_samplers(splits, cfg, transform)

    sample_batch    = train_sampler.sample()
    cond_keys       = list(sample_batch["condition"].keys())
    data_dim        = sample_batch["src_cell_data"].shape[-1]
    cond_input_dims = {k: sample_batch["condition"][k].shape[-1] for k in cond_keys}
    print(f"  data_dim={data_dim}  condition_keys={cond_keys}")

    print("Building model …")
    arch = cfg["arch"]
    sf = ScaleFlow(solver=cfg["solver"])
    prepare_kwargs = dict(
        sample_batch=sample_batch,
        max_combination_length=arch["max_combination_length"],
        conditioning=arch["conditioning"],
        hidden_dims=tuple(arch["hidden_dims"]),
        decoder_dims=tuple(arch["decoder_dims"]),
        condition_embedding_dim=arch["condition_embedding_dim"],
        match_fn=partial(match_linear, epsilon=arch["match_fn_epsilon"]),
        optimizer=build_optimizer(cfg),
    )
    if arch.get("constant_noise") is not None:
        prepare_kwargs["probability_path"] = {"constant_noise": arch["constant_noise"]}
    sf.prepare_model(**prepare_kwargs)

    ocfg = cfg["optimizer"]
    print(f"  optimizer: adam warmup-cosine  peak={ocfg['peak_lr']:.1e}  "
          f"end={ocfg['end_lr']:.1e}  warmup={ocfg['warmup_iterations']}  "
          f"grad_accum={ocfg.get('grad_accumulation', 1)}")

    vf       = sf.solver.vf
    n_params = sum(x.size for x in jax.tree.leaves(sf.solver.vf_state.params))

    prophet_status = {"default": "NO  ✗", "prophet": "YES ✓", "random": "RANDOM ✓"}[mode]
    print(f"\n{'─'*60}")
    print(f"  Model architecture")
    print(f"{'─'*60}")
    print(f"  Total parameters      : {n_params:,}")
    print(f"  Data dim              : {data_dim}")
    print(f"  Prophet embedding     : {prophet_status}")
    print(f"  Solver                : {cfg['solver']}")
    print(f"  Conditioning          : {arch['conditioning']}")
    print(f"  Hidden dims           : {list(arch['hidden_dims'])}")
    print(f"  Decoder dims          : {list(arch['decoder_dims'])}")
    print(f"  Condition emb dim     : {arch['condition_embedding_dim']}")
    print(f"  Max combination length: {arch['max_combination_length']}")
    print(f"  Match fn epsilon      : {arch['match_fn_epsilon']}")
    print(f"  Constant noise        : {arch.get('constant_noise')}")
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

    # Metrics callback names keys "{val_dataset}_{metric}_mean", so monitor must
    # follow the actual dataset names (single fallback dataset is named "gd").
    monitor_metrics = ["loss"]
    for ds_name in val_samplers:
        monitor_metrics += [
            f"{ds_name}_r_squared_mean",
            f"{ds_name}_e_distance_mean",
            f"{ds_name}_mmd_mean",
        ]

    print(f"Training {tcfg['num_iterations']} iterations "
          f"(val every {valid_freq} steps, {tcfg['n_val_conditions']} conditions) …")
    t0 = time.perf_counter()
    sf.train(
        train_dataloader=train_sampler,
        val_dataloader=val_samplers,
        num_iterations=tcfg["num_iterations"],
        valid_freq=valid_freq,
        callbacks=callbacks,
        monitor_metrics=monitor_metrics,
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
    test_metrics = evaluate_test(best_solver, test_samplers)

    result_path = output_dir / f"{name}_results.pkl"
    with open(result_path, "wb") as f:
        cloudpickle.dump(test_metrics, f)
    print(f"  test results saved → {result_path}")

    if wandb_run is not None:
        test_log = {f"test_{metric}": val for metric, val in test_metrics["aggregated"].items()}
        # per-dataset breakdown
        for dsname, dsres in test_metrics["per_dataset"].items():
            for metric, val in dsres["aggregated"].items():
                test_log[f"test_{dsname}_{metric}"] = val
        wandb_run.log(test_log)                      # shows up as a chart point
        for k, v in test_log.items():
            wandb_run.summary[k] = v                 # also pinned in Summary panel

    return {"solver": best_solver, "test_metrics": test_metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None,
                        help="Optional YAML overlay on top of experiments/base.yaml "
                             "(e.g. multi_dataset.yaml — resolved next to the script)")
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
            val = _parse_cli_value(val)
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