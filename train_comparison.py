"""
train_comparison.py

Trains a CellFlow2 model on Tahoe data.

Usage
─────
  # Train with prophet embeddings (model_1):
  python train_comparison.py --model prophet

  # Train without prophet embeddings (model_default):
  python train_comparison.py --model default

Both variants read from the same zarr. When --model default is chosen,
the "prophet" key is stripped from every condition dict at sample time,
so the ConditionEncoder never sees it.

Prerequisites
─────────────
  A zarr built by scripts/process_tahoe_prophet.py, e.g.
  /storage/pancellflow/tahoe_prophet.zarr

Output (written to OUTPUT_DIR)
──────────────────────────────
  {name}_best.pkl               best checkpoint (by val R²)
  {name}_results.pkl            test-set metrics
  {name}_training_logs.json     per-step loss + val metrics
  {name}_training_curves.png    loss / val-metric plots
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

# ── JAX persistent compilation cache ─────────────────────────────────────────
# Must be set before any JAX code runs. On first run the ODE solver (~10-30 min
# JIT compile) is saved; subsequent runs load it from disk instantly.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/storage/jax_cache")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "5")
# ─────────────────────────────────────────────────────────────────────────────

import cloudpickle
import diffrax
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
# Config
# ─────────────────────────────────────────────────────────────────────────────
ZARR_PATH        = Path("/storage/pancellflow/tahoe.zarr")
OUTPUT_DIR       = Path("/storage/pancellflow/outputs/test_working")

SEED             = 42
SPLIT_RATIOS     = [0.7, 0.2, 0.1]   # train / val / test

BATCH_SIZE       = 1024
POOL_FRACTION    = 0.7
REPLACEMENT_PROB = 0.5

NUM_ITERATIONS   = 200_000
VALID_FREQ       = 20_000
N_VAL_CONDITIONS = 20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Condition key filter wrapper
# Strips specified keys from condition dicts so model_default never sees prophet
# ─────────────────────────────────────────────────────────────────────────────
class ConditionFilterSampler:
    """Wraps any sampler and removes unwanted keys from condition dicts."""

    def __init__(self, sampler, drop_keys: list[str]):
        self._sampler   = sampler
        self._drop_keys = set(drop_keys)

    def __getattr__(self, name):
        return getattr(self._sampler, name)

    def sample(self, *args, **kwargs) -> dict[str, Any]:
        batch = self._sampler.sample(*args, **kwargs)
        # Training batch: condition is a flat dict {key: array}
        if "condition" in batch and isinstance(batch["condition"], dict):
            batch = dict(batch)
            batch["condition"] = {k: v for k, v in batch["condition"].items()
                                  if k not in self._drop_keys}
        return batch

    def init_sampler(self):
        self._sampler.init_sampler()

    @property
    def initialized(self):
        return self._sampler.initialized


class ConditionFilterValidationSampler:
    """Wraps ValidationSampler and removes unwanted keys from condition dicts."""

    def __init__(self, sampler: ValidationSampler, drop_keys: list[str]):
        self._sampler   = sampler
        self._drop_keys = set(drop_keys)

    def __getattr__(self, name):
        return getattr(self._sampler, name)

    def sample(self, *args, **kwargs) -> dict[str, Any]:
        batch = self._sampler.sample(*args, **kwargs)
        # Validation batch: condition is {cond_key: {emb_key: array}}
        if "condition" in batch:
            batch = dict(batch)
            batch["condition"] = {
                ck: {k: v for k, v in cond.items() if k not in self._drop_keys}
                for ck, cond in batch["condition"].items()
            }
        return batch

    def init_sampler(self):
        self._sampler.init_sampler()

    @property
    def initialized(self):
        return self._sampler.initialized


# ─────────────────────────────────────────────────────────────────────────────
# Random-embedding wrappers
# Replaces specified condition keys with random vectors of the same shape.
# Training sampler: new random vector each step (stochastic ablation).
# Validation sampler: fixed random vector per (cond_key, emb_key) so metrics
#   are comparable across val steps.
# ─────────────────────────────────────────────────────────────────────────────
class ConditionRandomizeSampler:
    """Wraps any sampler and replaces specified condition keys with random vectors."""

    def __init__(self, sampler, randomize_keys: list[str], seed: int = 0):
        self._sampler        = sampler
        self._randomize_keys = set(randomize_keys)
        self._rng            = np.random.default_rng(seed)

    def __getattr__(self, name):
        return getattr(self._sampler, name)

    def sample(self, *args, **kwargs) -> dict[str, Any]:
        batch = self._sampler.sample(*args, **kwargs)
        if "condition" in batch and isinstance(batch["condition"], dict):
            batch = dict(batch)
            new_cond = {}
            for k, v in batch["condition"].items():
                if k in self._randomize_keys:
                    arr = self._rng.standard_normal(v.shape).astype(v.dtype)
                    print(f"  [random] replaced '{k}'  original norm={np.linalg.norm(v):.4f}  random norm={np.linalg.norm(arr):.4f}")
                    new_cond[k] = arr
                else:
                    new_cond[k] = v
            batch["condition"] = new_cond
        return batch

    def init_sampler(self):
        self._sampler.init_sampler()

    @property
    def initialized(self):
        return self._sampler.initialized


class ConditionRandomizeValidationSampler:
    """Wraps ValidationSampler and replaces specified condition keys with
    fixed-per-condition random vectors (same drug → same random vector
    across all val steps)."""

    def __init__(self, sampler: ValidationSampler, randomize_keys: list[str], seed: int = 0):
        self._sampler        = sampler
        self._randomize_keys = set(randomize_keys)
        self._seed           = seed
        self._cache: dict    = {}   # (cond_key, emb_key, shape) → array

    def __getattr__(self, name):
        return getattr(self._sampler, name)

    def _get_random(self, cond_key: str, emb_key: str, shape: tuple, dtype) -> np.ndarray:
        cache_key = (cond_key, emb_key, shape)
        if cache_key not in self._cache:
            # Deterministic per (condition, emb_key) — reproducible across runs
            int_seed = abs(hash(cond_key + emb_key + str(self._seed))) % (2 ** 31)
            rng = np.random.default_rng(int_seed)
            self._cache[cache_key] = rng.standard_normal(shape).astype(dtype)
        return self._cache[cache_key]

    def sample(self, *args, **kwargs) -> dict[str, Any]:
        batch = self._sampler.sample(*args, **kwargs)
        if "condition" in batch:
            batch = dict(batch)
            batch["condition"] = {
                ck: {
                    k: self._get_random(str(ck), k, v.shape, v.dtype)
                    if k in self._randomize_keys else v
                    for k, v in cond.items()
                }
                for ck, cond in batch["condition"].items()
            }
        return batch

    def init_sampler(self):
        self._sampler.init_sampler()

    @property
    def initialized(self):
        return self._sampler.initialized


# ─────────────────────────────────────────────────────────────────────────────
# Dataset statistics
# ─────────────────────────────────────────────────────────────────────────────
def print_dataset_stats(gd: GroupedDistribution, label: str = "Full dataset") -> None:
    ann  = gd.annotation
    data = gd.data

    n_src  = len(data.src_data)
    n_tgt  = len(data.tgt_data)

    src_labels = list(ann.src_dist_idx_to_labels.values())
    tgt_labels = list(ann.tgt_dist_idx_to_labels.values())

    # cell lines: first element of each src label tuple
    cell_lines = sorted({str(lbl[0]) if isinstance(lbl, (list, tuple)) else str(lbl)
                         for lbl in src_labels})
    # drugs: second element of each tgt label tuple (or first if only one key)
    drugs = sorted({str(lbl[1]) if isinstance(lbl, (list, tuple)) and len(lbl) > 1
                    else str(lbl[0]) if isinstance(lbl, (list, tuple)) else str(lbl)
                    for lbl in tgt_labels})

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


def print_split_stats(train_gd, val_gd, test_gd) -> None:
    def drug_set(gd):
        tgt_labels = list(gd.annotation.tgt_dist_idx_to_labels.values())
        return {str(lbl[1]) if isinstance(lbl, (list, tuple)) and len(lbl) > 1
                else str(lbl) for lbl in tgt_labels}

    tr, va, te = drug_set(train_gd), drug_set(val_gd), drug_set(test_gd)
    print(f"\n  Split (by drug)")
    print(f"    Train : {len(train_gd.data.tgt_data):>5} conditions  |  {len(tr):>4} drugs")
    print(f"    Val   : {len(val_gd.data.tgt_data):>5} conditions  |  {len(va):>4} drugs")
    print(f"    Test  : {len(test_gd.data.tgt_data):>5} conditions  |  {len(te):>4} drugs")
    print(f"    Sample test drugs : {sorted(te)[:5]} ...")


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_logs(name: str, logs: dict) -> None:
    path = OUTPUT_DIR / f"{name}_training_logs.json"
    serialisable = {k: [float(v) for v in vals] for k, vals in logs.items() if vals}
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  logs saved  → {path}")




# ─────────────────────────────────────────────────────────────────────────────
# ValMetricsLogger callback — appends metrics to JSON after every val step
# ─────────────────────────────────────────────────────────────────────────────
class ValMetricsLogger(ComputationCallback):
    """Appends val metrics to a JSON file immediately after each validation step."""

    def __init__(self, save_path: str):
        self.save_path  = save_path
        self._step      = 0

    def on_train_begin(self, *args, **kwargs) -> None:
        self._step = 0

    def _compute_and_save(self, valid_true_data, valid_pred_data) -> None:
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
            return

        entry = {
            "step":       self._step,
            "n_conditions": len(r2s),
            "r_squared":  float(np.mean(r2s)),
            "e_distance": float(np.mean(eds)),
            "mmd":        float(np.mean(mmds)),
        }

        # Load existing entries, append, write back
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

    def on_log_iteration(self, valid_source_data, valid_true_data,
                         valid_pred_data, solver, **kwargs) -> dict:
        self._step += VALID_FREQ
        self._compute_and_save(valid_true_data, valid_pred_data)
        return {}

    def on_train_end(self, valid_source_data, valid_true_data,
                     valid_pred_data, solver, **kwargs) -> dict:
        self._compute_and_save(valid_true_data, valid_pred_data)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# BestModelCheckpoint callback
# ─────────────────────────────────────────────────────────────────────────────
class BestModelCheckpoint(ComputationCallback):
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.best_r2   = -np.inf

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
        return {"best_val_r2": self.best_r2}

    def on_train_end(self, valid_source_data, valid_true_data,
                     valid_pred_data, solver, **kwargs) -> dict:
        return self.on_log_iteration(valid_source_data, valid_true_data,
                                     valid_pred_data, solver)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_split(zarr_path: Path):
    gd = GroupedDistribution.read_zarr(zarr_path)
    print_dataset_stats(gd, "Full dataset")
    splits = split_datasets(
        {"gd": gd},
        split_by=["drug"],
        split_key="split",
        ratios=SPLIT_RATIOS,
        random_state=SEED,
        holdout_combinations=False,
    )
    train_gd = splits["gd"]["train"]
    val_gd   = splits["gd"]["val"]
    test_gd  = splits["gd"]["test"]
    print_split_stats(train_gd, val_gd, test_gd)
    return train_gd, val_gd, test_gd


def make_samplers(train_gd, val_gd, test_gd,
                  drop_keys: list[str],
                  randomize_keys: list[str]):
    rng = np.random.default_rng(SEED)

    raw_train = CombinedSampler(
        samplers={"gd": ReservoirSampler(
            train_gd, rng,
            batch_size=BATCH_SIZE,
            pool_fraction=POOL_FRACTION,
            replacement_prob=REPLACEMENT_PROB,
        )},
        rng=rng,
    )
    raw_val  = ValidationSampler(val_gd,  n_conditions_on_log_iteration=N_VAL_CONDITIONS,
                                           n_conditions_on_train_end=N_VAL_CONDITIONS, seed=SEED)
    raw_test = ValidationSampler(test_gd, n_conditions_on_log_iteration=None,
                                           n_conditions_on_train_end=None, seed=SEED)

    if drop_keys:
        train_sampler = ConditionFilterSampler(raw_train, drop_keys)
        val_sampler   = ConditionFilterValidationSampler(raw_val,  drop_keys)
        test_sampler  = ConditionFilterValidationSampler(raw_test, drop_keys)
    elif randomize_keys:
        train_sampler = ConditionRandomizeSampler(raw_train, randomize_keys, seed=SEED)
        val_sampler   = ConditionRandomizeValidationSampler(raw_val,  randomize_keys, seed=SEED)
        test_sampler  = ConditionRandomizeValidationSampler(raw_test, randomize_keys, seed=SEED)
    else:
        train_sampler, val_sampler, test_sampler = raw_train, raw_val, raw_test

    train_sampler.init_sampler()
    val_sampler.init_sampler()
    test_sampler.init_sampler()

    print(f"  [diag] val conditions dict  : {len(val_gd.data.conditions)}")
    print(f"  [diag] val tgt_data dict    : {len(val_gd.data.tgt_data)}")
    print(f"  [diag] val src_to_tgt map   : {sum(len(v) for v in val_gd.data.src_to_tgt_dist_map.values())}")
    print(f"  [diag] val _tgt_to_src      : {len(raw_val._tgt_to_src)}")

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
    for cond_key in tqdm(sorted(true.keys()), desc="  test metrics"):
        y_true = np.array(true[cond_key])
        y_pred = np.array(pred[cond_key])
        per_condition[cond_key] = {
            "r_squared":  float(compute_r_squared(y_true, y_pred)),
            "e_distance": float(compute_e_distance_fast(y_true, y_pred)),
            "mmd":        float(compute_scalar_mmd(y_true, y_pred)),
        }

    metrics = ["r_squared", "e_distance", "mmd"]
    aggregated = {m: float(np.mean([v[m] for v in per_condition.values()])) for m in metrics}
    return {"per_condition": per_condition, "aggregated": aggregated}


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────
def train_model(name: str, mode: str) -> dict:
    """
    mode: "default"  — no prophet key (dropped)
          "prophet"  — real prophet embeddings
          "random"   — prophet key present but values replaced with random vectors
    """
    print(f"\n{'='*64}")
    print(f"  {name}  |  mode={mode}")
    print(f"{'='*64}")

    drop_keys      = ["prophet"] if mode == "default" else []
    randomize_keys = ["prophet"] if mode == "random"  else []
    ckpt_path      = str(OUTPUT_DIR / f"{name}_best.pkl")

    t0 = time.perf_counter()
    print("Loading & splitting data …")
    train_gd, val_gd, test_gd = make_split(ZARR_PATH)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    print("Building samplers …")
    train_sampler, val_sampler, test_sampler = make_samplers(
        train_gd, val_gd, test_gd, drop_keys, randomize_keys
    )

    sample_batch = train_sampler.sample()
    cond_keys    = list(sample_batch["condition"].keys())
    data_dim     = sample_batch["src_cell_data"].shape[-1]
    print(f"  data_dim={data_dim}  condition_keys={cond_keys}")

    print("Building model …")
    sf = ScaleFlow()
    sf.prepare_model(sample_batch=sample_batch, max_combination_length=1)

    vf = sf.solver.vf
    n_params = sum(x.size for x in jax.tree.leaves(sf.solver.vf_state.params))
    cond_input_dims = {k: sample_batch["condition"][k].shape[-1]
                       for k in sample_batch["condition"]}
    print(f"\n{'─'*60}")
    print(f"  Model architecture")
    print(f"{'─'*60}")
    print(f"  Solver                : {type(sf.solver).__name__}")
    print(f"  Total parameters      : {n_params:,}")
    print(f"")
    print(f"  ── Data path ──────────────────────────────────────────")
    print(f"  Data dim (input)      : {data_dim}")
    print(f"  x_encoder (hidden)    : {tuple(vf.hidden_dims)}")
    print(f"  time_encoder dims     : {tuple(vf.time_encoder_dims)}")
    print(f"  conditioning          : {vf.conditioning}")
    print(f"  decoder dims          : {tuple(vf.decoder_dims)}")
    print(f"  Data dim (output)     : {vf.output_dim}")
    print(f"")
    print(f"  ── Condition encoder ──────────────────────────────────")
    print(f"  Condition keys        : {list(cond_input_dims.keys())}")
    prophet_status = ("YES ✓" if mode == "prophet" else
                      "RANDOM ✓" if mode == "random" else "NO  ✗")
    print(f"  Prophet embedding     : {prophet_status}")
    for k, d in cond_input_dims.items():
        print(f"  Input dim [{k:<10}]: {d}")
    print(f"  condition_mode        : {vf.condition_mode}")
    print(f"  pooling               : {vf.pooling}")
    print(f"  embed_dim (output)    : {vf.condition_embedding_dim}")
    print(f"  max_combination_len   : {vf.max_combination_length}")
    print(f"{'─'*60}")

    # sinkhorn_div is not in metric_to_func_gpu, so it can't be used with
    # use_gpu_optimized=True.  The three below cover what matters at val time;
    # sinkhorn_div is still computed at test time in evaluate_test().
    val_log_path = str(OUTPUT_DIR / f"{name}_val_metrics.json")
    callbacks = [
        Metrics(
            metrics=["r_squared", "e_distance", "mmd"],
            metric_aggregations=["mean"],
            use_gpu_optimized=True,
            precision="bfloat16",
        ),
        ValMetricsLogger(save_path=val_log_path),
        BestModelCheckpoint(save_path=ckpt_path),
    ]

    print(f"Training {NUM_ITERATIONS} iterations "
          f"(val every {VALID_FREQ} steps, {N_VAL_CONDITIONS} conditions) …")
    print(f"  tracking : loss | val_r_squared_mean | val_e_distance_mean | val_mmd_mean")
    t0 = time.perf_counter()
    sf.train(
        train_dataloader=train_sampler,
        val_dataloader={"val": val_sampler},
        num_iterations=NUM_ITERATIONS,
        valid_freq=VALID_FREQ,
        callbacks=callbacks,
        monitor_metrics=["loss", "val_r_squared_mean", "val_e_distance_mean", "val_mmd_mean"],
    )
    elapsed = (time.perf_counter() - t0) / 60
    print(f"  training done in {elapsed:.1f} min")

    save_logs(name, sf.trainer.training_logs)

    if os.path.exists(ckpt_path):
        print(f"Loading best checkpoint from {ckpt_path} …")
        with open(ckpt_path, "rb") as f:
            best_solver = cloudpickle.load(f)
    else:
        print("  no checkpoint found – using final iterate")
        best_solver = sf.solver

    print("Evaluating on test set …")
    test_metrics = evaluate_test(best_solver, test_sampler)

    result_path = OUTPUT_DIR / f"{name}_results.pkl"
    with open(result_path, "wb") as f:
        cloudpickle.dump(test_metrics, f)
    print(f"  test results saved to {result_path}")

    return {"solver": best_solver, "test_metrics": test_metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["default", "prophet", "random"],
        required=True,
        help=(
            "'default' = no prophet embedding  |  "
            "'prophet' = real prophet embeddings  |  "
            "'random'  = random embeddings same size as prophet"
        ),
    )
    args = parser.parse_args()

    name_map = {
        "default": "model_default",
        "prophet": "model_prophet",
        "random":  "model_random",
    }
    name   = name_map[args.model]
    result = train_model(name, mode=args.model)

    print(f"\n{'='*64}")
    print(f"  Final test metrics — {name}")
    print(f"{'='*64}")
    for metric, val in result["test_metrics"]["aggregated"].items():
        print(f"  {metric:<20} {val:.4f}")