"""
eval_test.py

Load a saved best checkpoint (.pkl) and evaluate it on the held-out test set
using the exact same zarr / split / seed as train_comparison.py.

Usage:
  python eval_test.py                                    # prophet model (default)
  python eval_test.py --model default                   # default model
  python eval_test.py --ckpt /path/to/custom_best.pkl   # explicit checkpoint path
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

# ── JAX compilation cache ─────────────────────────────────────────────────────
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/storage/jax_cache")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "5")
# ─────────────────────────────────────────────────────────────────────────────

import cloudpickle
import jax
import numpy as np
from tqdm import tqdm

from scaleflow.data import GroupedDistribution, split_datasets
from scaleflow.data._dataloader import ValidationSampler
from scaleflow.metrics._metrics import (
    compute_r_squared,
    compute_e_distance_fast,
    compute_scalar_mmd,
)

# ── Config — must match train_comparison.py exactly ──────────────────────────
ZARR_PATH   = Path("/storage/pancellflow/tahoe.zarr")
OUTPUT_DIR  = Path("/storage/pancellflow/outputs/cell_line_div")
SEED        = 42
SPLIT_RATIOS = [5/7, 1/7, 1/7]   # train / val / test  (5 cell lines train, 1 val, 1 test)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Condition filter (needed for --model default)
# ─────────────────────────────────────────────────────────────────────────────
class ConditionFilterValidationSampler:
    """Strips drop_keys from nested condition dicts returned by ValidationSampler."""

    def __init__(self, sampler, drop_keys: list[str]):
        self._sampler   = sampler
        self._drop_keys = set(drop_keys)

    def __getattr__(self, name):
        return getattr(self._sampler, name)

    def sample(self, *args, **kwargs) -> dict[str, Any]:
        batch = self._sampler.sample(*args, **kwargs)
        if "condition" in batch and isinstance(batch["condition"], dict) and batch["condition"]:
            batch = dict(batch)
            cond  = batch["condition"]
            first = next(iter(cond.values()))
            if isinstance(first, dict):
                # Validation batch: {cond_id: {key: array}} — strip from inner dicts
                batch["condition"] = {
                    cid: {k: v for k, v in emb.items() if k not in self._drop_keys}
                    for cid, emb in cond.items()
                }
            else:
                # Flat batch: {key: array}
                batch["condition"] = {k: v for k, v in cond.items()
                                      if k not in self._drop_keys}
        return batch

    def init_sampler(self):
        self._sampler.init_sampler()

    @property
    def initialized(self):
        return self._sampler.initialized


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["default", "prophet"], default="prophet")
parser.add_argument("--ckpt",  default=None,
                    help="Path to .pkl checkpoint. Defaults to OUTPUT_DIR/{name}_best.pkl")
args = parser.parse_args()

use_prophet = args.model == "prophet"
drop_keys   = [] if use_prophet else ["prophet"]
name        = f"model_{args.model}"
ckpt_path   = args.ckpt or str(OUTPUT_DIR / f"{name}_best.pkl")

print(f"Model   : {args.model}")
print(f"Checkpoint : {ckpt_path}")
print(f"Zarr    : {ZARR_PATH}")

# ── Load & split data ─────────────────────────────────────────────────────────
print("\nLoading zarr …")
gd = GroupedDistribution.read_zarr(ZARR_PATH)

print("Splitting data …")
splits = split_datasets(
    {"gd": gd},
    split_by=["cell_line"],
    split_key="split",
    ratios=SPLIT_RATIOS,
    random_state=SEED,
    holdout_combinations=False,
)
test_gd = splits["gd"]["test"]

src_labels = list(test_gd.annotation.src_dist_idx_to_labels.values())
test_cell_lines = {str(lbl[0]) if isinstance(lbl, (list, tuple)) else str(lbl) for lbl in src_labels}
print(f"  Test set : {len(test_gd.data.tgt_data)} conditions | cell lines: {', '.join(sorted(test_cell_lines))}")

# ── Build test sampler ────────────────────────────────────────────────────────
raw_test = ValidationSampler(test_gd,
                             n_conditions_on_log_iteration=None,
                             n_conditions_on_train_end=None,
                             seed=SEED)
test_sampler = ConditionFilterValidationSampler(raw_test, drop_keys) if drop_keys else raw_test
test_sampler.init_sampler()

# ── Load checkpoint ───────────────────────────────────────────────────────────
print(f"\nLoading checkpoint from {ckpt_path} …")
with open(ckpt_path, "rb") as f:
    solver = cloudpickle.load(f)
print("  checkpoint loaded.")

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\nSampling test batch …")
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

metrics     = ["r_squared", "e_distance", "mmd"]
aggregated  = {m: float(np.mean([v[m] for v in per_condition.values()])) for m in metrics}

# ── Print results ─────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"  Test results — {name}")
print(f"{'─'*50}")
for m in metrics:
    print(f"  {m:<15}: {aggregated[m]:.4f}")
print(f"{'─'*50}")

# ── Save results ──────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result_path = OUTPUT_DIR / f"{name}_test_results.json"
with open(result_path, "w") as f:
    json.dump({
        "aggregated":    aggregated,
        "per_condition": {str(k): v for k, v in per_condition.items()},
    }, f, indent=2)
print(f"\nResults saved → {result_path}")