"""
smoke_test.py

Quick end-to-end check: load zarr → print stats → build model → train 10 steps → save plot.
Validation is skipped (valid_freq > num_iterations) to avoid the long ODE JIT compile.

Usage:
  python smoke_test.py                  # with prophet embeddings
  python smoke_test.py --model default  # without prophet embeddings
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

# ── JAX persistent compilation cache ─────────────────────────────────────────
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/storage/jax_cache")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "5")
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scaleflow.data import GroupedDistribution, split_datasets
from scaleflow.data._dataloader import CombinedSampler, ReservoirSampler, ValidationSampler
from scaleflow.model import ScaleFlow
from scaleflow.training import Metrics

ZARR   = Path("/storage/pancellflow/tahoe.zarr")
OUTPUT = Path("/storage/pancellflow/smoke_outputs")
SEED   = 42

OUTPUT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Condition filter (same as train_comparison.py)
# ─────────────────────────────────────────────────────────────────────────────
class ConditionFilterSampler:
    """Strips drop_keys from condition dicts for both training and validation batches.

    Training batches have a flat condition dict:
        {"cell_line": arr, "drug": arr, "prophet": arr}
    Validation batches have a nested condition dict:
        {"cond_id_1": {"cell_line": arr, "drug": arr, "prophet": arr}, ...}
    Both cases are handled by checking the type of the first value.
    """

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
                # Validation batch: outer keys are condition IDs, strip from inner dicts
                batch["condition"] = {
                    cid: {k: v for k, v in emb.items() if k not in self._drop_keys}
                    for cid, emb in cond.items()
                }
            else:
                # Training batch: outer keys are embedding names, strip directly
                batch["condition"] = {k: v for k, v in cond.items()
                                      if k not in self._drop_keys}
        return batch

    def init_sampler(self):
        self._sampler.init_sampler()

    @property
    def initialized(self):
        return self._sampler.initialized


# ─────────────────────────────────────────────────────────────────────────────
# Dataset statistics
# ─────────────────────────────────────────────────────────────────────────────
def print_dataset_stats(gd: GroupedDistribution, label: str = "Dataset") -> None:
    data = gd.data
    ann  = gd.annotation

    src_labels = list(ann.src_dist_idx_to_labels.values())
    tgt_labels = list(ann.tgt_dist_idx_to_labels.values())

    cell_lines = sorted({str(lbl[0]) if isinstance(lbl, (list, tuple)) else str(lbl)
                         for lbl in src_labels})
    drugs = sorted({str(lbl[1]) if isinstance(lbl, (list, tuple)) and len(lbl) > 1
                    else str(lbl[0]) if isinstance(lbl, (list, tuple)) else str(lbl)
                    for lbl in tgt_labels})

    src_sizes = [v.shape[0] for v in data.src_data.values()]
    tgt_sizes = [v.shape[0] for v in data.tgt_data.values()]
    cond_keys = list(next(iter(data.conditions.values())).keys()) if data.conditions else []

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Cell lines    : {len(cell_lines)}  →  {', '.join(cell_lines)}")
    print(f"  Drugs         : {len(drugs)}")
    print(f"  Conditions    : {len(data.tgt_data)}  (cell_line × drug pairs)")
    print(f"  Control cells : {sum(src_sizes):,}  "
          f"(min={min(src_sizes):,}  max={max(src_sizes):,})")
    print(f"  Treated cells : {sum(tgt_sizes):,}  "
          f"(min={min(tgt_sizes):,}  max={max(tgt_sizes):,})")
    print(f"  Cond keys     : {cond_keys}")
    print(f"{'─'*60}")


def print_split_stats(train_gd, val_gd, test_gd) -> None:
    def drug_set(gd):
        tgt_labels = list(gd.annotation.tgt_dist_idx_to_labels.values())
        return {str(lbl[1]) if isinstance(lbl, (list, tuple)) and len(lbl) > 1
                else str(lbl) for lbl in tgt_labels}

    tr, va, te = drug_set(train_gd), drug_set(val_gd), drug_set(test_gd)
    print(f"\n  Train/Val/Test split (by drug)")
    print(f"    Train : {len(train_gd.data.tgt_data):>4} conditions  |  {len(tr):>3} drugs")
    print(f"    Val   : {len(val_gd.data.tgt_data):>4} conditions  |  {len(va):>3} drugs")
    print(f"    Test  : {len(test_gd.data.tgt_data):>4} conditions  |  {len(te):>3} drugs")
    print(f"    Sample test drugs : {sorted(te)[:5]} ...")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["default", "prophet"], default="prophet")
args = parser.parse_args()

use_prophet = args.model == "prophet"
drop_keys   = [] if use_prophet else ["prophet"]
name        = "smoke_prophet" if use_prophet else "smoke_default"

print(f"Model variant : {args.model}")
print(f"Loading zarr  : {ZARR}")
gd = GroupedDistribution.read_zarr(ZARR)
print_dataset_stats(gd, "Full dataset")

splits = split_datasets(
    {"gd": gd},
    split_by=["drug"],
    split_key="split",
    ratios=[0.7, 0.2, 0.1],
    random_state=SEED,
    holdout_combinations=False,
)
train_gd = splits["gd"]["train"]
val_gd   = splits["gd"]["val"]
test_gd  = splits["gd"]["test"]
print_split_stats(train_gd, val_gd, test_gd)

rng = np.random.default_rng(SEED)
raw_train = CombinedSampler(
    samplers={"gd": ReservoirSampler(
        train_gd, rng, batch_size=256, pool_fraction=0.7, replacement_prob=0.5,
    )},
    rng=rng,
)
raw_val = ValidationSampler(val_gd, n_conditions_on_log_iteration=5,
                             n_conditions_on_train_end=5, seed=SEED)

if drop_keys:
    train_sampler = ConditionFilterSampler(raw_train, drop_keys)
    val_sampler   = ConditionFilterSampler(raw_val,   drop_keys)
else:
    train_sampler, val_sampler = raw_train, raw_val

train_sampler.init_sampler()
val_sampler.init_sampler()

sample_batch = train_sampler.sample()
data_dim  = sample_batch["src_cell_data"].shape[-1]
cond_keys = list(sample_batch["condition"].keys())
print(f"\ndata_dim     : {data_dim}")
print(f"cond_keys    : {cond_keys}")

print("\nBuilding model …")
sf = ScaleFlow()
sf.prepare_model(sample_batch=sample_batch, max_combination_length=1)

import jax
vf = sf.solver.vf
n_params = sum(x.size for x in jax.tree.leaves(sf.solver.vf_state.params))
cond_input_dims = {k: sample_batch["condition"][k].shape[-1] for k in sample_batch["condition"]}

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
print(f"  Prophet embedding     : {'YES ✓' if 'prophet' in cond_input_dims else 'NO  ✗'}")
for k, d in cond_input_dims.items():
    print(f"  Input dim [{k:<10}]: {d}")
print(f"  condition_mode        : {vf.condition_mode}")
print(f"  pooling               : {vf.pooling}")
print(f"  embed_dim (output)    : {vf.condition_embedding_dim}")
print(f"  max_combination_len   : {vf.max_combination_length}")
print(f"{'─'*60}")

print("Training 10 steps (val only at train_end) …")
sf.train(
    train_dataloader=train_sampler,
    val_dataloader={"val": val_sampler},
    num_iterations=10,
    valid_freq=1000,   # > num_iterations, so mid-run validation is skipped
    callbacks=[
        Metrics(
            metrics=["r_squared", "e_distance", "mmd"],
            metric_aggregations=["mean"],
            use_gpu_optimized=True,
            precision="bfloat16",
        ),
    ],
    monitor_metrics=["loss", "val_r_squared_mean", "val_e_distance_mean", "val_mmd_mean"],
)

# ── Save logs ────────────────────────────────────────────────────────────────
logs = sf.trainer.training_logs
logs_path = OUTPUT / f"{name}_training_logs.json"
with open(logs_path, "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in logs.items() if vals}, f, indent=2)
print(f"\nLogs saved  → {logs_path}")

# ── Plot ─────────────────────────────────────────────────────────────────────
label_map = {
    "loss":                "Train loss",
    "val_r_squared_mean":  "Val R²  (mean)",
    "val_e_distance_mean": "Val E-distance (mean)",
    "val_mmd_mean":        "Val MMD (mean)",
}
keys_to_plot = [k for k in label_map if k in logs and logs[k]]
if keys_to_plot:
    fig, axes = plt.subplots(1, len(keys_to_plot), figsize=(5 * len(keys_to_plot), 4))
    if len(keys_to_plot) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys_to_plot):
        vals = logs[key]
        color = "#2196F3" if key == "loss" else "#E91E63"
        ax.plot(vals, marker="o", color=color, linewidth=1.2)
        ax.set_title(label_map[key], fontsize=10)
        ax.set_xlabel("step" if key == "loss" else "val checkpoint")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Smoke test — {args.model}", fontsize=12)
    fig.tight_layout()
    plot_path = OUTPUT / f"{name}_training_curves.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Plot saved  → {plot_path}")

print("\nDone — pipeline works!")