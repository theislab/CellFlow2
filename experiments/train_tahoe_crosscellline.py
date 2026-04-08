"""Cross-cell-line scaling experiment on Tahoe-100M.

Tests whether training on multiple cell lines (with shared 368-drug library)
improves prediction of held-out drugs in an evaluation cell line.

Split strategy:
- Anchor cell lines: train on ALL 368 drugs
- Eval cell lines: train on Set A (184 drugs), evaluate on Set B (184 drugs)
- The model sees each Set B drug in anchor cell lines but NOT in the eval cell line

Example experiments:
  A1: --eval-celllines SW48               (baseline: SW48 alone)
  B1: --anchor-celllines A549 --eval-celllines SW48  (does A549 help SW48?)
  C1: --anchor-celllines A549,SNU-1,H4,ASPC-1 --eval-celllines SW48
  D1: --anchor-celllines A549,SNU-1,H4,ASPC-1 --eval-celllines HOP62 --zero-shot

Usage:
  python train_tahoe_crosscellline.py \
      --anchor-celllines A549 \
      --eval-celllines SW48 \
      --experiment-id B1 \
      --output-dir ./results/crosscellline
"""

import argparse
import os
import sys
import cloudpickle
import numpy as np
import optax
import pandas as pd
import scanpy as sc
import anndata as ad

from scaleflow.data import AnnDataLocation, DataManager
from scaleflow.data._dataloader import (
    CombinedSampler,
    InMemorySampler,
    ValidationSampler,
)
from scaleflow.model import ScaleFlow
from scaleflow.training import (
    LearningRateMonitor,
    LoggingCallback,
    Metrics,
    WandbLogger,
)
from scaleflow.utils import match_linear
from functools import partial
from typing import Any


class ModelCheckpoint(LoggingCallback):
    """Save best solver params when a monitored metric improves."""

    def __init__(self, solver, monitor: str, save_path: str, mode: str = "min"):
        self.solver = solver
        self.monitor = monitor
        self.save_path = save_path
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def on_train_begin(self) -> None:
        self.best_value = float("inf") if self.mode == "min" else float("-inf")

    def on_train_end(self, dict_to_log: dict[str, Any] = None, **_: Any) -> dict[str, Any]:
        return dict_to_log or {}

    def on_log_iteration(self, dict_to_log: dict[str, Any], iteration: int = None, **_: Any) -> dict[str, Any]:
        if self.monitor not in dict_to_log:
            return dict_to_log
        current = float(dict_to_log[self.monitor])
        improved = (self.mode == "min" and current < self.best_value) or (
            self.mode == "max" and current > self.best_value
        )
        if improved:
            self.best_value = current
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            with open(self.save_path, "wb") as f:
                cloudpickle.dump(self.solver.vf_state_inference.params, f)
            print(f"  ModelCheckpoint: saved best params ({self.monitor}={current:.6f})")
        return dict_to_log


# ── Data paths ──────────────────────────────────────────────────────────────
DATA_DIR = "/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/unipert"

CELLLINE_FILES = {
    "A549":   "tahoe_a549_w_emb.h5ad",
    "SW48":   "tahoe_sw48_w_emb.h5ad",
    "SNU-1":  "tahoe_snu_1_w_emb.h5ad",
    "H4":     "tahoe_h4_w_emb.h5ad",
    "ASPC-1": "tahoe_aspc_1_w_emb.h5ad",
    "HOP62":  "tahoe_hop62_w_emb.h5ad",
    "PANC-1": "tahoe_panc_1_w_emb.h5ad",
}


# ── Utility functions (from train_and_predict.py) ───────────────────────────
def mark_control(adata):
    adata.obs["control"] = (adata.obs["drug_0"] == "control")
    if "drug_1" in adata.obs.columns:
        adata.obs["control"] = adata.obs["control"] & (adata.obs["drug_1"] == "control")
    return adata


def resample_cells(adata, target_cells, rng):
    """Stratified subsampling to target_cells, proportional within each drug group."""
    if target_cells >= adata.n_obs:
        return adata
    fraction = target_cells / adata.n_obs
    indices = []
    group_cols = ["drug_0"]
    if "drug_1" in adata.obs.columns:
        group_cols.append("drug_1")
    for _, group in adata.obs.groupby(group_cols, observed=True):
        group_indices = group.index.tolist()
        n_sample = max(1, int(len(group_indices) * fraction))
        sampled = rng.choice(group_indices, size=n_sample, replace=False)
        indices.extend(sampled)
    return adata[indices].copy()


def compute_global_drug_split(all_drugs, seed=42):
    """Split 368 drugs into Set A (train) and Set B (test), 50/50.

    Returns (set_a, set_b) where each is a set of drug names.
    """
    drugs_sorted = sorted(all_drugs)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(drugs_sorted))
    half = len(drugs_sorted) // 2
    set_a = {drugs_sorted[i] for i in idx[:half]}
    set_b = {drugs_sorted[i] for i in idx[half:]}
    return set_a, set_b


def filter_drugs(adata, keep_drugs, keep_control=True):
    """Keep only cells from specified drugs (and optionally control)."""
    mask = adata.obs["drug_0"].isin(keep_drugs)
    if keep_control:
        mask = mask | adata.obs["control"]
    return adata[mask].copy()


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Tahoe cross-cell-line scaling experiment")
    parser.add_argument("--anchor-celllines", type=str, default="",
                        help="Comma-separated anchor cell lines (train on ALL drugs). Empty = no anchors.")
    parser.add_argument("--eval-celllines", type=str, required=True,
                        help="Comma-separated eval cell lines (Set A train, Set B test).")
    parser.add_argument("--zero-shot", action="store_true",
                        help="Zero-shot mode: eval cell lines have NO training drugs (only anchor lines train).")
    parser.add_argument("--drug-split-seed", type=int, default=42,
                        help="Seed for global drug split into Set A/B.")
    parser.add_argument("--max-cells-per-cellline", type=int, default=200000,
                        help="Max cells per cell line after subsampling.")
    parser.add_argument("--experiment-id", type=str, required=True,
                        help="Experiment ID for logging (e.g., A1, B1, C1).")
    parser.add_argument("--output-dir", type=str, default="./results/crosscellline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-iterations", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--valid-freq", type=int, default=5000)
    parser.add_argument("--wandb-project", type=str, default="tahoe-crosscellline")
    args = parser.parse_args()

    anchor_cls = [c.strip() for c in args.anchor_celllines.split(",") if c.strip()]
    eval_cls = [c.strip() for c in args.eval_celllines.split(",") if c.strip()]
    all_cls = anchor_cls + eval_cls

    print(f"=" * 60)
    print(f"Experiment: {args.experiment_id}")
    print(f"Anchor cell lines (all drugs): {anchor_cls or 'NONE'}")
    print(f"Eval cell lines (Set A train, Set B test): {eval_cls}")
    print(f"Zero-shot mode: {args.zero_shot}")
    print(f"=" * 60)

    # Validate cell line names
    for cl in all_cls:
        if cl not in CELLLINE_FILES:
            raise ValueError(f"Unknown cell line: {cl}. Available: {list(CELLLINE_FILES.keys())}")

    rng = np.random.default_rng(args.seed)
    out_dir = os.path.join(args.output_dir, args.experiment_id)
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Get global drug list from all cell lines (backed loading) ─
    print("\n[Step 1] Scanning drug lists (backed loading)...")
    all_drugs = set()
    for cl in all_cls:
        path = os.path.join(DATA_DIR, CELLLINE_FILES[cl])
        adata_backed = ad.read_h5ad(path, backed="r")
        drugs = [d for d in adata_backed.obs["drug_0"].unique() if d != "control"]
        all_drugs.update(drugs)
        print(f"  {cl}: {len(drugs)} drugs, {adata_backed.n_obs:,} cells")
        del adata_backed

    print(f"\nGlobal drug library: {len(all_drugs)} drugs")
    set_a, set_b = compute_global_drug_split(all_drugs, seed=args.drug_split_seed)
    print(f"Drug split: Set A (train in eval lines) = {len(set_a)}, Set B (test) = {len(set_b)}")

    # ── Step 2: Load, filter, and subsample each cell line ────────────
    print("\n[Step 2] Loading and preparing datasets (memory-efficient)...")
    train_adatas = {}
    eval_adatas = {}

    for cl in anchor_cls:
        path = os.path.join(DATA_DIR, CELLLINE_FILES[cl])
        print(f"  Loading {cl} (anchor, all drugs)...")
        adata = sc.read_h5ad(path)
        adata = mark_control(adata)
        adata = resample_cells(adata, args.max_cells_per_cellline, rng)
        train_adatas[cl] = adata
        print(f"    → {adata.n_obs:,} cells after subsampling")

    for cl in eval_cls:
        path = os.path.join(DATA_DIR, CELLLINE_FILES[cl])
        print(f"  Loading {cl} (eval)...")
        adata_full = sc.read_h5ad(path)
        adata_full = mark_control(adata_full)

        if args.zero_shot:
            eval_adatas[cl] = adata_full
            print(f"    → zero-shot: eval on all {len(all_drugs)} drugs ({adata_full.n_obs:,} cells)")
        else:
            # Set A for training, Set B for evaluation
            train_adata = filter_drugs(adata_full, set_a, keep_control=True)
            train_adata = resample_cells(train_adata, args.max_cells_per_cellline, rng)
            train_adatas[cl] = train_adata

            eval_adata = filter_drugs(adata_full, set_b, keep_control=True)
            eval_adatas[cl] = eval_adata

            n_train_drugs = len([d for d in train_adata.obs["drug_0"].unique() if d != "control"])
            n_eval_drugs = len([d for d in eval_adata.obs["drug_0"].unique() if d != "control"])
            print(f"    → train: {train_adata.n_obs:,} cells ({n_train_drugs} drugs)")
            print(f"    → eval:  {eval_adata.n_obs:,} cells ({n_eval_drugs} drugs)")
            del adata_full  # free memory

    # ── Step 3: Create DataManager and samplers ─────────────────────────
    print("\n[Step 3] Creating DataManager and samplers...")

    adl = AnnDataLocation()
    data_manager = DataManager(
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys=["drug_0"],
        rep_keys={
            "cell_line": "cell_line_ccle_embeddings",
            "drug_0": "drug_0_embeddings",
        },
        data_location=adl.obsm["X_state"],
    )

    # Prepare grouped distributions for training
    train_gds = {}
    for cl, adata in train_adatas.items():
        train_gds[cl] = data_manager.prepare_data(adata)

    # Create train samplers
    train_samplers = {}
    for cl in train_adatas:
        train_samplers[cl] = InMemorySampler(
            train_gds[cl], np.random.default_rng(args.seed + hash(cl) % 1000),
            batch_size=args.batch_size,
        )

    if len(train_samplers) > 1:
        sampler = CombinedSampler(
            samplers=train_samplers,
            rng=np.random.default_rng(args.seed),
        )
    else:
        sampler = list(train_samplers.values())[0]

    # Create validation samplers:
    # 1. Holdout (Set B) — all conditions, the real evaluation
    # 2. Train sanity check — 20 conditions from training data, to confirm model is learning
    val_samplers = {}
    for cl, adata in eval_adatas.items():
        gd = data_manager.prepare_data(adata)
        n_eval_conditions = len(gd.data.conditions)
        val_samplers[f"{cl}_holdout"] = ValidationSampler(
            gd,
            n_conditions_on_log_iteration=min(n_eval_conditions, 184),
            n_conditions_on_train_end=None,
            seed=args.seed,
        )
        print(f"  Validation {cl}_holdout: {n_eval_conditions} conditions "
              f"(using {min(n_eval_conditions, 184)} per log step)")

    # Add train sanity check samplers (20 conditions from training data)
    for cl, gd in train_gds.items():
        n_train_conditions = len(gd.data.conditions)
        val_samplers[f"{cl}_train"] = ValidationSampler(
            gd,
            n_conditions_on_log_iteration=min(20, n_train_conditions),
            n_conditions_on_train_end=None,
            seed=args.seed,
        )
        print(f"  Validation {cl}_train (sanity): {n_train_conditions} conditions "
              f"(sampling 20 per log step)")

    # Initialize and get a sample batch
    sampler.init_sampler()
    sample_batch = sampler.sample()
    print(f"Condition shapes: {[(k, v.shape) for k, v in sample_batch['condition'].items()]}")

    # ── Step 4: Create and train model ──────────────────────────────────
    print("\n[Step 4] Creating model...")

    sf = ScaleFlow(solver="otfm")

    NUM_ITERATIONS = args.num_iterations
    WARMUP = 2000
    PEAK_LR = 1e-3
    END_LR = 1e-5

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=PEAK_LR,
        warmup_steps=WARMUP, decay_steps=NUM_ITERATIONS, end_value=END_LR,
    )

    match_fn = partial(match_linear, epsilon=10.0)

    sf.prepare_model(
        sample_batch=sample_batch,
        max_combination_length=1,
        conditioning="concatenation",
        decoder_dims=(256, 256, 256),
        hidden_dims=(256, 256, 256),
        conditioning_kwargs={},
        match_fn=match_fn,
        probability_path={"constant_noise": 0.1},
        optimizer=optax.MultiSteps(optax.adam(learning_rate=lr_schedule), 20),
    )

    # Callbacks
    ckpt_path = os.path.join(out_dir, "best_params.pkl")
    primary_eval = f"{eval_cls[0]}_holdout"

    callbacks = [
        Metrics(["e_distance", "r_squared", "mmd"]),
        LearningRateMonitor(schedule=lr_schedule),
        ModelCheckpoint(
            solver=sf._solver,
            monitor=f"{primary_eval}_mmd_mean",
            save_path=ckpt_path,
            mode="min",
        ),
        WandbLogger(
            project=args.wandb_project,
            out_dir=os.path.join(out_dir, "wandb"),
            config={
                "experiment_id": args.experiment_id,
                "anchor_celllines": anchor_cls,
                "eval_celllines": eval_cls,
                "zero_shot": args.zero_shot,
                "num_iterations": NUM_ITERATIONS,
                "batch_size": args.batch_size,
                "drug_split_seed": args.drug_split_seed,
                "max_cells_per_cellline": args.max_cells_per_cellline,
                "n_set_a": len(set_a),
                "n_set_b": len(set_b),
                "seed": args.seed,
            },
            entity="pancellflow",
        ),
    ]

    print(f"\n[Step 5] Training ({NUM_ITERATIONS} iterations)...")
    sf.train(
        val_dataloader=val_samplers,
        train_dataloader=sampler,
        num_iterations=NUM_ITERATIONS,
        valid_freq=args.valid_freq,
        callbacks=callbacks,
        monitor_metrics=["loss"],
    )

    # ── Step 6: Load best checkpoint and predict ────────────────────────
    print("\n[Step 6] Loading best checkpoint and predicting...")

    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            best_params = cloudpickle.load(f)
        sf._solver.vf_state = sf._solver.vf_state.replace(params=best_params)
        sf._solver.vf_state_inference = sf._solver.vf_state_inference.replace(params=best_params)
        print("Best params restored.")
    else:
        print("WARNING: No checkpoint found, using final params.")

    # Predict on eval cell lines
    for cl_name, adata_eval in eval_adatas.items():
        print(f"\nPredicting {cl_name}...")
        gd_eval = data_manager.prepare_data(adata_eval)
        eval_sampler = ValidationSampler(
            gd_eval, n_conditions_on_log_iteration=None,
            n_conditions_on_train_end=None, seed=args.seed,
        )
        if not eval_sampler.initialized:
            eval_sampler.init_sampler()

        annotation = eval_sampler._data.annotation
        tgt_labels = annotation.tgt_dist_idx_to_labels
        src_labels = annotation.src_dist_idx_to_labels
        tgt_to_src = eval_sampler._tgt_to_src
        gd_data = eval_sampler._data.data

        rows_X, rows_obs = [], []
        for tgt_idx in gd_data.conditions:
            src_idx = tgt_to_src.get(tgt_idx)
            if src_idx is None:
                continue
            drug = tgt_labels[tgt_idx][0] if tgt_idx in tgt_labels else str(tgt_idx)
            cell_line = src_labels[src_idx][0] if src_idx in src_labels else str(src_idx)

            pred = np.array(sf._solver.predict(gd_data.src_data[src_idx], gd_data.conditions[tgt_idx]))
            tgt = np.array(gd_data.tgt_data[tgt_idx])
            src = np.array(gd_data.src_data[src_idx])

            in_set_b = drug in set_b
            in_train = drug not in set_b  # drug was in training for this eval cell line

            for arr, split_name in [(pred, "prediction"), (tgt, "target"), (src, "source")]:
                rows_X.append(arr)
                rows_obs.append(pd.DataFrame({
                    "drug_0": drug, "cell_line": cell_line, "eval_cellline": cl_name,
                    "split": split_name, "in_train": in_train, "in_set_b": in_set_b,
                    "experiment_id": args.experiment_id,
                }, index=range(arr.shape[0])))

        obs = pd.concat(rows_obs, ignore_index=True)
        X = np.concatenate(rows_X, axis=0)
        adata_pred = ad.AnnData(X=X, obs=obs)

        pred_path = os.path.join(out_dir, f"predictions_{cl_name}.h5ad")
        adata_pred.write_h5ad(pred_path)
        print(f"Saved {adata_pred.n_obs:,} cells to {pred_path}")

    # Save experiment config
    import json
    config = {
        "experiment_id": args.experiment_id,
        "anchor_celllines": anchor_cls,
        "eval_celllines": eval_cls,
        "zero_shot": args.zero_shot,
        "set_a_drugs": sorted(set_a),
        "set_b_drugs": sorted(set_b),
        "train_cells_per_cellline": {cl: adata.n_obs for cl, adata in train_adatas.items()},
        "eval_cells_per_cellline": {cl: adata.n_obs for cl, adata in eval_adatas.items()},
        "seed": args.seed,
        "num_iterations": NUM_ITERATIONS,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\nDone! Results saved to {out_dir}")


if __name__ == "__main__":
    main()
