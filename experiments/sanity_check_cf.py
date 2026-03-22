"""Sanity check using CellFlow with one-hot condition encoding.

Same structure as train_sanity_check.py but uses CellFlow instead of ScaleFlow.
Supports both PCA (computed on the fly) and pre-computed obsm embeddings.

Usage:
    # PCA-100 space
    python sanity_check_cf.py --dataset tahoe --obsm-key X_pca \
        --epsilon 0.01 --noise 0.0 --output-dir /path/to/output

    # Pre-computed embedding space
    python sanity_check_cf.py --dataset tahoe --obsm-key X_scimilarity \
        --epsilon 0.01 --noise 0.0 --output-dir /path/to/output
"""
import argparse
import os
from typing import Any

import anndata as ad
import cloudpickle
import jax
import numpy as np
import optax
import pandas as pd
import scanpy as sc
from functools import partial

import cellflow.preprocessing as cfpp
from cellflow.metrics import compute_scalar_mmd
from cellflow.model import CellFlow
from cellflow.training import LoggingCallback, Metrics, WandbLogger
from cellflow.training import ComputationCallback
from cellflow.utils import match_linear


# ---------------------------------------------------------------------------
# Callbacks  (identical to train_sanity_check.py — solver interface is the same)
# ---------------------------------------------------------------------------

class ModelCheckpoint(LoggingCallback):
    def __init__(self, solver, monitor: str, save_path: str, mode="min", verbose=True):
        self.solver = solver
        self.monitor = monitor
        self.save_path = save_path
        self.mode = mode
        self.verbose = verbose
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def on_train_begin(self) -> None:
        self.best_value = float("inf") if self.mode == "min" else float("-inf")

    def on_log_iteration(self, dict_to_log, iteration=None, **_):
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
            if self.verbose:
                print(f"  ModelCheckpoint: {self.monitor}={current:.6f}")
        return dict_to_log

    def on_train_end(self, dict_to_log):
        return dict_to_log


class SaveAllCheckpoints(LoggingCallback):
    """Save checkpoint at every validation step.

    CellFlow's CallbackRunner does not pass `iteration` to LoggingCallbacks,
    so we track the call count internally.  Call N corresponds to validation
    after iteration  N * valid_freq  (the first call is N=1).
    """

    def __init__(self, solver, save_dir, valid_freq, verbose=False):
        self.solver = solver
        self.save_dir = save_dir
        self.valid_freq = valid_freq
        self.verbose = verbose
        self._call_count = 0

    def on_train_begin(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self._call_count = 0

    def on_log_iteration(self, dict_to_log, **_):
        self._call_count += 1
        step = self._call_count
        path = os.path.join(self.save_dir, f"checkpoint_{step:04d}.pkl")
        with open(path, "wb") as f:
            cloudpickle.dump(self.solver.vf_state_inference.params, f)
        if self.verbose:
            print(f"  SaveAll: call {step} -> {path}")
        return dict_to_log

    def on_train_end(self, dict_to_log):
        return dict_to_log


class PerturbationMetrics(ComputationCallback):
    """Log MMD(pred, src), MMD(src, tgt), and gap-closure ratio during training.

    gap_closure = 1 - MMD(pred, tgt) / MMD(src, tgt)
        1.0  -> perfect (pred matches target)
        0.0  -> trivial (pred == control)
        <0   -> worse than returning control unchanged
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def on_train_begin(self):
        pass

    def on_log_iteration(self, valid_source_data, valid_true_data, valid_pred_data, solver):
        metrics = {}
        for k in valid_true_data:
            src = valid_source_data[k]
            tgt = valid_true_data[k]
            pred = valid_pred_data[k]
            mmd_ps_list, mmd_st_list, gc_list = [], [], []
            for cond_key in tgt:
                if cond_key not in pred or cond_key not in src:
                    continue
                pt = float(compute_scalar_mmd(pred[cond_key], tgt[cond_key]))
                ps = float(compute_scalar_mmd(pred[cond_key], src[cond_key]))
                st = float(compute_scalar_mmd(src[cond_key], tgt[cond_key]))
                mmd_ps_list.append(ps)
                mmd_st_list.append(st)
                if st > self.eps:
                    gc_list.append(1.0 - pt / st)
            metrics[f"{k}_mmd_pred_src_mean"] = float(np.mean(mmd_ps_list)) if mmd_ps_list else 0.0
            metrics[f"{k}_mmd_src_tgt_mean"] = float(np.mean(mmd_st_list)) if mmd_st_list else 0.0
            metrics[f"{k}_gap_closure_mean"] = float(np.mean(gc_list)) if gc_list else 0.0
        return metrics

    def on_train_end(self, valid_source_data, valid_true_data, valid_pred_data, solver):
        return self.on_log_iteration(valid_source_data, valid_true_data, valid_pred_data, solver)


class LossLogger(ComputationCallback):
    """Log mean training loss at each validation step via the trainer's internal buffer."""

    def __init__(self, cf_model, valid_freq: int):
        self._cf = cf_model
        self._valid_freq = valid_freq

    def on_train_begin(self):
        pass

    def on_log_iteration(self, valid_source_data, valid_true_data, valid_pred_data, solver):
        losses = self._cf._trainer.training_logs["loss"]
        mean_loss = float(np.mean(losses[-self._valid_freq:]))
        return {"mean_loss": mean_loss}

    def on_train_end(self, valid_source_data, valid_true_data, valid_pred_data, solver):
        return self.on_log_iteration(valid_source_data, valid_true_data, valid_pred_data, solver)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mark_control(adata):
    adata.obs["control"] = (
        (adata.obs["drug_0"] == "control") & (adata.obs["drug_1"] == "control")
    )
    return adata


DATASETS = {
    "tahoe": "/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/unipert/tahoe_a549_w_emb.h5ad"
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-iterations", type=int, default=100_000)
    parser.add_argument("--valid-freq", type=int, default=500)
    parser.add_argument("--checkpoint-steps", default="1,4,8,16,32")
    parser.add_argument("--obsm-key", default="X_pca",
                        help="Cell representation: 'X_pca' computes PCA on the fly; "
                             "anything else (X_state, X_scimilarity, ...) uses the "
                             "existing obsm key directly")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="OT matching regularization (small=tight matching, large=random)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Constant noise sigma on the flow path")
    parser.add_argument("--n-pca", type=int, default=100,
                        help="Number of PCA components (only used when --obsm-key X_pca)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    ckpt_path = os.path.join(args.output_dir, "best_checkpoint.pkl")
    step_list = [int(s) for s in args.checkpoint_steps.split(",")]

    # ------------------------------------------------------------------
    # Load data  (X is already normalized + log1p)
    # ------------------------------------------------------------------
    print(f"Loading {args.dataset} ...")
    adata = sc.read_h5ad(DATASETS[args.dataset])
    adata = mark_control(adata)
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")

    # ------------------------------------------------------------------
    # Cell representation
    # ------------------------------------------------------------------
    sample_rep = args.obsm_key

    if args.obsm_key == "X_pca":
        print(f"Computing PCA-{args.n_pca} on 2000 HVGs ...")
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata_hvg = adata[:, adata.var["highly_variable"]].copy()
        cfpp.centered_pca(adata_hvg, n_comps=args.n_pca, method="scanpy",
                          keep_centered_data=False)
        adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
        adata.uns["pca"] = adata_hvg.uns["pca"]
        adata.varm["PCs"] = np.zeros((adata.n_vars, args.n_pca))
        adata.varm["PCs"][adata.var["highly_variable"].values] = adata_hvg.varm["PCs"]
        adata.varm["X_mean"] = np.zeros((adata.n_vars, 1))
        adata.varm["X_mean"][adata.var["highly_variable"].values] = adata_hvg.varm["X_mean"]
        del adata_hvg
        print(f"  X_pca shape: {adata.obsm['X_pca'].shape}")
    else:
        assert args.obsm_key in adata.obsm, (
            f"{args.obsm_key} not found in obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )
        print(f"Using pre-computed obsm['{args.obsm_key}'], "
              f"shape={adata.obsm[args.obsm_key].shape}")

    # ------------------------------------------------------------------
    # One-hot encode conditions
    # ------------------------------------------------------------------
    print("One-hot encoding drugs ...")
    # Include "control" so DataManager can look up representation for control cells
    cfpp.encode_onehot(adata, covariate_keys=["drug_0", "drug_1"],
                       uns_key_added="drug_onehot")
    n_oh = len(adata.uns["drug_onehot"])
    oh_dim = len(next(iter(adata.uns["drug_onehot"].values())))
    print(f"  {n_oh} drugs -> one-hot dim={oh_dim}")

    # ------------------------------------------------------------------
    # No train/val split — one-hot conditions can't generalise to
    # unseen drugs, so we train on ALL data and validate on the same.
    # The question is purely: can the model overfit?
    # ------------------------------------------------------------------
    train_drugs = set(adata.obs.loc[~adata.obs["control"], "drug_0"].unique())
    print(f"  All data used for train+val ({len(train_drugs)} drugs)")

    # ------------------------------------------------------------------
    # CellFlow data setup
    # ------------------------------------------------------------------
    cf = CellFlow(adata, solver="otfm")

    cf.prepare_data(
        sample_rep=sample_rep,
        control_key="control",
        perturbation_covariates={"drug_treatment": ("drug_0", "drug_1")},
        perturbation_covariate_reps={"drug_treatment": "drug_onehot"},
        max_combination_length=2,
        null_value=0.0,
    )

    cf.prepare_validation_data(
        adata, name=f"{args.dataset}_train",
        n_conditions_on_log_iteration=20,
        n_conditions_on_train_end=None,
    )

    print(f"  Train data: {cf.train_data.n_perturbations} perturbation conditions, "
          f"{cf.train_data.n_controls} control groups")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    batch_size = 1024
    hidden_dims = (1024, 1024)
    decoder_dims = (1024, 1024)

    layers_before_pool = [{
        "layer_type": "mlp",
        "dims": [1024, 1024],
        "dropout_rate": 0.0,
    }]

    print(f"  OT epsilon={args.epsilon}, path noise={args.noise}")
    cf.prepare_model(
        condition_mode="deterministic",
        pooling="mean",
        layers_before_pool=layers_before_pool,
        condition_embedding_dim=256,
        cond_output_dropout=0.0,
        hidden_dims=hidden_dims,
        hidden_dropout=0.0,
        conditioning="concatenation",
        decoder_dims=decoder_dims,
        decoder_dropout=0.0,
        probability_path={"constant_noise": args.noise},
        match_fn=partial(match_linear, epsilon=args.epsilon),
        optimizer=optax.MultiSteps(optax.adam(learning_rate=1e-4), 20),
        seed=args.seed,
    )

    total_params = sum(x.size for x in jax.tree_util.tree_leaves(cf._solver.vf_state.params))
    print(f"Total parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    VALID_FREQ = args.valid_freq
    monitor_metric = f"{args.dataset}_train_mmd_mean"

    callbacks = [
        Metrics(["r_squared", "mmd"]),
        PerturbationMetrics(),
        LossLogger(cf, valid_freq=VALID_FREQ),
        WandbLogger(
            project="sanity-check-cellflow",
            out_dir="./wandb_logs",
            config={
                "dataset": args.dataset,
                "sample_rep": sample_rep,
                "n_pca": args.n_pca if sample_rep == "X_pca" else None,
                "condition_encoding": "onehot",
                "num_iterations": args.num_iterations,
                "batch_size": batch_size,
                "hidden_dims": list(hidden_dims),
                "decoder_dims": list(decoder_dims),
                "cond_output_dropout": 0.0,
                "seed": args.seed,
                "total_params": total_params,
                "epsilon": args.epsilon,
                "noise": args.noise,
            },
            entity="pancellflow",
        ),
        ModelCheckpoint(
            solver=cf._solver,
            monitor=monitor_metric,
            save_path=ckpt_path,
            mode="min",
            verbose=True,
        ),
        SaveAllCheckpoints(cf._solver, ckpt_dir, valid_freq=VALID_FREQ, verbose=False),
    ]

    print(f"\nTraining for {args.num_iterations} iters, valid every {VALID_FREQ} ...")
    cf.train(
        num_iterations=args.num_iterations,
        batch_size=batch_size,
        valid_freq=VALID_FREQ,
        callbacks=callbacks,
        monitor_metrics=["loss"],
    )

    # ==================================================================
    # Ad-hoc MMD: compare with wandb metrics
    # ==================================================================
    print("\n" + "=" * 70)
    print("AD-HOC MMD COMPUTATION (same as analyze_checkpoints notebook)")
    print("=" * 70)

    td = cf.train_data

    # Map perturbation_idx -> control_idx  (reverse of control_to_perturbation)
    ptb_to_ctrl = {}
    for ctrl_idx, ptb_indices in td.control_to_perturbation.items():
        for pidx in ptb_indices:
            ptb_to_ctrl[pidx] = ctrl_idx

    src_dict, condition_dict, target_dict = {}, {}, {}
    for ptb_idx in range(td.n_perturbations):
        ctrl_idx = ptb_to_ctrl.get(ptb_idx)
        if ctrl_idx is None:
            continue
        cov = td.perturbation_idx_to_covariates[ptb_idx]
        key = "|".join(str(c) for c in cov)

        src_dict[key] = td.cell_data[td.split_covariates_mask == ctrl_idx]
        target_dict[key] = td.cell_data[td.perturbation_covariates_mask == ptb_idx]
        condition_dict[key] = {k: v[[ptb_idx]] for k, v in td.condition_data.items()}

    print(f"  {len(src_dict)} conditions for ad-hoc prediction")

    MAX_PRED_CELLS = 512
    rng_pred = np.random.default_rng(args.seed + 99)

    def _subsample(arr, max_n=MAX_PRED_CELLS):
        if arr.shape[0] <= max_n:
            return arr
        return arr[rng_pred.choice(arr.shape[0], max_n, replace=False)]

    def _inject_params(params):
        cf._solver.vf_state = cf._solver.vf_state.replace(params=params)
        cf._solver.vf_state_inference = cf._solver.vf_state_inference.replace(params=params)

    def adhoc_mmd(label="", condition_keys=None):
        """Predict conditions, return DataFrame with per-condition MMDs."""
        keys = condition_keys if condition_keys is not None else list(src_dict.keys())
        rows = []
        for key in keys:
            drug = key.split("|")[0]
            src_sub = _subsample(src_dict[key])
            tgt_sub = _subsample(target_dict[key])
            pred = np.array(cf._solver.predict(src_sub, condition=condition_dict[key]))
            rows.append({
                "drug_0": drug, "condition_key": key,
                "mmd_pred_tgt": float(compute_scalar_mmd(pred, tgt_sub)),
                "mmd_pred_ctrl": float(compute_scalar_mmd(pred, src_sub)),
            })
        df = pd.DataFrame(rows)
        print(f"  {label:>12}  MMD(pred,tgt)={df.mmd_pred_tgt.mean():.5f}  "
              f"MMD(pred,ctrl)={df.mmd_pred_ctrl.mean():.5f}  ({len(keys)} conds)")
        return df

    # Compute baseline first (no ODE needed) and save immediately
    ctrl_data = {}
    for key in src_dict:
        ctrl_data[key] = _subsample(src_dict[key])
    baseline_rows = []
    for key in target_dict:
        tgt_sub = _subsample(target_dict[key])
        ctrl_sub = ctrl_data.get(key, _subsample(list(src_dict.values())[0]))
        baseline_rows.append({
            "condition_key": key,
            "mmd_ctrl_tgt": float(compute_scalar_mmd(ctrl_sub, tgt_sub)),
        })
    df_baseline = pd.DataFrame(baseline_rows)
    df_baseline.to_csv(os.path.join(args.output_dir, "mmd_baseline.csv"), index=False)
    print(f"\n  Baseline MMD(ctrl,tgt): mean={df_baseline.mmd_ctrl_tgt.mean():.5f} [saved]\n")

    # Subsample conditions for per-step analysis to avoid timeout
    MAX_ADHOC_CONDS = 100
    all_keys = list(src_dict.keys())
    if len(all_keys) > MAX_ADHOC_CONDS:
        adhoc_keys = list(rng_pred.choice(all_keys, MAX_ADHOC_CONDS, replace=False))
        print(f"  Subsampled {MAX_ADHOC_CONDS}/{len(all_keys)} conditions for per-step ad-hoc")
    else:
        adhoc_keys = all_keys

    all_step_dfs = []

    # Per-step ad-hoc
    for step in step_list:
        ckpt_file = os.path.join(ckpt_dir, f"checkpoint_{step:04d}.pkl")
        if not os.path.isfile(ckpt_file):
            print(f"  Checkpoint step {step} not found, skipping")
            continue
        with open(ckpt_file, "rb") as f:
            step_params = cloudpickle.load(f)
        _inject_params(step_params)
        df = adhoc_mmd(f"step {step}", condition_keys=adhoc_keys)
        df["checkpoint_step"] = step
        all_step_dfs.append(df)

    if all_step_dfs:
        mmd_along = pd.concat(all_step_dfs, ignore_index=True)
        mmd_along.to_csv(os.path.join(args.output_dir, "mmd_along_training.csv"), index=False)
        print(f"  mmd_along_training.csv saved ({len(mmd_along)} rows)")

    # Best checkpoint — use all conditions
    if os.path.isfile(ckpt_path):
        with open(ckpt_path, "rb") as f:
            best_params = cloudpickle.load(f)
        _inject_params(best_params)
        df_best = adhoc_mmd("best ckpt")
        df_best.to_csv(os.path.join(args.output_dir, "mmd_best.csv"), index=False)
        print(f"  mmd_best.csv saved ({len(df_best)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
