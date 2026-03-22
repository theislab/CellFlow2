"""Sanity check: Tahoe A549 with X_state embedding.

Training restricted to top-N drugs (by mean-shift from control in X_state) + control.
Large model: attention_token pooling, hidden_dims [2048,2048,2048], decoder_dims [4096,4096,4096].

Usage:
    python sanity_check_cf_tahoe_X_state.py --output-dir /path/to/output
    python sanity_check_cf_tahoe_X_state.py --output-dir /path/to/output --top-n-drugs 20
"""
import argparse
import functools
import os
from typing import Any

import anndata as ad
import cloudpickle
import flax.linen as nn
import jax
import numpy as np
import optax
import pandas as pd
import scanpy as sc

import cellflow.preprocessing as cfpp
from cellflow.metrics import compute_scalar_mmd
from cellflow.model import CellFlow
from cellflow.training import LoggingCallback, Metrics, WandbLogger
from cellflow.training import ComputationCallback
from cellflow.utils import match_linear


# ---------------------------------------------------------------------------
# Callbacks  (same as sanity_check_cf.py)
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
    """Log MMD(pred, src), MMD(src, tgt), and gap-closure ratio during training."""

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


class WandbPerStepLossLogger(LoggingCallback):
    """Log per-step training loss directly to wandb between validation checkpoints.

    Must be placed *after* WandbLogger in the callbacks list so that wandb is
    already initialised when on_train_begin fires.  Uses a custom step metric
    (train_step) so the per-step curve does not conflict with WandbLogger's
    auto-incremented step counter.
    """

    def __init__(self, cf_model):
        self._cf = cf_model
        self._logged_steps = 0

    def on_train_begin(self):
        import wandb
        wandb.define_metric("train/loss", step_metric="train_step")
        self._logged_steps = 0

    def on_log_iteration(self, dict_to_log, **_):
        import wandb
        losses = self._cf._trainer.training_logs["loss"]
        for i, loss in enumerate(losses[self._logged_steps:], start=self._logged_steps):
            wandb.log({"train/loss": float(loss), "train_step": i + 1})
        self._logged_steps = len(losses)
        return dict_to_log

    def on_train_end(self, dict_to_log):
        return self.on_log_iteration(dict_to_log)


# ---------------------------------------------------------------------------
# Data path
# ---------------------------------------------------------------------------

TAHOE_PATH = "/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/unipert/tahoe_a549_w_emb.h5ad"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n-drugs", type=int, default=10,
                        help="Number of top drugs (by mean-shift in X_state) to use for validation")
    parser.add_argument("--top-n-train-drugs", type=int, default=None,
                        help="Number of top drugs to use for training (default: all eligible drugs)")
    parser.add_argument("--extra-n-drugs", type=int, default=None,
                        help="Add N extra drugs (beyond top-N val drugs) to the training set. "
                             "Requires --extra-drug-type.")
    parser.add_argument("--extra-drug-type", choices=["strong", "weak"], default=None,
                        help="Whether extra drugs are the strongest or weakest outside the top-N val set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-iterations", type=int, default=250_000)
    parser.add_argument("--valid-freq", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epsilon", type=float, default=0.5,
                        help="OT matching regularization")
    parser.add_argument("--noise", type=float, default=0.5,
                        help="Constant noise sigma on the flow path")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Adam learning rate")
    args = parser.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import warnings
    from pandas.errors import SettingWithCopyWarning
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", SettingWithCopyWarning)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    ckpt_path = os.path.join(args.output_dir, "best_checkpoint.pkl")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading Tahoe A549 from {TAHOE_PATH} ...")
    adata = sc.read_h5ad(TAHOE_PATH)
    adata.obs["control"] = (
        (adata.obs["drug_0"] == "control") & (adata.obs["drug_1"] == "control")
    )
    assert "X_state" in adata.obsm, (
        f"X_state not in obsm. Available: {list(adata.obsm.keys())}"
    )
    print(f"  {adata.n_obs} cells, X_state shape {adata.obsm['X_state'].shape}")

    # ------------------------------------------------------------------
    # Drug effect ranking (all single-drug conditions with >= batch_size cells)
    # Only include drugs with >= batch_size cells so minibatches can be
    # drawn without replacement issues (sampling is in cellflow/data/_dataloader.py
    # TrainSampler._sample_from_mask, line ~58: replace=True).
    # ------------------------------------------------------------------
    TOP_N_DRUGS = args.top_n_drugs
    TOP_N_TRAIN_DRUGS = args.top_n_train_drugs
    batch_size = args.batch_size
    X = np.asarray(adata.obsm["X_state"])
    ctrl_mask = adata.obs["control"].values
    ctrl_mean = X[ctrl_mask].mean(axis=0)
    drug_arr = adata.obs["drug_0"].values.astype(str)
    drug_effect = {}
    drug_count = {}
    for drug in np.unique(drug_arr[~ctrl_mask]):
        mask = (drug_arr == drug) & ~ctrl_mask
        n_cells = mask.sum()
        drug_count[drug] = int(n_cells)
        drug_effect[drug] = float(np.linalg.norm(X[mask].mean(axis=0) - ctrl_mean))

    # Restrict to drugs with enough samples for minibatch
    drugs_with_enough = sorted(
        [d for d in drug_effect if drug_count[d] >= batch_size],
        key=drug_effect.get, reverse=True,
    )

    # Validation drugs: top-N by effect
    top_drugs = drugs_with_enough[:TOP_N_DRUGS]
    if len(top_drugs) < TOP_N_DRUGS:
        print(f"  Warning: only {len(top_drugs)} drugs have >= {batch_size} cells; "
              f"requested top {TOP_N_DRUGS} for validation.")

    # non-top-N pool sorted strong → weak (drugs_with_enough is already sorted descending)
    non_top_drugs = drugs_with_enough[TOP_N_DRUGS:]

    # Training drugs: all eligible, top-N, or top-N val + N strong/weak extras
    if args.extra_n_drugs is not None:
        if args.extra_drug_type is None:
            raise ValueError("--extra-drug-type must be set when --extra-n-drugs is used")
        if args.extra_drug_type == "strong":
            extra_drugs = non_top_drugs[:args.extra_n_drugs]
        else:  # weak
            extra_drugs = non_top_drugs[-args.extra_n_drugs:]
        if len(extra_drugs) < args.extra_n_drugs:
            print(f"  Warning: only {len(extra_drugs)} {args.extra_drug_type} extras available "
                  f"(requested {args.extra_n_drugs})")
        train_drugs = list(top_drugs) + list(extra_drugs)
        train_mode_label = f"top{TOP_N_DRUGS}+{len(extra_drugs)}{args.extra_drug_type}"
    elif TOP_N_TRAIN_DRUGS is None:
        train_drugs = drugs_with_enough
        train_mode_label = "all"
    else:
        train_drugs = drugs_with_enough[:TOP_N_TRAIN_DRUGS]
        train_mode_label = f"top{TOP_N_TRAIN_DRUGS}"
        if len(train_drugs) < TOP_N_TRAIN_DRUGS:
            print(f"  Warning: only {len(train_drugs)} drugs have >= {batch_size} cells; "
                  f"requested top {TOP_N_TRAIN_DRUGS} for training.")

    print(f"\nValidation drugs ({len(top_drugs)} top by effect, min {batch_size} cells):")
    for d in top_drugs:
        print(f"  {d:30s}  effect={drug_effect[d]:.4f}  n_cells={drug_count[d]}")
    if args.extra_n_drugs is not None:
        print(f"\nExtra {args.extra_drug_type} drugs ({len(extra_drugs)}):")
        for d in extra_drugs:
            print(f"  {d:30s}  effect={drug_effect[d]:.4f}  n_cells={drug_count[d]}")
    print(f"\nTraining drugs: {len(train_drugs)} ({train_mode_label})")

    # ------------------------------------------------------------------
    # Build train / validation adatas
    # One-hot encoding is done on adata_train so the vocabulary covers all
    # training drugs.  adata_val shares the same uns via .copy(), ensuring
    # the model sees consistent encodings at validation time.
    # ------------------------------------------------------------------
    single_drug_mask = (adata.obs["drug_1"] == "control")
    train_keep = ctrl_mask | (
        adata.obs["drug_0"].isin(train_drugs) & single_drug_mask
    )
    adata_train = adata[train_keep].copy()
    print(f"\nTraining set: {adata_train.n_obs} cells, "
          f"{adata_train.obs['control'].sum()} control, "
          f"{(~adata_train.obs['control']).sum()} treated")

    print("\nOne-hot encoding drugs (vocabulary from training set) ...")
    cfpp.encode_onehot(adata_train, covariate_keys=["drug_0", "drug_1"],
                       uns_key_added="drug_onehot")
    n_oh = len(adata_train.uns["drug_onehot"])
    oh_dim = len(next(iter(adata_train.uns["drug_onehot"].values())))
    print(f"  {n_oh} drug combinations -> one-hot dim={oh_dim}")

    # Validation set: top-N drugs, same vocabulary
    val_keep = ctrl_mask | (
        adata.obs["drug_0"].isin(top_drugs) & single_drug_mask
    )
    adata_val = adata[val_keep].copy()
    adata_val.uns["drug_onehot"] = adata_train.uns["drug_onehot"]
    print(f"Validation set: {adata_val.n_obs} cells, "
          f"{adata_val.obs['control'].sum()} control, "
          f"{(~adata_val.obs['control']).sum()} treated")

    # ------------------------------------------------------------------
    # CellFlow data setup
    # ------------------------------------------------------------------
    cf = CellFlow(adata_train, solver="otfm")

    cf.prepare_data(
        sample_rep="X_state",
        control_key="control",
        perturbation_covariates={"drug_treatment": ("drug_0", "drug_1")},
        perturbation_covariate_reps={"drug_treatment": "drug_onehot"},
        max_combination_length=2,
        null_value=0.0,
    )

    n_conds_log = min(20, len(top_drugs))
    cf.prepare_validation_data(
        adata_val,
        name="tahoe_train",
        n_conditions_on_log_iteration=n_conds_log,
        n_conditions_on_train_end=None,
    )

    print(f"  Train data: {cf.train_data.n_perturbations} perturbation conditions, "
          f"{cf.train_data.n_controls} control groups")

    # ------------------------------------------------------------------
    # Model  (large architecture from notebook)
    # ------------------------------------------------------------------
    layers_before_pool = {
        "drug_treatment": {"layer_type": "mlp", "dims": [1024], "dropout_rate": 0.5},
    }
    layers_after_pool = {
        "layer_type": "mlp", "dims": [1024], "dropout_rate": 0.0,
    }

    match_fn = functools.partial(
        match_linear, epsilon=args.epsilon, tau_a=1.0, tau_b=1.0
    )

    cf.prepare_model(
        condition_mode="deterministic",
        regularization=0.0,
        pooling="attention_token",
        pooling_kwargs={},
        layers_before_pool=layers_before_pool,
        layers_after_pool=layers_after_pool,
        condition_embedding_dim=256,
        cond_output_dropout=0.9,
        condition_encoder_kwargs={},
        pool_sample_covariates=True,
        time_freqs=1024,
        time_encoder_dims=[1024, 1024, 1024],
        time_encoder_dropout=0.0,
        hidden_dims=[2048, 2048, 2048],
        hidden_dropout=0.0,
        conditioning="concatenation",
        decoder_dims=[4096, 4096, 4096],
        vf_act_fn=nn.silu,
        vf_kwargs=None,
        probability_path={"constant_noise": args.noise},
        match_fn=match_fn,
        optimizer=optax.MultiSteps(optax.adam(args.lr), 20),
        solver_kwargs={},
        layer_norm_before_concatenation=False,
        linear_projection_before_concatenation=False,
        seed=args.seed,
    )

    total_params = sum(
        x.size for x in jax.tree_util.tree_leaves(cf._solver.vf_state.params)
    )
    print(f"Total parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    VALID_FREQ = args.valid_freq
    monitor_metric = "tahoe_train_mmd_mean"

    callbacks = [
        Metrics(metrics=["r_squared", "mmd", "e_distance"]),
        PerturbationMetrics(),
        LossLogger(cf, valid_freq=VALID_FREQ),
        WandbLogger(
            project="sanity-check-cellflow",
            out_dir="./wandb_logs",
            config={
                "dataset": "tahoe",
                "sample_rep": "X_state",
                "top_n_drugs_val": TOP_N_DRUGS,
                "train_mode": train_mode_label,
                "n_train_drugs": len(train_drugs),
                "extra_n_drugs": args.extra_n_drugs,
                "extra_drug_type": args.extra_drug_type,
                "condition_encoding": "onehot",
                "num_iterations": args.num_iterations,
                "batch_size": args.batch_size,
                "hidden_dims": [2048, 2048, 2048],
                "decoder_dims": [4096, 4096, 4096],
                "cond_output_dropout": 0.9,
                "seed": args.seed,
                "total_params": total_params,
                "epsilon": args.epsilon,
                "noise": args.noise,
                "lr": args.lr,
            },
            entity="pancellflow",
        ),
        WandbPerStepLossLogger(cf),
        ModelCheckpoint(
            solver=cf._solver,
            monitor=monitor_metric,
            save_path=ckpt_path,
            mode="min",
            verbose=True,
        ),
        SaveAllCheckpoints(cf._solver, ckpt_dir, valid_freq=VALID_FREQ, verbose=False),
    ]

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.num_iterations} iters, valid every {VALID_FREQ} ...")
    cf.train(
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        valid_freq=VALID_FREQ,
        callbacks=callbacks,
        monitor_metrics=["loss"],
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
