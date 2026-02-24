"""Train a model and generate predictions from the best checkpoint.

Trains Tahoe dataset with subsampling options using the Equilibrium Matching solver.
After training, loads the checkpoint that achieved the best monitored metric and
runs predictions over all conditions (not just validation), storing results to disk.

Options:
    --cell-fraction F: Target total cells as fraction of original dataset
    --condition-fraction F: Keep F of conditions (randomly selected)
    --output-dir DIR: Directory to store model checkpoint and predictions

The two subsampling options combine: first select conditions, then resample cells to
reach the target count (upsampling with replacement if needed).

When condition-fraction < 1.0, 60 perturbations are held out for validation (fixed
across all experiments) to evaluate generalization to unseen conditions.
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

from scaleflow.data import AnnDataLocation, DataManager, split_datasets
from scaleflow.data._dataloader import InMemorySampler, ValidationSampler
from scaleflow.model import ScaleFlow
from scaleflow.training import EarlyStopping, LoggingCallback, Metrics, WandbLogger


# ---------------------------------------------------------------------------
# ModelCheckpoint callback
# ---------------------------------------------------------------------------

class ModelCheckpoint(LoggingCallback):
    """Save the best solver params when a monitored metric improves.

    Parameters
    ----------
    solver
        The solver instance (reference). Params are read at each log step.
    monitor
        Metric name to monitor.
    save_path
        File path to write the best params.
    mode
        'min' if lower is better, 'max' if higher is better.
    verbose
        Print a message when a new best is saved.
    """

    def __init__(self, solver, monitor: str, save_path: str, mode: str = "min", verbose: bool = True):
        self.solver = solver
        self.monitor = monitor
        self.save_path = save_path
        self.mode = mode
        self.verbose = verbose
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def on_train_begin(self) -> None:
        self.best_value = float("inf") if self.mode == "min" else float("-inf")

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
            if self.verbose:
                print(f"ModelCheckpoint: saved best params ({self.monitor}={current:.6f})")
        return dict_to_log

    def on_train_end(self, dict_to_log: dict[str, Any]) -> dict[str, Any]:
        return dict_to_log


# ---------------------------------------------------------------------------
# Data helpers (identical to updated_together_training.py)
# ---------------------------------------------------------------------------

def resample_cells(adata, target_cells, rng):
    if target_cells == adata.n_obs:
        return adata
    fraction = target_cells / adata.n_obs
    replace = fraction > 1.0
    indices = []
    for _, group in adata.obs.groupby(["drug_0", "drug_1"], observed=True):
        group_indices = group.index.tolist()
        n_sample = max(1, int(len(group_indices) * fraction))
        sampled = rng.choice(group_indices, size=n_sample, replace=replace)
        indices.extend(sampled)
    return adata[indices].copy()


def holdout_conditions(adata, n_holdout, seed):
    holdout_rng = np.random.default_rng(seed)
    condition_list = adata.obs.groupby(["drug_0", "drug_1"], observed=True).size().index.tolist()
    control = ("control", "control")
    non_control = [c for c in condition_list if c != control]
    holdout_idx = holdout_rng.choice(len(non_control), size=n_holdout, replace=False)
    holdout = [non_control[i] for i in holdout_idx]
    assert control not in holdout
    holdout_mask = adata.obs.set_index(["drug_0", "drug_1"]).index.isin(holdout)
    control_mask = adata.obs["control"]
    train_adata = adata[~holdout_mask].copy()
    val_adata = adata[holdout_mask | control_mask].copy()
    assert train_adata.obs["control"].any()
    assert val_adata.obs["control"].any()
    return train_adata, val_adata


def subsample_conditions(adata, fraction, rng):
    if fraction >= 1.0:
        return adata
    condition_list = adata.obs.groupby(["drug_0", "drug_1"], observed=True).size().index.tolist()
    control = ("control", "control")
    non_control = [c for c in condition_list if c != control]
    n_select = max(1, int(len(condition_list) * fraction) - 1)
    selected_idx = rng.choice(len(non_control), size=min(n_select, len(non_control)), replace=False)
    selected = [non_control[i] for i in selected_idx] + [control]
    assert control in selected
    mask = adata.obs.set_index(["drug_0", "drug_1"]).index.isin(selected)
    result = adata[mask].copy()
    assert result.obs["control"].any()
    return result


def holdout_train_nn_mean(adata_ref, train_conditions, holdout_conditions_list):
    if not train_conditions or not holdout_conditions_list:
        return None
    def vec(d0, d1):
        return np.asarray(adata_ref.uns["drug_0_embeddings"][d0]).ravel()
    train_arr = np.stack([vec(c[0], c[1]) for c in train_conditions])
    holdout_arr = np.stack([vec(c[0], c[1]) for c in holdout_conditions_list])
    pairwise = np.linalg.norm(holdout_arr[:, None, :] - train_arr[None, :, :], axis=2)
    return float(pairwise.min(axis=1).mean())


def randomize_embeddings(adata, rng):
    for key in ["drug_0_embeddings", "drug_1_embeddings"]:
        emb_dict = adata.uns[key]
        for name, vec in emb_dict.items():
            vec = np.asarray(vec)
            emb_dict[name] = rng.standard_normal(vec.shape).astype(vec.dtype)
    return adata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--cell-fraction", type=float, default=1.0)
parser.add_argument("--condition-fraction", type=float, default=1.0)
parser.add_argument("--experiment-type", type=str, default="full")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--random-embeddings", action="store_true")
parser.add_argument("--solver", type=str, default="otfm")
parser.add_argument("--obsm-key", type=str, default="X_state")
parser.add_argument("--output-dir", type=str, default="./outputs/predictions",
                    help="Directory to store model checkpoint and predictions")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
ckpt_path = os.path.join(args.output_dir, "best_params.pkl")
predictions_path = os.path.join(args.output_dir, "predictions.h5ad")

rng = np.random.default_rng(args.seed)

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

print("Loading Tahoe dataset...")
#adata = sc.read_h5ad('/lustre/groups/ml01/workspace/alejandro.tejada/tahoe_with_embeddings_normalized.h5ad')
adata = sc.read_h5ad('/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/unipert/tahoe_a549_w_se.h5ad')
adata.obs["control"] = (
    (adata.obs["drug_0"] == "control") &
    (adata.obs["drug_1"] == "control")
)

n_cells_original = adata.n_obs
n_conditions_original = adata.obs.groupby(["drug_0", "drug_1"], observed=True).ngroups
print(f"Original dataset: {n_cells_original} cells, {n_conditions_original} conditions")

if args.random_embeddings:
    print("Replacing drug embeddings with random vectors...")
    adata = randomize_embeddings(adata, rng)

adata_full = adata

N_HOLDOUT = 60
adata_holdout = None

if args.experiment_type == "condition_subsampling":
    print(f"Holding out {N_HOLDOUT} conditions for validation...")
    adata, adata_holdout = holdout_conditions(adata, N_HOLDOUT, seed=args.seed)
    n_conditions_pool = adata.obs.groupby(["drug_0", "drug_1"], observed=True).ngroups
    print(f"After holdout: {adata.n_obs} cells, {n_conditions_pool} conditions for training pool")
    print(f"Holdout validation set: {adata_holdout.n_obs} cells, {N_HOLDOUT} conditions")

    print(f"Subsampling to {args.condition_fraction:.0%} of training pool conditions...")
    adata = subsample_conditions(adata, args.condition_fraction, rng)
    n_conditions_after = adata.obs.groupby(["drug_0", "drug_1"], observed=True).ngroups
    print(f"After condition subsampling: {adata.n_obs} cells, {n_conditions_after} conditions")

if args.cell_fraction < 1.0:
    target_cells = int(n_cells_original * args.cell_fraction)
    if target_cells == adata.n_obs:
        print(f"Already at target {target_cells} cells, skipping resampling")
    else:
        mode = "upsampling" if target_cells > adata.n_obs else "downsampling"
        print(f"Resampling ({mode}) to {target_cells} cells ({args.cell_fraction:.0%} of original)...")
        adata = resample_cells(adata, target_cells, rng)
        print(f"After resampling: {adata.n_obs} cells")

holdout_train_nn_mean_dist = None
if adata_holdout is not None:
    train_cond_list = adata.obs.groupby(["drug_0", "drug_1"], observed=True).size().index.tolist()
    holdout_cond_list = [
        c for c in adata_holdout.obs.groupby(["drug_0", "drug_1"], observed=True).size().index.tolist()
        if c != ("control", "control")
    ]
    holdout_train_nn_mean_dist = holdout_train_nn_mean(adata, train_cond_list, holdout_cond_list)
    if holdout_train_nn_mean_dist is not None:
        print(f"Holdout-train mean NN distance (embedding space): {holdout_train_nn_mean_dist:.4f}")

adl = AnnDataLocation()
data_manager = DataManager(
    dist_flag_key="control",
    src_dist_keys=["cell_line"],
    tgt_dist_keys=["drug_0"],
    rep_keys={
        "cell_line": "cell_line_embeddings",
        "drug_0": "drug_0_embeddings",
    },
    data_location=adl.obsm[args.obsm_key],
)

print("Preparing data...")
gd = data_manager.prepare_data(adata)

if adata_holdout is not None:
    print("Preparing held-out validation data...")
    gd_holdout = data_manager.prepare_data(adata_holdout)

print("Splitting datasets...")
if adata_holdout is not None:
    train_split = gd
else:
    data = split_datasets(
        {"tahoe": gd},
        split_by=["drug_0"],
        split_key="split",
        ratios=[0.4, 0.3, 0.3],
        random_state=args.seed,
        holdout_combinations=False,
    )
    train_split = data["tahoe"]["train"]
    val_split = data["tahoe"]["val"]

print("Creating samplers...")
sampler = InMemorySampler(train_split, np.random.default_rng(args.seed), batch_size=512)

val_sampler_name = "tahoe_val_holdout" if adata_holdout is not None else "tahoe_val"
val_gd = gd_holdout if adata_holdout is not None else val_split
val_samplers = {
    val_sampler_name: ValidationSampler(
        val_gd,
        n_conditions_on_log_iteration=None,
        n_conditions_on_train_end=None,
        seed=args.seed,
    ),
}

print("Initializing sampler...")
sampler.init_sampler()

print("Sampling batch...")
sample_batch = sampler.sample()
print(f"Condition shapes: {[(k, v.shape) for k, v in sample_batch['condition'].items()]}")

solver = args.solver
print(f"Creating model with {solver} solver...")
sf = ScaleFlow(solver=solver)

NUM_ITERATIONS = 500000
WARMUP_STEPS = 2000
PEAK_LR = 1e-3
END_LR = 1e-5

lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=PEAK_LR,
    warmup_steps=WARMUP_STEPS,
    decay_steps=NUM_ITERATIONS,
    end_value=END_LR,
)

print("Preparing model with smaller encoder dimensions...")
decoder_dims = (256, 256, 256)
hidden_dims = (256, 256, 256)
time_encoder_dims = (256, 256, 256)
sf.prepare_model(
    sample_batch=sample_batch,
    max_combination_length=2,
    conditioning="film",
    hidden_dims=hidden_dims,
    hidden_dropout=0.1,
    time_encoder_dims=time_encoder_dims,
    time_encoder_dropout=0.1,
    condition_embedding_dim=256,
    cond_output_dropout=0.1,
    decoder_dims=decoder_dims,
    optimizer=optax.MultiSteps(optax.adamw(learning_rate=1e-4), 20),
)

print("\n=== Model Parameter Count ===")
params = sf._solver.vf_state.params


def count_params(param_dict):
    return sum(x.size for x in jax.tree_util.tree_leaves(param_dict))


total_params = count_params(params)
print(f"Total parameters: {total_params:,}")

condition_encoder_params = count_params(params['condition_encoder']) if 'condition_encoder' in params else 0
time_encoder_params = count_params(params['time_encoder']) if 'time_encoder' in params else 0
x_encoder_params = count_params(params['x_encoder']) if 'x_encoder' in params else 0
decoder_params = count_params(params['decoder']) if 'decoder' in params else 0
output_layer_params = count_params(params['output_layer']) if 'output_layer' in params else 0

print(f"Condition encoder: {condition_encoder_params:,}")
print(f"Time encoder: {time_encoder_params:,}")
print(f"X encoder: {x_encoder_params:,}")
print(f"Decoder: {decoder_params:,}")
print(f"Output layer: {output_layer_params:,}")
print("=" * 30 + "\n")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

print("Training...")

monitor_metric = f"{val_sampler_name}_mmd_mean"
print(f"Early stopping enabled: monitoring '{monitor_metric}' with patience=10")

callbacks = [
    Metrics(["e_distance", "r_squared", "mmd"]),
    WandbLogger(
        project="perturbation-scaling",
        out_dir="./wandb_logs",
        config={
            "solver": solver,
            "conditioning": "concatenation",
            "num_iterations": NUM_ITERATIONS,
            "batch_size": 512,
            "hidden_dims": hidden_dims,
            "time_encoder_dims": time_encoder_dims,
            "condition_embedding_dim": 256,
            "decoder_dims": decoder_dims,
            "peak_lr": PEAK_LR,
            "end_lr": END_LR,
            "warmup_steps": WARMUP_STEPS,
            "dataset": "tahoe",
            "cell_fraction": args.cell_fraction,
            "condition_fraction": args.condition_fraction,
            "experiment_type": args.experiment_type,
            "seed": args.seed,
            "obsm_key": args.obsm_key,
            "random_embeddings": args.random_embeddings,
            "n_cells": adata.n_obs,
            "n_conditions": adata.obs.groupby(["drug_0", "drug_1"], observed=True).ngroups,
            "n_holdout_conditions": N_HOLDOUT if adata_holdout is not None else 0,
            "n_holdout_cells": adata_holdout.n_obs if adata_holdout is not None else 0,
            "holdout_train_nn_mean": holdout_train_nn_mean_dist,
            "total_params": total_params,
            "condition_encoder_params": condition_encoder_params,
            "time_encoder_params": time_encoder_params,
            "x_encoder_params": x_encoder_params,
            "decoder_params": decoder_params,
            "output_layer_params": output_layer_params,
            "weight_decay": 1e-3,
            "output_dir": args.output_dir,
        },
        entity="pancellflow",
    ),
    ModelCheckpoint(
        solver=sf._solver,
        monitor=monitor_metric,
        save_path=ckpt_path,
        mode="min",
        verbose=True,
    ),
    EarlyStopping(
        monitor=monitor_metric,
        patience=10,
        mode="min",
        min_delta=0.0,
        verbose=True,
    ),
]

sf.train(
    val_dataloader=val_samplers,
    train_dataloader=sampler,
    num_iterations=NUM_ITERATIONS,
    valid_freq=5000,
    callbacks=callbacks,
    monitor_metrics=["loss"],
)

# ---------------------------------------------------------------------------
# Load best checkpoint and run predictions
# ---------------------------------------------------------------------------

print(f"\nLoading best checkpoint from {ckpt_path}...")
with open(ckpt_path, "rb") as f:
    best_params = cloudpickle.load(f)

sf._solver.vf_state = sf._solver.vf_state.replace(params=best_params)
sf._solver.vf_state_inference = sf._solver.vf_state_inference.replace(params=best_params)
print("Best params restored into solver.")

print("Preparing full dataset for predictions...")
gd_full = data_manager.prepare_data(adata_full)
full_sampler = ValidationSampler(
    gd_full,
    n_conditions_on_log_iteration=None,
    n_conditions_on_train_end=None,
    seed=args.seed,
)
if not full_sampler.initialized:
    full_sampler.init_sampler()

train_drugs = set(
    adata.obs.groupby("drug_0", observed=True).groups.keys()
)

annotation = full_sampler._data.annotation
tgt_labels = annotation.tgt_dist_idx_to_labels
src_labels = annotation.src_dist_idx_to_labels
tgt_to_src = full_sampler._tgt_to_src

full_gd_data = full_sampler._data.data
all_tgt_indices = list(full_gd_data.conditions.keys())

src_dict: dict[str, np.ndarray] = {}
condition_dict: dict[str, dict] = {}
target_dict: dict[str, np.ndarray] = {}

for tgt_idx in all_tgt_indices:
    src_idx = tgt_to_src.get(tgt_idx)
    if src_idx is None:
        continue
    drug = tgt_labels[tgt_idx][0] if tgt_idx in tgt_labels else str(tgt_idx)
    cell_line = src_labels[src_idx][0] if src_idx in src_labels else str(src_idx)
    key = f"{drug}|{cell_line}"
    src_dict[key] = full_gd_data.src_data[src_idx]
    condition_dict[key] = full_gd_data.conditions[tgt_idx]
    target_dict[key] = full_gd_data.tgt_data[tgt_idx]

print(f"Predicting {len(src_dict)} (drug, cell_line) conditions...")

rows_X = []
rows_obs = []

for key in src_dict:
    drug, cell_line = key.split("|", 1)
    pred = np.array(sf._solver.predict(src_dict[key], condition_dict[key]))
    tgt = np.array(target_dict[key])
    src_arr = np.array(src_dict[key])

    n_pred, n_tgt, n_src = pred.shape[0], tgt.shape[0], src_arr.shape[0]
    in_train = drug in train_drugs

    rows_X.append(pred)
    rows_obs.append(pd.DataFrame({
        "drug_0": drug, "cell_line": cell_line,
        "split": "prediction", "in_train": in_train,
    }, index=range(n_pred)))

    rows_X.append(tgt)
    rows_obs.append(pd.DataFrame({
        "drug_0": drug, "cell_line": cell_line,
        "split": "target", "in_train": in_train,
    }, index=range(n_tgt)))

    rows_X.append(src_arr)
    rows_obs.append(pd.DataFrame({
        "drug_0": drug, "cell_line": cell_line,
        "split": "source", "in_train": in_train,
    }, index=range(n_src)))

obs = pd.concat(rows_obs, ignore_index=True)
X = np.concatenate(rows_X, axis=0)

adata_pred = ad.AnnData(X=X, obs=obs)
adata_pred.uns["obsm_key"] = args.obsm_key

print(f"Saving predictions to {predictions_path}...")
adata_pred.write_h5ad(predictions_path)
print(f"Done! AnnData with {adata_pred.n_obs} cells across {len(src_dict)} conditions saved to {predictions_path}")
