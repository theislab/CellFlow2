"""
train_comparison.py

Trains and compares two CellFlow2 model variants on Tahoe:

  model_default  –  ConditionEncoder uses drug, cell_line, dosage
  model_1        –  ConditionEncoder uses drug, cell_line, dosage + prophet_emb

Both models share an identical train / val / test split (random_state=42,
split_by=["drug"]), so unseen test drugs are the same for both.

During training:
  - val metrics (R², E-distance, MMD) are logged every VALID_FREQ steps
  - best checkpoint (by val R²) is saved per model

Post-training:
  - both best checkpoints are evaluated on the held-out test split
  - metrics: R², Energy Distance, MMD, Sinkhorn Divergence (per condition + mean)
  - results printed as a comparison table and saved to OUTPUT_DIR

Prerequisites
─────────────
  /data/tahoe.zarr          base GroupedDistribution  (no prophet)
  /data/tahoe_prophet.zarr  GroupedDistribution with prophet_emb

  Create the prophet zarr by running scripts/process_tahoe_prophet.py,
  which is process_tahoe.py extended with:
      dm = DataManager(
          ...
          extra_rep_keys={"prophet": ("drug", "prophet_emb")},
      )
  The prophet embeddings must be stored in adata.uns["prophet_emb"] as
  {drug_name: np.ndarray} before calling prepare_data.
"""

import os
import time
from pathlib import Path

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
    compute_sinkhorn_div,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
BASE_ZARR        = Path("/data/tahoe.zarr")
PROPHET_ZARR     = Path("/data/tahoe_prophet.zarr")
OUTPUT_DIR       = Path("./outputs")

SEED             = 42
SPLIT_RATIOS     = [0.7, 0.2, 0.1]   # train / val / test

BATCH_SIZE       = 1024
POOL_FRACTION    = 0.7
REPLACEMENT_PROB = 0.5

NUM_ITERATIONS   = 50_000
VALID_FREQ       = 500                # validate every N steps
N_VAL_CONDITIONS = 100                # conditions sampled per val step

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# BestModelCheckpoint callback
# (not yet in scaleflow.training – defined inline here)
# ─────────────────────────────────────────────────────────────────────────────
class BestModelCheckpoint(ComputationCallback):
    """Save the solver state whenever mean val R² improves.

    Inherits from ComputationCallback so it plugs into the existing
    CallbackRunner / CellFlowTrainer without any changes to the trainer.
    """

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.best_r2   = -np.inf

    def on_train_begin(self, *args, **kwargs) -> None:
        self.best_r2 = -np.inf

    def on_log_iteration(
        self,
        valid_source_data,   # {dataset_key: {cond_key: array}}
        valid_true_data,     # {dataset_key: {cond_key: array}}
        valid_pred_data,     # {dataset_key: {cond_key: array}}
        solver,
        **kwargs,
    ) -> dict:
        scores = []
        for dataset_key in valid_true_data:
            for cond_key, true_arr in valid_true_data[dataset_key].items():
                pred_arr = valid_pred_data[dataset_key].get(cond_key)
                if pred_arr is None:
                    continue
                scores.append(
                    compute_r_squared(np.array(true_arr), np.array(pred_arr))
                )

        if not scores:
            return {}

        mean_r2 = float(np.mean(scores))
        if mean_r2 > self.best_r2:
            self.best_r2 = mean_r2
            with open(self.save_path, "wb") as f:
                cloudpickle.dump(solver, f)
            print(f"    ✓ checkpoint saved  (val R²={mean_r2:.4f})")

        return {"best_val_r2": self.best_r2}

    def on_train_end(
        self,
        valid_source_data,
        valid_true_data,
        valid_pred_data,
        solver,
        **kwargs,
    ) -> dict:
        return self.on_log_iteration(
            valid_source_data, valid_true_data, valid_pred_data, solver
        )


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_split(zarr_path: Path):
    """Load zarr and return (train_gd, val_gd, test_gd)."""
    gd = GroupedDistribution.read_zarr(zarr_path)
    splits = split_datasets(
        {"gd": gd},
        split_by=["drug"],
        split_key="split",
        ratios=SPLIT_RATIOS,
        random_state=SEED,          # same seed → same drug assignment for both models
        holdout_combinations=False,
    )
    return splits["gd"]["train"], splits["gd"]["val"], splits["gd"]["test"]


def make_train_sampler(train_gd: GroupedDistribution) -> CombinedSampler:
    rng = np.random.default_rng(SEED)
    return CombinedSampler(
        samplers={
            "gd": ReservoirSampler(
                train_gd, rng,
                batch_size=BATCH_SIZE,
                pool_fraction=POOL_FRACTION,
                replacement_prob=REPLACEMENT_PROB,
            )
        },
        rng=rng,
    )


def make_val_sampler(val_gd: GroupedDistribution) -> ValidationSampler:
    """Finite sampler used during training validation steps."""
    return ValidationSampler(
        val_gd,
        n_conditions_on_log_iteration=N_VAL_CONDITIONS,
        n_conditions_on_train_end=N_VAL_CONDITIONS,
        seed=SEED,
    )


def make_test_sampler(test_gd: GroupedDistribution) -> ValidationSampler:
    """Returns all test conditions (used for post-training eval only)."""
    return ValidationSampler(
        test_gd,
        n_conditions_on_log_iteration=None,  # all conditions
        n_conditions_on_train_end=None,
        seed=SEED,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_test(solver, test_sampler: ValidationSampler) -> dict:
    """
    Predict on all held-out test conditions and compute all four metrics.

    Returns
    -------
    dict with keys:
      "per_condition" – {cond_key: {metric: value}}
      "aggregated"    – {metric: mean_over_conditions}
    """
    batch = test_sampler.sample(mode="on_train_end")
    src   = batch["source"]    # {cond_key: (n_cells, pca_dim)}
    cond  = batch["condition"] # {cond_key: {covariate: embedding}}
    true  = batch["target"]    # {cond_key: (n_cells, pca_dim)}

    print(f"  predicting {len(src)} test conditions …")
    pred = jax.tree.map(solver.predict, src, cond)

    per_condition = {}
    rng_key = jax.random.PRNGKey(0)

    for cond_key in tqdm(sorted(true.keys()), desc="  test metrics"):
        y_true = np.array(true[cond_key])
        y_pred = np.array(pred[cond_key])
        rng_key, sub_key = jax.random.split(rng_key)

        per_condition[cond_key] = {
            "r_squared":    float(compute_r_squared(y_true, y_pred)),
            "e_distance":   float(compute_e_distance_fast(y_true, y_pred)),
            "mmd":          float(compute_scalar_mmd(y_true, y_pred)),
            "sinkhorn_div": float(compute_sinkhorn_div(y_true, y_pred, epsilon=1.0)),
        }

    metrics = ["r_squared", "e_distance", "mmd", "sinkhorn_div"]
    aggregated = {
        m: float(np.mean([v[m] for v in per_condition.values()]))
        for m in metrics
    }

    return {"per_condition": per_condition, "aggregated": aggregated}


# ─────────────────────────────────────────────────────────────────────────────
# Train one model
# ─────────────────────────────────────────────────────────────────────────────
def train_model(name: str, zarr_path: Path) -> dict:
    """
    Full pipeline for one model variant.

    Parameters
    ----------
    name
        Human-readable name used for checkpoint filenames and logging.
    zarr_path
        Path to the GroupedDistribution zarr for this model variant.
        model_default → BASE_ZARR    (no prophet key in conditions)
        model_1       → PROPHET_ZARR (prophet key present in conditions)

    Returns
    -------
    dict with keys "solver" and "test_metrics".
    """
    print(f"\n{'='*64}")
    print(f"  {name}  |  zarr: {zarr_path}")
    print(f"{'='*64}")
    ckpt_path = str(OUTPUT_DIR / f"{name}_best.pkl")

    # ── Load & split ──────────────────────────────────────────────
    t0 = time.perf_counter()
    print("Loading & splitting data …")
    train_gd, val_gd, test_gd = make_split(zarr_path)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    # ── Samplers ──────────────────────────────────────────────────
    print("Building samplers …")
    train_sampler = make_train_sampler(train_gd)
    val_sampler   = make_val_sampler(val_gd)
    test_sampler  = make_test_sampler(test_gd)

    train_sampler.init_sampler()
    val_sampler.init_sampler()
    test_sampler.init_sampler()

    # ── Sample one batch → infer condition shape for model init ───
    sample_batch = train_sampler.sample()
    cond_keys = list(sample_batch["condition"].keys())
    data_dim  = sample_batch["src_cell_data"].shape[-1]
    print(f"  data_dim={data_dim}  condition_keys={cond_keys}")

    # ── Build model ───────────────────────────────────────────────
    # ConditionEncoder adapts automatically to whatever keys are in
    # sample_batch["condition"]:
    #   model_default → {drug, cell_line, dosage}
    #   model_1       → {drug, cell_line, dosage, prophet}
    # No architectural code change needed.
    print("Building model …")
    sf = ScaleFlow()
    sf.prepare_model(
        sample_batch=sample_batch,
        max_combination_length=1,
    )

    # ── Callbacks ─────────────────────────────────────────────────
    callbacks = [
        Metrics(
            metrics=["r_squared", "e_distance", "mmd"],
            metric_aggregations=["mean"],
            use_gpu_optimized=True,
            precision="bfloat16",
        ),
        BestModelCheckpoint(save_path=ckpt_path),
    ]

    # ── Train ─────────────────────────────────────────────────────
    print(
        f"Training {NUM_ITERATIONS} iterations "
        f"(val every {VALID_FREQ} steps, {N_VAL_CONDITIONS} conditions) …"
    )
    t0 = time.perf_counter()
    sf.train(
        train_dataloader=train_sampler,
        val_dataloader={"val": val_sampler},
        num_iterations=NUM_ITERATIONS,
        valid_freq=VALID_FREQ,
        callbacks=callbacks,
    )
    elapsed = (time.perf_counter() - t0) / 60
    print(f"  training done in {elapsed:.1f} min")

    # ── Load best checkpoint ──────────────────────────────────────
    if os.path.exists(ckpt_path):
        print(f"Loading best checkpoint from {ckpt_path} …")
        with open(ckpt_path, "rb") as f:
            best_solver = cloudpickle.load(f)
    else:
        print("  no checkpoint found – using final iterate")
        best_solver = sf.solver

    # ── Test evaluation ───────────────────────────────────────────
    print("Evaluating on test set …")
    test_metrics = evaluate_test(best_solver, test_sampler)

    # Save full results for this model
    result_path = OUTPUT_DIR / f"{name}_results.pkl"
    with open(result_path, "wb") as f:
        cloudpickle.dump(test_metrics, f)
    print(f"  test results saved to {result_path}")

    return {"solver": best_solver, "test_metrics": test_metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
def print_comparison(all_results: dict[str, dict]) -> None:
    """Print a side-by-side aggregated metric comparison."""
    metrics = ["r_squared", "e_distance", "mmd", "sinkhorn_div"]
    col_w = 20

    print("\n" + "=" * (20 + col_w * len(all_results)))
    print("  TEST SET  (mean over all unseen conditions)")
    print("=" * (20 + col_w * len(all_results)))

    header = f"{'metric':<20}" + "".join(f"{n:>{col_w}}" for n in all_results)
    print(header)
    print("-" * (20 + col_w * len(all_results)))

    for m in metrics:
        row = f"{m:<20}"
        for model_res in all_results.values():
            val = model_res["test_metrics"]["aggregated"][m]
            row += f"{val:>{col_w}.4f}"
        print(row)

    print("=" * (20 + col_w * len(all_results)))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_results = {}

    # model_default – base zarr, no prophet in condition dict
    all_results["model_default"] = train_model("model_default", BASE_ZARR)

    # model_1 – prophet zarr, prophet key present in condition dict
    all_results["model_1"] = train_model("model_1", PROPHET_ZARR)

    # Save combined results
    combined_path = OUTPUT_DIR / "comparison_results.pkl"
    with open(combined_path, "wb") as f:
        cloudpickle.dump({k: v["test_metrics"] for k, v in all_results.items()}, f)
    print(f"\nCombined results saved to {combined_path}")

    print_comparison(all_results)
