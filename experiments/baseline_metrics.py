"""Baseline metrics: control predictions for all perturbed conditions.

For every non-control condition, compares the control cells against the
perturbed cells using MMD, E-distance, and R-squared.
"""
import argparse
import json

import numpy as np
import scanpy as sc
from tqdm import tqdm

from scaleflow.metrics import compute_e_distance_fast, compute_r_squared, compute_scalar_mmd

DATA_PATH = "/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/unipert/tahoe_a549_w_se.h5ad"

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="baseline_metrics.json")
args = parser.parse_args()

print("Loading dataset...")
adata = sc.read_h5ad(DATA_PATH)
adata.obs["control"] = (
    (adata.obs["drug_0"] == "control") &
    (adata.obs["drug_1"] == "control")
)

condition_list = adata.obs.groupby(["drug_0", "drug_1"], observed=True).size().index.tolist()
control = ("control", "control")
conditions = sorted(c for c in condition_list if c != control)

control_mask = adata.obs["control"].to_numpy()
X_state = adata.obsm["X_state"].copy()
drug_0_col = adata.obs["drug_0"].to_numpy()
drug_1_col = adata.obs["drug_1"].to_numpy()
del adata

X_control = X_state[control_mask]
print(f"Control cells: {len(X_control)}")
print(f"Conditions to evaluate: {len(conditions)}")

per_condition = {}

for drug_0, drug_1 in tqdm(conditions):
    mask = (drug_0_col == drug_0) & (drug_1_col == drug_1)
    cells = X_state[mask]
    n = len(cells)
    if n < 2:
        print(f"Skipping {drug_0}+{drug_1}: only {n} cells")
        continue

    ctrl_mmd = float(compute_scalar_mmd(X_control, cells))
    ctrl_edist = float(compute_e_distance_fast(X_control, cells))
    ctrl_r2 = float(compute_r_squared(X_control, cells))

    per_condition[f"{drug_0}+{drug_1}"] = {
        "n_cells": n,
        "mmd": ctrl_mmd,
        "e_distance": ctrl_edist,
        "r_squared": ctrl_r2,
    }

condition_names = list(per_condition.keys())
mmds = [v["mmd"] for v in per_condition.values()]
edists = [v["e_distance"] for v in per_condition.values()]
r2s = [v["r_squared"] for v in per_condition.values()]

summary = {
    "n_conditions_evaluated": len(per_condition),
    "conditions": condition_names,
    "control": {
        "mmd_mean": float(np.mean(mmds)),
        "mmd_std": float(np.std(mmds)),
        "mmd_values": mmds,
        "e_distance_mean": float(np.mean(edists)),
        "e_distance_std": float(np.std(edists)),
        "e_distance_values": edists,
        "r_squared_mean": float(np.mean(r2s)),
        "r_squared_std": float(np.std(r2s)),
        "r_squared_values": r2s,
    },
    "per_condition": per_condition,
}

print("\n=== Control baseline ===")
print(f"MMD:        {summary['control']['mmd_mean']:.4f} +/- {summary['control']['mmd_std']:.4f}")
print(f"E-distance: {summary['control']['e_distance_mean']:.4f} +/- {summary['control']['e_distance_std']:.4f}")
print(f"R2:         {summary['control']['r_squared_mean']:.4f} +/- {summary['control']['r_squared_std']:.4f}")

with open(args.output, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {args.output}")
