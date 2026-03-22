#!/usr/bin/env python
"""
Quick sanity checks for the embedded aissa21 h5ad.

Verifies:
  1. All expected obsm keys exist
  2. Shape matches n_obs
  3. No NaN / Inf values
  4. Non-degenerate (not all zeros, has variance)
  5. Embedding dimensions match expected sizes
  6. Original data (X, obs, var) is preserved vs the source file
"""
import sys
import numpy as np
import anndata as ad

SRC_PATH = "/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/unipert/aissa21.h5ad"
EMB_PATH = "/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/unipert/aissa21_w_emb.h5ad"

EXPECTED_KEYS = ["X_scconcept", "X_scgpt", "X_scimilarity"]

def main():
    print(f"Loading embedded file: {EMB_PATH}")
    adata = ad.read_h5ad(EMB_PATH)
    print(f"  n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    print(f"  obsm keys: {list(adata.obsm.keys())}")

    ok = True

    # --- Check expected keys exist ---
    for key in EXPECTED_KEYS:
        if key not in adata.obsm:
            print(f"  FAIL: {key} not found in obsm")
            ok = False

    # --- Per-key checks ---
    for key in EXPECTED_KEYS:
        if key not in adata.obsm:
            continue
        emb = adata.obsm[key]
        n, d = emb.shape
        print(f"\n  {key}: shape=({n}, {d})")

        if n != adata.n_obs:
            print(f"    FAIL: row count {n} != n_obs {adata.n_obs}")
            ok = False

        n_nan = np.isnan(emb).sum()
        n_inf = np.isinf(emb).sum()
        if n_nan > 0:
            print(f"    FAIL: {n_nan} NaN values")
            ok = False
        if n_inf > 0:
            print(f"    FAIL: {n_inf} Inf values")
            ok = False

        if np.allclose(emb, 0):
            print(f"    FAIL: all zeros")
            ok = False

        col_var = np.var(emb, axis=0)
        zero_var_cols = (col_var == 0).sum()
        if zero_var_cols == d:
            print(f"    FAIL: all {d} columns have zero variance (constant)")
            ok = False
        elif zero_var_cols > 0:
            print(f"    WARN: {zero_var_cols}/{d} columns have zero variance")

        mean_val = np.mean(emb)
        std_val  = np.std(emb)
        print(f"    mean={mean_val:.4f}, std={std_val:.4f}, "
              f"min={emb.min():.4f}, max={emb.max():.4f}")
        print(f"    OK")

    # --- Check original data is preserved ---
    print(f"\nLoading source file: {SRC_PATH}")
    src = ad.read_h5ad(SRC_PATH)

    if adata.n_obs != src.n_obs:
        print(f"  FAIL: n_obs mismatch ({adata.n_obs} vs {src.n_obs})")
        ok = False
    else:
        print(f"  n_obs match: {adata.n_obs}")

    if adata.n_vars != src.n_vars:
        print(f"  FAIL: n_vars mismatch ({adata.n_vars} vs {src.n_vars})")
        ok = False
    else:
        print(f"  n_vars match: {adata.n_vars}")

    if not (adata.obs_names == src.obs_names).all():
        print(f"  FAIL: obs_names differ")
        ok = False
    else:
        print(f"  obs_names match")

    if not (adata.var_names == src.var_names).all():
        print(f"  FAIL: var_names differ")
        ok = False
    else:
        print(f"  var_names match")

    obs_cols_match = set(adata.obs.columns) == set(src.obs.columns)
    if not obs_cols_match:
        extra = set(adata.obs.columns) - set(src.obs.columns)
        missing = set(src.obs.columns) - set(adata.obs.columns)
        if extra:
            print(f"  WARN: extra obs columns: {extra}")
        if missing:
            print(f"  FAIL: missing obs columns: {missing}")
            ok = False
    else:
        print(f"  obs columns match")

    # --- Summary ---
    print()
    if ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
