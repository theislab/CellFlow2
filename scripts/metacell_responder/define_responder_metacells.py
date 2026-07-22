"""
define_responder_metacells.py
=============================

Label each metacell in a metacell GroupedDistribution zarr as a RESPONDER or
not, using the "option-3" control-reference k-NN test from
build_filtered_metacell_zarr_option3.py (here made standalone + parameterised).

Method
------
Per cell line, using its RAW control single cells (the zarr's source data):
  1. split controls into a REFERENCE set (`--reference` cells) and a held-out POOL,
  2. build a NULL distribution of "no-effect" k-NN distances from control
     PSEUDO-METACELLS = bootstrap means of `--n-eff` control cells from the pool,
  3. threshold = the `--percentile`-th percentile of that null distribution.
Then every metacell x is a RESPONDER iff its distance to its `--k`-th nearest
control-reference neighbour exceeds the threshold (i.e. it sits farther from the
control population than a typical no-effect metacell of the same aggregation size).

`--n-eff` MUST match the metacell aggregation level (cells averaged per metacell):
sciplex metacells here aggregate ~25 cells -> `--n-eff 25` (tahoe used 76). A
mismatch miscalibrates the null and hence the responder threshold.

Outputs
-------
  <out_prefix>.csv             per-condition: labels + n_metacells, n_responders, fraction, threshold
  <out_prefix>_permetacell.csv per-metacell:  labels + knn_dist, threshold, responder
  <out_prefix>_responders.zarr (optional, --write-zarr) metacell zarr keeping ONLY responders

Assumes source distributions are keyed by cell_line (src label[0]) and that the
first element of each target label is the cell_line. Works for 2-tuple
(cell_line, drug) or 3-tuple (cell_line, drug, dose) labels.

    conda run -n cfp python scripts/define_responder_metacells.py \
        --input  outputs/sciplex_metacell_dose_concept.zarr \
        --out-prefix outputs/responder_labels_sciplex_dose_concept \
        --n-eff 25
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from scaleflow.data import GroupedDistribution


def fit_control_models(src_data: dict, src_dist_idx_to_labels: dict, *,
                       n_eff: int, k: int, n_reference: int, n_null: int,
                       percentile: float, seed: int) -> dict:
    """Per-cell-line (k-NN reference index, null-distance threshold).
    Identical logic to build_filtered_metacell_zarr_option3.fit_control_models,
    with the knobs passed in. A SINGLE rng is shared across cell lines (its state
    advances between them) — matching the original."""
    models = {}
    rng = np.random.default_rng(seed)
    for src_idx, ctrl_X in src_data.items():
        cell_line = src_dist_idx_to_labels[src_idx][0]
        X = np.asarray(ctrl_X, dtype=np.float32)
        n = X.shape[0]

        n_ref = min(n_reference, n // 2)
        perm = rng.permutation(n)
        X_ref, X_pool = X[perm[:n_ref]], X[perm[n_ref:]]

        boot_idx = rng.integers(0, X_pool.shape[0], size=(n_null, n_eff))
        X_null = X_pool[boot_idx].mean(axis=1)                 # control pseudo-metacells

        nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X_ref)
        null_knn_dist = nn.kneighbors(X_null)[0][:, -1]
        threshold = float(np.quantile(null_knn_dist, percentile / 100.0))

        models[cell_line] = (nn, threshold)
        print(f"    [{cell_line}] n_ctrl={n:>6d} n_ref={n_ref} n_null={n_null} "
              f"(bootstrap means of {n_eff}) k={k} thr(p{percentile:g})={threshold:.3f}", flush=True)
    return models


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True, help="metacell GroupedDistribution zarr")
    p.add_argument("--out-prefix", required=True, help="output path prefix (no extension)")
    p.add_argument("--n-eff",      type=int,   default=25,   help="cells aggregated per metacell (match construction!)")
    p.add_argument("--k",          type=int,   default=15,   help="k-th nearest reference neighbour")
    p.add_argument("--reference",  type=int,   default=5000, help="control cells in the k-NN reference set")
    p.add_argument("--null",       type=int,   default=2000, help="control pseudo-metacells for the null")
    p.add_argument("--percentile", type=float, default=95,   help="null percentile -> responder threshold")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--write-zarr", action="store_true", help="also write a responders-only metacell zarr")
    a = p.parse_args()

    print(f"loading {a.input} …", flush=True)
    gd = GroupedDistribution.read_zarr(a.input)
    gd.to_memory()
    ann, dat = gd.annotation, gd.data
    print(f"  {len(dat.src_data)} control distributions, {len(dat.tgt_data)} metacell conditions", flush=True)

    print(f"\nfitting control models (n_eff={a.n_eff}, k={a.k}, ref={a.reference}, "
          f"null={a.null}, p{a.percentile:g}) …", flush=True)
    models = fit_control_models(
        dat.src_data, ann.src_dist_idx_to_labels,
        n_eff=a.n_eff, k=a.k, n_reference=a.reference, n_null=a.null,
        percentile=a.percentile, seed=a.seed,
    )

    # label name(s) beyond cell_line, inferred from label tuple length
    sample_label = next(iter(ann.tgt_dist_idx_to_labels.values()))
    extra_cols = ["drug", "dose"][: max(0, len(sample_label) - 1)]

    cond_rows, permeta_rows = [], []
    for tgt_idx in list(dat.tgt_data.keys()):
        labels = ann.tgt_dist_idx_to_labels[tgt_idx]
        cell_line = labels[0]
        nn, thr = models[cell_line]
        X = np.asarray(dat.tgt_data[tgt_idx], dtype=np.float32)
        knn = nn.kneighbors(X)[0][:, -1]
        resp = knn > thr

        base = {"cell_line": cell_line}
        for col, val in zip(extra_cols, labels[1:]):
            base[col] = float(val) if col == "dose" else val
        cond_rows.append({**base, "n_metacells": int(X.shape[0]),
                          "n_responders": int(resp.sum()),
                          "responder_fraction": float(resp.mean()), "threshold": float(thr)})
        for j in range(X.shape[0]):
            permeta_rows.append({**base, "metacell_in_cond": j, "knn_dist": float(knn[j]),
                                 "threshold": float(thr), "responder": bool(resp[j])})

        if a.write_zarr:
            dat.tgt_data[tgt_idx] = X[resp].astype(np.float32) if resp.any() else X[:0]

    cond = pd.DataFrame(cond_rows)
    pm = pd.DataFrame(permeta_rows)
    cond.to_csv(f"{a.out_prefix}.csv", index=False)
    pm.to_csv(f"{a.out_prefix}_permetacell.csv", index=False)
    print(f"\nwrote {a.out_prefix}.csv ({len(cond)} conditions) and "
          f"{a.out_prefix}_permetacell.csv ({len(pm)} metacells)", flush=True)
    print(f"total metacells={len(pm):,}  responders={int(pm.responder.sum()):,} "
          f"({pm.responder.mean():.1%})", flush=True)

    if a.write_zarr:
        # drop conditions left with zero responders, then write
        empty = [t for t in list(dat.tgt_data.keys()) if dat.tgt_data[t].shape[0] == 0]
        for t in empty:
            del dat.tgt_data[t]; del ann.tgt_dist_idx_to_labels[t]
            if t in dat.conditions: del dat.conditions[t]
        out = f"{a.out_prefix}_responders.zarr"
        gd.write_zarr(out, chunk_size=131072, shard_size=131072 * 8)
        print(f"wrote responders-only zarr -> {out} ({len(empty)} conditions had 0 responders)", flush=True)


if __name__ == "__main__":
    main()
