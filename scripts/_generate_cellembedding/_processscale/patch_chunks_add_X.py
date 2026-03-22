#!/usr/bin/env python
"""
One-off script: rewrite chunk h5ads with a placeholder sparse X
so that ad.experimental.concat_on_disk can process them.

Reads obsm via h5py, then writes a proper AnnData via anndata
to guarantee correct on-disk encoding.
"""
import argparse
import glob

import h5py
import numpy as np
import scipy.sparse as sp
import anndata as ad


def rewrite_with_placeholder_X(path: str, obsm_key: str) -> None:
    with h5py.File(path, "r") as f:
        if "obsm" not in f or obsm_key not in f["obsm"]:
            raise ValueError(f"{path}: missing obsm/{obsm_key}")
        emb = f["obsm"][obsm_key][:]

    n_obs = emb.shape[0]
    adata = ad.AnnData(
        X=sp.csr_matrix((n_obs, 1)),
        obsm={obsm_key: emb},
    )
    adata.var_names = ["_placeholder"]
    adata.write_h5ad(path, compression="gzip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indirec", required=True)
    ap.add_argument("--obsm-key", default="X_state")
    args = ap.parse_args()

    files = sorted(glob.glob(args.indirec + "/*.h5ad"))
    print(f"Found {len(files)} chunk files")
    for i, fp in enumerate(files):
        rewrite_with_placeholder_X(fp, args.obsm_key)
        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            print(f"  patched {i + 1}/{len(files)}")
    print("Done.")


if __name__ == "__main__":
    main()
