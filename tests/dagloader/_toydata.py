"""Shared toy fixtures for the dagloader tests (no jax/optax — importable by every test file here).

Two cell lines × three drugs, one contiguous block per (cell_line, drug) combination, so the on-disk
layout is already sorted by condition (each leaf is one run) — which is what ``chunk_size > 1`` reads
efficiently. Embeddings for both covariates are provided so a condition can be a *combination* of a
cell-line embedding and a drug embedding (the sc-flow-tools multi-covariate case).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

G, DRUG_D, CELL_D, PCA_D = 8, 4, 2, 5
DRUG_SHIFT = {"control": np.zeros(G), "d1": np.eye(G)[0] * 4.0, "d2": np.eye(G)[1] * 4.0}
NOISE_STD = 0.4
DRUG_EMB = {"control": np.zeros(DRUG_D), "d1": np.eye(DRUG_D)[0], "d2": np.eye(DRUG_D)[1]}
CELL_EMB = {"A": np.eye(CELL_D)[0], "B": np.eye(CELL_D)[1]}
CELL_LINES = ["A", "B"]
DRUGS = ["control", "d1", "d2"]
N_PER_BLOCK = 60


def toy_adata(seed: int = 0) -> ad.AnnData:
    """Cell-line means ±1; a single-coordinate drug spike. Blocks are contiguous ⇒ condition-sorted."""
    rng = np.random.default_rng(seed)
    rows, blocks = [], []
    for cl in CELL_LINES:
        cl_mean = np.full(G, 1.0 if cl == "A" else -1.0)
        for drug in DRUGS:
            blocks.append(rng.normal(cl_mean + DRUG_SHIFT[drug], NOISE_STD, size=(N_PER_BLOCK, G)).astype(np.float32))
            rows += [{"cell_line": cl, "drug": drug}] * N_PER_BLOCK
    obs = pd.DataFrame(rows)
    for c in ("cell_line", "drug"):
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=np.vstack(blocks), obs=obs)
    adata.obsm["pca"] = adata.X[:, :PCA_D].copy()  # a lower-dim rep carrying the same conditional signal
    adata.uns["drug_emb"] = DRUG_EMB
    return adata


def shuffled_adata(seed: int = 1) -> ad.AnnData:
    """Same cells, obs order permuted → leaves are no longer contiguous runs (unsorted layout)."""
    adata = toy_adata()
    perm = np.random.default_rng(seed).permutation(adata.n_obs)
    out = adata[perm].copy()
    out.obs = out.obs.reset_index(drop=True)
    return out


def write_collection(adata: ad.AnnData, tmp_path):
    """Persist ``adata`` as an on-disk annbatch DatasetCollection, preserving row order."""
    ad.settings.zarr_write_format = 3  # annbatch sharding needs zarr v3
    from annbatch import DatasetCollection

    ap, cp = tmp_path / "a.zarr", tmp_path / "coll.zarr"
    adata.write_zarr(str(ap))
    DatasetCollection(str(cp), mode="a").add_adatas(adata_paths=[str(ap)], shuffle=False)  # preserve row order
    return DatasetCollection(str(cp), mode="r")


def drug_cond_fn(cols):
    """Condition = drug embedding (single-covariate condition)."""
    return lambda leaf: DRUG_EMB[leaf[cols.index("drug")]]


def cell_drug_cond_fn(cols):
    """Condition = [cell-line embedding ‖ drug embedding] (multi-covariate combination)."""
    return lambda leaf: np.concatenate([CELL_EMB[leaf[cols.index("cell_line")]], DRUG_EMB[leaf[cols.index("drug")]]])
