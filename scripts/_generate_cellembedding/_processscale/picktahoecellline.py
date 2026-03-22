# %% [markdown]
# # Extract cell lines from Tahoe
# 
# Select cell lines at different OOD levels and save each as a standalone
# `.h5ad` with the original expression data (`X`) and the SE embedding in
# `obsm['X_state']`.

# %%
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd
import dask.array as da
from tqdm.auto import tqdm

INPUT_H5AD  = Path("/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe.h5ad")
CHUNKS_DIR  = Path("/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe/test_batched")
OUT_DIR     = Path("/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/unipert")
OBSM_KEY    = "X_state"

CELL_LINES = {
    "HOP62":  "CVCL_1285",
    "AsPC-1": "CVCL_0152",
    "PANC-1": "CVCL_0480",
    "SW48":   "CVCL_1724",
    "H4":     "CVCL_1239",
    "SNU-1":  "CVCL_0099",
}

OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
SPARSE_CHUNK_SIZE = 500_000
with h5py.File(INPUT_H5AD, "r") as f:
    adata = ad.AnnData(
        obs=ad.io.read_elem(f["obs"]),
        var=ad.io.read_elem(f["var"]),
    )
    adata.X = ad.experimental.read_elem_lazy(
        f["X"], chunks=(SPARSE_CHUNK_SIZE, adata.shape[1])
    )
print(f"Total cells: {adata.n_obs:,}  |  Genes: {adata.n_vars:,}")
print(f"Cell lines in data: {adata.obs['cell_line'].nunique()}")

# %%
DASK_CHUNK_ROWS = 500_000

chunk_files = sorted(CHUNKS_DIR.glob("batch_*.h5ad"))
print(f"Found {len(chunk_files)} SE chunk files")

emb_handles = []
emb_arrays = []
for fp in chunk_files:
    fh = h5py.File(fp, "r")
    emb_handles.append(fh)
    ds = fh[f"obsm/{OBSM_KEY}"]
    emb_arrays.append(da.from_array(ds, chunks=(DASK_CHUNK_ROWS, ds.shape[1])))

emb_lazy = da.concatenate(emb_arrays, axis=0)
print(f"Lazy SE shape: {emb_lazy.shape}, dtype: {emb_lazy.dtype}")
assert emb_lazy.shape[0] == adata.n_obs, (
    f"Row mismatch: SE has {emb_lazy.shape[0]:,} but adata has {adata.n_obs:,}"
)

# %%
# (SE loaded lazily via Dask in the cell above)

# %%
for name, cvcl in CELL_LINES.items():
    mask = adata.obs["cell_line"].astype(str).eq(cvcl)
    n = int(mask.sum())
    print(f"\n{'='*50}")
    print(f"{name} ({cvcl}): {n:,} cells")
    if n == 0:
        print(f"  WARNING: no cells found, skipping")
        continue

    out_path = OUT_DIR / f"tahoe_{name.lower().replace('-', '_').replace(' ', '_')}.h5ad"
    if out_path.exists():
        print(f"  {out_path.name} already exists, skipping")
        continue

    mask_np = mask.values

    print(f"  Reading X ({n:,} × {adata.n_vars:,}) ...")
    sub_adata = adata[mask_np].copy()

    print(f"  Reading {OBSM_KEY} ({n:,} × {emb_lazy.shape[1]}) ...")
    sub_adata.obsm[OBSM_KEY] = emb_lazy[mask_np].compute()

    print(f"  Writing {out_path.name} ...")
    sub_adata.write_h5ad(out_path, compression="gzip")
    print(f"  Done: {sub_adata.n_obs:,} cells, {sub_adata.n_vars:,} genes, "
          f"obsm['{OBSM_KEY}'] = {sub_adata.obsm[OBSM_KEY].shape}")
    del sub_adata

# %%
for fh in emb_handles:
    fh.close()
del adata, emb_lazy, emb_arrays, emb_handles

print("\n=== Verification ===")
for name, cvcl in CELL_LINES.items():
    stem = name.lower().replace('-', '_').replace(' ', '_')
    fp = OUT_DIR / f"tahoe_{stem}.h5ad"
    if not fp.exists():
        print(f"{name}: MISSING")
        continue
    a = ad.read_h5ad(fp, backed="r")
    has_se = OBSM_KEY in a.obsm
    se_shape = a.obsm[OBSM_KEY].shape if has_se else None
    print(f"{name:8s}  cells={a.n_obs:>10,}  genes={a.n_vars:>6,}  "
          f"{OBSM_KEY}={'OK '+str(se_shape) if has_se else 'MISSING'}")
    del a

# %%



