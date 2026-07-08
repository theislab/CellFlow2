"""Extract the A549 (CVCL_0023) subset from the 228 GB tahoe_prophet_filtered.h5ad into a compact h5ad
for the transfer runs: keep obs + obsm['X_state'] + uns (cell_line_ccle_embeddings / drug_0_embeddings /
prophet_emb). Drops the 62k-gene X (training uses X_state; recon is on sciplex). Reads selectively via
h5py so we never materialize the full obsm/X."""
import time
import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp

SRC = "/lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/filtered_data/tahoe_prophet_filtered.h5ad"
OUT = "/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/unipert/tahoe_a549_prophet.h5ad"
A549 = "CVCL_0023"

t0 = time.time()
with h5py.File(SRC, "r") as f:
    print("reading obs …", flush=True)
    obs = ad.io.read_elem(f["obs"])
    idx = np.where(obs["cell_line"].astype(str).values == A549)[0]
    print(f"A549 rows: {len(idx):,} of {len(obs):,}", flush=True)
    obs = obs.iloc[idx].copy()
    print("reading obsm['X_state'] subset …", flush=True)
    xs = f["obsm/X_state"][np.sort(idx)]           # (n_a549, 2058)
    print(f"X_state subset: {xs.shape}", flush=True)
    uns = ad.io.read_elem(f["uns"])

A = ad.AnnData(X=sp.csr_matrix((len(idx), 1), dtype="float32"), obs=obs, uns=uns)
A.obsm["X_state"] = np.asarray(xs, dtype="float32")
print(f"writing {OUT} …", flush=True)
A.write_h5ad(OUT, compression="gzip")
print(f"done: {A.n_obs:,} cells | drugs={A.obs['drug'].nunique()} | "
      f"prophet_emb n={len(A.uns.get('prophet_emb', {}))} | {(time.time()-t0)/60:.1f} min", flush=True)
