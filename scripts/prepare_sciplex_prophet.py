# %%
"""
process_sciplex_prophet.py

Single-dataset (sciPlex3) version of process_tahoe_prophet.py. Builds a
GroupedDistribution zarr from the prophet-filtered h5ad produced by
prepare_sciplex_prophet.ipynb, with

    extra_rep_keys={"prophet": ("drug", "prophet_emb")}

so each condition dict gets a "prophet" key looked up from
adata.uns["prophet_emb"][drug_name]. If "prophet_emb" is absent the DataManager
silently skips it.

⚠️  Verify against your sciPlex h5ad (the Inspect cell of the notebook prints these):
    - data_location  : obsm key holding the cell representation (Tahoe used "X_state")
    - rep_keys        : uns keys for the cell-line / drug embeddings
    - src/tgt_dist_keys: obs columns ("cell_line", "drug")
If sciPlex uses different names, change them HERE (not in the notebook).

Output
──────
/storage/pancellflow/sciplex3.zarr
"""

from scaleflow.data import DataManager, AnnDataLocation
from pathlib import Path
import anndata as ad
import h5py
import time
import numpy as np

start_time = time.time()
print("loading data")

OUTPUT_PATH = Path("/storage/pancellflow/sciplex3.zarr")
DATA_PATH   = Path("/storage/pancellflow/sciplex3_prophet_filtered.h5ad")

with h5py.File(DATA_PATH, "r") as f:
    adata = ad.AnnData(
        obs=ad.io.read_elem(f["obs"]),
        obsm=ad.io.read_elem(f["obsm"]),
        uns=ad.io.read_elem(f["uns"]),
    )

load_time = time.time() - start_time
print(f"data loaded (took {load_time:.2f} seconds)")

# ── Dose as a raw scalar condition ────────────────────────────────────────────
# Rename dose_value -> dose and cast to float. dose is NOT given a rep_key, so
# DataManager._col_to_repr returns np.array([dose]) (a 1-d scalar) — it only takes
# that path when the label is a float. Controls (excluded from target grouping)
# may be NaN; perturbed cells must have a real dose else they'd form a NaN group.
adata.obs = adata.obs.rename(columns={"dose_value": "dose"})
adata.obs["dose"] = adata.obs["dose"].astype("float32")
adata.obs["dose"] = np.log1p(adata.obs["dose"].astype("float32")) 
n_bad = int(((~adata.obs["control"]) & adata.obs["dose"].isna()).sum())
if n_bad:
    print(f"WARNING: {n_bad:,} perturbed cells have NaN dose (will form a bad group)")

# ── DataManager — dose-resolved: target = (cell_line, drug, dose) ──────────────
adl = AnnDataLocation()
dm  = DataManager(
    dist_flag_key="control",
    src_dist_keys=["cell_line"],
    tgt_dist_keys=["drug", "dose"],             # ← dose now defines the distribution
    rep_keys={
        "cell_line": "cell_line_ccle_embeddings",  
        "drug":      "drug_0_embeddings",      
    },
    data_location=adl.obsm["X_state"],          
    extra_rep_keys={"prophet": ("drug", "prophet_emb")},
)

prepare_start = time.time()
gd = dm.prepare_data(adata=adata)

prepare_time = time.time() - prepare_start
print(f"data prepared (took {prepare_time:.2f} seconds)")

chunk_size = 131072
shard_size = chunk_size * 8

write_start = time.time()
gd.write_zarr(
    path=OUTPUT_PATH,
    chunk_size=chunk_size,
    shard_size=shard_size,
)

write_time = time.time() - write_start
print(f"data written (took {write_time:.2f} seconds)")

total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.2f} seconds")
print(f"  - Loading:     {load_time:.2f} seconds ({load_time/total_time*100:.1f}%)")
print(f"  - Preparation: {prepare_time:.2f} seconds ({prepare_time/total_time*100:.1f}%)")
print(f"  - Writing:     {write_time:.2f} seconds ({write_time/total_time*100:.1f}%)")