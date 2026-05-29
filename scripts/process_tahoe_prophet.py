# %%
"""
process_tahoe_prophet.py

Identical to process_tahoe.py except that the DataManager is configured with
    extra_rep_keys={"prophet": ("drug", "prophet_emb")}

This causes each condition dict to contain a "prophet" key whose embedding is
looked up from adata.uns["prophet_emb"][drug_name].  The resulting zarr can be
used directly by train_comparison.py (model_1).

Prerequisites
─────────────
adata.uns["prophet_emb"] must exist in the source h5ad and must be a dict of
the form  {drug_name: np.ndarray}  with one vector per drug.

If the key is absent the DataManager will silently skip it (the silent-skip
logic lives in DataManager._prepare_data), so the same DataManager definition
is safe to run even if prophet embeddings have not yet been added.

Output
──────
/data/tahoe_prophet.zarr   –  GroupedDistribution with prophet key in conditions
"""

from scaleflow.data import DataManager, AnnDataLocation
from pathlib import Path
import anndata as ad
import h5py
import time

start_time = time.time()
print("loading data")

OUTPUT_PATH = Path("/data/tahoe_prophet.zarr")
# This should point at the filtered h5ad produced by prepare_tahoe_prophet.ipynb
# (contains adata.uns["prophet_emb"] and only cells whose drug has an embedding)
DATA_PATH   = Path("/data/100m_prophet_filtered.h5ad")

with h5py.File(DATA_PATH, "r") as f:
    adata = ad.AnnData(
        obs=ad.io.read_elem(f["obs"]),
        obsm=ad.io.read_elem(f["obsm"]),
        uns=ad.io.read_elem(f["uns"]),
    )

load_time = time.time() - start_time
print(f"data loaded (took {load_time:.2f} seconds)")

# ── DataManager – same as process_tahoe.py plus extra_rep_keys ────────────────
adl = AnnDataLocation()
dm  = DataManager(
    dist_flag_key="control",
    src_dist_keys=["cell_line"],
    tgt_dist_keys=["drug"],
    rep_keys={
        "cell_line": "cell_line_embeddings",
        "drug":      "drug_0_embeddings",
    },
    data_location=adl.obsm["X_state"],
    # Look up adata.uns["prophet_emb"][drug_name] and store it under the
    # condition key "prophet".  Silently skipped if "prophet_emb" is absent
    # from adata.uns (so the script is safe to run on a base h5ad as well).
    extra_rep_keys={"prophet": ("drug", "prophet_emb")},
)

# %%
prepare_start = time.time()
gd = dm.prepare_data(adata=adata)

prepare_time = time.time() - prepare_start
print(f"data prepared (took {prepare_time:.2f} seconds)")

# %%
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
