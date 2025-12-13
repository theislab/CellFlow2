# %%
from scaleflow.data import DataManager, AnnDataLocation
from pathlib import Path
import anndata as ad
import h5py
import time

start_time = time.time()
print("loading data")
OUTPUT_PATH = Path("/data/tahoe.zarr")
DATA_PATH = Path("/data/100m_int_indices.h5ad")

with h5py.File(DATA_PATH, "r") as f:
    adata = ad.AnnData(
        obs=ad.io.read_elem(f["obs"]),
        obsm=ad.experimental.read_lazy(f["obsm"]),
        uns=ad.io.read_elem(f["uns"]),
    )

adl = AnnDataLocation()
dm = DataManager(
    dist_flag_key="control",
    src_dist_keys=["cell_line"],
    tgt_dist_keys=["drug", "dosage"],
    rep_keys={
        "cell_line": "cell_line_embeddings",
        "drug": "drug_embeddings",
    },
    data_location=adl.obsm["X_pca"][:, :50],
)

load_time = time.time() - start_time
print(f"data loaded (took {load_time:.2f} seconds)")

# %%
prepare_start = time.time()
gd = dm.prepare_data(
    adata=adata,
)

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
print(f"  - Loading: {load_time:.2f} seconds ({load_time/total_time*100:.1f}%)")
print(f"  - Preparation: {prepare_time:.2f} seconds ({prepare_time/total_time*100:.1f}%)")
print(f"  - Writing: {write_time:.2f} seconds ({write_time/total_time*100:.1f}%)")
