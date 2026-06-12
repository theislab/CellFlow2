from scaleflow.data import DataManager, AnnDataLocation
from pathlib import Path
import anndata as ad
import h5py
import time

start_time = time.time()
print("loading data")

OUTPUT_PATH = Path("/storage/pancellflow/tahoe.zarr")
DATA_PATH   = Path("/storage/pancellflow/100m_prophet_filtered.h5ad")

with h5py.File(DATA_PATH, "r") as f:
    adata = ad.AnnData(
        obs=ad.io.read_elem(f["obs"]),
        obsm=ad.io.read_elem(f["obsm"]),   # read_lazy not available
        uns=ad.io.read_elem(f["uns"]),
    )

load_time = time.time() - start_time
print(f"data loaded (took {load_time:.2f} seconds)")

adl = AnnDataLocation()
dm = DataManager(
    dist_flag_key="control",              # boolean col added during concat
    src_dist_keys=["cell_line"],
    tgt_dist_keys=["drug"],               # no dosage in per-cell-line h5ads
    rep_keys={
        "cell_line": "cell_line_embeddings",
        "drug":      "drug_0_embeddings", # uns key in per-cell-line h5ads
    },
    data_location=adl.obsm["X_state"],    # 2058-dim scVI latent
)

prepare_start = time.time()
gd = dm.prepare_data(adata=adata, verbose=True)
prepare_time = time.time() - prepare_start
print(f"data prepared (took {prepare_time:.2f} seconds)")

chunk_size = 131072
shard_size = chunk_size * 8

write_start = time.time()
gd.write_zarr(path=OUTPUT_PATH, chunk_size=chunk_size, shard_size=shard_size)
write_time = time.time() - write_start
print(f"data written (took {write_time:.2f} seconds)")

total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.2f} seconds")
print(f"  - Loading:     {load_time:.2f} seconds ({load_time/total_time*100:.1f}%)")
print(f"  - Preparation: {prepare_time:.2f} seconds ({prepare_time/total_time*100:.1f}%)")
print(f"  - Writing:     {write_time:.2f} seconds ({write_time/total_time*100:.1f}%)")