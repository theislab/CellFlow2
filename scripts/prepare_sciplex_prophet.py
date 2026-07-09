# %%
"""
process_sciplex_prophet.py

Single-dataset (sciPlex3) version of process_tahoe_prophet.py. Builds a
GroupedDistribution zarr from the emb h5ad (sciplex3_with_emb.h5ad), with

    extra_rep_keys={"prophet": ("drug", "prophet_emb")}

so each condition dict gets a "prophet" key looked up from
adata.uns["prophet_emb"][drug_name]. If "prophet_emb" is absent the DataManager
silently skips it.

The cell representation stored in the zarr is chosen by ``--embedding_key`` and
must exist in ``adata.obsm``. sciplex3_with_emb.h5ad carries both the foundation
-model embeddings (X_state, X_scgpt, X_scconcept, X_scimilarity[_correct]) and
the autoencoder embeddings (AE_10, AE_32, AE_128, AE_128_opt), so any of those
can be passed. The script validates the key up-front and lists what's available
if it's missing.

Usage
─────
python prepare_sciplex_prophet.py --embedding_key X_state
python prepare_sciplex_prophet.py --embedding_key AE_128
python prepare_sciplex_prophet.py --embedding_key AE_128 --output_path /storage/pancellflow/sciplex3_AE_128.zarr
python prepare_sciplex_prophet.py --embedding_key X_scgpt --output_path /storage/pancellflow/sciplex3_X_scgpt.zarr
"""

import argparse

from scaleflow.data import DataManager, AnnDataLocation
from pathlib import Path
import anndata as ad
import h5py
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_key", type=str, default="X_state",
                    help="cell state representation. Pass 'X' to train directly on gene "
                         "expression (adata.X). Otherwise an obsm key — foundation-model: "
                         "X_state, X_scgpt, X_scconcept, X_scimilarity[_correct]; "
                         "autoencoder: AE_10, AE_32, AE_128, AE_128_opt.")
parser.add_argument("--output_path", type=str, default=None,
                    help="Output zarr path (default: /storage/pancellflow/sciplex3_<embedding_key>.zarr)")
parser.add_argument("--data_path", type=str, default="/storage/pancellflow/sciplex3_with_emb.h5ad",
                    help="Input h5ad path (must contain the chosen --embedding_key in obsm, or X; "
                         "sciplex3_with_emb.h5ad carries X (2000 HVGs) plus X_* and AE_* embeddings)")
args = parser.parse_args()

EMBEDDING_KEY = args.embedding_key
USE_X         = EMBEDDING_KEY in ("X", "genes", "gene")   # train directly on gene expression
DATA_PATH     = Path(args.data_path)
OUTPUT_PATH   = Path(args.output_path) if args.output_path else Path(f"/storage/pancellflow/sciplex3_{EMBEDDING_KEY}.zarr")

print(f"embedding_key = {EMBEDDING_KEY}")
print(f"output_path   = {OUTPUT_PATH}")
print(f"data_path     = {DATA_PATH}")

start_time = time.time()
print("loading data")

with h5py.File(DATA_PATH, "r") as f:
    elems = dict(
        obs=ad.io.read_elem(f["obs"]),
        obsm=ad.io.read_elem(f["obsm"]),
        uns=ad.io.read_elem(f["uns"]),
    )
    if USE_X:
        elems["X"] = ad.io.read_elem(f["X"])   # gene-expression matrix (only loaded when needed)
    adata = ad.AnnData(**elems)

load_time = time.time() - start_time
print(f"data loaded (took {load_time:.2f} seconds)")

# ── resolve the cell representation to store: gene space (adata.X) or an obsm embedding ──
adl = AnnDataLocation()
if USE_X:
    if adata.X is None or getattr(adata.X, "shape", (0, 0))[1] == 0:
        raise KeyError(f"--embedding_key '{EMBEDDING_KEY}' requested but adata.X is empty in {DATA_PATH}")
    if hasattr(adata.X, "toarray"):        # densify sparse X so the zarr writer gets ndarrays
        adata.X = np.asarray(adata.X.toarray(), dtype=np.float32)
    print(f"  X: gene-space matrix shape {adata.X.shape}")
    cell_location = adl.X
else:
    if EMBEDDING_KEY not in adata.obsm:
        raise KeyError(
            f"embedding_key '{EMBEDDING_KEY}' not found in adata.obsm of {DATA_PATH}. "
            f"Available obsm keys: {sorted(adata.obsm.keys())}"
        )
    print(f"  {EMBEDDING_KEY}: obsm shape {adata.obsm[EMBEDDING_KEY].shape}")
    cell_location = adl.obsm[EMBEDDING_KEY]

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
dm  = DataManager(
    dist_flag_key="control",
    src_dist_keys=["cell_line"],
    tgt_dist_keys=["drug", "dose"],             # ← dose now defines the distribution
    rep_keys={
        "cell_line": "cell_line_ccle_embeddings",
        "drug":      "drug_0_embeddings",
    },
    data_location=cell_location,
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
