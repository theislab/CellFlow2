# %%
# %load_ext autoreload
# %autoreload 2


# %%
import anndata as ad
import h5py
import zarr
from scaleflow.data._utils import write_sharded
from anndata.experimental import read_lazy
from scaleflow.data import DataManager
import cupy as cp
import tqdm
import dask
import concurrent.futures
from functools import partial
import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar

print("loading data")
with h5py.File("/lustre/groups/ml01/workspace/100mil/100m_int_indices.h5ad", "r") as f:
    adata_all = ad.AnnData(
        obs=ad.io.read_elem(f["obs"]),
        var=read_lazy(f["var"]),
        uns = read_lazy(f["uns"]),
        obsm = read_lazy(f["obsm"]),
    )

dm = DataManager(adata_all,  
    sample_rep="X_pca",
    control_key="control",
    perturbation_covariates={"drugs": ("drug",), "dosage": ("dosage",)},
    perturbation_covariate_reps={"drugs": "drug_embeddings"},
    sample_covariates=["cell_line"],
    sample_covariate_reps={"cell_line": "cell_line_embeddings"},
    split_covariates=["cell_line"],
    max_combination_length=None,
    null_value=0.0
)
print("data loaded")

# %%
cond_data = dm._get_condition_data(adata=adata_all)
cell_data = dm._get_cell_data(adata_all)

# %%
n_source_dists = len(cond_data.split_idx_to_covariates)
n_target_dists = len(cond_data.perturbation_idx_to_covariates)

tgt_cell_data = {}
src_cell_data = {}
gpu_per_cov_mask = cp.asarray(cond_data.perturbation_covariates_mask)
gpu_spl_cov_mask = cp.asarray(cond_data.split_covariates_mask)

for src_idx in tqdm.tqdm(range(n_source_dists), desc="Computing source to cell data idcs"):
    mask = gpu_spl_cov_mask == src_idx
    src_cell_data[str(src_idx)] = {
        "cell_data_index": cp.where(mask)[0].get(),
    }

for tgt_idx in tqdm.tqdm(range(n_target_dists), desc="Computing target to cell data idcs"):
    mask = gpu_per_cov_mask == tgt_idx
    tgt_cell_data[str(tgt_idx)] = {
        "cell_data_index": cp.where(mask)[0].get(),
    }

# %%

print("Computing cell data")
cell_data = cell_data.compute()
print("cell data computed")

for src_idx in tqdm.tqdm(range(n_source_dists), desc="Computing source to cell data"):
    indices = src_cell_data[str(src_idx)]["cell_data_index"]
    src_cell_data[str(src_idx)]["cell_data"] = cell_data[indices]

for tgt_idx in tqdm.tqdm(range(n_target_dists), desc="Computing target to cell data"):
    indices = tgt_cell_data[str(tgt_idx)]["cell_data_index"]
    tgt_cell_data[str(tgt_idx)]["cell_data"] = cell_data[indices]


# %%

split_covariates_mask = np.asarray(cond_data.split_covariates_mask)
perturbation_covariates_mask = np.asarray(cond_data.perturbation_covariates_mask)
condition_data = {str(k): np.asarray(v) for k, v in (cond_data.condition_data or {}).items()}
control_to_perturbation = {str(k): np.asarray(v) for k, v in (cond_data.control_to_perturbation or {}).items()}
split_idx_to_covariates = {str(k): np.asarray(v) for k, v in (cond_data.split_idx_to_covariates or {}).items()}
perturbation_idx_to_covariates = {
    str(k): np.asarray(v) for k, v in (cond_data.perturbation_idx_to_covariates or {}).items()
}
perturbation_idx_to_id = {str(k): v for k, v in (cond_data.perturbation_idx_to_id or {}).items()}

train_data_dict = {
    "split_covariates_mask": split_covariates_mask,
    "perturbation_covariates_mask": perturbation_covariates_mask,
    "split_idx_to_covariates": split_idx_to_covariates,
    "perturbation_idx_to_covariates": perturbation_idx_to_covariates,
    "perturbation_idx_to_id": perturbation_idx_to_id,
    "condition_data": condition_data,
    "control_to_perturbation": control_to_perturbation,
    "max_combination_length": int(cond_data.max_combination_length),
    # "src_cell_data": src_cell_data,
    # "tgt_cell_data": tgt_cell_data,
}

print("prepared train_data_dict")
# %%
path = "/lustre/groups/ml01/workspace/100mil/tahoe.zarr"
zgroup = zarr.open_group(path, mode="w")
chunk_size = 131072
shard_size = chunk_size * 8

ad.settings.zarr_write_format = 3  # Needed to support sharding in Zarr

def get_size(shape: tuple[int, ...], chunk_size: int, shard_size: int) -> tuple[int, int]:
    shard_size_used = shard_size
    chunk_size_used = chunk_size
    if chunk_size > shape[0]:
        chunk_size_used = shard_size_used = shape[0]
    elif chunk_size < shape[0] or shard_size > shape[0]:
        chunk_size_used = shard_size_used = shape[0]
    return chunk_size_used, shard_size_used




def write_single_array(group, key, arr, idxs, chunk_size, shard_size):
    """Write a single array - designed for threading"""
    chunk_size_used, shard_size_used = get_size(arr.shape, chunk_size, shard_size)
    
    group.create_array(
        name=key,
        data=arr,
        chunks=(chunk_size_used, arr.shape[1]),
        shards=(shard_size_used, arr.shape[1]),
        compressors=None,
    )

    group.create_array(
        name=f"{key}_index",
        data=idxs,
        chunks=(len(idxs),),
        shards=(len(idxs),),
        compressors=None,
    )
    return key

def write_cell_data_threaded(group, cell_data, chunk_size, shard_size, max_workers=8):
    """Write cell data using threading for I/O parallelism"""
    
    write_func = partial(write_single_array, group, chunk_size=chunk_size, shard_size=shard_size)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all write tasks
        future_to_key = {
            executor.submit(write_single_array, group, k, cell_data[k]["cell_data"], cell_data[k]["cell_data_index"], chunk_size, shard_size): k
            for k in cell_data.keys()
        }
        
        # Process results with progress bar
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_key), 
            total=len(future_to_key),
            desc=f"Writing {group.name}"
        ):
            key = future_to_key[future]
            try:
                future.result()  # This will raise any exceptions
            except Exception as exc:
                print(f'Array {key} generated an exception: {exc}')
                raise

# %%


src_group = zgroup.create_group("src_cell_data", overwrite=True)
tgt_group = zgroup.create_group("tgt_cell_data", overwrite=True)


# Use the fast threaded approach you already implemented
write_cell_data_threaded(src_group, src_cell_data, chunk_size, shard_size, max_workers=24)
print("done writing src_cell_data")
write_cell_data_threaded(tgt_group, tgt_cell_data, chunk_size, shard_size, max_workers=24)
print("done writing tgt_cell_data")






# %%

print("Writing mapping data")
mapping_data = zgroup.create_group("mapping_data", overwrite=True)


write_sharded(
    group=mapping_data,
    name="mapping_data",
    data=train_data_dict,
    chunk_size=chunk_size,
    shard_size=shard_size,
    compressors=None,
)
print("done")


