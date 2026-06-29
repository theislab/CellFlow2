import time

import numpy as np

from scaleflow.data import (
    AnnDataLocation,
    DataManager,
    GroupedDistribution,
    GroupedAnnbatchSampler,
    CombinedSampler,
    ValidationSampler,
    split_datasets,
)
from scaleflow.model import ScaleFlow

# Sorted DatasetCollection of cells, built via scaleflow.data.write_sorted_collection.
COLLECTION_PATH = "/data/tahoe_collection.zarr"

# ClassSampler chunk size; must be <= smallest trained condition's cell count.
CHUNK_SIZE = 256

script_start = time.perf_counter()

# Create sample data
print("Load data...")
t0 = time.perf_counter()


gd1 = GroupedDistribution.read_zarr("/data/tahoe.zarr")
gd2 = GroupedDistribution.read_zarr("/data/tahoe.zarr")
print(f"  Load data took {(time.perf_counter() - t0) / 60:.2f}min")


# Split datasets
print("Splitting datasets...")
t0 = time.perf_counter()
data = split_datasets({"gd1": gd1, "gd2": gd2}, split_by=["drug"], split_key="split", ratios=[0.7, 0.2, 0.1], random_state=42, holdout_combinations=False)
train_splits = {k:v["train"] for k,v in data.items()}
val_splits = {k:v["val"] for k,v in data.items()}
ds1_train, ds2_train = train_splits["gd1"], train_splits["gd2"]
ds1_val, ds2_val = val_splits["gd1"], val_splits["gd2"]
print(f"  Splitting datasets took {(time.perf_counter() - t0) / 60:.2f}min")


# Create training samplers
print("Creating samplers...")
t0 = time.perf_counter()
rng = np.random.default_rng(42)
sampler = CombinedSampler(
    samplers={
        "gd1": GroupedAnnbatchSampler(
            COLLECTION_PATH, ds1_train, batch_size=1024, chunk_size=CHUNK_SIZE, seed=42
        ),
        "gd2": GroupedAnnbatchSampler(
            COLLECTION_PATH, ds2_train, batch_size=1024, chunk_size=CHUNK_SIZE, seed=42
        ),
    },
    rng=rng,
)

# Create validation sampler - returns all conditions at once (finite, not infinite)

val_samplers = {
    "gd1": ValidationSampler(
        COLLECTION_PATH,
        ds1_val,  # Use one split for validation
        seed=42,
        n_conditions_on_train_end=10,
        n_conditions_on_log_iteration=10,
    ),
    "gd2": ValidationSampler(
        COLLECTION_PATH,
        ds2_val,  # Use one split for validation
        n_conditions_on_train_end=100,
        n_conditions_on_log_iteration=100,
        seed=42,
    ),
}
print(f"  Creating samplers took {(time.perf_counter() - t0) / 60:.2f}min")



# Initialize sampler
print("Initializing sampler...")
t0 = time.perf_counter()
sampler.init_sampler()
print(f"  Initializing sampler took {(time.perf_counter() - t0) / 60:.2f}min")

# Sample a batch
print("Sampling batch...")
t0 = time.perf_counter()
sample_batch = sampler.sample()
print(f"  Sampling batch took {(time.perf_counter() - t0) / 60:.2f}min")
print(f"Condition shapes: {[(k, v.shape) for k, v in sample_batch['condition'].items()]}")

# Create and prepare model
print("Creating model...")
t0 = time.perf_counter()
sf = ScaleFlow()
print(f"  Creating model took {(time.perf_counter() - t0) / 60:.2f}min")

print("Preparing model...")
t0 = time.perf_counter()
sf.prepare_model(
    sample_batch=sample_batch,
    max_combination_length=1,  # Single perturbations
)
print(f"  Preparing model took {(time.perf_counter() - t0) / 60:.2f}min")

# Train
print("Training...")
t0 = time.perf_counter()
sf.train(
    val_dataloader=val_samplers,
    train_dataloader=sampler,
    num_iterations=4,
)
print(f"  Training took {(time.perf_counter() - t0) / 60:.2f}min")

print(f"Done! Total time: {(time.perf_counter() - script_start) / 60:.2f}min")
