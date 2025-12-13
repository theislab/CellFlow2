import time

import numpy as np

from scaleflow.data import AnnDataLocation, DataManager, GroupedDistribution, split_datasets
from scaleflow.data._dataloader import CombinedSampler, ReservoirSampler, ValidationSampler
from scaleflow.model import ScaleFlow

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
        "gd1": ReservoirSampler(
            ds1_train, rng, batch_size=1024, pool_fraction=0.7, replacement_prob=0.5
        ),
        "gd2": ReservoirSampler(
            ds2_train, rng, batch_size=1024, pool_fraction=0.7, replacement_prob=0.5
        ),
    },
    rng=rng,
)

# Create validation sampler - returns all conditions at once (finite, not infinite)

val_samplers = {
    "gd1": ValidationSampler(
        ds1_val,  # Use one split for validation
        seed=42,
        n_conditions_on_train_end=10,
        n_conditions_on_log_iteration=10,
    ),
    "gd2": ValidationSampler(
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
