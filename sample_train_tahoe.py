import numpy as np

from scaleflow.data import AnnDataLocation, DataManager, GroupedDistribution, split_datasets
from scaleflow.data._dataloader import CombinedSampler, ReservoirSampler, ValidationSampler
from scaleflow.model import ScaleFlow

# Create sample data
print("Load data...")


gd1 = GroupedDistribution.read_zarr("/data/tahoe.zarr")
gd2 = GroupedDistribution.read_zarr("/data/tahoe.zarr")


# Split datasets
print("Splitting datasets...")
data = split_datasets({"gd1": gd1, "gd2": gd2}, split_by=["drug"], split_key="split", ratios=[0.7, 0.2, 0.1], random_state=42, holdout_combinations=False)
train_splits = {k:v["train"] for k,v in data.items()}
val_splits = {k:v["val"] for k,v in data.items()}
ds1_train, ds2_train = train_splits["gd1"], train_splits["gd2"]
ds1_val, ds2_val = val_splits["gd1"], val_splits["gd2"]


# Create training samplers
print("Creating samplers...")
rng = np.random.default_rng(42)
sampler = CombinedSampler(
    samplers={
        "gd1": ReservoirSampler(
            ds1_train, rng, batch_size=1024, pool_fraction=0.7, replacement_prob=0.3
        ),
        "gd2": ReservoirSampler(
            ds2_train, rng, batch_size=1024, pool_fraction=0.7, replacement_prob=0.3
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
        n_conditions_on_train_end=10,
        n_conditions_on_log_iteration=10,
        seed=42,
    ),
}



# Initialize sampler
print("Initializing sampler...")
sampler.init_sampler()

# Sample a batch
print("Sampling batch...")
sample_batch = sampler.sample()
print(f"Condition shapes: {[(k, v.shape) for k, v in sample_batch['condition'].items()]}")

# Create and prepare model
print("Creating model...")
sf = ScaleFlow()

print("Preparing model...")
sf.prepare_model(
    sample_batch=sample_batch,
    max_combination_length=1,  # Single perturbations
)

# Train
print("Training...")
sf.train(
    val_dataloader=val_samplers,
    train_dataloader=sampler,
    num_iterations=4,
)

print("Done!")
