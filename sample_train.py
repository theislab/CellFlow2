import numpy as np

from scaleflow.data import AnnDataLocation, DataManager, GroupedDistribution, prepare_datasets, split_datasets
from scaleflow.data._dataloader import CombinedSampler, InMemorySampler, ReservoirSampler, ValidationSampler
from scaleflow.datasets import sample_adata
from scaleflow.model import ScaleFlow
from scaleflow.training import Metrics

# Create sample data
print("Creating sample data...")
adata1, adata2, adata3 = sample_adata(n_pca=60), sample_adata(n_pca=60), sample_adata(n_pca=60)

# Setup data manager
adl = AnnDataLocation()
data_manager = DataManager(
    dist_flag_key="control",
    src_dist_keys=["cell_line"],
    tgt_dist_keys=["drug", "gene"],
    rep_keys={
        "cell_line": "cell_line_embeddings",
        "drug": "drug_embeddings",
        "gene": "gene_embeddings",
    },
    data_location=adl.obsm["X_pca"],
)

# Prepare data
print("Preparing data...")
gd1_mem = data_manager.prepare_data(adata1)
gd1_mem.write_zarr("data/gd1.zarr")
gd1 = GroupedDistribution.read_zarr("data/gd1.zarr")
del gd1_mem

gd2 = data_manager.prepare_data(adata2)
gd3 = data_manager.prepare_data(adata3)

# Split datasets
print("Splitting datasets...")
data = split_datasets({"gd1": gd1, "gd2": gd2, "gd3": gd3}, split_by=["drug"], split_key="split", ratios=[0.4, 0.3, 0.3], random_state=42, holdout_combinations=False)
train_splits = {k:v["train"] for k,v in data.items()}
val_splits = {k:v["val"] for k,v in data.items()}
ds1, ds2, ds3 = train_splits["gd1"], train_splits["gd2"], train_splits["gd3"]

# Create training samplers
print("Creating samplers...")
sampler = CombinedSampler(
    samplers={
        "gd1": ReservoirSampler(
            gd1, np.random.default_rng(42), batch_size=8, pool_fraction=0.5, replacement_prob=0.1
        ),
        "gd2": InMemorySampler(gd2, np.random.default_rng(43), batch_size=8),
        "gd3": InMemorySampler(gd3, np.random.default_rng(44), batch_size=8),
    },
    rng=np.random.default_rng(42),
)

# Create validation sampler - returns all conditions at once (finite, not infinite)

val_samplers = {
    "gd1": ValidationSampler(
        val_splits["gd1"],  # Use one split for validation
        n_conditions=5,  # Limit to 5 conditions for faster testing
        seed=42,
    ),
    "gd2": ValidationSampler(
        val_splits["gd2"],  # Use one split for validation
        n_conditions=5,  # Limit to 5 conditions for faster testing
        seed=42,
    ),
    "gd3": ValidationSampler(
        val_splits["gd3"],  # Use one split for validation
        n_conditions=5,  # Limit to 5 conditions for faster testing
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
    num_iterations=10,
)

print("Done!")
