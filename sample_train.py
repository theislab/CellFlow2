import numpy as np

from scaleflow.data import (
    AnnDataLocation,
    DataManager,
    GroupedDistribution,
    GroupedAnnbatchSampler,
    CombinedSampler,
    ValidationSampler,
    write_sorted_collection,
    split_datasets,
)
from scaleflow.datasets import sample_adata
from scaleflow.model import ScaleFlow
from scaleflow.training import Metrics

# ClassSampler chunk size; must be <= smallest trained condition's cell count.
# Synthetic conditions are tiny, so keep this small.
CHUNK_SIZE = 2

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

# Build sorted collections + grouped-distribution metadata from each synthetic adata
print("Preparing data...")
collections = {}
gds = {}
for i, adata in enumerate((adata1, adata2, adata3), start=1):
    coll = write_sorted_collection(
        adata,
        f"data/coll{i}.zarr",
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys=["drug", "gene"],
    )
    collections[f"gd{i}"] = coll
    gds[f"gd{i}"] = data_manager.prepare_data_from_collection(coll, rep_dict=adata.uns)

gd1, gd2, gd3 = gds["gd1"], gds["gd2"], gds["gd3"]

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
        "gd1": GroupedAnnbatchSampler(
            collections["gd1"], train_splits["gd1"], batch_size=8, chunk_size=CHUNK_SIZE, seed=42
        ),
        "gd2": GroupedAnnbatchSampler(
            collections["gd2"], train_splits["gd2"], batch_size=8, chunk_size=CHUNK_SIZE, seed=43
        ),
        "gd3": GroupedAnnbatchSampler(
            collections["gd3"], train_splits["gd3"], batch_size=8, chunk_size=CHUNK_SIZE, seed=44
        ),
    },
    rng=np.random.default_rng(42),
)

# Create validation sampler - returns all conditions at once (finite, not infinite)

val_samplers = {
    "gd1": ValidationSampler(
        collections["gd1"],
        val_splits["gd1"],  # Use one split for validation
        n_conditions_on_log_iteration=5,  # Limit to 5 conditions for faster testing
        n_conditions_on_train_end=5,
        seed=42,
    ),
    "gd2": ValidationSampler(
        collections["gd2"],
        val_splits["gd2"],  # Use one split for validation
        n_conditions_on_log_iteration=5,  # Limit to 5 conditions for faster testing
        n_conditions_on_train_end=5,
        seed=42,
    ),
    "gd3": ValidationSampler(
        collections["gd3"],
        val_splits["gd3"],  # Use one split for validation
        n_conditions_on_log_iteration=5,  # Limit to 5 conditions for faster testing
        n_conditions_on_train_end=5,
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
