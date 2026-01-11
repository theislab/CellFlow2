# %% [markdown]
# ## Test script for cross datasets training - Tahoe & combosciplex

# %%
import numpy as np

from scaleflow.data import AnnDataLocation, DataManager, GroupedDistribution, prepare_datasets, split_datasets
from scaleflow.data._dataloader import CombinedSampler, InMemorySampler, ReservoirSampler, ValidationSampler
from scaleflow.datasets import sample_adata
from scaleflow.model import ScaleFlow
from scaleflow.training import Metrics
import scanpy as sc


# %%
adata1 = sc.read_h5ad('/lustre/groups/ml01/projects/big_perturbation/dataset_w_embeddings/combosciplex_with_embeddings.h5ad')
adata2 = sc.read_h5ad('/lustre/groups/ml01/projects/big_perturbation/dataset_w_embeddings/tahoe_with_embeddings.h5ad')
adata1.obs["control"] = (
    (adata1.obs["drug_0"] == "control") &
    (adata1.obs["drug_1"] == "control")
)
adata2.obs["control"] = (
    (adata2.obs["drug_0"] == "control") &
    (adata2.obs["drug_1"] == "control")
)

# %%
adl = AnnDataLocation()
data_manager = DataManager(
    dist_flag_key="control",
    src_dist_keys=["cell_line"],
    tgt_dist_keys=["drug_0", "drug_1"],
    rep_keys={
        "cell_line": "cell_line_embeddings",
        "drug_0": "drug_0_embeddings",
        "drug_1": "drug_1_embeddings",
    },
    data_location=adl.obsm["X_state"],
)

# %%

# Prepare data
print("Preparing data...")
gd1 = data_manager.prepare_data(adata1)
gd2 = data_manager.prepare_data(adata2)

# Split datasets
print("Splitting datasets...")
data = split_datasets({"gd1": gd1, "gd2": gd2}, split_by=["drug_0","drug_1"], split_key="split", ratios=[0.4, 0.3,0.3], random_state=42, holdout_combinations=False)
train_splits = {k:v["train"] for k,v in data.items()}
val_splits = {k:v["val"] for k,v in data.items()}
ds1, ds2 = train_splits["gd1"], train_splits["gd2"]
# Create training samplers
print("Creating samplers...")
sampler = CombinedSampler(
    samplers={
        "gd1": InMemorySampler(gd1, np.random.default_rng(43), batch_size=8),
        "gd2": InMemorySampler(gd2, np.random.default_rng(44), batch_size=8),
    },
    rng=np.random.default_rng(42),
)

# Create validation sampler - returns all conditions at once (finite, not infinite)

val_samplers = {
    "gd1": ValidationSampler(
        val_splits["gd1"],  # Use one split for validation
        n_conditions_on_log_iteration=3,  # Limit to 5 conditions for faster testing
        n_conditions_on_train_end=3,
        seed=42,
    ),
    "gd2": ValidationSampler(
        val_splits["gd2"],  # Use one split for validation
        n_conditions_on_log_iteration=3,  # Limit to 5 conditions for faster testing
        n_conditions_on_train_end=3,
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

# %%

# Train
print("Training...")
sf.train(
    val_dataloader=val_samplers,
    train_dataloader=sampler,
    num_iterations=100000,
    valid_freq=5000,
    callbacks=[Metrics(["e_distance"])],  
    monitor_metrics=["loss_functional"] 
)

print("Done!")




