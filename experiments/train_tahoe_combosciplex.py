"""Updated training script for cross datasets - Tahoe & combosciplex.

Uses Equilibrium Matching solver with AdaLN-Zero architecture.
"""
import numpy as np
import optax
import scanpy as sc
import ast

from scaleflow.data import AnnDataLocation, DataManager, split_datasets
from scaleflow.data._dataloader import CombinedSampler, InMemorySampler, ValidationSampler
from scaleflow.model import ScaleFlow
from scaleflow.training import LearningRateMonitor, Metrics, WandbLogger


import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# TODO: integrate hydra config for datasets
@hydra.main(config_path="/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/experiments/config", config_name="base_train", version_base=None)
def train(cfg: DictConfig) -> None:
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
        data_location=adl.obsm[cfg.cell_embedding.key],
    )

    print("Preparing data...")
    gd1 = data_manager.prepare_data(adata1)
    gd2 = data_manager.prepare_data(adata2)

    print("Splitting datasets...")
    data = split_datasets(
        {"gd1": gd1, "gd2": gd2},
        split_by=["drug_0", "drug_1"],
        split_key="split",
        ratios=[0.4, 0.3, 0.3],
        random_state=42,
        holdout_combinations=False,
    )
    train_splits = {k: v["train"] for k, v in data.items()}
    val_splits = {k: v["val"] for k, v in data.items()}

    print("Creating samplers...")
    sampler = CombinedSampler(
        samplers={
            "gd1": InMemorySampler(gd1, np.random.default_rng(43), batch_size=512),
            "gd2": InMemorySampler(gd2, np.random.default_rng(44), batch_size=512),
        },
        rng=np.random.default_rng(42),
    )

    val_samplers = {
        "gd1": ValidationSampler(
            val_splits["gd1"],
            n_conditions_on_log_iteration=20,
            n_conditions_on_train_end=None,
            seed=42,
        ),
        "gd2": ValidationSampler(
            val_splits["gd2"],
            n_conditions_on_log_iteration=20,
            n_conditions_on_train_end=None,
            seed=42,
        ),
    }

    print("Initializing sampler...")
    sampler.init_sampler()

    print("Sampling batch...")
    sample_batch = sampler.sample()
    print(f"Condition shapes: {[(k, v.shape) for k, v in sample_batch['condition'].items()]}")

    print("Creating model with Equilibrium Matching solver...")
    solver_key = cfg.solver.solver_key
    sf = ScaleFlow(solver=solver_key)

    # Learning rate schedule: warmup + cosine decay
    NUM_ITERATIONS = cfg.training.num_iterations
    WARMUP_STEPS = cfg.training.warmup_iterations
    PEAK_LR = cfg.training.peak_lr
    END_LR = cfg.training.end_lr
    BATCH_SIZE = cfg.training.batch_size

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=PEAK_LR,
        warmup_steps=WARMUP_STEPS,
        decay_steps=NUM_ITERATIONS,
        end_value=END_LR,
    )

    print("Preparing model with AdaLN-Zero architecture...")
    decoder_dims = tuple(map(int, ast.literal_eval(cfg.model.decoder_dims)))  # e.g., (2048, 2048, 2048)
    conditioning_key = cfg.model.conditioning_key
    conditioning_kwargs = OmegaConf.to_container(cfg.model.conditioning_kwargs, resolve=True)
    sf.prepare_model(
        sample_batch=sample_batch,
        max_combination_length=2,
        conditioning=conditioning_key,
        decoder_dims=decoder_dims,  # Must match hidden_dims[-1] for adaln_zero
        conditioning_kwargs=conditioning_kwargs,
        optimizer=optax.MultiSteps(optax.adam(learning_rate=lr_schedule), 20),
    )

    print("Training...")
    sf.train(
        val_dataloader=val_samplers,
        train_dataloader=sampler,
        num_iterations=NUM_ITERATIONS,
        valid_freq=5000,
        callbacks=[
            Metrics(["e_distance", "r_squared"]),
            LearningRateMonitor(schedule=lr_schedule),
            WandbLogger(
                project="tahoe-combosciplex",
                out_dir="./wandb_logs",
                config={
                    "solver": solver_key,
                    "conditioning": conditioning_key,
                    "num_iterations": NUM_ITERATIONS,
                    "batch_size": BATCH_SIZE,
                    "adaln_blocks": len(decoder_dims),
                    "peak_lr": PEAK_LR,
                    "end_lr": END_LR,
                    "warmup_steps": WARMUP_STEPS,
                    "datasets": ["combosciplex", "tahoe"],
                },
                entity="pancellflow",
            ),
        ],
        monitor_metrics=["loss"],
    )

    print("Done!")

if __name__ == "__main__":
    train()