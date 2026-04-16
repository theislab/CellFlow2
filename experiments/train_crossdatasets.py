"""Updated training script for cross datasets - Tahoe & combosciplex.

Uses Equilibrium Matching solver with AdaLN-Zero architecture.
"""
import numpy as np
import optax
import scanpy as sc
import ast
from functools import partial
from scaleflow.data import AnnDataLocation, DataManager, split_datasets
from scaleflow.data._dataloader import CombinedSampler, InMemorySampler, ValidationSampler
from scaleflow.model import ScaleFlow
from scaleflow.training import LearningRateMonitor, Metrics, WandbLogger
from scaleflow.utils import match_linear

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  

def mark_control(adata):
    # control = both drugs are control
    adata.obs["control"] = (adata.obs["drug_0"] == "control") & (adata.obs["drug_1"] == "control")
    return adata

# TODO: integrate hydra config for datasets
@hydra.main(config_path="/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/experiments/config", config_name="base_train_crossdatasets", version_base=None)
def train(cfg: DictConfig) -> None:

    datasets_to_use = []
    for name in cfg.selected_datasets:
        info = cfg.datasets[name]
        datasets_to_use.append({
            "name": name,
            "path": info.path,
            "seed": info.seed,
        })

    adatas = {}
    for dcfg in datasets_to_use:
        name = str(dcfg["name"])         
        path = str(dcfg["path"])
        print(f"Reading {name}: {path}")
        adata = sc.read_h5ad(path)
        adata = mark_control(adata)
        adatas[name] = adata


    adl = AnnDataLocation()
    data_manager = DataManager(
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys=["drug_0", "drug_1"],
        rep_keys={
            "cell_line": cfg.cell_embedding.cell_line_rep_key,
            "drug_0": "drug_0_embeddings",
            "drug_1": "drug_1_embeddings",
        },
        data_location=adl.obsm[cfg.cell_embedding.key],
    )

    print("Preparing data...")
    gds = {}
    for name, adata in adatas.items():
        gds[name] = data_manager.prepare_data(adata)

    print("Splitting datasets...")
    data = split_datasets(
        gds,
        split_by=["drug_0", "drug_1"],
        split_key="split",
        ratios=list(cfg.data.split_ratios) if "data" in cfg and "split_ratios" in cfg.data else [0.4, 0.3, 0.3],
        random_state=int(cfg.data.random_state) if "data" in cfg and "random_state" in cfg.data else 42,
        holdout_combinations=bool(cfg.data.holdout_combinations) if "data" in cfg and "holdout_combinations" in cfg.data else False,
    )
    train_splits = {k: v["train"] for k, v in data.items()}
    val_splits   = {k: v["val"]   for k, v in data.items()}


    print("Creating samplers...")
    batch_size = int(cfg.training.batch_size)

    samplers = {}
    weights  = {}
    for dcfg in datasets_to_use:
        name = str(dcfg["name"])
        seed = int(dcfg["seed"]) if "seed" in dcfg.keys() else 42
        w    = float(dcfg["weight"]) if "weight" in dcfg.keys() else 1.0

        samplers[name] = InMemorySampler(train_splits[name], np.random.default_rng(seed), batch_size=batch_size)
        weights[name]  = w

    sampler = CombinedSampler(
        samplers=samplers,
        rng=np.random.default_rng(int(cfg.data.random_state) if "data" in cfg and "random_state" in cfg.data else 42),
        # weights=weights, # Uncomment to use dataset weights in sampling. By default, datasets are sampled uniformly.
    )

    val_samplers = {
        name: ValidationSampler(
            val_splits[name],
            n_conditions_on_log_iteration=int(cfg.training.n_conditions_on_log_iteration) if ("n_conditions_on_log_iteration" in cfg.training and cfg.training.n_conditions_on_log_iteration is not None) else None,
            n_conditions_on_train_end=None,
            seed=int(cfg.data.random_state) if "data" in cfg and "random_state" in cfg.data else 42,
        )
        for name in gds.keys()
    }

    sampler.init_sampler()
    sample_batch = sampler.sample()
    print(f"Condition shapes: {[(k, v.shape) for k, v in sample_batch['condition'].items()]}")


    solver_key = cfg.solver.solver_key
    sf = ScaleFlow(solver=solver_key)
    print(f"Creating model with {solver_key} solver...")

    # Learning rate schedule: warmup + cosine decay
    NUM_ITERATIONS = cfg.training.num_iterations
    WARMUP_STEPS = min(cfg.training.warmup_iterations, NUM_ITERATIONS - 1)
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
    decoder_dims = tuple(int(x) for x in cfg.model.decoder_dims)
    hidden_dims  = tuple(int(x) for x in cfg.model.hidden_dims)

    conditioning_key = cfg.model.conditioning_key
    conditioning_kwargs = OmegaConf.to_container(cfg.model.conditioning_kwargs, resolve=True)

    if conditioning_key == "adaln_zero" and decoder_dims[-1] != hidden_dims[-1]:
        raise ValueError(
            f"For 'adaln_zero' conditioning, the last dimension of decoder_dims "
            f"must match the last dimension of hidden_dims. Got decoder_dims[-1]={decoder_dims[-1]}, "
            f"hidden_dims[-1]={hidden_dims[-1]}"
        )

    print(f"Preparing model with {conditioning_key} architecture...")
    match_fn = partial(
        match_linear,
        epsilon=cfg.match_fn.epsilon, 
    )
    sf.prepare_model(
        sample_batch=sample_batch,
        max_combination_length=2,
        conditioning=conditioning_key,
        decoder_dims=decoder_dims,  # Must match hidden_dims[-1] for adaln_zero
        conditioning_kwargs=conditioning_kwargs,
        match_fn=match_fn,
        probability_path=cfg.model.probability_path_kwargs,
        optimizer=optax.MultiSteps(optax.adam(learning_rate=1e-4), 20),
    )


    print("Training...")
    dataset_names = [str(dcfg["name"]) for dcfg in datasets_to_use]
    sf.train(
        val_dataloader=val_samplers,
        train_dataloader=sampler,
        num_iterations=NUM_ITERATIONS,
        valid_freq=cfg.training.valid_freq,
        callbacks=[
            Metrics(["e_distance", "r_squared"]),
            LearningRateMonitor(schedule=lr_schedule),
            WandbLogger(
                project="crossdatasets-unipert-256",
                out_dir="./wandb_logs",
                config={
                    "solver": solver_key,
                    "conditioning": conditioning_key,
                    "num_iterations": NUM_ITERATIONS,
                    "batch_size": BATCH_SIZE,
                    "decoder_dims": decoder_dims,
                    "hidden_dims": hidden_dims,
                    "noise_level": cfg.model.probability_path_kwargs,
                    "match_fn_epsilon": cfg.match_fn.epsilon,
                    "adaln_blocks": len(decoder_dims),
                    "peak_lr": PEAK_LR,
                    "end_lr": END_LR,
                    "warmup_steps": WARMUP_STEPS,
                    "datasets": dataset_names,
                },
                entity="pancellflow",
            ),
        ],
        monitor_metrics=["loss"],
    )

    print("Done!")

if __name__ == "__main__":
    train()