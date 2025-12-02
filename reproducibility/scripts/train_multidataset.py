import functools
import gc
import sys
from pathlib import Path

import hydra
import optax
import psutil
from omegaconf import DictConfig, OmegaConf
from ott.solvers import utils as solver_utils

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import scaleflow as cfp


def print_memory_usage(label=""):
    """Print current memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024 ** 3)
    available = psutil.virtual_memory().available / (1024 ** 3)
    total = psutil.virtual_memory().total / (1024 ** 3)
    print(f"[MEMORY {label}] Used: {mem_gb:.2f} GB | Available: {available:.2f} GB | Total: {total:.2f} GB")
    return mem_gb


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration for multiple datasets."""
    print("=" * 80)
    print("CellFlow Training with Hydra - Multiple Datasets")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    print("\n[1/7] Loading data...")
    print_memory_usage("Before loading")
    import anndata as ad
    import h5py

    datasets = {}
    dataset_names = ["dataset_1", "dataset_2"]

    for dataset_name in dataset_names:
        print(f"\n  Loading {dataset_name}...")
        print_memory_usage(f"Before loading {dataset_name}")

        with h5py.File(cfg.data.adata_path, "r") as f:
            datasets[dataset_name] = ad.AnnData(
                obs=ad.io.read_elem(f["obs"]),
                var=ad.io.read_elem(f["var"]),
                varm=ad.io.read_elem(f["varm"]),
                uns=ad.io.read_elem(f["uns"]),
                obsm=ad.io.read_elem(f["obsm"]),
            )

        print_memory_usage(f"After loading {dataset_name}")
        gc.collect()
        print_memory_usage(f"After GC for {dataset_name}")

    print(f"\nLoaded {len(datasets)} datasets:")
    for name, adata in datasets.items():
        print(f"  {name}: {adata.shape}")
        print(f"    Control cells: {adata.obs['control'].sum()}")
        print(f"    Perturbed cells: {(~adata.obs['control']).sum()}")

    adata = datasets["dataset_1"]
    print_memory_usage("After loading all datasets")

    print("\n[2/7] Preparing data with DataManager...")
    print_memory_usage("Before DataManager setup")
    from scaleflow.data import DataManager, ReservoirSampler, prepare_and_split_multiple_datasets
    from scaleflow.data._anndata_location import AnnDataLocation

    adl = AnnDataLocation()

    perturbation_covariate_reps = OmegaConf.to_container(
        cfg.data.get("perturbation_covariate_reps", {})
    )

    dm = DataManager(
        dist_flag_key=cfg.data.control_key,
        src_dist_keys=list(cfg.data.get("src_dist_keys", ["cell_line"])),
        tgt_dist_keys=list(cfg.data.get("tgt_dist_keys", ["drug", "dosage"])),
        rep_keys=perturbation_covariate_reps,
        data_location=adl.obsm[cfg.data.sample_rep],
    )
    print_memory_usage("After DataManager setup")

    print("\n[3/7] Preparing and splitting multiple datasets...")
    print_memory_usage("Before prepare_and_split")

    if cfg.data.get("split_by"):
        print(f"  Processing {len(datasets)} datasets with prepare_and_split_multiple_datasets...")
        print(f"  Dataset names: {list(datasets.keys())}")
        print(f"  Split by: {list(cfg.data.split_by)}")
        print_memory_usage("Starting prepare_and_split_multiple_datasets")

        splits_all = prepare_and_split_multiple_datasets(
            datasets=datasets,
            data_manager=dm,
            holdout_combinations=False,
            split_by=list(cfg.data.split_by),
            ratios=list(cfg.data.get("split_ratios", [0.7, 0.15, 0.15])),
            random_state=cfg.seed,
            verbose=True,
        )

        print_memory_usage("After prepare_and_split_multiple_datasets")
        gc.collect()
        print_memory_usage("After GC post-split")

        print("\n  Split summary:")
        split_column = list(cfg.data.split_by)[0]
        for dataset_name in dataset_names:
            print(f"\n  Dataset: {dataset_name}")
            print_memory_usage(f"Processing split summary for {dataset_name}")
            print(f"    Train sources: {len(splits_all[dataset_name]['train'].data.src_data)}")
            print(f"    Train targets: {len(splits_all[dataset_name]['train'].data.tgt_data)}")
            print(f"    Val sources: {len(splits_all[dataset_name]['val'].data.src_data)}")
            print(f"    Val targets: {len(splits_all[dataset_name]['val'].data.tgt_data)}")
            print(f"    Test sources: {len(splits_all[dataset_name]['test'].data.src_data)}")
            print(f"    Test targets: {len(splits_all[dataset_name]['test'].data.tgt_data)}")

            train_df = splits_all[dataset_name]['train'].annotation.src_tgt_dist_df
            val_df = splits_all[dataset_name]['val'].annotation.src_tgt_dist_df
            test_df = splits_all[dataset_name]['test'].annotation.src_tgt_dist_df

            if split_column in train_df.columns:
                train_values = sorted(train_df[split_column].unique())
                val_values = sorted(val_df[split_column].unique())
                test_values = sorted(test_df[split_column].unique())

                print(f"    Train {split_column}s ({len(train_values)}): {train_values}")
                print(f"    Val {split_column}s ({len(val_values)}): {val_values}")
                print(f"    Test {split_column}s ({len(test_values)}): {test_values}")

        print("\n  Using first dataset for training...")
        print_memory_usage("Before extracting splits")

        train_data = splits_all[dataset_names[0]]['train']
        val_data = splits_all[dataset_names[0]]['val']
        test_data = splits_all[dataset_names[0]]['test']

        print(f"  Training samples: {len(train_data.data.src_data)} sources")
        print(f"  Validation samples: {len(val_data.data.src_data)} sources")
        print(f"  Test samples: {len(test_data.data.src_data)} sources")

        additional_test_data = {}
        for dataset_name in dataset_names[1:]:
            additional_test_data[dataset_name] = splits_all[dataset_name]['test']
            print(f"  Additional test data ({dataset_name}): {len(splits_all[dataset_name]['test'].data.src_data)} sources")

        print_memory_usage("After extracting splits")

        print("\n  Cleaning up original datasets from memory...")
        del datasets
        gc.collect()
        print_memory_usage("After deleting original datasets")

    else:
        from scaleflow.data import prepare_multiple_datasets
        gd_dict = prepare_multiple_datasets(
            datasets=datasets,
            data_manager=dm,
            verbose=True,
        )
        train_data = gd_dict[dataset_names[0]]
        val_data = None
        test_data = None
        additional_test_data = {}
        print("  No splitting - using all data from first dataset for training")

    print("\n[4/7] Initializing CellFlow...")
    print_memory_usage("Before CellFlow init")
    cf = cfp.model.CellFlow(adata, solver=cfg.solver.type)
    cf.train_data = train_data

    first_src_idx = next(iter(train_data.data.src_data.keys()))
    cf._data_dim = train_data.data.src_data[first_src_idx].shape[-1]

    train_data.max_combination_length = 1

    print_memory_usage("Before ReservoirSampler")
    cf._dataloader = ReservoirSampler(
        data=train_data,
        batch_size=cfg.training.batch_size
    )
    print_memory_usage("After ReservoirSampler")

    if val_data is not None:
        cf._validation_data["val"] = val_data
    if test_data is not None:
        cf._validation_data["test"] = test_data

    if cfg.data.get("split_by"):
        for dataset_name in dataset_names[1:]:
            cf._validation_data[f"test_{dataset_name}"] = additional_test_data[dataset_name]
            print(f"  Added additional test data: test_{dataset_name}")

    cf._dm = None

    print("\n[5/7] Preparing model...")
    print_memory_usage("Before model preparation")

    match_fn = functools.partial(
        solver_utils.match_linear,
        epsilon=cfg.solver.epsilon,
        scale_cost="mean",
        tau_a=cfg.solver.tau_a,
        tau_b=cfg.solver.tau_b,
    )

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.training.optimizer.learning_rate,
        warmup_steps=cfg.training.optimizer.warmup_steps,
        decay_steps=cfg.training.num_iterations - cfg.training.optimizer.warmup_steps,
        end_value=cfg.training.optimizer.end_value,
    )

    optimizer = optax.MultiSteps(
        optax.adamw(learning_rate=lr_schedule, weight_decay=cfg.training.optimizer.weight_decay),
        cfg.training.optimizer.multi_steps
    )

    probability_path = (
        {cfg.solver.flow_type: cfg.solver.flow_noise}
        if cfg.solver.type == "otfm"
        else None
    )

    layers_before_pool = OmegaConf.to_container(cfg.condition_encoder.layers_before_pool)
    layers_after_pool = OmegaConf.to_container(cfg.condition_encoder.layers_after_pool)

    solver_kwargs = {"ema": cfg.training.ema}

    conditioning_kwargs = {}
    if cfg.architecture.conditioning.method == "adaln_zero":
        conditioning_kwargs = {
            "num_heads": cfg.architecture.conditioning.num_heads,
            "qkv_dim": cfg.architecture.conditioning.qkv_dim,
        }
    elif cfg.architecture.conditioning.method in ["concatenation", "film", "resnet"]:
        conditioning_kwargs = {
            "layer_norm_before_concatenation": cfg.architecture.conditioning.get("layer_norm_before", False),
            "linear_projection_before_concatenation": cfg.architecture.conditioning.get("linear_projection_before", False),
        }

    cf.prepare_model(
        condition_mode=cfg.condition_encoder.mode,
        regularization=cfg.condition_encoder.regularization,
        pooling=cfg.condition_encoder.pooling.method,
        pooling_kwargs={
            k: v for k, v in OmegaConf.to_container(cfg.condition_encoder.pooling).items()
            if k != "method"
        },
        layers_before_pool=layers_before_pool,
        layers_after_pool=layers_after_pool,
        condition_embedding_dim=cfg.condition_encoder.embedding_dim,
        cond_output_dropout=cfg.condition_encoder.output_dropout,
        time_freqs=cfg.architecture.time_encoder.freqs,
        time_max_period=cfg.architecture.time_encoder.max_period,
        time_encoder_dims=list(cfg.architecture.time_encoder.dims),
        time_encoder_dropout=cfg.architecture.time_encoder.dropout,
        hidden_dims=list(cfg.architecture.cell_encoder.dims),
        hidden_dropout=cfg.architecture.cell_encoder.dropout,
        cell_transformer_layers=cfg.architecture.cell_transformer.layers,
        cell_transformer_heads=cfg.architecture.cell_transformer.heads,
        cell_transformer_dim=cfg.architecture.cell_transformer.dim,
        cell_transformer_dropout=cfg.architecture.cell_transformer.dropout,
        cell_transformer_mode=cfg.architecture.cell_transformer.mode,
        conditioning=cfg.architecture.conditioning.method,
        conditioning_kwargs=conditioning_kwargs,
        decoder_dims=list(cfg.architecture.decoder.dims),
        decoder_dropout=cfg.architecture.decoder.dropout,
        probability_path=probability_path,
        solver_kwargs=solver_kwargs,
        match_fn=match_fn,
        optimizer=optimizer,
        layer_norm_before_concatenation=cfg.architecture.conditioning.get("layer_norm_before", False),
        linear_projection_before_concatenation=cfg.architecture.conditioning.get("linear_projection_before", False),
    )

    print_memory_usage("After model preparation")

    import jax
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(cf._solver.vf_state.params))

    print("\nModel architecture:")
    print(f"  Conditioning: {cfg.architecture.conditioning.method}")
    print(f"  Cell encoder: {cfg.architecture.cell_encoder.dims}")
    print(f"  Time encoder: {cfg.architecture.time_encoder.dims}")
    print(f"  Decoder: {cfg.architecture.decoder.dims}")
    print(f"  Condition embedding: {cfg.condition_encoder.embedding_dim}")
    print("\nModel info:")
    print(f"  Input data dimension: {cf._data_dim}")
    print(f"  Total trainable parameters: {num_params:,}")

    print("\n[6/7] Setting up callbacks...")
    callbacks = []

    if cfg.training.callbacks.use_metrics:
        metrics_callback = cfp.training.Metrics(
            metrics=list(cfg.training.callbacks.metrics),
        )
        callbacks.append(metrics_callback)
        print(f"  Added Metrics callback: {list(cfg.training.callbacks.metrics)}")

    if cfg.training.callbacks.use_wandb:
        wandb_callback = cfp.training.WandbLogger(
            project=cfg.training.callbacks.wandb_project,
            out_dir=cfg.training.callbacks.wandb_dir,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        callbacks.append(wandb_callback)
        print(f"  Added WandB callback: {cfg.training.callbacks.wandb_project}")

    print("\n[7/7] Training...")
    print(f"  Iterations: {cfg.training.num_iterations}")
    print(f"  Batch size: {cfg.training.batch_size}")
    print(f"  Validation batch size: {cfg.training.validation_batch_size}")
    print(f"  Validation frequency: {cfg.training.logging.eval_every}")
    print(f"  Learning rate: {cfg.training.optimizer.learning_rate}")
    print(f"  Multi-steps: {cfg.training.optimizer.multi_steps}")
    print(f"  EMA: {cfg.training.ema}")
    print()
    print_memory_usage("Before starting training")

    cf.train(
        num_iterations=cfg.training.num_iterations,
        batch_size=cfg.training.batch_size,
        valid_freq=cfg.training.logging.eval_every,
        validation_batch_size=cfg.training.validation_batch_size,
        callbacks=callbacks,
    )

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    if cfg.training.get("save_model", False):
        import os
        out_dir = cfg.training.out_dir
        os.makedirs(out_dir, exist_ok=True)

        run_name = f"run_{cfg.seed}"
        if cfg.training.callbacks.use_wandb:
            import wandb
            if wandb.run is not None:
                run_name = wandb.run.name

        print(f"\nSaving model to {out_dir}...")
        cf.save(out_dir, file_prefix=run_name)
        print(f"Model saved as {run_name}")

        split_info = {
            "seed": cfg.seed,
            "split_by": list(cfg.data.get("split_by", [])),
            "split_ratios": list(cfg.data.get("split_ratios", [])),
            "num_datasets": len(dataset_names),
            "dataset_names": dataset_names,
        }
        import json
        split_path = os.path.join(out_dir, f"splits_{run_name}.json")
        with open(split_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"Split info saved to {split_path}")


if __name__ == "__main__":
    train()

