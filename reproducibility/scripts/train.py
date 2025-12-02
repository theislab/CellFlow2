import functools
import sys
from pathlib import Path

import hydra
import optax
from omegaconf import DictConfig, OmegaConf
from ott.solvers import utils as solver_utils

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import scaleflow as cfp


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration."""
    print("=" * 80)
    print("CellFlow Training with Hydra")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    print("\n[1/7] Loading data...")
    import anndata as ad
    import h5py

    with h5py.File(cfg.data.adata_path, "r") as f:
        adata = ad.AnnData(
            obs=ad.io.read_elem(f["obs"]),
            var=ad.io.read_elem(f["var"]),
            varm=ad.io.read_elem(f["varm"]),
            uns=ad.io.read_elem(f["uns"]),
            obsm=ad.io.read_elem(f["obsm"]),
        )

    print(f"Loaded AnnData: {adata.shape}")
    print(f"  Control cells: {adata.obs['control'].sum()}")
    print(f"  Perturbed cells: {(~adata.obs['control']).sum()}")

    print("\n[2/7] Preparing data with DataManager...")
    from scaleflow.data import DataManager, ReservoirSampler
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

    gd_full = dm.prepare_data(adata=adata, verbose=True)

    print("\nGroupedDistribution prepared:")
    print(f"  Source distributions: {len(gd_full.data.src_data)}")
    print(f"  Target distributions: {len(gd_full.data.tgt_data)}")

    print("\n[3/7] Splitting data...")
    if cfg.data.get("split_by"):
        from scaleflow.data import GroupedDistribution, GroupedDistributionAnnotation
        from scaleflow.data._data_splitter import AnnotationSplitter

        split_column = list(cfg.data.split_by)[0]

        splitter = AnnotationSplitter(
            annotation=gd_full.annotation,
            holdout_combinations=False,
            split_by=list(cfg.data.split_by),
            split_key="split",
            force_training_values={},
            ratios=list(cfg.data.get("split_ratios", [0.7, 0.15, 0.15])),
            random_state=cfg.seed,
        )

        split_df_unique = splitter._split_df()

        train_values = split_df_unique[split_df_unique['split'] == 'train'][split_column].unique().tolist()
        val_values = split_df_unique[split_df_unique['split'] == 'val'][split_column].unique().tolist()
        test_values = split_df_unique[split_df_unique['split'] == 'test'][split_column].unique().tolist()

        print("  Split assignment:")
        print(f"    Train {split_column}s ({len(train_values)}): {train_values}")
        print(f"    Val {split_column}s ({len(val_values)}): {val_values}")
        print(f"    Test {split_column}s ({len(test_values)}): {test_values}")

        split_df = gd_full.annotation.src_tgt_dist_df.merge(
            split_df_unique[[*splitter.split_by, 'split']],
            on=splitter.split_by,
            how='left'
        )

        splits_data = gd_full.split_by_dist_df(split_df, "split")

        def create_split_annotation(split_name):
            filtered_df = split_df[split_df['split'] == split_name].copy()
            involved_src_dists = set(filtered_df['src_dist_idx'].unique())
            involved_tgt_dists = set(filtered_df['tgt_dist_idx'].unique())
            filtered_src_labels = {k: v for k, v in gd_full.annotation.src_dist_idx_to_labels.items() if k in involved_src_dists}
            filtered_tgt_labels = {k: v for k, v in gd_full.annotation.tgt_dist_idx_to_labels.items() if k in involved_tgt_dists}
            return GroupedDistributionAnnotation(
                old_obs_index=gd_full.annotation.old_obs_index,
                src_dist_idx_to_labels=filtered_src_labels,
                tgt_dist_idx_to_labels=filtered_tgt_labels,
                src_tgt_dist_df=filtered_df,
                tgt_dist_keys=gd_full.annotation.tgt_dist_keys,
                src_dist_keys=gd_full.annotation.src_dist_keys,
                dist_flag_key=gd_full.annotation.dist_flag_key,
                default_values=gd_full.annotation.default_values,
                condition_structure=gd_full.annotation.condition_structure,
            )

        train_data = GroupedDistribution(
            data=splits_data['train'],
            annotation=create_split_annotation('train')
        )
        val_data = GroupedDistribution(
            data=splits_data['val'],
            annotation=create_split_annotation('val')
        )
        test_data = GroupedDistribution(
            data=splits_data['test'],
            annotation=create_split_annotation('test')
        )

        print(f"  Training samples: {len(train_data.data.src_data)} sources")
        print(f"  Validation samples: {len(val_data.data.src_data)} sources")
        print(f"  Test samples: {len(test_data.data.src_data)} sources")
    else:
        train_data = gd_full
        val_data = None
        test_data = None
        print("  No splitting - using all data for training")

    print("\n[4/7] Initializing CellFlow...")
    cf = cfp.model.CellFlow(adata, solver=cfg.solver.type)
    cf.train_data = train_data

    first_src_idx = next(iter(train_data.data.src_data.keys()))
    cf._data_dim = train_data.data.src_data[first_src_idx].shape[-1]

    train_data.max_combination_length = 1

    cf._dataloader = ReservoirSampler(
        data=train_data,
        batch_size=cfg.training.batch_size
    )

    if val_data is not None:
        cf._validation_data["val"] = val_data
    if test_data is not None:
        cf._validation_data["test"] = test_data
    cf._dm = None

    print("\n[5/7] Preparing model...")

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

    import jax
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(cf._solver.vf_state.params))

    print("\nModel architecture:")
    print(f"  Conditioning: {cfg.architecture.conditioning.method}")
    print(f"  Cell encoder: {cfg.architecture.cell_encoder.dims}")
    print(f"  Time encoder: {cfg.architecture.time_encoder.dims}")
    print(f"  Decoder: {cfg.architecture.decoder.dims}")
    print(f"  Condition embedding: {cfg.condition_encoder.embedding_dim}")
    print(f"\nModel info:")
    print(f"  Input data dimension: {cf._data_dim}")
    print(f"  Total trainable parameters: {num_params:,}")

    print("\n[6/6] Setting up callbacks...")
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
        }
        import json
        split_path = os.path.join(out_dir, f"splits_{run_name}.json")
        with open(split_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"Split info saved to {split_path}")


if __name__ == "__main__":
    train()

