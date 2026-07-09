"""Qualitatively compare trained CellFlow models on the highest-effect held-out test conditions.

Loads N run checkpoints (label, ckpt dir, conditioning, ablation), rebuilds each model exactly as
train_comparison.py did, restores its best params, predicts the sciplex held-out test set, then draws
a per-condition UMAP (utils.umap_effect_panels) for the top-k conditions by true effect:
control (grey) + true-perturbed (blue) + each model's prediction (colour) over a light-grey backdrop
of the cell line's other perturbations — so you can *see* whether prophet's better effect-size
calibration is real (its predicted cloud reaches the true-perturbed cloud rather than under-shooting).

    python experiments/model_comparison.py                       # 3 adaln runs (prophet/no/random), latent UMAP
    python experiments/model_comparison.py --decode --k 8        # UMAP in decoded gene space, 8 conditions
"""
import os

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/train_comparison/jax_cache")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "5")

import argparse
from collections import defaultdict
from functools import partial
from pathlib import Path

import jax
import numpy as np
import optax
from omegaconf import OmegaConf

from scaleflow.data import split_datasets
from scaleflow.data._dataloader import ReservoirSampler, ValidationSampler
from scaleflow.model import ScaleFlow
from scaleflow.model._recon import ReconDecoder
from scaleflow.utils import match_linear

import utils
import callbacks
from train_comparison import build_gd_from_h5ad

HERE = Path(__file__).resolve().parent
CFG_DIR = str(HERE / "config")
OUT = Path("/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/train_comparison")

# (label, checkpoint dir under OUT, conditioning group, ablation group, condition_encoder group)
# Transformer (prophet/default @ cond_output_dropout 0.4) vs the best MLP-encoder baselines
# (prophet @ 0.9 and 0.4). Each run is rebuilt with its own encoder group so the orbax restore
# matches the trained architecture (mlp → single-MHA pool w/ qkv_dim=64; transformer → 2-layer CLS).
RUNS = [
    ("tf_prophet_do0.4",  "model_prophet_7y61v13v_best_ckpt", "adaln_zero", "prophet", "transformer"),
    ("tf_default_do0.4",  "model_default_ace0pr9w_best_ckpt", "adaln_zero", "default", "transformer"),
    ("mlp_prophet_do0.9", "model_prophet_0olbggwi_best_ckpt", "adaln_zero", "prophet", "mlp"),
    ("mlp_prophet_do0.4", "model_prophet_lfhyawlu_best_ckpt", "adaln_zero", "prophet", "mlp"),
]


def load_cfg(conditioning: str, ablation: str, encoder: str = "transformer"):
    """Compose the train_comparison config with this run's conditioning + ablation + encoder groups."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        return compose(config_name="train_comparison",
                       overrides=[f"conditioning={conditioning}", f"ablation={ablation}",
                                  f"condition_encoder={encoder}"])


def load_solver(cfg, gd_train, transform, ckpt_path: Path):
    """Rebuild the model as train_comparison.run() does, restore best params from the orbax ckpt."""
    import orbax.checkpoint as ocp
    bs = int(cfg.training.batch_size)
    samp = ReservoirSampler(gd_train, np.random.default_rng(0), batch_size=bs,
                            pool_fraction=float(cfg.training.pool_fraction),
                            replacement_prob=float(cfg.training.replacement_prob),
                            condition_transform=transform)
    samp.init_sampler()
    sample_batch = samp.sample()

    m, ce = cfg.model, cfg.model.condition_encoder
    encoder_arch = OmegaConf.to_container(ce.encoder_arch, resolve=True)
    layers_before_pool = {k: encoder_arch for k in sample_batch["condition"]}
    sf = ScaleFlow(solver=cfg.solver.solver_key)
    sf._validation_data["predict_kwargs"] = OmegaConf.to_container(cfg.solver.get("predict_kwargs", {}), resolve=True)
    sf.prepare_model(
        sample_batch=sample_batch,
        max_combination_length=int(m.max_combination_length),
        conditioning=m.conditioning_key,
        conditioning_kwargs=OmegaConf.to_container(m.conditioning_kwargs, resolve=True),
        pooling=ce.pooling,
        pooling_kwargs=OmegaConf.to_container(ce.pooling_kwargs, resolve=True),
        layers_before_pool=layers_before_pool,
        layers_after_pool=OmegaConf.to_container(ce.layers_after_pool, resolve=True),
        cond_output_dropout=float(ce.cond_output_dropout),
        hidden_dims=tuple(int(x) for x in m.hidden_dims),
        decoder_dims=tuple(int(x) for x in m.decoder_dims),
        condition_embedding_dim=int(m.condition_embedding_dim),
        match_fn=partial(match_linear, epsilon=float(cfg.match_fn.epsilon)),
        probability_path=OmegaConf.to_container(m.probability_path_kwargs, resolve=True),
        optimizer=optax.adam(1e-4),  # dummy — inference only
    )
    target = callbacks._solver_params(sf.solver)
    params = ocp.PyTreeCheckpointer().restore(str(ckpt_path), item=target)
    callbacks.restore_solver_params(sf.solver, params)
    return sf.solver, sf._validation_data["predict_kwargs"]


def cellline_background(splits: dict, max_per_cl: int = 5000, seed: int = 0):
    """Pool each cell line's cells (control + all perturbations, all splits) → light-grey UMAP backdrop."""
    rng = np.random.default_rng(seed)
    pools = defaultdict(list)
    for sp in splits.values():
        for part in ("train", "val", "test"):
            gd = sp.get(part)
            if gd is None:
                continue
            for sidx, arr in gd.data.src_data.items():
                cl = str(gd.annotation.src_dist_idx_to_labels[sidx][0])
                pools[cl].append(np.asarray(arr, np.float32))
            for tidx, arr in gd.data.tgt_data.items():
                cl = str(gd.annotation.tgt_dist_idx_to_labels[tidx][0])
                pools[cl].append(np.asarray(arr, np.float32))
    bg = {}
    for cl, parts in pools.items():
        X = np.concatenate(parts, 0)
        bg[cl] = X[rng.choice(X.shape[0], max_per_cl, replace=False)] if X.shape[0] > max_per_cl else X
    return bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6, help="top-k highest-effect test conditions")
    ap.add_argument("--decode", action="store_true", help="UMAP in decoded gene space (else raw latent)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "model_comparison_umap.png"))
    args = ap.parse_args()

    base = load_cfg("adaln_zero", "default")   # data-defining config (shared across runs)

    print("Building GroupedDistributions …", flush=True)
    gds = {name: build_gd_from_h5ad(str(base.datasets[name].path), base.data, base.datasets[name])
           for name in base.selected_datasets}
    splits = {}
    for ds in gds:
        if bool(base.datasets[ds].get("holdout", True)):
            splits[ds] = split_datasets({ds: gds[ds]}, holdout_combinations=bool(base.split.holdout_combinations),
                                         split_by=list(base.split.by), split_key="split",
                                         ratios=list(base.split.ratios), random_state=int(base.split.random_state))[ds]
        else:
            splits[ds] = {"train": gds[ds], "val": None, "test": None}
    hold = [ds for ds in splits if splits[ds]["test"] is not None]
    ds0 = hold[0]
    gd_train, gd_test = splits[ds0]["train"], splits[ds0]["test"]
    print(f"  holdout dataset for eval: {ds0}")

    bg = cellline_background(splits, seed=args.seed)
    print(f"  background cell lines: {[(cl, v.shape[0]) for cl, v in bg.items()]}")

    src = true = None
    preds = {}
    for label, ckpt, cond, abl, enc in RUNS:
        ckpt_path = OUT / ckpt
        if not ckpt_path.exists():
            print(f"  !! skip {label}: {ckpt_path} missing"); continue
        print(f"Loading {label}  ({cond}/{abl}/{enc})  ← {ckpt}", flush=True)
        cfg = load_cfg(cond, abl, enc)
        transform = utils.ConditionTransform(abl, seed=int(cfg.seed)) if abl != "prophet" else None
        solver, pk = load_solver(cfg, gd_train, transform, ckpt_path)
        tsamp = ValidationSampler(gd_test, n_conditions_on_log_iteration=None,
                                  n_conditions_on_train_end=None, seed=args.seed, condition_transform=transform)
        tsamp.init_sampler()
        batch = tsamp.sample(mode="on_train_end")
        if src is None:
            src, true = batch["source"], batch["target"]   # same cells regardless of ablation
        preds[label] = jax.tree.map(partial(solver.predict, **pk), batch["source"], batch["condition"])
        print(f"  predicted {len(preds[label])} conditions")

    decoder = ReconDecoder.load(str(base.recon.decoder_path)) if args.decode else None
    space = "gene" if args.decode else "AE_128 latent"
    df = utils.umap_effect_panels(
        src, true, preds, background=bg, k=args.k, decoder=decoder, seed=args.seed,
        out_path=args.out, title=f"high-effect test conditions — {space} space",
    )
    csv = args.out.replace(".png", ".csv")
    df.to_csv(csv, index=False)
    print(f"\nTop conditions (effect + per-model effect-ratio) → {csv}")
    print(df.head(args.k).to_string(index=False))


if __name__ == "__main__":
    main()
