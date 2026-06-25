"""Train a reconstruction embedding/decoder and save the weights (+ learned embedding).

Two modes (set ``mode`` in ``config/recon_emb.yaml``):

1. ``ae``         — train an :class:`~scaleflow.model._recon.Autoencoder`
                    (genes -> latent -> genes) with a chosen latent size and encoder/decoder
                    architecture. Saves the (encoder+decoder) weights AND writes the learned
                    per-cell embedding into an h5ad as ``AE_<latent_dim>`` (obsm).
2. ``pretrained`` — train a :class:`~scaleflow.model._recon.Decoder` mapping an existing h5ad
                    embedding (``pretrained_key``, e.g. ``X_state``) -> genes. Saves the
                    decoder weights.

The **train/val/test split is inherited from ``train_zarr.py``** (the same ``split: drug``
config group): a held-out-drug split — unique treated drugs are shuffled with
``split.random_state`` and partitioned by ``split.ratios``; control cells go to train. The
model trains on the train split, and **pure reconstruction** metrics are reported on train,
val and test. Weights go to ``weights_dir/<run_name>/`` (OUTSIDE the repo).

    python experiments/recon_emb.py                                    # AE mode (defaults)
    python experiments/recon_emb.py mode=pretrained pretrained_key=X_state
    python experiments/recon_emb.py --multirun latent_dim=10,32,128
"""
import os

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import json
import time
from pathlib import Path

import h5py
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
from flax import serialization
from omegaconf import DictConfig, OmegaConf

import recon_metrics as rm
from scaleflow.model._recon import Autoencoder, Decoder, ReconDecoder

# anndata moved write_elem across versions; import defensively.
try:
    from anndata.io import write_elem
except Exception:  # noqa: BLE001
    try:
        from anndata.experimental import write_elem
    except Exception:  # noqa: BLE001
        from anndata._io.specs import write_elem

SPLITS = ("train", "val", "test")


# ──────────────────────────────── data ────────────────────────────────
def load_and_preprocess(cfg: DictConfig):
    """Read the h5ad, optional subsample, then normalize_total + log1p + optional HVG."""
    print(f"Reading {cfg.data_path} …")
    adata = sc.read_h5ad(cfg.data_path)
    print(f"  loaded: {adata.shape}")
    n_sub = cfg.get("n_subsample", None)
    if n_sub:
        rng = np.random.default_rng(int(cfg.seed))
        idx = np.sort(rng.choice(adata.n_obs, size=min(int(n_sub), adata.n_obs), replace=False))
        adata = adata[idx].copy()
        print(f"  subsampled: {adata.shape}")
    sc.pp.normalize_total(adata, target_sum=float(cfg.preprocess.target_sum))
    sc.pp.log1p(adata)
    ntg = cfg.preprocess.get("n_top_genes", None)
    if ntg:
        sc.pp.highly_variable_genes(adata, n_top_genes=int(ntg))
        adata = adata[:, adata.var.highly_variable].copy()
        print(f"  HVG -> {adata.shape}")
    return adata


def to_dense_f32(X) -> np.ndarray:
    return np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)


def assign_splits(adata, split_by: list[str], ratios: list[float], random_state: int) -> np.ndarray:
    """Held-out-condition split replicating ``scaleflow``'s ``GroupedDistributionSplitter``.

    Unique combinations of ``split_by`` (over treated cells) are shuffled with
    ``random_state`` and partitioned into train/val/test by ``ratios``; **control cells go to
    train**. Returns a per-cell array of "train"/"val"/"test". This reproduces the same split
    config ``train_zarr.py`` uses (``split/drug.yaml``).
    """
    keys = list(split_by)
    treated = ~np.asarray(adata.obs["control"].values).astype(bool)
    obs = adata.obs[keys].astype(str).reset_index(drop=True)

    df_unique = obs[treated].drop_duplicates(subset=keys).copy()
    n = len(df_unique)
    train_size = round(ratios[0] * n)
    val_size = round(ratios[1] * n)
    test_size = n - train_size - val_size
    if min(train_size, val_size, test_size) == 0:
        raise ValueError(f"a split is empty for n={n} unique combos and ratios={list(ratios)}")

    # 1st shuffle -> first train_size go to train, rest to test_val
    df_unique["split"] = "test_val"
    sh = df_unique.sample(frac=1, random_state=random_state).reset_index(drop=True)
    sh.loc[: train_size - 1, "split"] = "train"
    # 2nd shuffle of the test_val remainder -> test then val
    tv = sh[sh["split"] == "test_val"].copy().sample(frac=1, random_state=random_state).reset_index(drop=True)
    tv.loc[: test_size - 1, "split"] = "test"
    tv.loc[test_size:, "split"] = "val"

    combo2split = {tuple(r[k] for k in keys): "train" for _, r in sh[sh["split"] == "train"].iterrows()}
    combo2split.update({tuple(r[k] for k in keys): r["split"] for _, r in tv.iterrows()})

    cell_keys = list(zip(*[obs[k].values for k in keys]))
    splits = np.array(["train"] * adata.n_obs, dtype=object)
    splits[treated] = [combo2split[cell_keys[i]] for i in np.where(treated)[0]]
    return splits.astype(str)


# ──────────────────────────────── model ───────────────────────────────
def build_module(cfg: DictConfig, mode: str, gene_dim: int):
    """Return ``(module, input_key, run_name)`` (``input_key`` is the obsm key for decoders)."""
    dropout = float(cfg.dropout_rate)
    if mode == "ae":
        latent = int(cfg.latent_dim)
        module = Autoencoder(
            gene_dim=gene_dim, latent_dim=latent, dropout_rate=dropout,
            encoder_hidden=tuple(int(x) for x in cfg.encoder_hidden),
            decoder_hidden=tuple(int(x) for x in cfg.decoder_hidden),
        )
        return module, None, (cfg.get("run_name") or f"AE_{latent}")
    pkey = str(cfg.pretrained_key)
    module = Decoder(
        output_dim=gene_dim, dropout_rate=dropout,
        hidden_dims=tuple(int(x) for x in cfg.pretrained_decoder_hidden),
    )
    return module, pkey, (cfg.get("run_name") or f"dec_{pkey}")


def batched_apply(module, params, arr, *, method=None, batch=16384) -> np.ndarray:
    """Apply ``module`` over ``arr`` in batches (inference; deterministic)."""
    kwargs = {"method": method} if method is not None else {}
    outs = []
    for i in range(0, len(arr), batch):
        out = module.apply({"params": params}, jnp.asarray(arr[i:i + batch]), training=False, **kwargs)
        out = out[0] if isinstance(out, tuple) else out  # Autoencoder returns (recon, z)
        outs.append(np.asarray(out))
    return np.concatenate(outs, axis=0)


# ──────────────────────────────── io ──────────────────────────────────
def save_run(weights_dir: Path, module, params, meta: dict, losses) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)
    # unified, self-contained, picklable model (ScaleFlow-style .pkl): module + params + metadata
    ReconDecoder(module=module, params=jax.device_get(params), metadata=meta).save(str(weights_dir), overwrite=True)
    with open(weights_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(weights_dir / "train_loss.json", "w") as f:
        json.dump([float(x) for x in losses], f)
    print(f"  ReconDecoder.pkl + metadata -> {weights_dir}")


def _h5ad_n_obs(f: "h5py.File") -> int:
    obs = f["obs"]
    return obs[obs.attrs.get("_index", "_index")].shape[0]


def write_embedding(adata, key: str, Z: np.ndarray, out_path: str, source: str) -> None:
    """Store the learned embedding ``key`` in an h5ad (append in place if it exists, else write)."""
    Z = np.ascontiguousarray(np.asarray(Z, dtype=np.float32))
    adata.obsm[key] = Z
    if Path(out_path).exists():
        with h5py.File(out_path, "r+") as f:
            if Z.shape[0] != _h5ad_n_obs(f):
                raise ValueError(
                    f"embedding has {Z.shape[0]} rows but {out_path} has {_h5ad_n_obs(f)} cells "
                    f"— set embedding_out to a fresh path (you are likely subsampling)."
                )
            obsm = f.require_group("obsm")
            if key in obsm:
                del obsm[key]
            write_elem(obsm, key, Z)
        warn = "  (== data_path; source modified)" if out_path == source else ""
        print(f"  appended obsm['{key}'] {Z.shape} -> {out_path}{warn}")
    else:
        adata.write_h5ad(out_path)
        print(f"  wrote adata with obsm['{key}'] {Z.shape} -> {out_path}")


# ──────────────────────────────── run ─────────────────────────────────
@hydra.main(config_path="config", config_name="recon_emb", version_base=None)
def main(cfg: DictConfig) -> None:
    mode = str(cfg.mode)
    assert mode in ("ae", "pretrained"), f"mode must be 'ae' or 'pretrained', got {mode!r}"

    adata = load_and_preprocess(cfg)
    X = to_dense_f32(adata.X)
    gene_dim = int(X.shape[1])

    # inherit train_zarr's held-out-drug split
    splits = assign_splits(adata, list(cfg.split.by), list(cfg.split.ratios), int(cfg.split.random_state))
    idx = {s: np.where(splits == s)[0] for s in SPLITS}

    module, input_key, run_name = build_module(cfg, mode, gene_dim)
    inp = X if input_key is None else np.asarray(adata.obsm[input_key], dtype=np.float32)

    print(f"\n{'=' * 70}\n  recon_emb | mode={mode} | run={run_name} | "
          f"split_by={list(cfg.split.by)} ratios={list(cfg.split.ratios)}")
    print(f"  splits: " + "  ".join(f"{s}={len(idx[s])}" for s in SPLITS)
          + f" | in{inp.shape}->genes{X.shape}\n{'=' * 70}")

    # ── train on the train split ──
    t0 = time.perf_counter()
    state, losses = module.train(
        inp[idx["train"]], X[idx["train"]],
        n_iters=int(cfg.n_iters), batch_size=int(cfg.batch_size),
        lr=float(cfg.lr), seed=int(cfg.seed), log_every=int(cfg.log_every),
    )

    # ── pure-reconstruction metrics on train / val / test ──
    metrics = {}
    for s in SPLITS:
        pred = batched_apply(module, state.params, inp[idx[s]])
        metrics[s] = rm.pure_reconstruction(X[idx[s]], pred)
    print(f"\n  pure reconstruction ({run_name}):")
    print(f"  {'split':<6}{'n':>9}{'MSE':>10}{'recon_R2':>11}{'R2/gene':>10}{'med_r':>9}")
    for s in SPLITS:
        m = metrics[s]
        print(f"  {s:<6}{len(idx[s]):>9}{m['mse']:>10.4f}{m['reconstruction_r2']:>11.3f}"
              f"{m['reconstruction_r2_per_gene']:>10.3f}{m['median_per_gene_r']:>9.3f}")

    # ── save weights + metadata ──
    meta = {
        "mode": mode, "run_name": run_name, "data_path": str(cfg.data_path),
        "gene_dim": gene_dim, "dropout_rate": float(cfg.dropout_rate), "loss_kind": "mse",
        "preprocess": {"target_sum": float(cfg.preprocess.target_sum), "log1p": True,
                       "n_top_genes": cfg.preprocess.get("n_top_genes", None)},
        "split": {**OmegaConf.to_container(cfg.split, resolve=True),
                  "sizes": {s: int(len(idx[s])) for s in SPLITS}},
        "n_iters": int(cfg.n_iters), "metrics": metrics,
        "var_names": list(map(str, adata.var_names)),
    }
    if mode == "ae":
        meta.update(latent_dim=int(cfg.latent_dim), embedding_key=run_name,
                    encoder_hidden=list(map(int, cfg.encoder_hidden)),
                    decoder_hidden=list(map(int, cfg.decoder_hidden)))
    else:
        meta.update(pretrained_key=input_key, pretrained_dim=int(inp.shape[1]),
                    decoder_hidden=list(map(int, cfg.pretrained_decoder_hidden)))
    # the input this decoder expects: the AE latent (ae) or the pretrained embedding (pretrained)
    meta["input_key"] = run_name if mode == "ae" else input_key
    meta["input_dim"] = int(cfg.latent_dim) if mode == "ae" else int(inp.shape[1])
    save_run(Path(cfg.weights_dir) / run_name, module, state.params, meta, losses)

    # ── AE: encode all cells and store the embedding ──
    if mode == "ae":
        print("Encoding all cells -> embedding …")
        Z = batched_apply(module, state.params, X, method="encode")
        out_path = cfg.get("embedding_out") or str(cfg.data_path)
        write_embedding(adata, run_name, Z, out_path, source=str(cfg.data_path))

    n_params = sum(int(x.size) for x in jax.tree.leaves(state.params))
    print(f"  params: {n_params:,}  final loss: {losses[-1]:.4f}  "
          f"done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
