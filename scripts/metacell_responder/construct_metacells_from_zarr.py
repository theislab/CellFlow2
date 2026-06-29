"""
construct_metacells_from_zarr.py
================================

Construct SEACells metacells DIRECTLY from a GroupedDistribution zarr of raw
single cells (e.g. sciplex3_X_scconcept.zarr / sciplex_raw_concept.zarr) and
write a metacell GroupedDistribution zarr with the SAME condition structure.

This is the zarr-native equivalent of `build_perturbation_metacells.py` (which
operates on an h5ad). In that script each (cell_line × drug [× dose]) group is
aggregated; here each TARGET distribution of the zarr already *is* one such
group (its cells = the perturbed single cells of one condition), so we simply
run SEACells on every target distribution.

What it does
------------
  • SOURCE (control) distributions  -> kept as raw single cells (unchanged).
  • TARGET (perturbed) distributions -> replaced by metacells = mean embeddings
    of SEACells hard-assignments (~`--cells-per-metacell` cells each).
  • Per-distribution condition vectors + the annotation are preserved, so a
    model trained on the output is directly comparable to the raw input zarr.
  • Groups too small for SEACells collapse to a single mean (no-op aggregation).
  • Optional per-distribution .npy checkpoints (`--checkpoint-dir`) -> resumable.

Pipeline position
-----------------
    raw single-cell zarr  --[THIS SCRIPT]-->  metacell zarr
    metacell zarr         --[define_responder_metacells.py]-->  responder labels

ENV
---
Needs BOTH `scaleflow` (zarr 3.x I/O) AND `SEACells`. SEACells is normally in the
`bio-agent` env (zarr 2.x) while GroupedDistribution I/O needs `cfp` (zarr 3.x).
If no single env has both, use the proven h5ad route instead
(build_perturbation_metacells.py [bio-agent] -> process_metacell_*_to_zarr.py [cfp]).
SEACells is deferred-imported, so everything except the actual aggregation runs
without it.

    conda run -n <env> python scripts/construct_metacells_from_zarr.py \
        --input  outputs/sciplex_raw_concept.zarr \
        --output outputs/sciplex_metacell_from_zarr.zarr \
        --cells-per-metacell 25 \
        --checkpoint-dir outputs/metacell_from_zarr_ckpts
"""
import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from scaleflow.data import GroupedDistribution

# defaults (match build_perturbation_metacells.py)
CELLS_PER_METACELL     = 25
MIN_CELLS_FOR_SEACELLS = 50
N_PCA_COMPONENTS       = 50
MAX_CELLS_FOR_KERNEL   = 20_000
SEED                   = 42
CHUNK_SIZE             = 131072
SHARD_SIZE             = CHUNK_SIZE * 8


def cells_to_metacells(emb: np.ndarray, *, cells_per_metacell: int, min_cells: int,
                       n_pca: int, max_kernel: int, seed: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Aggregate one distribution's single cells (emb: [n_cells, d]) into metacells
    = mean embeddings of SEACells hard-assignments. Identical core to
    build_perturbation_metacells.build_metacells."""
    import SEACells  # deferred: only needed for real aggregation

    n = emb.shape[0]
    # too few cells for SEACells -> collapse to a single mean "metacell"
    if n <= min_cells or (n // cells_per_metacell) < 2:
        return emb.mean(axis=0, keepdims=True).astype(np.float32)

    if n > max_kernel:                       # cap kernel cost on huge groups
        emb = emb[rng.choice(n, size=max_kernel, replace=False)]
        n = emb.shape[0]

    n_meta = max(2, n // cells_per_metacell)
    n_comp = min(n_pca, n - 1, emb.shape[1])
    emb_pca = PCA(n_components=n_comp, random_state=seed).fit_transform(emb)

    sub = ad.AnnData(
        X=np.zeros((n, 1), dtype=np.float32),
        obs=pd.DataFrame(index=[str(i) for i in range(n)]),
        obsm={"X_pca": emb_pca.astype(np.float32)},
    )
    model = SEACells.core.SEACells(
        sub, build_kernel_on="X_pca", n_SEACells=n_meta,
        n_neighbors=min(15, n - 1), verbose=False,
    )
    model.construct_kernel_matrix()
    model.initialize_archetypes()
    model.fit(min_iter=5, max_iter=30)
    assign = model.get_hard_assignments()["SEACell"]
    return np.stack([
        emb[sub.obs_names.get_indexer(members)].mean(axis=0)
        for _, members in assign.groupby(assign).groups.items()
    ]).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="raw single-cell GroupedDistribution zarr")
    p.add_argument("--output", required=True, help="output metacell zarr")
    p.add_argument("--cells-per-metacell", type=int, default=CELLS_PER_METACELL)
    p.add_argument("--min-cells",          type=int, default=MIN_CELLS_FOR_SEACELLS)
    p.add_argument("--n-pca",              type=int, default=N_PCA_COMPONENTS)
    p.add_argument("--max-kernel-cells",   type=int, default=MAX_CELLS_FOR_KERNEL)
    p.add_argument("--seed",               type=int, default=SEED)
    p.add_argument("--checkpoint-dir",     default=None, help="dir for per-distribution .npy checkpoints (resumable)")
    a = p.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    print(f"loading {a.input} …", flush=True)
    gd = GroupedDistribution.read_zarr(a.input)
    gd.to_memory()
    ann, dat = gd.annotation, gd.data
    tgt_idxs = list(dat.tgt_data.keys())
    print(f"  {len(dat.src_data)} source (control) distributions (kept raw)", flush=True)
    print(f"  {len(tgt_idxs)} target distributions to aggregate", flush=True)

    ckpt_dir = Path(a.checkpoint_dir) if a.checkpoint_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_cells_in = n_meta_out = 0
    for gi, tgt_idx in enumerate(tgt_idxs):
        labels = ann.tgt_dist_idx_to_labels[tgt_idx]
        emb = np.asarray(dat.tgt_data[tgt_idx], dtype=np.float32)
        n_cells_in += emb.shape[0]

        ckpt = ckpt_dir / f"tgt_{tgt_idx:06d}.npy" if ckpt_dir else None
        if ckpt is not None and ckpt.exists():
            meta = np.load(ckpt)
        else:
            meta = cells_to_metacells(
                emb, cells_per_metacell=a.cells_per_metacell, min_cells=a.min_cells,
                n_pca=a.n_pca, max_kernel=a.max_kernel_cells, seed=a.seed, rng=rng,
            )
            if ckpt is not None:
                np.save(ckpt, meta)

        dat.tgt_data[tgt_idx] = meta            # replace single cells with metacells
        n_meta_out += meta.shape[0]
        if gi % 50 == 0 or gi == len(tgt_idxs) - 1:
            print(f"  [{gi + 1}/{len(tgt_idxs)}] {labels}: {emb.shape[0]} cells -> {meta.shape[0]} metacells", flush=True)

    print(f"\naggregated {n_cells_in:,} perturbed cells -> {n_meta_out:,} metacells", flush=True)
    print(f"writing {a.output} …", flush=True)
    gd.write_zarr(a.output, chunk_size=CHUNK_SIZE, shard_size=SHARD_SIZE)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
