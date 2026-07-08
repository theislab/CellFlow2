"""
build_perturbation_metacells_X.py

Gene-space variant of build_perturbation_metacells.py.

Builds SEACells metacells whose VALUES are gene expression (adata.X, the 2000
log1p HVGs in sciplex3_with_emb.h5ad), while the metacell ASSIGNMENT / kernel is
built on a clean embedding (obsm['X_state'] by default) rather than on the noisy
gene matrix. Perturbed cells are aggregated into metacells per condition; control
cells are kept as individual single cells (they are the flow-matching source).

Why kernel on X_state, values on X:
  A neighbor graph over 2000 genes (or 58k) is dominated by technical noise, giving
  poor metacell assignments. A foundation-model embedding gives a much cleaner
  manifold. We assign on X_state, then average the gene expression (and X_state) of
  each SEACell's members. Pass --kernel-obsm X to build the kernel on gene PCA instead.

Output h5ad:
  - X   : gene expression (2000 HVG). Control cells raw; perturbed cells = metacell means.
  - obsm[X_state] : kept for controls; metacell = mean X_state over members.
  - var : the gene names (preserved, so a gene-space zarr can align genes).
  - uns : preserved (prophet_emb, cell_line_ccle_embeddings, drug_0_embeddings, log1p)
          so scripts/prepare_sciplex_prophet.py can build the paired zarr.
  - obs : cell_line, drug, dose_value, control (control=False for metacells).

A per-condition metacell-count CSV is written next to the output (or --log path):
  columns = group keys + n_cells, n_metacells, method.

Environment: SEACells (bio-agent conda env), NOT the jax container:
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate bio-agent

Reproduce sciplex HVG metacells (kernel on X_state, dose-resolved):
  python scripts/metacell/build_perturbation_metacells_X.py \
      --input  /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/sciplex3_with_emb.h5ad \
      --output /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/sciplex3_hvg_metacell.h5ad
"""
import argparse
import time
import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── defaults ────────────────────────────────────────────────────────────────
_BASE = "/lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow"
CELLS_PER_METACELL     = 75
MIN_CELLS_FOR_SEACELLS = 50
N_PCA_COMPONENTS       = 50
MAX_CELLS_FOR_KERNEL   = 20_000
SEED                   = 42
KERNEL_OBSM            = "X_state"   # embedding used for the neighbor graph / assignment
CTRL_KEY               = "control"
DRUG_KEY               = "drug"
DOSE_KEY               = "dose_value"
CELL_LINE_KEY          = "cell_line"


def _mean_dense(mat, rows):
    """Mean over `rows` of a (possibly sparse) matrix → 1-D float32 array."""
    sub = mat[rows]
    m = sub.mean(axis=0)
    return np.asarray(m).ravel().astype(np.float32)


def build_metacells(
    adata: ad.AnnData,
    group_keys: list[str],
    kernel_obsm: str = KERNEL_OBSM,
    ctrl_key: str = CTRL_KEY,
    cells_per_metacell: int = CELLS_PER_METACELL,
    min_cells_for_seacells: int = MIN_CELLS_FOR_SEACELLS,
    n_pca_components: int = N_PCA_COMPONENTS,
    max_cells_for_kernel: int = MAX_CELLS_FOR_KERNEL,
    seed: int = SEED,
    checkpoint_dir: Path | None = None,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Replace perturbed cells with SEACells metacells (gene-space values); keep controls raw.

    Assignment is computed on ``obsm[kernel_obsm]``; each metacell stores the mean gene
    expression (adata.X) AND the mean kernel embedding of its member cells.
    Returns (combined AnnData, per-condition count DataFrame).
    """
    import SEACells  # deferred: only available in the bio-agent env

    rng = np.random.default_rng(seed)
    ctrl_mask = adata.obs[ctrl_key].to_numpy().astype(bool)
    ctrl = adata[ctrl_mask].copy()
    pert = adata[~ctrl_mask]

    Xg = pert.X                                   # (n_pert, G) genes, sparse
    Xk = np.asarray(pert.obsm[kernel_obsm])       # (n_pert, K) kernel embedding, dense
    groups = pert.obs.groupby(list(group_keys), observed=True).indices

    meta_gene_rows: list[np.ndarray] = []
    meta_kern_rows: list[np.ndarray] = []
    meta_obs: dict[str, list] = {k: [] for k in group_keys}
    log_rows: list[dict] = []

    for gi, (label, idx) in enumerate(groups.items()):
        idx = np.asarray(idx)
        n = len(idx)
        label_tuple = label if isinstance(label, tuple) else (label,)
        label_strs = np.array([str(v) for v in label_tuple])
        print(f"  [{gi + 1}/{len(groups)}] {dict(zip(group_keys, label_tuple))}: {n} cells", flush=True)

        # ── crash-recovery checkpoint (per group) ──
        ckpt = checkpoint_dir / f"group_{gi:04d}.npz" if checkpoint_dir else None
        if ckpt and ckpt.exists():
            cached = np.load(ckpt, allow_pickle=True)
            if np.array_equal(cached["label"], label_strs):
                g_rows, k_rows, method = cached["Xg"], cached["Xk"], str(cached["method"])
                for gr, kr in zip(g_rows, k_rows):
                    meta_gene_rows.append(gr); meta_kern_rows.append(kr)
                    for k, v in zip(group_keys, label_tuple):
                        meta_obs[k].append(v)
                log_rows.append({**dict(zip(group_keys, label_tuple)),
                                 "n_cells": int(cached["n_cells"]), "n_metacells": int(g_rows.shape[0]),
                                 "method": method})
                print(f"    -> resumed {g_rows.shape[0]} metacell(s)", flush=True)
                continue

        if n <= min_cells_for_seacells or (n // cells_per_metacell) < 2:
            # too few cells for SEACells: collapse the whole condition to a single mean
            g_rows = _mean_dense(Xg, idx).reshape(1, -1)
            k_rows = Xk[idx].mean(axis=0, keepdims=True).astype(np.float32)
            method = "single_mean"
        else:
            sub_idx = idx
            if n > max_cells_for_kernel:
                sub_idx = rng.choice(idx, size=max_cells_for_kernel, replace=False)
            m = len(sub_idx)
            emb = Xk[sub_idx]
            n_meta = max(2, m // cells_per_metacell)
            n_comp = min(n_pca_components, m - 1, emb.shape[1])
            emb_pca = PCA(n_components=n_comp, random_state=seed).fit_transform(emb)
            sub = ad.AnnData(
                X=np.zeros((m, 1), dtype=np.float32),
                obs=pd.DataFrame(index=[str(i) for i in range(m)]),
                obsm={"X_pca": emb_pca.astype(np.float32)},
            )
            model = SEACells.core.SEACells(
                sub, build_kernel_on="X_pca", n_SEACells=n_meta,
                n_neighbors=min(15, m - 1), verbose=False,
            )
            model.construct_kernel_matrix()
            model.initialize_archetypes()
            model.fit(min_iter=5, max_iter=30)
            assign = model.get_hard_assignments()["SEACell"]
            g_list, k_list = [], []
            for _, members in assign.groupby(assign).groups.items():
                pos = sub.obs_names.get_indexer(members)   # positions into sub / sub_idx
                cell_ids = sub_idx[pos]
                g_list.append(_mean_dense(Xg, cell_ids))
                k_list.append(Xk[cell_ids].mean(axis=0).astype(np.float32))
            g_rows = np.stack(g_list)
            k_rows = np.stack(k_list)
            method = "seacells"

        g_rows = g_rows.astype(np.float32)
        k_rows = k_rows.astype(np.float32)
        for gr, kr in zip(g_rows, k_rows):
            meta_gene_rows.append(gr); meta_kern_rows.append(kr)
            for k, v in zip(group_keys, label_tuple):
                meta_obs[k].append(v)
        log_rows.append({**dict(zip(group_keys, label_tuple)),
                         "n_cells": n, "n_metacells": int(g_rows.shape[0]), "method": method})

        if ckpt:
            np.savez(ckpt, Xg=g_rows, Xk=k_rows, label=label_strs, n_cells=n, method=method)

    # ── assemble output: control single cells + perturbed metacells ──
    meta_Xg = sp.csr_matrix(np.stack(meta_gene_rows).astype(np.float32))
    meta_Xk = np.stack(meta_kern_rows).astype(np.float32)
    meta_obs_df = pd.DataFrame(meta_obs)
    meta_obs_df[ctrl_key] = False
    for k in group_keys:
        if adata.obs[k].dtype.name == "category":
            meta_obs_df[k] = pd.Categorical(meta_obs_df[k], categories=adata.obs[k].cat.categories)

    meta_adata = ad.AnnData(
        X=meta_Xg, obs=meta_obs_df, obsm={kernel_obsm: meta_Xk}, var=adata.var.copy()
    )
    combined = ad.concat([ctrl, meta_adata], join="outer")
    combined.var = adata.var.copy()               # preserve gene names
    combined.uns = dict(adata.uns)                 # preserve prophet_emb / embeddings / log1p
    combined.obs[ctrl_key] = combined.obs[ctrl_key].fillna(False).astype(bool)
    return combined, pd.DataFrame(log_rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Build gene-space SEACells metacells (values=adata.X, kernel=embedding).")
    p.add_argument("--input",  default=f"{_BASE}/sciplex3_with_emb.h5ad")
    p.add_argument("--output", default=f"{_BASE}/sciplex3_hvg_metacell.h5ad")
    p.add_argument("--log", default=None, help="per-condition metacell-count CSV (default: <output_stem>_metacell_counts.csv)")
    p.add_argument("--kernel-obsm", default=KERNEL_OBSM,
                   help=f"obsm key for the assignment kernel (default: {KERNEL_OBSM}; use 'X' for gene PCA).")
    p.add_argument("--cell-line", default=None, help="Filter to a single cell line (e.g. A549). Default: all lines.")
    p.add_argument("--group-by", nargs="+", default=[DRUG_KEY, DOSE_KEY],
                   help=f"obs columns defining a condition (default: {DRUG_KEY} {DOSE_KEY}). cell_line auto-added if --cell-line unset.")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--ctrl-key", default=CTRL_KEY)
    p.add_argument("--cell-line-key", default=CELL_LINE_KEY)
    p.add_argument("--cells-per-metacell", type=int, default=CELLS_PER_METACELL)
    p.add_argument("--min-cells",          type=int, default=MIN_CELLS_FOR_SEACELLS)
    p.add_argument("--n-pca",              type=int, default=N_PCA_COMPONENTS)
    p.add_argument("--max-kernel-cells",   type=int, default=MAX_CELLS_FOR_KERNEL)
    p.add_argument("--seed",               type=int, default=SEED)
    args = p.parse_args()

    t0 = time.time()
    input_path, output_path = Path(args.input), Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else output_path.with_name(f"{output_path.stem}_metacell_counts.csv")
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_path.parent / f"{output_path.stem}_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── load X (genes) + var + kernel embedding + obs + uns ──
    print(f"loading {input_path} …", flush=True)
    with h5py.File(input_path, "r") as f:
        obs = ad.io.read_elem(f["obs"])
        var = ad.io.read_elem(f["var"])
        X   = ad.io.read_elem(f["X"])                       # sparse CSR (n, G) genes
        Xk  = f["obsm"][args.kernel_obsm][:]                # dense (n, K) kernel embedding
        uns = ad.io.read_elem(f["uns"])
    print(f"  n_obs={obs.shape[0]:,}  genes={var.shape[0]:,}  kernel[{args.kernel_obsm}]={Xk.shape}", flush=True)

    # ── cell-line filter ──
    is_ctrl = obs[args.ctrl_key].to_numpy().astype(bool)
    if args.cell_line:
        is_cl = (obs[args.cell_line_key] == args.cell_line).to_numpy()
    else:
        is_cl = np.ones(len(obs), dtype=bool)
    keep = is_cl                                            # controls + perturbed of this line
    if args.cell_line:
        print(f"  cell-line filter '{args.cell_line}': {keep.sum():,} cells", flush=True)

    obs_sub = obs.loc[keep].copy().reset_index(drop=True)
    adata = ad.AnnData(X=X[keep], obs=obs_sub, var=var, obsm={args.kernel_obsm: Xk[keep].astype(np.float32)}, uns=uns)

    # ── grouping keys ──
    group_keys = list(args.group_by)
    if not args.cell_line and args.cell_line_key in adata.obs.columns and args.cell_line_key not in group_keys:
        group_keys.insert(0, args.cell_line_key)
    for k in group_keys:
        if adata.obs[k].dtype.name != "category":
            adata.obs[k] = adata.obs[k].astype("category")

    n_ctrl = int(is_ctrl[keep].sum())
    n_pert = int((~is_ctrl[keep]).sum())
    print(f"\nbuilding metacells (SEACells)  kernel={args.kernel_obsm}  group_by={group_keys}", flush=True)
    print(f"  {n_ctrl:,} control cells (kept raw) + {n_pert:,} perturbed cells → metacells", flush=True)

    adata_meta, counts = build_metacells(
        adata, group_keys=group_keys, kernel_obsm=args.kernel_obsm, ctrl_key=args.ctrl_key,
        cells_per_metacell=args.cells_per_metacell, min_cells_for_seacells=args.min_cells,
        n_pca_components=args.n_pca, max_cells_for_kernel=args.max_kernel_cells,
        seed=args.seed, checkpoint_dir=ckpt_dir,
    )
    if args.cell_line and args.cell_line_key not in adata_meta.obs.columns:
        adata_meta.obs[args.cell_line_key] = args.cell_line

    # ── per-condition metacell-count log + summary ──
    counts.to_csv(log_path, index=False)
    nmc = counts["n_metacells"].to_numpy()
    print(f"\n  metacell-count log → {log_path}", flush=True)
    print(f"  conditions: {len(counts)}   total metacells: {int(nmc.sum())}", flush=True)
    print(f"  metacells/condition  min={nmc.min()}  median={int(np.median(nmc))}  max={nmc.max()}", flush=True)
    print(f"  single-mean (degenerate, n_metacells==1): {(nmc == 1).sum()} conditions", flush=True)

    n_ctrl_out = int(adata_meta.obs[args.ctrl_key].sum())
    n_meta_out = int((~adata_meta.obs[args.ctrl_key]).sum())
    print(f"\n  output: {n_ctrl_out:,} control single cells + {n_meta_out:,} metacells  (genes={adata_meta.n_vars:,})", flush=True)
    print(f"\nwriting {output_path} …", flush=True)
    adata_meta.write_h5ad(output_path)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
