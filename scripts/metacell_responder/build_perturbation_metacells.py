"""
build_perturbation_metacells.py

Build SEACells metacells from a single-cell perturbation h5ad and save a
compact h5ad containing raw control cells + per-condition metacells.

Required input h5ad structure
──────────────────────────────
  adata.obs columns:
    - cell_line  : str / category  (e.g. "A549")
    - drug       : str / category  (e.g. "Trametinib"; "Vehicle" for controls)
    - control    : bool            (True = DMSO/vehicle control cell)
  adata.obsm["X_state"] : (n_cells, D) float  — foundation-model embedding

Output h5ad
──────────────────────────────
  Same obsm["X_state"] key; obs keeps cell_line, drug, control columns.
  Control cells are kept as-is; perturbed cells are replaced by metacells
  (mean embeddings of SEACell assignments).

Environment
──────────────────────────────
  Requires bio-agent conda env (SEACells + zarr 2.x):
    conda run -n bio-agent python scripts/build_perturbation_metacells.py [args]

Basic usage (Sciplex, reproduce original output):
    python scripts/build_perturbation_metacells.py \\
        --input     /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/sciplex3_prophet_filtered.h5ad \\
        --output    /lustre/groups/ml01/workspace/ken_wang/meta_cellflow/outputs/sciplex_a549_metacell.h5ad \\
        --cell-line A549 \\
        --obsm      X_state \\
        --drug-map  scripts/sci_to_tahoe_drugs.tsv

Filter to specific drugs only (pass a text file, one drug name per line):
    python scripts/build_perturbation_metacells.py \\
        --input     mydata.h5ad \\
        --output    out_metacells.h5ad \\
        --cell-line MCF7 \\
        --obsm      X_state \\
        --drug-list my_drugs.txt
"""
import argparse
import time
import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── default metacell knobs ─────────────────────────────────────────────────
CELLS_PER_METACELL     = 75
MIN_CELLS_FOR_SEACELLS = 50
N_PCA_COMPONENTS       = 50
MAX_CELLS_FOR_KERNEL   = 20_000
SEED                   = 42
EMB_KEY                = "X_state"
CTRL_KEY               = "control"
DRUG_KEY               = "drug"
CELL_LINE_KEY          = "cell_line"


def build_metacells(
    adata: ad.AnnData,
    group_keys: list[str],
    emb_key: str = EMB_KEY,
    ctrl_key: str = CTRL_KEY,
    cells_per_metacell: int = CELLS_PER_METACELL,
    min_cells_for_seacells: int = MIN_CELLS_FOR_SEACELLS,
    n_pca_components: int = N_PCA_COMPONENTS,
    max_cells_for_kernel: int = MAX_CELLS_FOR_KERNEL,
    seed: int = SEED,
    checkpoint_dir: Path | None = None,
) -> ad.AnnData:
    """
    Replace perturbed cells with SEACells metacells; keep control cells raw.

    Returns a new AnnData with:
      - all control cells (unchanged)
      - one metacell row per SEACell per condition group
    """
    import SEACells  # deferred: not available outside bio-agent env

    rng = np.random.default_rng(seed)
    ctrl_mask = adata.obs[ctrl_key].to_numpy().astype(bool)
    ctrl = adata[ctrl_mask].copy()
    pert = adata[~ctrl_mask]

    X = np.asarray(pert.obsm[emb_key])
    groups = pert.obs.groupby(list(group_keys), observed=True).indices

    meta_rows: list[np.ndarray] = []
    meta_obs: dict[str, list] = {k: [] for k in group_keys}

    for gi, (label, idx) in enumerate(groups.items()):
        idx = np.asarray(idx)
        n = len(idx)
        label_tuple = label if isinstance(label, tuple) else (label,)
        label_strs = np.array([str(v) for v in label_tuple])

        print(
            f"  [{gi + 1}/{len(groups)}] {dict(zip(group_keys, label_tuple))}: {n} cells",
            flush=True,
        )

        ckpt = checkpoint_dir / f"group_{gi:04d}.npz" if checkpoint_dir else None
        if ckpt and ckpt.exists():
            cached = np.load(ckpt)
            if np.array_equal(cached["label"], label_strs):
                for row in cached["X"]:
                    meta_rows.append(row)
                    for k, v in zip(group_keys, label_tuple):
                        meta_obs[k].append(v)
                print(f"    -> resumed {cached['X'].shape[0]} metacell(s)", flush=True)
                continue
            print("    -> checkpoint label mismatch, recomputing", flush=True)

        if n <= min_cells_for_seacells or (n // cells_per_metacell) < 2:
            # too few cells for SEACells: collapse to a single mean metacell
            group_rows = X[idx].mean(axis=0, keepdims=True)
        else:
            if n > max_cells_for_kernel:
                idx = rng.choice(idx, size=max_cells_for_kernel, replace=False)
                n = len(idx)
            emb = X[idx]
            n_meta = max(2, n // cells_per_metacell)
            n_comp = min(n_pca_components, n - 1, emb.shape[1])
            emb_pca = PCA(n_components=n_comp, random_state=seed).fit_transform(emb)
            sub = ad.AnnData(
                X=np.zeros((n, 1), dtype=np.float32),
                obs=pd.DataFrame(index=[str(i) for i in range(n)]),
                obsm={"X_pca": emb_pca.astype(np.float32)},
            )
            model = SEACells.core.SEACells(
                sub,
                build_kernel_on="X_pca",
                n_SEACells=n_meta,
                n_neighbors=min(15, n - 1),
                verbose=False,
            )
            model.construct_kernel_matrix()
            model.initialize_archetypes()
            model.fit(min_iter=5, max_iter=30)
            assign = model.get_hard_assignments()["SEACell"]
            group_rows = np.stack([
                emb[sub.obs_names.get_indexer(members)].mean(axis=0)
                for _, members in assign.groupby(assign).groups.items()
            ])

        group_rows = group_rows.astype(np.float32)
        for row in group_rows:
            meta_rows.append(row)
            for k, v in zip(group_keys, label_tuple):
                meta_obs[k].append(v)

        if ckpt:
            np.savez(ckpt, X=group_rows, label=label_strs)

    meta_X = np.stack(meta_rows).astype(np.float32)
    meta_obs_df = pd.DataFrame(meta_obs)
    meta_obs_df[ctrl_key] = False
    for k in group_keys:
        if adata.obs[k].dtype.name == "category":
            meta_obs_df[k] = pd.Categorical(
                meta_obs_df[k], categories=adata.obs[k].cat.categories
            )

    meta_adata = ad.AnnData(obs=meta_obs_df, obsm={emb_key: meta_X})
    combined = ad.concat([ctrl, meta_adata], join="outer")
    combined.uns = dict(adata.uns)
    combined.obs[ctrl_key] = combined.obs[ctrl_key].fillna(False).astype(bool)
    return combined


def load_drug_map(path: str) -> dict[str, str]:
    """Load a two-column TSV (source_name → target_name) drug name mapping."""
    df = pd.read_csv(path, sep="\t", header=None, names=["src", "tgt"])
    return dict(zip(df["src"], df["tgt"]))


def load_drug_list(path: str) -> set[str]:
    """Load a plain text file with one drug name per line."""
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SEACells metacells from a perturbation h5ad."
    )
    parser.add_argument("--input",  required=True, help="Path to input h5ad file.")
    parser.add_argument("--output", required=True, help="Path for output h5ad file.")
    parser.add_argument(
        "--cell-line", default=None,
        help="Filter to a single cell line value (e.g. 'A549'). "
             "If omitted, all cell lines are processed together.",
    )
    parser.add_argument(
        "--drug-map", default=None,
        help="Optional two-column TSV (no header): source_drug_name<TAB>target_drug_name. "
             "Renames drugs in the output (e.g. to match a reference dataset).",
    )
    parser.add_argument(
        "--drug-list", default=None,
        help="Optional text file with one drug name per line (after any --drug-map renaming). "
             "Only these drugs will be included. Controls are always kept.",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Directory to store per-group .npz checkpoints for crash recovery. "
             "Defaults to <output_stem>_checkpoints/ next to the output file.",
    )
    parser.add_argument("--obsm", dest="emb_key", default=EMB_KEY,
                        help=f"obsm key for the cell embedding (default: {EMB_KEY}).")
    parser.add_argument("--ctrl-key", default=CTRL_KEY, help=f"obs column marking controls (default: {CTRL_KEY}).")
    parser.add_argument("--drug-key", default=DRUG_KEY, help=f"obs column for drug names (default: {DRUG_KEY}).")
    parser.add_argument("--cell-line-key", default=CELL_LINE_KEY,
                        help=f"obs column for cell line (default: {CELL_LINE_KEY}).")
    parser.add_argument("--extra-group-keys", default=None,
                        help="Comma-separated extra obs columns to add to the metacell grouping "
                             "(e.g. 'dose_value' for dose-resolved metacells). Kept in the output obs.")
    parser.add_argument("--cells-per-metacell", type=int, default=CELLS_PER_METACELL)
    parser.add_argument("--min-cells",          type=int, default=MIN_CELLS_FOR_SEACELLS)
    parser.add_argument("--n-pca",              type=int, default=N_PCA_COMPONENTS)
    parser.add_argument("--max-kernel-cells",   type=int, default=MAX_CELLS_FOR_KERNEL)
    parser.add_argument("--seed",               type=int, default=SEED)
    args = parser.parse_args()

    t0 = time.time()
    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else output_path.parent / f"{output_path.stem}_checkpoints"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── fast load: read obs, embedding, and uns (condition embeddings) ────
    print(f"loading {input_path} …", flush=True)
    with h5py.File(input_path, "r") as f:
        obs     = ad.io.read_elem(f["obs"])
        X_state = f["obsm"][args.emb_key][:]
        uns     = ad.io.read_elem(f["uns"]) if "uns" in f else {}
    print(f"  n_obs={len(obs):,}  embedding shape={X_state.shape}", flush=True)
    print(f"  uns keys: {list(uns.keys())}", flush=True)

    # ── optional drug name remapping ──────────────────────────────────────
    if args.drug_map:
        drug_map = load_drug_map(args.drug_map)
        obs[args.drug_key] = obs[args.drug_key].map(lambda d: drug_map.get(d, d))
        print(f"  applied drug name map from {args.drug_map}", flush=True)

    # ── cell line filter ──────────────────────────────────────────────────
    is_ctrl = obs[args.ctrl_key].to_numpy().astype(bool)

    if args.cell_line:
        is_cl = (obs[args.cell_line_key] == args.cell_line).to_numpy()
        ctrl_keep = is_ctrl & is_cl
    else:
        ctrl_keep = is_ctrl
        is_cl = np.ones(len(obs), dtype=bool)

    # ── drug filter ───────────────────────────────────────────────────────
    if args.drug_list:
        wanted_drugs = load_drug_list(args.drug_list)
        is_drug = obs[args.drug_key].isin(wanted_drugs).to_numpy()
    else:
        is_drug = ~is_ctrl  # all perturbed cells

    pert_keep = is_cl & is_drug & ~is_ctrl
    keep = ctrl_keep | pert_keep

    n_ctrl_in = ctrl_keep.sum()
    n_pert_in = pert_keep.sum()
    print(
        f"  subset: {n_ctrl_in:,} control cells + {n_pert_in:,} perturbed cells "
        f"({keep.sum():,} total)",
        flush=True,
    )
    if n_pert_in == 0:
        raise ValueError("No perturbed cells matched the given filters — check --cell-line and --drug-list.")

    obs_sub = obs.loc[keep].copy().reset_index(drop=True)
    X_sub   = X_state[keep]

    obs_sub[args.drug_key]     = obs_sub[args.drug_key].astype("category")
    if args.cell_line_key in obs_sub.columns:
        obs_sub[args.cell_line_key] = obs_sub[args.cell_line_key].astype("category")

    adata = ad.AnnData(obs=obs_sub, obsm={args.emb_key: X_sub.astype(np.float32)}, uns=uns)

    n_resumed = len(list(ckpt_dir.glob("group_*.npz")))
    if n_resumed:
        print(f"  resuming: {n_resumed} group checkpoint(s) found", flush=True)

    # ── build metacells ───────────────────────────────────────────────────
    group_keys = [args.drug_key]
    if not args.cell_line and args.cell_line_key in adata.obs.columns:
        group_keys.insert(0, args.cell_line_key)
    if args.extra_group_keys:
        for k in (s.strip() for s in args.extra_group_keys.split(",")):
            if k and k not in group_keys:
                group_keys.append(k)

    print(f"\nbuilding metacells (SEACells), grouping by {group_keys} …", flush=True)
    adata_meta = build_metacells(
        adata,
        group_keys=group_keys,
        emb_key=args.emb_key,
        ctrl_key=args.ctrl_key,
        cells_per_metacell=args.cells_per_metacell,
        min_cells_for_seacells=args.min_cells,
        n_pca_components=args.n_pca,
        max_cells_for_kernel=args.max_kernel_cells,
        seed=args.seed,
        checkpoint_dir=ckpt_dir,
    )

    if args.cell_line and args.cell_line_key not in adata_meta.obs.columns:
        adata_meta.obs[args.cell_line_key] = args.cell_line

    n_ctrl_out = int(adata_meta.obs[args.ctrl_key].sum())
    n_meta_out = int((~adata_meta.obs[args.ctrl_key]).sum())
    drugs_built = sorted(
        adata_meta.obs.loc[~adata_meta.obs[args.ctrl_key], args.drug_key].unique()
    )
    print(f"\n  output: {n_ctrl_out:,} control cells + {n_meta_out:,} metacells", flush=True)
    print(f"  conditions ({len(drugs_built)}): {drugs_built}", flush=True)

    print(f"\nwriting {output_path} …", flush=True)
    adata_meta.write_h5ad(output_path)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
