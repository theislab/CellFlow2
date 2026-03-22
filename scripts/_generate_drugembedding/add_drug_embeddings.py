#!/usr/bin/env python
"""
Append drug and cell-line embeddings to h5ad files.

For each dataset in the config:
  1) Map the drug column to (drug_0, drug_1) via pert_compound splitting
  2) Filter out cells whose drugs are not in the embedding dict
  3) Store per-dataset drug dicts in uns['drug_{0,1}_embeddings']
  4) Store one-hot cell-line embeddings in uns['cell_line_embeddings']
  5) Write to output_dir (or overwrite in-place if output_dir == input dir)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import yaml


# ---------------------------------------------------------------------------
# Drug-name helpers
# ---------------------------------------------------------------------------

def _build_control_set(aliases: list[str]) -> set[str]:
    return {a.strip().lower() for a in aliases} | {""}


def normalize_drug_name(name: str, control_set: set[str]) -> str:
    if pd.isna(name) or str(name).strip().lower() in control_set:
        return "control"
    return str(name)


def split_pert_compound(x: str, control_set: set[str]) -> tuple[str, str]:
    """Split a pert_compound string into (drug_0, drug_1)."""
    if pd.isna(x):
        return ("", "")
    s = str(x)

    if "|" in s:
        parts = [normalize_drug_name(p, control_set) for p in s.split("|") if p]
    elif s.upper() == "DMSO_TF":
        parts = ["control"]
    elif "_" in s:
        parts = [normalize_drug_name(p, control_set) for p in s.split("_") if p]
    else:
        parts = [normalize_drug_name(s, control_set)] if s else []

    if len(parts) == 0:
        return ("", "")
    if len(parts) == 1:
        return (parts[0], "control")
    return (parts[0], parts[1])


def is_control(drug: str, control_set: set[str]) -> bool:
    return drug.strip().lower() in control_set


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def add_embeddings_to_adata(
    adata,
    embeddings: dict[str, np.ndarray],
    control_set: set[str],
    drug_col: str = "pert_compound",
):
    """
    Mutates *adata* in-place (after a filtered copy):
      - adds drug_org / drug_0 / drug_1 obs columns
      - filters cells with unknown drugs
      - stores drug & cell-line embedding dicts in uns
    Returns the (possibly smaller) AnnData.
    """
    if drug_col != "pert_compound":
        adata.obs["pert_compound"] = adata.obs[drug_col].astype(str)

    adata.obs["drug_org"] = adata.obs["pert_compound"].astype(str)

    pc_str = adata.obs["pert_compound"].astype(str).to_numpy()
    d0, d1 = zip(*(split_pert_compound(x, control_set) for x in pc_str))
    adata.obs["drug_0"] = pd.Series(d0, index=adata.obs_names).astype(str)
    adata.obs["drug_1"] = pd.Series(d1, index=adata.obs_names).astype(str)

    emb_keys = set(embeddings.keys())

    def is_known(drug: str) -> bool:
        return is_control(drug, control_set) or drug in emb_keys

    mask = adata.obs["drug_0"].map(is_known) & adata.obs["drug_1"].map(is_known)

    before = adata.n_obs
    adata = adata[mask].copy()
    dropped = before - adata.n_obs
    print(f"  Filter drugs: {before:,} -> {adata.n_obs:,} (dropped {dropped:,})")

    # --- drug embeddings ---
    drug_dim = int(np.asarray(next(iter(embeddings.values()))).shape[0])
    zero = np.zeros((drug_dim,), dtype=np.float32)

    drugs0 = np.sort(pd.unique(adata.obs["drug_0"].astype(str)))
    drugs1 = np.sort(pd.unique(adata.obs["drug_1"].astype(str)))

    adata.uns["drug_embedding_dim"] = drug_dim
    adata.uns["drug_0_embeddings"] = {
        d: np.asarray(embeddings[d], dtype=np.float32)
        for d in drugs0
        if (not is_control(d, control_set)) and (d in embeddings)
    }
    adata.uns["drug_1_embeddings"] = {
        d: np.asarray(embeddings[d], dtype=np.float32)
        for d in drugs1
        if (not is_control(d, control_set)) and (d in embeddings)
    }
    adata.uns["drug_0_embeddings"]["control"] = zero
    adata.uns["drug_1_embeddings"]["control"] = zero

    # --- cell-line one-hot embeddings ---
    cl_series = (
        adata.obs["cell_line"]
        .astype("string")
        .fillna("NA")
        .astype(str)
    )
    n_na = (cl_series == "NA").sum()
    if n_na > 0:
        print(f"  [cell_line] {n_na:,} NA values -> class 'NA'")
    adata.obs["cell_line"] = cl_series

    celllines = np.sort(cl_series.unique())
    eye = np.eye(len(celllines), dtype=np.float32)
    adata.uns["cell_line_embedding_dim"] = int(len(celllines))
    adata.uns["cell_line_embeddings"] = {
        cl: eye[i] for i, cl in enumerate(celllines)
    }

    return adata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to config YAML"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip datasets whose output file already exists",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    emb_path = cfg["embedding_path"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    control_set = _build_control_set(cfg.get("control_aliases", []))

    print(f"Loading drug embeddings from {emb_path} ...")
    embeddings = pd.read_pickle(emb_path)
    print(f"  {len(embeddings)} drugs, dim={np.asarray(next(iter(embeddings.values()))).shape[0]}")

    for ds_cfg in cfg["datasets"]:
        in_path = Path(ds_cfg["path"])
        drug_col = ds_cfg.get("drug_col", "pert_compound")
        stem = in_path.stem
        out_path = out_dir / f"{stem}.h5ad"

        print(f"\n{'='*60}")
        print(f"{stem}  (drug_col={drug_col!r})")

        if args.skip_existing and out_path.exists() and out_path != in_path:
            print(f"  {out_path.name} already exists, skipping")
            continue

        print(f"  Reading {in_path} ...")
        adata = sc.read_h5ad(str(in_path))

        adata = add_embeddings_to_adata(
            adata,
            embeddings=embeddings,
            control_set=control_set,
            drug_col=drug_col,
        )

        print(f"  Writing {out_path} ...")
        adata.write_h5ad(out_path)
        print(f"  Done: {adata.n_obs:,} cells | "
              f"drug_0: {len(adata.uns['drug_0_embeddings'])} | "
              f"drug_1: {len(adata.uns['drug_1_embeddings'])} | "
              f"cell_lines: {len(adata.uns['cell_line_embeddings'])}")
        del adata

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
