"""
prophet_overlap.py — inventory the Prophet (bulk/phenotype) datasets and the
single-cell (unipert) datasets, and quantify their overlaps.

Questions answered:
  1. Per dataset: # cell lines, # perturbations (drugs/genes), readout type, size.
  2. Cell-line overlap across all datasets (Jaccard + raw shared counts).
  3. Drug overlap across the drug datasets.
  4. KEY: how much of each single-cell (unipert) dataset's (cell_line, drug)
     pairs / cell lines / drugs are covered by each Prophet dataset — i.e. how
     much phenotype supervision is actually reachable for the pilot.
  5. Phenotype inventory: readout type + #readout-dims per dataset, shared vs unique.

Reads h5ad via h5py (no anndata dependency) and CSVs in chunks (LINCS is ~8GB).
Outputs CSV tables to analysis/tables/ and PNG figures to analysis/figures/.

Run:  python analysis/prophet_overlap.py
"""
from __future__ import annotations

import re
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA   = Path("/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow")
PROPHET = DATA / "prophet"
UNIPERT = DATA / "unipert"
HERE   = Path(__file__).resolve().parent
TAB    = HERE / "tables";   TAB.mkdir(parents=True, exist_ok=True)
FIG    = HERE / "figures";  FIG.mkdir(parents=True, exist_ok=True)
CACHE  = HERE / "_cache";   CACHE.mkdir(parents=True, exist_ok=True)

CONTROLS = {"control", "dmso", "none", "nan", "", "non-targeting", "nontargeting", "neg", "vehicle"}


# ── normalization ────────────────────────────────────────────────────────────
def norm_cell(s) -> str:
    s = str(s).upper().strip()
    s = re.sub(r"\s*CELL LINE\s*$", "", s)
    s = re.sub(r"\s+CELL\s*$", "", s)
    return re.sub(r"[^A-Z0-9]", "", s)

def norm_drug(s) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\(.*?\)", "", s)          # drop parenthetical aliases
    return re.sub(r"[^a-z0-9]", "", s)

def split_perts(label) -> list[str]:
    """Split a (possibly combo) perturbation label into individual perts."""
    parts = re.split(r"[;_+]| and ", str(label))
    out = []
    for p in parts:
        n = norm_drug(p)
        if n and p.strip().lower() not in CONTROLS:
            out.append(n)
    return out


# ── h5ad obs reader (h5py) ───────────────────────────────────────────────────
def _read_cat(node):
    """Return (categories: np.ndarray[str], codes: np.ndarray[int]) for an obs col."""
    if isinstance(node, h5py.Group):  # categorical
        cats = node["categories"][:]
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in cats])
        codes = node["codes"][:]
    else:
        vals = node[:]
        vals = np.array([v.decode() if isinstance(v, bytes) else v for v in vals])
        cats, codes = np.unique(vals, return_inverse=True)
    return cats, codes

def load_h5ad(path, cell_col, drug_col, cell_override=None):
    with h5py.File(path, "r") as f:
        obs = f["obs"]
        cl_cats, cl_codes = _read_cat(obs[cell_col])
        dr_cats, dr_codes = _read_cat(obs[drug_col])
        n = len(cl_codes)
        try:
            n_readout = f["var"]["_index"].shape[0]
        except Exception:
            n_readout = None
    if cell_override is not None:
        # file uses an opaque cell id (e.g. Cellosaurus CVCL_*) — pin the name
        one = norm_cell(cell_override)
        cl_norm = np.array([one] * len(cl_cats))
        cell_lines = {one}
    else:
        cl_norm = np.array([norm_cell(c) for c in cl_cats])
        cell_lines = {c for c in cl_norm if c}
    drugs = set()
    for d in dr_cats:
        drugs |= set(split_perts(d))
    dr_full = np.array([norm_drug(d) for d in dr_cats])
    m = (cl_codes >= 0) & (dr_codes >= 0)
    pairs = set(zip(cl_norm[cl_codes[m]].tolist(), dr_full[dr_codes[m]].tolist()))
    pairs = {(c, d) for c, d in pairs if c and d and d not in CONTROLS}
    return dict(cell_lines=cell_lines, drugs=drugs, pairs=pairs, n_rows=int(n), n_readout=n_readout)


# ── CSV reader (chunked) ─────────────────────────────────────────────────────
def load_csv(path, chunksize=1_000_000):
    cell_lines, drugs, pairs, phenos = set(), set(), set(), set()
    n = 0
    cols = None
    keep = {"cell_line", "iv1", "iv2", "phenotype"}
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False,
                             usecols=lambda c: c in keep):
        if cols is None:
            cols = set(chunk.columns)
        n += len(chunk)
        if "cell_line" in chunk:
            cell_lines |= {norm_cell(c) for c in chunk["cell_line"].dropna().unique()}
        for col in ("iv1", "iv2"):
            if col in chunk:
                for v in chunk[col].dropna().unique():
                    drugs |= set(split_perts(v))
        if "iv1" in chunk:
            pp = chunk[["cell_line", "iv1"]].dropna().drop_duplicates()
            for c, d in zip(pp["cell_line"], pp["iv1"]):
                nc, nd = norm_cell(c), norm_drug(d)
                if nc and nd and nd not in CONTROLS:
                    pairs.add((nc, nd))
        if "phenotype" in chunk:
            phenos |= set(chunk["phenotype"].dropna().astype(str).unique())
    cell_lines.discard("")
    return dict(cell_lines=cell_lines, drugs=drugs, pairs=pairs, n_rows=int(n), n_readout=len(phenos))


# ── dataset registry ─────────────────────────────────────────────────────────
# (name, kind, loader-args, perturbation, readout)
PROPHET_H5AD = [
    ("GDSC",  "pert_compound", "drug singleton",     "ln(IC50) / viability"),
    ("GDSC2", "pert_compound", "drug combination",   "ln(IC50) / viability"),
    ("CTRP",  "pert_compound", "drug singleton",     "IC50 (CellTiter-Glo)"),
    ("PRISM", "pert_compound", "drug singleton",     "Log2FC / MFI"),
    ("SCORE", "pert_name",     "CRISPR singleton",   "essentiality (CERES)"),
]
PROPHET_CSV = [
    ("Horlbeck", "Horlbeck_dataset.csv", "CRISPR combination", "viability"),
    ("JUMP",     "JUMP_dataset.csv",     "drug/CRISPR",        "Cell Painting"),
    ("JUMPcr",   "JUMPcr_dataset.csv",   "CRISPR",             "Cell Painting"),
    ("JUMPsm",   "JUMPsm_dataset.csv",   "drug",               "Cell Painting"),
    ("LINCS",    "LINCS_dataset.csv",    "drug singleton",     "RNA (fluorescence)"),
    ("Shifrut",  "Shifrut_dataset.csv",  "CRISPR",             "proliferation"),
    ("PRISM_csv","PRISM_dataset.csv",    "drug singleton",     "Log2FC"),
]


def _tahoe_cellname(stem):
    """tahoe_* files carry Cellosaurus ids in obs; recover the name from the filename."""
    if stem.startswith("tahoe_"):
        return stem[len("tahoe_"):].replace("_", "")   # tahoe_panc_1 -> panc1
    return None


def build_inventory(use_cache=True):
    rows, sets = [], {}
    pcache = CACHE / "prophet_sets.pkl"

    # Prophet datasets are the slow part (LINCS ~8GB) — cache them.
    if use_cache and pcache.exists():
        prophet_sets, prophet_rows = pickle.load(open(pcache, "rb"))
        print("  [cache] loaded Prophet sets from", pcache)
    else:
        prophet_sets, prophet_rows = {}, []
        for name, drug_col, pert, readout in PROPHET_H5AD:
            path = PROPHET / f"{name}.h5ad"
            if not path.exists():
                print(f"  [skip] {path}"); continue
            print(f"  loading prophet/{name}.h5ad …")
            d = load_h5ad(path, "cell_line", drug_col)
            prophet_sets[name] = d
            prophet_rows.append(dict(dataset=name, source="prophet", perturbation=pert, readout=readout,
                                     n_cell_lines=len(d["cell_lines"]), n_drugs=len(d["drugs"]),
                                     n_pairs=len(d["pairs"]), n_rows=d["n_rows"], n_readout_dims=d["n_readout"]))
        for name, fname, pert, readout in PROPHET_CSV:
            path = PROPHET / fname
            if not path.exists():
                print(f"  [skip] {path}"); continue
            print(f"  loading prophet/{fname} …")
            d = load_csv(path)
            prophet_sets[name] = d
            prophet_rows.append(dict(dataset=name, source="prophet", perturbation=pert, readout=readout,
                                     n_cell_lines=len(d["cell_lines"]), n_drugs=len(d["drugs"]),
                                     n_pairs=len(d["pairs"]), n_rows=d["n_rows"], n_readout_dims=d["n_readout"]))
        pickle.dump((prophet_sets, prophet_rows), open(pcache, "wb"))
        print("  [cache] saved Prophet sets →", pcache)

    sets.update(prophet_sets); rows.extend(prophet_rows)

    # Single-cell (cheap) — always fresh, with cell-line harmonization.
    for path in sorted(UNIPERT.glob("*_w_emb.h5ad")):
        stem = path.stem.replace("_w_emb", "")
        name = "sc:" + stem
        override = _tahoe_cellname(stem)
        print(f"  loading unipert/{path.name} …" + (f"  [cell={override}]" if override else ""))
        try:
            d = load_h5ad(path, "cell_line", "drug_0", cell_override=override)
        except Exception as e:
            print(f"     [warn] {e}"); continue
        # flag opaque ids we didn't harmonize
        if override is None and any(c.startswith("CVCL") for c in d["cell_lines"]):
            print(f"     [warn] {name}: cell_line looks like Cellosaurus ids — overlap may be understated")
        sets[name] = d
        rows.append(dict(dataset=name, source="single-cell", perturbation="drug (sc)", readout="scRNA",
                         n_cell_lines=len(d["cell_lines"]), n_drugs=len(d["drugs"]),
                         n_pairs=len(d["pairs"]), n_rows=d["n_rows"], n_readout_dims=d["n_readout"]))
    return pd.DataFrame(rows), sets


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_matrix(sets, key, names):
    M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            M[i, j] = jaccard(sets[a][key], sets[b][key])
    return pd.DataFrame(M, index=names, columns=names)


def shared_counts(sets, key, names):
    M = np.zeros((len(names), len(names)), dtype=int)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            M[i, j] = len(sets[a][key] & sets[b][key])
    return pd.DataFrame(M, index=names, columns=names)


def heatmap(df, title, path, fmt="{:.2f}", cmap="viridis"):
    fig, ax = plt.subplots(figsize=(0.6 * len(df.columns) + 4, 0.5 * len(df.index) + 3))
    im = ax.imshow(df.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(df.index)));   ax.set_yticklabels(df.index, fontsize=7)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            v = df.values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6,
                    color="white" if im.norm(v) < 0.6 else "black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  figure → {path}")


def main():
    print("Building inventory …")
    inv, sets = build_inventory()
    inv = inv.sort_values(["source", "dataset"]).reset_index(drop=True)
    inv.to_csv(TAB / "inventory.csv", index=False)
    print("\n=== INVENTORY ===")
    print(inv.to_string(index=False))

    all_names     = list(inv["dataset"])
    prophet_names = list(inv[inv.source == "prophet"]["dataset"])
    sc_names      = list(inv[inv.source == "single-cell"]["dataset"])
    # drug datasets = those with any drugs
    drug_names    = [n for n in all_names if sets[n]["drugs"]]

    # 2/3. overlap matrices
    cl_jac = overlap_matrix(sets, "cell_lines", all_names)
    cl_cnt = shared_counts(sets, "cell_lines", all_names)
    dr_jac = overlap_matrix(sets, "drugs", drug_names)
    dr_cnt = shared_counts(sets, "drugs", drug_names)
    cl_jac.to_csv(TAB / "cellline_jaccard.csv"); cl_cnt.to_csv(TAB / "cellline_shared_counts.csv")
    dr_jac.to_csv(TAB / "drug_jaccard.csv");     dr_cnt.to_csv(TAB / "drug_shared_counts.csv")
    heatmap(cl_jac, "Cell-line overlap (Jaccard)", FIG / "cellline_jaccard.png")
    heatmap(dr_jac, "Drug overlap (Jaccard)", FIG / "drug_jaccard.png")

    # 4. single-cell → prophet coverage
    cov_rows = []
    for sc in sc_names:
        S = sets[sc]
        for pn in prophet_names:
            P = sets[pn]
            cl_ov = S["cell_lines"] & P["cell_lines"]
            dr_ov = S["drugs"] & P["drugs"]
            pr_ov = S["pairs"] & P["pairs"]
            cov_rows.append(dict(
                single_cell=sc, prophet=pn,
                sc_cell_lines=len(S["cell_lines"]), cl_covered=len(cl_ov),
                sc_drugs=len(S["drugs"]), drug_covered=len(dr_ov),
                drug_cov_frac=round(len(dr_ov) / max(len(S["drugs"]), 1), 3),
                sc_pairs=len(S["pairs"]), pair_covered=len(pr_ov),
                pair_cov_frac=round(len(pr_ov) / max(len(S["pairs"]), 1), 3),
            ))
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(TAB / "singlecell_to_prophet_coverage.csv", index=False)
    print("\n=== SINGLE-CELL → PROPHET COVERAGE (drug & pair fractions) ===")
    print(cov[["single_cell", "prophet", "cl_covered", "drug_covered",
               "drug_cov_frac", "pair_covered", "pair_cov_frac"]].to_string(index=False))

    if sc_names and prophet_names:
        piv = cov.pivot(index="single_cell", columns="prophet", values="drug_cov_frac")
        heatmap(piv.fillna(0), "Drug coverage of single-cell by Prophet (fraction)",
                FIG / "sc_drug_coverage.png", fmt="{:.2f}", cmap="magma")
        pivp = cov.pivot(index="single_cell", columns="prophet", values="pair_cov_frac")
        heatmap(pivp.fillna(0), "(cell_line,drug) pair coverage of single-cell by Prophet",
                FIG / "sc_pair_coverage.png", fmt="{:.2f}", cmap="magma")

    # cell-line membership across datasets (shared vs unique)
    from collections import Counter
    cl_membership = Counter()
    for n in all_names:
        for c in sets[n]["cell_lines"]:
            cl_membership[c] += 1
    memb = pd.Series(Counter(cl_membership.values())).sort_index()
    memb.to_csv(TAB / "cellline_membership_hist.csv")
    print("\n=== CELL LINES BY #DATASETS THEY APPEAR IN ===")
    print(memb.to_string())

    print("\nDone. Tables → analysis/tables/, figures → analysis/figures/")


if __name__ == "__main__":
    main()
