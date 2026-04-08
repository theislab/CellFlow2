"""
Compatibility check for all datasets listed in config_drug_emb.yaml against:
  1. config_drug_emb.yaml  — drug embedding coverage + control alias handling
  2. sanity_check_cf_tahoe_X_state.py — CellFlow input requirements

For each dataset checks:
  - Source h5ad: exists, drug_col present, drug coverage in embedding pkl
  - Output *_w_emb.h5ad: exists, drug_0/drug_1/X_state columns, uns embedding dicts,
    control cells, drug-effect ranking feasibility

Prints a pass/fail report for every dataset.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "config_drug_emb.yaml"

BATCH_SIZE  = 1024  # min cells per drug for CellFlow top-N eval

# ── Helpers ───────────────────────────────────────────────────────────────
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

def check(label, ok, detail=""):
    sym = PASS if ok else FAIL
    print(f"    {sym}  {label}" + (f"\n         {detail}" if detail else ""))
    return ok

# ── Load config ────────────────────────────────────────────────────────────
cfg = yaml.safe_load(CONFIG_PATH.read_text())
control_set = {a.strip().lower() for a in cfg.get("control_aliases", [])} | {""}

output_dir = Path(cfg["output_dir"])
emb_path   = cfg["embedding_path"]

print(f"\nLoading drug embeddings from {emb_path} ...")
embeddings: dict = pd.read_pickle(emb_path)
emb_keys = set(embeddings.keys())
emb_dim  = int(np.asarray(next(iter(embeddings.values()))).shape[0])
print(f"  {len(emb_keys):,} drugs, dim={emb_dim}")

datasets = cfg.get("datasets", [])
print(f"\n{len(datasets)} datasets in config\n")

# ═══════════════════════════════════════════════════════════════════════════
# Per-dataset checks
# ═══════════════════════════════════════════════════════════════════════════

summary_rows = []

for ds in datasets:
    src_path  = Path(ds["path"])
    drug_col  = ds.get("drug_col", "drug")
    stem      = src_path.stem                           # e.g. tahoe_hop62
    w_emb_path = output_dir / f"{stem}_w_emb.h5ad"

    print(f"══ {stem} ═══════════════════════════════════════════════════")

    # ── Source h5ad ──────────────────────────────────────────────────────
    print(f"  [source] {src_path.name}")
    src_ok = check("Source h5ad exists", src_path.exists())

    src_coverage = None
    if src_ok:
        adata_src = sc.read_h5ad(str(src_path))
        print(f"    {adata_src.n_obs:,} cells × {adata_src.n_vars:,} genes")
        check(f"'{drug_col}' column present", drug_col in adata_src.obs.columns)

        if drug_col in adata_src.obs.columns:
            drugs_raw = adata_src.obs[drug_col].astype(str)
            # map control aliases
            is_ctrl = drugs_raw.str.strip().str.lower().isin(control_set)
            non_ctrl_drugs = set(drugs_raw[~is_ctrl].unique())
            in_emb  = non_ctrl_drugs & emb_keys
            missing = non_ctrl_drugs - emb_keys
            src_coverage = len(in_emb) / len(non_ctrl_drugs) if non_ctrl_drugs else 1.0
            check(
                f"Drug embedding coverage ({len(in_emb)}/{len(non_ctrl_drugs)} drugs)",
                len(missing) == 0,
                detail=(f"Missing ({len(missing)}): {sorted(missing)[:10]}"
                        + ("..." if len(missing) > 10 else "")) if missing else "",
            )
        del adata_src

    # ── Output *_w_emb.h5ad ──────────────────────────────────────────────
    print(f"  [output] {w_emb_path.name}")
    wemb_ok = check("Output *_w_emb.h5ad exists", w_emb_path.exists())

    n_eligible = 0
    if wemb_ok:
        adata = sc.read_h5ad(str(w_emb_path))
        print(f"    {adata.n_obs:,} cells × {adata.n_vars:,} genes")

        # CellFlow required fields
        check("drug_0 column present",    "drug_0"  in adata.obs.columns)
        check("drug_1 column present",    "drug_1"  in adata.obs.columns)
        check("X_state in obsm",          "X_state" in adata.obsm,
              detail=f"available: {list(adata.obsm.keys())}")

        if "drug_0" in adata.obs.columns and "drug_1" in adata.obs.columns:
            ctrl_mask = (adata.obs["drug_0"] == "control") & (adata.obs["drug_1"] == "control")
            n_ctrl = int(ctrl_mask.sum())
            check("Control cells present (drug_0==drug_1=='control')", n_ctrl > 0,
                  detail=f"{n_ctrl:,} control cells ({100*n_ctrl/adata.n_obs:.1f}%)")

            d0 = adata.obs["drug_0"].astype(str)
            nan_d0 = (d0.str.strip() == "") | (d0 == "nan") | (d0 == "None")
            check("No empty/NaN in drug_0", not nan_d0.any(),
                  detail=f"{nan_d0.sum():,} bad values" if nan_d0.any() else "")

            all_drugs_d0 = set(d0.unique()) - {"control"}

            # uns embeddings
            check("uns['drug_0_embeddings'] present", "drug_0_embeddings" in adata.uns)
            check("uns['drug_1_embeddings'] present", "drug_1_embeddings" in adata.uns)
            check("uns['cell_line_embeddings'] present", "cell_line_embeddings" in adata.uns)

            if "drug_0_embeddings" in adata.uns:
                d0_keys   = set(adata.uns["drug_0_embeddings"].keys())
                d0_miss   = all_drugs_d0 - d0_keys
                check(f"All drug_0 drugs in uns['drug_0_embeddings']",
                      len(d0_miss) == 0,
                      detail=f"missing: {sorted(d0_miss)[:10]}" if d0_miss else "")

            if "cell_line_embeddings" in adata.uns and "cell_line" in adata.obs:
                obs_cls   = set(adata.obs["cell_line"].astype(str).unique())
                miss_cl   = obs_cls - set(adata.uns["cell_line_embeddings"].keys())
                check("All cell_lines covered in uns['cell_line_embeddings']",
                      len(miss_cl) == 0,
                      detail=f"missing: {miss_cl}" if miss_cl else "")

            # Drug-effect ranking feasibility
            if "X_state" in adata.obsm:
                drug_arr = d0.values
                single   = adata.obs["drug_1"].astype(str) == "control"
                eligible = [
                    dr for dr in np.unique(drug_arr[~ctrl_mask.values])
                    if ((drug_arr == dr) & (~ctrl_mask.values) & single.values).sum() >= BATCH_SIZE
                ]
                n_eligible = len(eligible)
                check(f"≥{BATCH_SIZE} cells for ≥10 drugs (top-N eval)", n_eligible >= 10,
                      detail=f"{n_eligible} eligible drugs")
                if eligible:
                    top5 = sorted(eligible,
                                  key=lambda dr: (drug_arr == dr).sum(), reverse=True)[:5]
                    print(f"         {WARN}  Top-5: " +
                          ", ".join(f"{dr}({(drug_arr==dr).sum():,})" for dr in top5))

        del adata

    summary_rows.append({
        "dataset":    stem,
        "src_exists": src_ok,
        "coverage":   f"{100*src_coverage:.1f}%" if src_coverage is not None else "—",
        "wemb_exists": wemb_ok,
        "eligible_drugs": n_eligible,
    })
    print()

# ═══════════════════════════════════════════════════════════════════════════
print("══ Summary ═══════════════════════════════════════════════════════")
print(f"  {'Dataset':<20} {'src':>5} {'coverage':>10} {'w_emb':>7} {'eligible_drugs':>14}")
print(f"  {'-'*20} {'-'*5} {'-'*10} {'-'*7} {'-'*14}")
for r in summary_rows:
    src_sym  = "✓" if r["src_exists"]  else "✗"
    wemb_sym = "✓" if r["wemb_exists"] else "✗"
    print(f"  {r['dataset']:<20} {src_sym:>5} {r['coverage']:>10} {wemb_sym:>7} {r['eligible_drugs']:>14}")
