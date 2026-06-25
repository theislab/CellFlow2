"""Build a cell_line × drug × SMILES × viability table from the prophet drug-response
datasets (GDSC, CTRP, PRISM) and compare coverage. Singletons only (iv2 == 'negative_drug').

Columns in the source CSVs: cell_line, iv1 (SMILES), iv_name (drug), iv2 (2nd pert), value
(viability 0-1), phenotype (dataset tag).
"""
from pathlib import Path
import pandas as pd

DATA = Path("/lustre/groups/ml01/projects/super_rad_project/HF/data_hf")
OUT = Path("/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/sub/analysis")
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = {"GDSC": "GDSC_dataset.csv", "CTRP": "CTRP_dataset.csv", "PRISM": "PRISM_dataset.csv"}
USECOLS = ["cell_line", "iv1", "iv_name", "iv2", "value"]

frames, summary = [], []
for name, fn in DATASETS.items():
    df = pd.read_csv(DATA / fn, usecols=USECOLS)
    df = df[df["iv2"].astype(str) == "negative_drug"]          # singletons only
    t = (df.rename(columns={"iv_name": "drug", "iv1": "smiles", "value": "viability"})
            [["cell_line", "drug", "smiles", "viability"]])
    t["cell_line"] = t["cell_line"].astype(str)
    # one row per (cell_line, drug, smiles): mean viability over any replicates
    t = t.groupby(["cell_line", "drug", "smiles"], as_index=False)["viability"].mean()
    t["dataset"] = name
    frames.append(t)
    summary.append(dict(
        dataset=name,
        rows=len(t),
        cell_lines=t["cell_line"].nunique(),
        drugs_by_name=t["drug"].nunique(),
        drugs_by_smiles=t["smiles"].nunique(),
        pairs_cellline_x_smiles=t.drop_duplicates(["cell_line", "smiles"]).shape[0],
    ))

table = pd.concat(frames, ignore_index=True)
sdf = pd.DataFrame(summary).sort_values("pairs_cellline_x_smiles", ascending=False).reset_index(drop=True)

print("=== coverage per dataset (singletons; one row per cell_line × drug) ===")
print(sdf.to_string(index=False))
print(f"\nHighest coverage: {sdf.iloc[0]['dataset']} "
      f"({sdf.iloc[0]['pairs_cellline_x_smiles']:,} cell_line × drug pairs)")

# union (note: cell-line ids differ across datasets — CTRP can be numeric — so union is a loose upper bound)
print("\n=== union across the 3 datasets (loose; cell-line ids not harmonized) ===")
print(f"  unique cell_lines: {table['cell_line'].nunique():,}")
print(f"  unique drugs (smiles): {table['smiles'].nunique():,}")
print(f"  unique drugs (name):   {table['drug'].nunique():,}")
print(f"  unique cell_line × smiles pairs: {table.drop_duplicates(['cell_line','smiles']).shape[0]:,}")

table.to_parquet(OUT / "drug_viability_table.parquet", index=False)
sdf.to_csv(OUT / "drug_viability_coverage.csv", index=False)
print(f"\nsaved table ({len(table):,} rows) -> {OUT/'drug_viability_table.parquet'}")
print(f"saved coverage summary -> {OUT/'drug_viability_coverage.csv'}")
print("\nsample rows:")
print(table.sample(6, random_state=0).to_string(index=False))
