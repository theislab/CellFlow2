"""Cross-dataset concordance of viability for SHARED cell_line × drug measurements.

Order agreement (Spearman) is the headline because the assays use different viability scales.
Two levels:
  - global : all shared (cell_line, smiles) pairs pooled -> mixes within- and across-drug signal.
  - per-drug: within each shared drug, rank cell-line viabilities across the two datasets
              (the real "does the dose-response biology replicate" test).
"""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/sub/analysis")
df = pd.read_parquet(OUT / "drug_viability_table.parquet")
D = ["GDSC", "CTRP", "PRISM"]
# one viability per (cell_line, smiles) per dataset (collapse any drug-name variants of a SMILES)
agg = {d: df[df.dataset == d].groupby(["cell_line", "smiles"], as_index=False)["viability"].mean()
       for d in D}
piv = {d: agg[d].set_index(["cell_line", "smiles"])["viability"] for d in D}

# ── global concordance per dataset pair + scatter ──
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
rows = []
for ax, (a, b) in zip(axes, itertools.combinations(D, 2)):
    m = pd.concat([piv[a].rename("va"), piv[b].rename("vb")], axis=1, join="inner").dropna()
    sr = spearmanr(m.va, m.vb).statistic
    pr = pearsonr(m.va, m.vb)[0]
    rows.append(dict(pair=f"{a}-{b}", n_pairs=len(m), spearman=round(sr, 3), pearson=round(pr, 3)))
    ax.scatter(m.va, m.vb, s=4, alpha=0.25)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel(f"{a} viability"); ax.set_ylabel(f"{b} viability")
    ax.set_title(f"{a} vs {b}  (n={len(m)})\nSpearman ρ={sr:.2f}  Pearson r={pr:.2f}")
plt.tight_layout(); plt.savefig(OUT / "viability_concordance.png", dpi=120)

print("=== GLOBAL: shared cell_line × drug viability agreement (pooled) ===")
print(pd.DataFrame(rows).to_string(index=False))

# ── per-drug rank agreement (within drug, across cell lines) ──
print("\n=== PER-DRUG: rank agreement of cell-line sensitivities (drugs with >=10 shared lines) ===")
for a, b in itertools.combinations(D, 2):
    j = agg[a].merge(agg[b], on=["cell_line", "smiles"], suffixes=("_a", "_b"))
    rhos = []
    for _, g in j.groupby("smiles"):
        if g["cell_line"].nunique() >= 10 and g.viability_a.std() > 0 and g.viability_b.std() > 0:
            r = spearmanr(g.viability_a, g.viability_b).statistic
            if np.isfinite(r):
                rhos.append(r)
    if rhos:
        rhos = np.array(rhos)
        print(f"  {a}-{b}: {len(rhos):>3} drugs | median per-drug ρ={np.median(rhos):+.2f} | "
              f"mean={np.mean(rhos):+.2f} | frac ρ>0.5: {(rhos > 0.5).mean():.0%} | "
              f"frac ρ>0: {(rhos > 0).mean():.0%}")
    else:
        print(f"  {a}-{b}: no drug with >=10 shared cell lines")

# ── per-cell-line rank agreement (within cell line, across drugs) — the "generalize over drugs" test ──
print("\n=== PER-CELL-LINE: rank agreement of drug responses (cell lines with >=10 shared drugs) ===")
for a, b in itertools.combinations(D, 2):
    j = agg[a].merge(agg[b], on=["cell_line", "smiles"], suffixes=("_a", "_b"))
    rhos = []
    for _, g in j.groupby("cell_line"):
        if g["smiles"].nunique() >= 10 and g.viability_a.std() > 0 and g.viability_b.std() > 0:
            r = spearmanr(g.viability_a, g.viability_b).statistic
            if np.isfinite(r):
                rhos.append(r)
    if rhos:
        rhos = np.array(rhos)
        print(f"  {a}-{b}: {len(rhos):>3} cell lines | median per-line ρ={np.median(rhos):+.2f} | "
              f"mean={np.mean(rhos):+.2f} | frac ρ>0.5: {(rhos > 0.5).mean():.0%} | "
              f"frac ρ>0: {(rhos > 0).mean():.0%}")
    else:
        print(f"  {a}-{b}: no cell line with >=10 shared drugs")
print(f"\nsaved scatter -> {OUT / 'viability_concordance.png'}")
