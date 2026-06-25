"""Per-cell-line drug-response agreement between two screens: scatter several cell lines
spanning the agreement range, + the distribution of per-cell-line Spearman rho.

For a fixed cell line, each point is one shared drug: viability in dataset A vs dataset B.
A tight monotone cloud (high rho) = the cell line's drug ranking replicates; a diffuse cloud
(low rho) = it doesn't. Scales differ across assays, so rho (rank) is the agreement, not y=x.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/sub/analysis")
A, B = "GDSC", "CTRP"          # dataset pair to compare
MIN_DRUGS = 20                 # cell lines need >= this many shared drugs

df = pd.read_parquet(OUT / "drug_viability_table.parquet")
agg = {d: df[df.dataset == d].groupby(["cell_line", "smiles"], as_index=False)["viability"].mean()
       for d in (A, B)}
j = agg[A].merge(agg[B], on=["cell_line", "smiles"], suffixes=("_a", "_b"))

recs = []
for cl, g in j.groupby("cell_line"):
    if g["smiles"].nunique() >= MIN_DRUGS and g.viability_a.std() > 0 and g.viability_b.std() > 0:
        rho = spearmanr(g.viability_a, g.viability_b).statistic
        if np.isfinite(rho):
            recs.append((cl, rho, len(g)))
cc = pd.DataFrame(recs, columns=["cell_line", "rho", "n_drugs"]).sort_values("rho").reset_index(drop=True)
print(f"{A}-{B}: {len(cc)} cell lines (>= {MIN_DRUGS} shared drugs)")
print(f"  per-cell-line rho: median {cc.rho.median():.2f}, range [{cc.rho.min():.2f}, {cc.rho.max():.2f}], "
      f"IQR [{cc.rho.quantile(.25):.2f}, {cc.rho.quantile(.75):.2f}]")

# 9 cell lines evenly spanning low -> high rho
sel = cc.iloc[np.linspace(0, len(cc) - 1, 9).round().astype(int)]

fig, axes = plt.subplots(3, 3, figsize=(11, 11))
for ax, (_, r) in zip(axes.ravel(), sel.iterrows()):
    g = j[j.cell_line == r.cell_line]
    ax.scatter(g.viability_a, g.viability_b, s=16, alpha=0.6)
    ax.set_title(f"{r.cell_line}   ρ={r.rho:.2f}  (n={int(r.n_drugs)})", fontsize=10)
    ax.set_xlabel(f"{A} viability"); ax.set_ylabel(f"{B} viability")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
fig.suptitle(f"Per-cell-line drug-response agreement ({A} vs {B}) — low → high ρ", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(OUT / "percellline_scatter_grid.png", dpi=120)

plt.figure(figsize=(5.5, 3.2))
plt.hist(cc.rho, bins=30, color="steelblue")
plt.axvline(cc.rho.median(), color="r", ls="--", label=f"median {cc.rho.median():.2f}")
plt.xlabel("per-cell-line Spearman ρ (drug ranking)"); plt.ylabel("cell lines")
plt.title(f"{A} vs {B}: spread of per-cell-line agreement"); plt.legend()
plt.tight_layout(); plt.savefig(OUT / "percellline_rho_hist.png", dpi=120)

print("\nselected cell lines (low -> high ρ):")
print(sel.to_string(index=False))
print(f"\nlowest-agreement: {cc.iloc[:3]['cell_line'].tolist()} (ρ {cc.iloc[:3]['rho'].round(2).tolist()})")
print(f"highest-agreement: {cc.iloc[-3:]['cell_line'].tolist()} (ρ {cc.iloc[-3:]['rho'].round(2).tolist()})")
print("saved -> percellline_scatter_grid.png + percellline_rho_hist.png")
