"""Decoder ceiling: decode single-cell X_state -> genes and measure the perturbation-delta
fidelity vs real genes (no metacells). This is the upper bound the decoded-metacell metric
approaches, since metacell aggregation conserves the X_state pseudobulk. Runs in `pancellflow`.
"""
import sys
sys.path.insert(0, "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/sub/src")
import numpy as np, anndata as ad
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from scaleflow.model._recon import ReconDecoder

H5 = "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/metacell/sciplex_all_for_metacell.h5ad"
DEC = "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/recon_weights/dec_X_state_opt/ReconDecoder.pkl"
DEG_K = 50

print("loading decoder + data …", flush=True)
dec = ReconDecoder.load(DEC)
A = ad.read_h5ad(H5)
genes = np.asarray(A.X.todense(), np.float32) if hasattr(A.X, "todense") else np.asarray(A.X, np.float32)
Xst = np.asarray(A.obsm["X_state"], np.float32)
cl = A.obs["cell_line"].astype(str).to_numpy(); drug = A.obs["drug"].astype(str).to_numpy()
ctrl = A.obs["control"].to_numpy().astype(bool); split = A.obs["split"].astype(str).to_numpy()
pert = ~ctrl; LINES = sorted(set(cl))
drug_split = {d: split[pert & (drug == d)][0] for d in np.unique(drug[pert])}

print("decoding all cells X_state -> genes …", flush=True)
dgenes = dec.decode(Xst)                                  # (n, 2000) decoded gene space
dgenes = np.asarray(dgenes, np.float32)

real_ctrl = {c: genes[ctrl & (cl == c)].mean(0) for c in LINES}
dec_ctrl = {c: dgenes[ctrl & (cl == c)].mean(0) for c in LINES}

rows = []
for c in LINES:
    for d in np.unique(drug[pert & (cl == c)]):
        m = pert & (cl == c) & (drug == d)
        if m.sum() < 20:
            continue
        rd = genes[m].mean(0) - real_ctrl[c]              # real delta
        dd = dgenes[m].mean(0) - dec_ctrl[c]              # decoded delta
        if rd.std() == 0 or dd.std() == 0:
            continue
        jac = len(set(np.argsort(-np.abs(rd))[:DEG_K]) & set(np.argsort(-np.abs(dd))[:DEG_K])) \
            / len(set(np.argsort(-np.abs(rd))[:DEG_K]) | set(np.argsort(-np.abs(dd))[:DEG_K]))
        rows.append(dict(cell_line=c, split=drug_split[d], drug=d,
                         delta_r2=r2_score(rd, dd), delta_pearson=np.corrcoef(rd, dd)[0, 1],
                         deg_spearman=spearmanr(rd, dd).correlation, deg_jaccard=jac))

import pandas as pd
df = pd.DataFrame(rows)
print("\n=== DECODER CEILING (single-cell X_state -> decode -> genes vs real) ===")
print("overall (median over conditions):")
print(df[["delta_r2", "delta_pearson", "deg_spearman", "deg_jaccard"]].median().round(3).to_dict())
print("\nby split:")
print(df.groupby("split")[["delta_r2", "delta_pearson", "deg_spearman", "deg_jaccard"]].median().round(3))
print("\nby cell_line (test):")
print(df[df.split == "test"].groupby("cell_line")[["delta_r2", "delta_pearson", "deg_spearman", "deg_jaccard"]].median().round(3))
df.to_csv("/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/metacell/decoder_ceiling.csv", index=False)
print("\nsaved -> outputs/metacell/decoder_ceiling.csv")
