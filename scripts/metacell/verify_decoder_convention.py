"""Reconcile the decoder-ceiling delta R² with recon_emb_sanity's 0.665.

recon_emb_sanity scores: rd = decoded_treated - REAL_ctrl (rm.delta_metrics with ctrl_mean from
real genes). My ceiling used: rd = decoded_treated - DECODED_ctrl (cancels the decoder offset ->
higher). This recomputes X_state both ways on the same test split. Runs in `pancellflow`.
"""
import sys
sys.path.insert(0, "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/sub/src")
sys.path.insert(0, "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/sub/experiments")
import numpy as np, anndata as ad
import recon_metrics as rm
from scaleflow.model._recon import ReconDecoder

BASE = "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow"
A = ad.read_h5ad(f"{BASE}/outputs/metacell/sciplex_all_for_metacell.h5ad")
genes = np.asarray(A.X.todense(), np.float32) if hasattr(A.X, "todense") else np.asarray(A.X, np.float32)
Xst = np.asarray(A.obsm["X_state"], np.float32)
cl = A.obs["cell_line"].astype(str).to_numpy(); drug = A.obs["drug"].astype(str).to_numpy()
ctrl = A.obs["control"].to_numpy().astype(bool); split = A.obs["split"].astype(str).to_numpy()

dec = ReconDecoder.load(f"{BASE}/recon_weights/dec_X_state_opt/ReconDecoder.pkl")
te = np.where(split == "test")[0]                       # treated test cells (recon_emb_sanity's idx['test'])
pred = np.asarray(dec.decode(Xst[te]), np.float32)      # decoded genes
ctrl_mean = rm.control_means(genes, cl, ctrl)           # REAL per-line control mean

# (1) recon_emb_sanity convention: real ctrl for both true and pred
dm_real = rm.delta_metrics(genes[te], pred, cl[te], drug[te], (~ctrl)[te], ctrl_mean, min_cells=20)
print("X_state, REAL-ctrl convention (== recon_emb_sanity):")
print(f"  delta_r2_median={dm_real['r2_median']:.3f}  pearson_median={dm_real['pearson_median']:.3f}  n={dm_real['n']}")

# (2) my ceiling convention: decoded ctrl (cancels decoder offset)
dec_ctrl = {c: np.asarray(dec.decode(Xst[ctrl & (cl == c)].mean(0)[None]), np.float32)[0]
            for c in sorted(set(cl))}
from sklearn.metrics import r2_score
r2s = []
for c in sorted(set(cl)):
    for d in np.unique(drug[(~ctrl) & (cl == c) & (split == "test")]):
        m = (~ctrl) & (cl == c) & (drug == d)
        if m.sum() < 20:
            continue
        td = genes[m].mean(0) - ctrl_mean[c]
        rd = pred[(split[te] == "test")][None]  # placeholder; recompute below
# simpler decoded-ctrl pass over te
r2s = []
dec_all = pred  # decoded test cells aligned to te
for c in sorted(set(cl)):
    for d in np.unique(drug[te][cl[te] == c]):
        mm = (cl[te] == c) & (drug[te] == d)
        if mm.sum() < 20:
            continue
        td = genes[te][mm].mean(0) - ctrl_mean[c]
        rd = dec_all[mm].mean(0) - dec_ctrl[c]
        r2s.append(r2_score(td, rd))
print(f"\nX_state, DECODED-ctrl convention (my ceiling): delta_r2_median={np.median(r2s):.3f}  n={len(r2s)}")
