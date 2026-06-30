"""Pre-compute the gene-space arrays for the true/decoded/decoded-metacell UMAP overlay, so the
seacells-kernel metacell_effect.ipynb can render the UMAP without a decoder. Runs in `pancellflow`.
Saves outputs/metacell/umap_overlay.npz = {Z (genes), groups, condition}.
"""
import pickle, sys
sys.path.insert(0, "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/sub/src")
import anndata as ad, numpy as np
from scaleflow.model._recon import ReconDecoder

BASE = "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow"
dec = ReconDecoder.load(f"{BASE}/recon_weights/dec_X_state_opt/ReconDecoder.pkl")
pm = pickle.load(open(f"{BASE}/outputs/metacell/decode_inputs.pkl", "rb"))["X_state"]
cm = pickle.load(open(f"{BASE}/outputs/metacell/control_metacells.pkl", "rb"))
A = ad.read_h5ad(f"{BASE}/outputs/metacell/sciplex_all_for_metacell.h5ad")
genes = np.asarray(A.X.todense(), np.float32) if hasattr(A.X, "todense") else np.asarray(A.X, np.float32)
Xst = np.asarray(A.obsm["X_state"], np.float32)
cl = A.obs["cell_line"].astype(str).to_numpy(); drug = A.obs["drug"].astype(str).to_numpy()
ctrl = A.obs["control"].to_numpy().astype(bool); split = A.obs["split"].astype(str).to_numpy()
pert = ~ctrl
c = "A549"; real_ctrl = genes[ctrl & (cl == c)].mean(0)
tdr = [d for d in np.unique(drug[pert & (cl == c)]) if split[pert & (drug == d)][0] == "test"]
d = max(tdr, key=lambda d: np.linalg.norm(genes[pert & (cl == c) & (drug == d)].mean(0) - real_ctrl))
print(f"UMAP condition: {c} / {d}")

rng = np.random.default_rng(0)
ci = np.where(ctrl & (cl == c))[0]; pi = np.where(pert & (cl == c) & (drug == d))[0]
ci = rng.choice(ci, min(1200, len(ci)), replace=False); pi = rng.choice(pi, min(1200, len(pi)), replace=False)
selp = (pm["mc"]["cell_line"].values == c) & (pm["mc"]["drug"].values == d)
dec_ci = np.asarray(dec.decode(Xst[ci]), np.float32); dec_pi = np.asarray(dec.decode(Xst[pi]), np.float32)
mcc = np.asarray(dec.decode(cm[c]["xstate"]), np.float32)
mcp = np.asarray(dec.decode(np.asarray(pm["mc_xstate"], np.float32)[selp]), np.float32)

blocks = [("true_ctrl", genes[ci]), ("true_ptb", genes[pi]),
          ("dec_ctrl", dec_ci), ("dec_ptb", dec_pi), ("mc_ctrl", mcc), ("mc_ptb", mcp)]
Z = np.vstack([v for _, v in blocks]).astype(np.float32)
groups = np.concatenate([[k] * len(v) for k, v in blocks])
np.savez(f"{BASE}/outputs/metacell/umap_overlay.npz", Z=Z, groups=groups, condition=f"{c} / {d}")
print("saved umap_overlay.npz:", Z.shape, {k: int((groups == k).sum()) for k, _ in blocks})
