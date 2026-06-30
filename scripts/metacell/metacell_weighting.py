"""Equal-weight vs size-weighted decoded-metacell delta (convention B), per line.

equal : pseudobulk = mean over the condition's metacells of decode(centroid)      (flow's view:
        each metacell is one equal-weight sample)
sizew : pseudobulk = size-weighted mean (each metacell weighted by its #cells k == "repeat k
        times") -> recovers the cell-weighted population mean.
Both minus decode(mean control X_state). Runs in `pancellflow`.
"""
import pickle, sys
sys.path.insert(0, "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/sub/src")
import anndata as ad, numpy as np, pandas as pd
from sklearn.metrics import r2_score
from scaleflow.model._recon import ReconDecoder

BASE = "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow"
dec = ReconDecoder.load(f"{BASE}/recon_weights/dec_X_state_opt/ReconDecoder.pkl")
inp = pickle.load(open(f"{BASE}/outputs/metacell/decode_inputs.pkl", "rb"))
A = ad.read_h5ad(f"{BASE}/outputs/metacell/sciplex_all_for_metacell.h5ad")
genes = np.asarray(A.X.todense(), np.float32) if hasattr(A.X, "todense") else np.asarray(A.X, np.float32)
Xst = np.asarray(A.obsm["X_state"], np.float32)
cl = A.obs["cell_line"].astype(str).to_numpy(); drug = A.obs["drug"].astype(str).to_numpy()
ctrl = A.obs["control"].to_numpy().astype(bool); split = A.obs["split"].astype(str).to_numpy()
pert = ~ctrl; LINES = sorted(set(cl))
drug_split = {d: split[pert & (drug == d)][0] for d in np.unique(drug[pert])}
real_ctrl = {c: genes[ctrl & (cl == c)].mean(0) for c in LINES}
dec_ctrl = {c: np.asarray(dec.decode(Xst[ctrl & (cl == c)].mean(0)[None]), np.float32)[0] for c in LINES}
true_delta = {(c, d): genes[pert & (cl == c) & (drug == d)].mean(0) - real_ctrl[c]
              for c in LINES for d in np.unique(drug[pert & (cl == c)])
              if (pert & (cl == c) & (drug == d)).sum() >= 20}

rows = []
for k, blob in inp.items():
    mc = blob["mc"]; dmc = np.asarray(dec.decode(np.asarray(blob["mc_xstate"], np.float32)), np.float32)
    size = mc["size"].values.astype(float)
    for (c, d), td in true_delta.items():
        sel = (mc["cell_line"].values == c) & (mc["drug"].values == d)
        if not sel.any():
            continue
        eq = dmc[sel].mean(0) - dec_ctrl[c]
        sw = np.average(dmc[sel], axis=0, weights=size[sel]) - dec_ctrl[c]
        rows.append(dict(kernel=k, cell_line=c, split=drug_split[d],
                         r2_equal=float(r2_score(td, eq)), r2_sizew=float(r2_score(td, sw))))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n=== decoded-metacell delta R2 (conv B): EQUAL-weight vs SIZE-weighted, test, per kernel x line ===")
print(df[df.split == "test"].groupby(["kernel", "cell_line"])[["r2_equal", "r2_sizew"]].median().round(3))
print("\n=== overall (test), per kernel ===")
print(df[df.split == "test"].groupby("kernel")[["r2_equal", "r2_sizew"]].median().round(3))
df.to_csv(f"{BASE}/outputs/metacell/metacell_weighting.csv", index=False)
