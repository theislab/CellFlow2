"""Clean comparison: (a) per-cell decode  vs  size-weighted metacell  (== decode each cell by its
metacell centroid). Both cell-weighted, SAME control baseline, so the only difference is per-cell
own-X_state vs the cell's metacell centroid -> isolates the metacell-smoothing cost in decoded
gene space. Convention B (decoded control). Runs in `pancellflow`.
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

print("decoding all cells …", flush=True)
dgenes = np.asarray(dec.decode(Xst), np.float32)                       # per-cell decode (for (a) + shared ctrl)
real_ctrl = {c: genes[ctrl & (cl == c)].mean(0) for c in LINES}
dec_ctrl = {c: dgenes[ctrl & (cl == c)].mean(0) for c in LINES}        # SHARED control = mean-of-decode

dmc = {k: np.asarray(dec.decode(np.asarray(b["mc_xstate"], np.float32)), np.float32) for k, b in inp.items()}

rows = []
for c in LINES:
    for d in np.unique(drug[pert & (cl == c)]):
        m = pert & (cl == c) & (drug == d)
        if m.sum() < 20:
            continue
        td = genes[m].mean(0) - real_ctrl[c]
        da = dgenes[m].mean(0) - dec_ctrl[c]                            # (a) per-cell decode
        r = dict(cell_line=c, split=drug_split[d], r2_percell=float(r2_score(td, da)))
        for k, b in inp.items():
            sel = (b["mc"]["cell_line"].values == c) & (b["mc"]["drug"].values == d)
            if sel.any():
                sw = np.average(dmc[k][sel], axis=0, weights=b["mc"]["size"].values[sel]) - dec_ctrl[c]
                r[f"r2_metacell_{k}"] = float(r2_score(td, sw))         # size-weighted metacell
        rows.append(r)
df = pd.DataFrame(rows)
cols = ["r2_percell"] + [f"r2_metacell_{k}" for k in inp]
pd.set_option("display.width", 200)
print("\n=== (a) per-cell decode  vs  size-weighted metacell (conv B, same ctrl) — test, per line ===")
print(df[df.split == "test"].groupby("cell_line")[cols].median().round(3))
print("\n=== overall (test) ===")
print(df[df.split == "test"][cols].median().round(3).to_dict())
df.to_csv(f"{BASE}/outputs/metacell/decode_compare.csv", index=False)
