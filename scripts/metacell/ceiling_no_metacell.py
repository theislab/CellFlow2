"""Per-cell-line cell counts + the NO-metacell decoder ceiling (convention B), to compare against
the decoded-metacell numbers. Two no-metacell ceilings:
  (a) mean-of-decode : mean over cells of decode(cell X_state)  - decoded control  [from decoder_ceiling.csv]
  (b) decode-of-mean : decode(mean perturbed X_state) - decode(mean control X_state)
The metacell decodes a CENTROID (a mean), so (b) is the apples-to-apples single-cell ceiling;
(a) is the best-case (per-cell decode). Runs in `pancellflow`.
"""
import sys
sys.path.insert(0, "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/sub/src")
import anndata as ad, numpy as np, pandas as pd
from sklearn.metrics import r2_score
from scaleflow.model._recon import ReconDecoder

BASE = "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow"
A = ad.read_h5ad(f"{BASE}/outputs/metacell/sciplex_all_for_metacell.h5ad")
genes = np.asarray(A.X.todense(), np.float32) if hasattr(A.X, "todense") else np.asarray(A.X, np.float32)
Xst = np.asarray(A.obsm["X_state"], np.float32)
cl = A.obs["cell_line"].astype(str).to_numpy(); drug = A.obs["drug"].astype(str).to_numpy()
ctrl = A.obs["control"].to_numpy().astype(bool); split = A.obs["split"].astype(str).to_numpy()
pert = ~ctrl; LINES = sorted(set(cl))
drug_split = {d: split[pert & (drug == d)][0] for d in np.unique(drug[pert])}

print("=== cell counts per line ===")
for c in LINES:
    nctrl = int((ctrl & (cl == c)).sum()); npert = int((pert & (cl == c)).sum())
    per_drug = [int((pert & (cl == c) & (drug == d)).sum()) for d in np.unique(drug[pert & (cl == c)])]
    ntest = sum(drug_split[d] == "test" for d in np.unique(drug[pert & (cl == c)]))
    print(f"  {c:6s}: total={nctrl+npert:7,d}  control={nctrl:6,d}  perturbed={npert:7,d}  "
          f"median cells/drug={int(np.median(per_drug)):5d}  (test drugs={ntest})")

dec = ReconDecoder.load(f"{BASE}/recon_weights/dec_X_state_opt/ReconDecoder.pkl")
dec_ctrl = {c: np.asarray(dec.decode(Xst[ctrl & (cl == c)].mean(0)[None]), np.float32)[0] for c in LINES}

rows = []
for c in LINES:
    rc = genes[ctrl & (cl == c)].mean(0)
    for d in np.unique(drug[pert & (cl == c)]):
        m = pert & (cl == c) & (drug == d)
        if m.sum() < 20:
            continue
        true_md = genes[m].mean(0) - rc
        dec_md = np.asarray(dec.decode(Xst[m].mean(0)[None]), np.float32)[0] - dec_ctrl[c]   # (b) decode-of-mean, conv B
        rows.append(dict(cell_line=c, split=drug_split[d], delta_r2_b=float(r2_score(true_md, dec_md)),
                         delta_pearson_b=float(np.corrcoef(true_md, dec_md)[0, 1])))
df = pd.DataFrame(rows)
print("\n=== (b) decode-of-mean pseudobulk ceiling, NO metacell (convention B), test ===")
print(df[df.split == "test"].groupby("cell_line")[["delta_r2_b", "delta_pearson_b"]].median().round(3))

# (a) per-cell-decode ceiling from the earlier run
try:
    ca = pd.read_csv(f"{BASE}/outputs/metacell/decoder_ceiling.csv")
    print("\n=== (a) mean-of-decode (per-cell) ceiling, NO metacell (convention B), test ===")
    print(ca[ca.split == "test"].groupby("cell_line")[["delta_r2", "delta_pearson"]].median().round(3))
except Exception as e:
    print("decoder_ceiling.csv:", e)
