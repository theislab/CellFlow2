"""Deployment-relevant gene-space metric: decode each metacell's X_state centroid through the
ReconDecoder and score the perturbation delta vs real genes. This is the metric that applies to
flow-PREDICTED metacells (which have no member cells / no GT genes). Runs in `pancellflow`.

Two baseline conventions for the delta, reported side by side:
  A_realctrl : decoded_treated - REAL control mean   (== recon_emb_sanity; stricter, penalizes
               the decoder's absolute calibration offset; comparable to the 0.665 reference)
  B_decctrl  : decoded_treated - DECODED control     (symmetric: both arms through the decoder,
               so the decoder's shared per-gene offset cancels -> isolates the perturbation EFFECT)

Reads decode_inputs.pkl (per kernel: metacell table + X_state centroids) saved by the notebook.
"""
import pickle
import sys

sys.path.insert(0, "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/sub/src")
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from scaleflow.model._recon import ReconDecoder

BASE = "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow"
H5 = f"{BASE}/outputs/metacell/sciplex_all_for_metacell.h5ad"
DEC = f"{BASE}/recon_weights/dec_X_state_opt/ReconDecoder.pkl"
INP = f"{BASE}/outputs/metacell/decode_inputs.pkl"
DEG_K = 50


def deg_jaccard(a, b, k=DEG_K):
    sa = set(np.argsort(-np.abs(a))[:k]); sb = set(np.argsort(-np.abs(b))[:k])
    return len(sa & sb) / len(sa | sb)


def main():
    dec = ReconDecoder.load(DEC)
    inp = pickle.load(open(INP, "rb"))
    A = ad.read_h5ad(H5)
    genes = np.asarray(A.X.todense(), np.float32) if hasattr(A.X, "todense") else np.asarray(A.X, np.float32)
    Xst = np.asarray(A.obsm["X_state"], np.float32)
    cl = A.obs["cell_line"].astype(str).to_numpy(); drug = A.obs["drug"].astype(str).to_numpy()
    ctrl = A.obs["control"].to_numpy().astype(bool); split = A.obs["split"].astype(str).to_numpy()
    pert = ~ctrl; LINES = sorted(set(cl))
    drug_split = {d: split[pert & (drug == d)][0] for d in np.unique(drug[pert])}

    real_ctrl = {c: genes[ctrl & (cl == c)].mean(0) for c in LINES}                       # real control mean
    dec_ctrl = {c: np.asarray(dec.decode(Xst[ctrl & (cl == c)].mean(0)[None]), np.float32)[0]
                for c in LINES}                                                            # decode(mean control X_state)
    real_delta = {}
    for c in LINES:
        for d in np.unique(drug[pert & (cl == c)]):
            m = pert & (cl == c) & (drug == d)
            if m.sum() >= 20:
                real_delta[(c, d)] = genes[m].mean(0) - real_ctrl[c]

    rows = []
    for k, blob in inp.items():
        mc = blob["mc"]; dmc = np.asarray(dec.decode(np.asarray(blob["mc_xstate"], np.float32)), np.float32)
        for (c, d), rd in real_delta.items():
            sel = (mc["cell_line"].values == c) & (mc["drug"].values == d)
            if not sel.any() or rd.std() == 0:
                continue
            treated = dmc[sel].mean(0)                          # equal-weight decoded metacell pseudobulk
            row = dict(kernel=k, cell_line=c, split=drug_split[d], drug=d)
            for tag, base in (("A_realctrl", real_ctrl[c]), ("B_decctrl", dec_ctrl[c])):
                dd = treated - base
                if dd.std() == 0:
                    continue
                row[f"delta_r2_{tag}"] = float(r2_score(rd, dd))
                row[f"delta_pearson_{tag}"] = float(np.corrcoef(rd, dd)[0, 1])
                row[f"deg_jaccard_{tag}"] = float(deg_jaccard(rd, dd))
            rows.append(row)

    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c.startswith(("delta_", "deg_"))]
    print("\n=== DECODED METACELL (decode(metacell X_state) vs real genes) — median over conditions ===")
    print("\nby kernel x split:")
    print(df.groupby(["kernel", "split"])[cols].median().round(3))
    print("\nby kernel x cell_line (test):")
    print(df[df.split == "test"].groupby(["kernel", "cell_line"])[cols].median().round(3))
    df.to_csv(f"{BASE}/outputs/metacell/decoded_metacell.csv", index=False)
    print(f"\nsaved -> outputs/metacell/decoded_metacell.csv")


if __name__ == "__main__":
    main()
