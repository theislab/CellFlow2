"""Clean 3-way DEG/delta comparison, each arm self-consistent:
  1. truth          : mean(real ptb)              - mean(real ctrl)
  2. decode-then-avg : mean(decode(ptb cells))    - mean(decode(ctrl cells))   [correct decoded pipeline;
                       == per-cell == size-weighted decode-then-avg metacell]
  3. decode-of-mean  : decode(ptb metacell centroid, size-wt) - decode(ctrl metacell centroid, size-wt)
                       [deployment-forced: predicted metacell is a mean with no member cells]
Rows 2 & 3 are scored against row 1 (all-gene delta R²; and on the true significant DEGs: magnitude
R², sign-concordance, recall). Row3 needs control metacells (build_control_metacells.py). `pancellflow`.
"""
import pickle, sys
sys.path.insert(0, "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/sub/src")
import anndata as ad, numpy as np, pandas as pd
from sklearn.metrics import r2_score
from scaleflow.model._recon import ReconDecoder

BASE = "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow"
PADJ, LFC, MIN_PTB, MIN_DEG = 0.05, 0.25, 20, 10


def wilcoxon_padj(genes, var_names, cl, drug, ctrl, line, drugs):
    """Per-gene BH-adjusted Wilcoxon p (scanpy rank_genes_groups): each drug vs control, for `line`."""
    import scanpy as sc, anndata as ad
    msk = (cl == line) & (ctrl | np.isin(drug, drugs))
    grp = np.where(ctrl[msk], "control", drug[msk]).astype(object)
    sub = ad.AnnData(genes[msk].copy(), obs=pd.DataFrame({"group": pd.Categorical(grp)},
                                                         index=np.arange(int(msk.sum())).astype(str)))
    sub.var_names = var_names; sub.uns["log1p"] = {"base": None}
    sc.tl.rank_genes_groups(sub, "group", groups=list(drugs), reference="control", method="wilcoxon")
    return {d: np.nan_to_num(sc.get.rank_genes_groups_df(sub, group=d).set_index("names")["pvals_adj"]
                             .reindex(var_names).values, nan=1.0) for d in drugs}


def metrics(true_md, md, sig):
    topd = set(np.argsort(-np.abs(md))[:int(sig.sum())]); ts = set(np.where(sig)[0])
    return (float(r2_score(true_md, md)), float(r2_score(true_md[sig], md[sig])),
            float(np.mean(np.sign(true_md[sig]) == np.sign(md[sig]))), len(ts & topd) / len(ts))


def main():
    dec = ReconDecoder.load(f"{BASE}/recon_weights/dec_X_state_opt/ReconDecoder.pkl")
    pm = pickle.load(open(f"{BASE}/outputs/metacell/decode_inputs.pkl", "rb"))["X_state"]
    cm = pickle.load(open(f"{BASE}/outputs/metacell/control_metacells.pkl", "rb"))
    A = ad.read_h5ad(f"{BASE}/outputs/metacell/sciplex_all_for_metacell.h5ad")
    genes = np.asarray(A.X.todense(), np.float32) if hasattr(A.X, "todense") else np.asarray(A.X, np.float32)
    Xst = np.asarray(A.obsm["X_state"], np.float32)
    cl = A.obs["cell_line"].astype(str).to_numpy(); drug = A.obs["drug"].astype(str).to_numpy()
    ctrl = A.obs["control"].to_numpy().astype(bool); split = A.obs["split"].astype(str).to_numpy()
    pert = ~ctrl; LINES = sorted(set(cl)); drug_split = {d: split[pert & (drug == d)][0] for d in np.unique(drug[pert])}
    var_names = A.var_names.to_numpy()
    padj = {c: wilcoxon_padj(genes, var_names, cl, drug, ctrl, c, list(np.unique(drug[pert & (cl == c)]))) for c in LINES}

    print("decoding all cells (row 2) …", flush=True)
    dgenes = np.asarray(dec.decode(Xst), np.float32)
    real_ctrl = {c: genes[ctrl & (cl == c)].mean(0) for c in LINES}
    sc_ctrl = {c: dgenes[ctrl & (cl == c)].mean(0) for c in LINES}                    # row2 ctrl (decode-then-avg)
    mc_ctrl = {c: np.average(np.asarray(dec.decode(cm[c]["xstate"]), np.float32), axis=0,
                             weights=cm[c]["size"]) for c in LINES}                    # row3 ctrl (decode-of-mean, size-wt)
    dpm = np.asarray(dec.decode(np.asarray(pm["mc_xstate"], np.float32)), np.float32)  # perturbed metacell decoded

    rows = []
    for c in LINES:
        gc = genes[ctrl & (cl == c)]
        for d in np.unique(drug[pert & (cl == c)]):
            m = pert & (cl == c) & (drug == d)
            if m.sum() < MIN_PTB:
                continue
            true_md = genes[m].mean(0) - real_ctrl[c]
            sig = (padj[c][d] < PADJ) & (np.abs(true_md) >= LFC)
            if sig.sum() < MIN_DEG:
                continue
            r2a = dgenes[m].mean(0) - sc_ctrl[c]                                       # row2
            sel = (pm["mc"]["cell_line"].values == c) & (pm["mc"]["drug"].values == d)
            if not sel.any():
                continue
            r3 = np.average(dpm[sel], axis=0, weights=pm["mc"]["size"].values[sel]) - mc_ctrl[c]  # row3
            a_all, a_deg, a_sign, a_rec = metrics(true_md, r2a, sig)
            m_all, m_deg, m_sign, m_rec = metrics(true_md, r3, sig)
            rows.append(dict(cell_line=c, split=drug_split[d], n_deg=int(sig.sum()),
                             r2_all_decodeAvg=a_all, r2_all_decodeMean=m_all,
                             deg_r2_decodeAvg=a_deg, deg_r2_decodeMean=m_deg,
                             deg_sign_decodeAvg=a_sign, deg_sign_decodeMean=m_sign,
                             deg_recall_decodeAvg=a_rec, deg_recall_decodeMean=m_rec))
    df = pd.DataFrame(rows); pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    cols = [c for c in df.columns if c not in ("cell_line", "split")]
    print("\n=== 3-way (row1=truth ref) : decode-then-avg vs decode-of-mean metacell — test, per line (median) ===")
    print(df[df.split == "test"].groupby("cell_line")[cols].median().round(3).T)
    df.to_csv(f"{BASE}/outputs/metacell/decode_3way.csv", index=False)
    print("\nsaved -> outputs/metacell/decode_3way.csv")


if __name__ == "__main__":
    main()
