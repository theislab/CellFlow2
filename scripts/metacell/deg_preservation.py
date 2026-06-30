"""Significant-DEG preservation: overlap + magnitude correlation, for the per-cell-decode pipeline
and the size-weighted metacell pipeline (each self-consistent: both arms decoded the same way).

Per (cell_line, drug):
  - true DEGs S: Welch t-test (true perturbed cells vs true control cells), BH; padj<0.05 & |LFC|>=LFC.
  - true delta = mean(true_ptb) - mean(true_ctrl).
  - per-cell pipeline:  ptb = mean(decode(cell));            ctrl = mean(decode(control cell)).
  - metacell pipeline:  ptb = size-weighted decode(centroid); ctrl = decode(mean control X_state).
  Metrics on the significant-DEG subset S:
    deg_r2 / deg_pearson  : magnitude correlation of true vs decoded delta on S
    deg_sign              : fraction of S with matching sign
    deg_recall            : |S ∩ (decoded top-|S| genes)| / |S|   (overall DEG overlap)
Runs in `pancellflow`.
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


def subset_metrics(true_md, dec_md, sig):
    ts, ds = true_md[sig], dec_md[sig]
    topd = set(np.argsort(-np.abs(dec_md))[:int(sig.sum())]); trueset = set(np.where(sig)[0])
    return dict(deg_r2=float(r2_score(ts, ds)), deg_pearson=float(np.corrcoef(ts, ds)[0, 1]),
                deg_sign=float(np.mean(np.sign(ts) == np.sign(ds))),
                deg_recall=len(trueset & topd) / len(trueset))


def main():
    dec = ReconDecoder.load(f"{BASE}/recon_weights/dec_X_state_opt/ReconDecoder.pkl")
    inp = pickle.load(open(f"{BASE}/outputs/metacell/decode_inputs.pkl", "rb"))
    A = ad.read_h5ad(f"{BASE}/outputs/metacell/sciplex_all_for_metacell.h5ad")
    genes = np.asarray(A.X.todense(), np.float32) if hasattr(A.X, "todense") else np.asarray(A.X, np.float32)
    Xst = np.asarray(A.obsm["X_state"], np.float32)
    cl = A.obs["cell_line"].astype(str).to_numpy(); drug = A.obs["drug"].astype(str).to_numpy()
    ctrl = A.obs["control"].to_numpy().astype(bool); split = A.obs["split"].astype(str).to_numpy()
    pert = ~ctrl; LINES = sorted(set(cl)); drug_split = {d: split[pert & (drug == d)][0] for d in np.unique(drug[pert])}
    var_names = A.var_names.to_numpy()
    padj = {c: wilcoxon_padj(genes, var_names, cl, drug, ctrl, c, list(np.unique(drug[pert & (cl == c)]))) for c in LINES}

    print("decoding all cells …", flush=True)
    dgenes = np.asarray(dec.decode(Xst), np.float32)
    real_ctrl = {c: genes[ctrl & (cl == c)].mean(0) for c in LINES}
    pc_ctrl = {c: dgenes[ctrl & (cl == c)].mean(0) for c in LINES}                         # per-cell ctrl
    mc_ctrl = {c: np.asarray(dec.decode(Xst[ctrl & (cl == c)].mean(0)[None]), np.float32)[0] for c in LINES}  # metacell ctrl
    dmc = {k: np.asarray(dec.decode(np.asarray(b["mc_xstate"], np.float32)), np.float32) for k, b in inp.items()}

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
            row = dict(cell_line=c, split=drug_split[d], n_deg=int(sig.sum()))
            # per-cell pipeline
            row.update({f"pc_{k}": v for k, v in subset_metrics(true_md, dgenes[m].mean(0) - pc_ctrl[c], sig).items()})
            # size-weighted metacell pipeline (X_state kernel)
            b = inp["X_state"]; sel = (b["mc"]["cell_line"].values == c) & (b["mc"]["drug"].values == d)
            if sel.any():
                sw = np.average(dmc["X_state"][sel], axis=0, weights=b["mc"]["size"].values[sel]) - mc_ctrl[c]
                row.update({f"mc_{k}": v for k, v in subset_metrics(true_md, sw, sig).items()})
            rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    show = ["n_deg", "pc_deg_r2", "mc_deg_r2", "pc_deg_pearson", "mc_deg_pearson",
            "pc_deg_sign", "mc_deg_sign", "pc_deg_recall", "mc_deg_recall"]
    print(f"\n=== significant-DEG preservation (padj<{PADJ} & |LFC|>={LFC}); pc=per-cell, mc=size-wt metacell ===")
    print("\ntest, by cell_line (median):")
    print(df[df.split == "test"].groupby("cell_line")[show].median().round(3))
    print("\noverall test (median):", {k: round(float(df[df.split == "test"][k].median()), 3) for k in show})
    df.to_csv(f"{BASE}/outputs/metacell/deg_preservation.csv", index=False)


if __name__ == "__main__":
    main()
