"""Builder for notebooks/metacell_effect.ipynb (run with any env that has nbformat).

The notebook itself runs in the `seacells` Jupyter kernel. Full sciplex: all 3 cell lines,
metacells built per (cell_line, drug) under three kernels, evaluated against the single-cell
ground truth with the reconstruction-notebook metrics + DEG preservation, split train/test and
broken down per cell line.
"""
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
co = lambda s: cells.append(new_code_cell(s))

md(r"""# Metacell effect on full sciplex — does SEACells aggregation lose the perturbation signal?

The metacell builder (`scripts/metacell/build_perturbation_metacells.py`) replaces each
`(cell_line, drug)` condition's perturbed cells with **SEACells metacells** (~`n/75` per condition,
each = the mean of its members); the flow then trains on those. **Question:** how much single-cell /
DEG signal is lost, and **does the kernel space matter**?

We compare three kernels SEACells can be built on, on **all three sciplex lines (A549, K-562, MCF7)**:

| kernel | space | dim |
|---|---|---|
| `X_state` | FM latent (what the flow lives in) | 2058 |
| `AE_10` | learned AE latent | 10 |
| `gene_PCA` | PCA of log-norm HVG expression (classic SEACells) | 2000→50 |

For all three we keep the cell→metacell assignment and **aggregate the real genes**, so every metric
is in gene space (no decoder needed). We report the reconstruction-notebook metrics — per-cell
reconstruction R², pseudobulk R², perturbation-delta R²/Pearson — **plus DEG preservation**, split by
**train/test** (held-out) drugs and broken down **per cell line**.

*Each metacell is the mean of its members → the condition **mean** is conserved by construction, so
pseudobulk stays near-ceiling (DEGs-by-fold-change not lost). The discriminative metrics are the
**per-cell reconstruction R²** (cell-level resolution kept) and the **equal-weight delta/DEG** (how
faithfully the metacell set reproduces the perturbation when used as equal-weight samples).*""")

co(r"""import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import SEACells
sys.path.insert(0, "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/sub/experiments")
from recon_metrics import pure_reconstruction

# ---- config ----
DATA   = "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/metacell/sciplex_all_for_metacell.h5ad"
KERNELS            = ["X_state", "AE_10", "gene_PCA"]   # gene_PCA = PCA of X (genes)
CELLS_PER_METACELL = 75
N_PCA              = 50
MIN_CELLS          = 50
SEED               = 42
DEG_K              = 50          # top-K genes for DEG-overlap
DEV_N_DRUGS        = None        # None = ALL 128 drugs (train+val+test, ~5-6h); int = top-N train + few val + all test (quick smoke)
print("SEACells", SEACells.__version__)""")

md("## 1. Load full sciplex — genes (ground truth) + the two latents, all 3 lines")

co(r"""adata = ad.read_h5ad(DATA)
genes = np.asarray(adata.X.todense(), dtype=np.float32) if hasattr(adata.X, "todense") else np.asarray(adata.X, np.float32)
emb = {"gene_PCA": genes, "X_state": adata.obsm["X_state"], "AE_10": adata.obsm["AE_10"]}
cl_arr = adata.obs["cell_line"].astype(str).to_numpy()
drug   = adata.obs["drug"].astype(str).to_numpy()
ctrl   = adata.obs["control"].to_numpy().astype(bool)
split  = adata.obs["split"].astype(str).to_numpy()
pert   = ~ctrl
LINES  = sorted(set(cl_arr))

# per-cell-line control baseline (gene space)
ctrl_mean = {cl: genes[ctrl & (cl_arr == cl)].mean(0) for cl in LINES}

# drug -> split (1:1 by construction); pick used drugs (most-abundant first for stable metacells)
drug_split = {d: split[pert & (drug == d)][0] for d in np.unique(drug[pert])}
drugs_all = sorted(drug_split, key=lambda d: -(pert & (drug == d)).sum())
if DEV_N_DRUGS:   # quick smoke: top-N train + a few val + all test
    tr = [d for d in drugs_all if drug_split[d] == "train"][:DEV_N_DRUGS]
    va = [d for d in drugs_all if drug_split[d] == "val"][:max(2, DEV_N_DRUGS // 2)]
    te = [d for d in drugs_all if drug_split[d] == "test"]
    drugs_use = tr + va + te
else:             # full: all 128 treated drugs
    drugs_use = [d for d in drugs_all if drug_split[d] in ("train", "val", "test")]

# conditions = (cell_line, drug) that actually have perturbed cells
conditions = [(cl, d) for cl in LINES for d in drugs_use if (pert & (cl_arr == cl) & (drug == d)).any()]
print(f"cells={adata.n_obs:,}  lines={LINES}  control={ctrl.sum():,}")
print(f"drugs used: {len(drugs_use)} ({sum(drug_split[d]=='train' for d in drugs_use)} train / "
      f"{sum(drug_split[d]=='val' for d in drugs_use)} val / "
      f"{sum(drug_split[d]=='test' for d in drugs_use)} test) -> {len(conditions)} conditions")""")

md("""## 2. Metacell builder (one SEACells run per condition)

Mirrors `build_perturbation_metacells.py`: PCA the kernel embedding → build SEACells kernel → fit →
hard assignments. Tiny conditions (`< MIN_CELLS` or `< 2` metacells) collapse to a single mean.""")

co(r"""def build_metacell_labels(e, n_per=CELLS_PER_METACELL, n_pca=N_PCA, min_cells=MIN_CELLS, seed=SEED):
    # per-cell metacell label (0..M-1) for one condition's embedding e (n x d)
    n = e.shape[0]
    if n <= min_cells or (n // n_per) < 2:
        return np.zeros(n, dtype=int)
    n_meta = max(2, n // n_per)
    n_comp = min(n_pca, n - 1, e.shape[1])
    epca = PCA(n_components=n_comp, random_state=seed).fit_transform(e).astype(np.float32)
    sub = ad.AnnData(X=np.zeros((n, 1), np.float32),
                     obs=pd.DataFrame(index=[str(i) for i in range(n)]), obsm={"X_pca": epca})
    m = SEACells.core.SEACells(sub, build_kernel_on="X_pca", n_SEACells=n_meta,
                               n_neighbors=min(15, n - 1), verbose=False)
    m.construct_kernel_matrix(); m.initialize_archetypes(); m.fit(min_iter=5, max_iter=30)
    return pd.factorize(m.get_hard_assignments()["SEACell"].reindex([str(i) for i in range(n)]).values)[0]


def run_kernel(kernel):
    # build metacells for every (cell_line, drug); aggregate REAL genes + X_state; keep mc centroids
    E = emb[kernel]; Xst = adata.obsm["X_state"]
    tg, pg, tx, px, pcl, pdr = [], [], [], [], [], []
    mc = []   # (cell_line, drug, split, size, gene_profile, kernel_centroid)
    t0 = time.time()
    for j, (cl, d) in enumerate(conditions):
        idx = np.where(pert & (cl_arr == cl) & (drug == d))[0]
        lab = build_metacell_labels(E[idx])
        for L in np.unique(lab):
            mem = idx[lab == L]
            gmean = genes[mem].mean(0); xmean = Xst[mem].mean(0); kmean = E[mem].mean(0)
            tg.append(genes[mem]); pg.append(np.repeat(gmean[None], len(mem), 0))
            tx.append(Xst[mem]);   px.append(np.repeat(xmean[None], len(mem), 0))
            pcl.append(np.repeat(cl, len(mem))); pdr.append(np.repeat(d, len(mem)))
            mc.append((cl, d, drug_split[d], len(mem), gmean, kmean, xmean))   # xmean = X_state centroid (to decode)
        if (j + 1) % 30 == 0:
            print(f"  [{kernel}] {j+1}/{len(conditions)} conditions ({time.time()-t0:.0f}s)", flush=True)
    return dict(
        true_g=np.concatenate(tg), pred_g=np.concatenate(pg),
        true_x=np.concatenate(tx), pred_x=np.concatenate(px),
        cl=np.concatenate(pcl), drug=np.concatenate(pdr),
        mc=pd.DataFrame({"cell_line": [r[0] for r in mc], "drug": [r[1] for r in mc],
                         "split": [r[2] for r in mc], "size": [r[3] for r in mc]}),
        mc_profiles=np.stack([r[4] for r in mc]), mc_kernel=np.stack([r[5] for r in mc]),
        mc_xstate=np.stack([r[6] for r in mc]),   # per-metacell X_state centroid -> decode at deployment
    )""")

md("""## 3. Metrics — per kernel × {train,test} × {ALL + each cell line}

- **recon_r2_gene / recon_r2_xstate**: per-cell variance-weighted R² of replacing a cell by its
  metacell mean (within-metacell homogeneity = cell-level resolution kept).
- **pseudobulk_r2**: metacell-sample mean vs single-cell mean (gene space).
- **delta_r2 / delta_pearson** (median over conditions): metacell-sample delta vs single-cell delta
  (control-subtracted per cell line).
- **deg_spearman / deg_jaccard@K**: Δ rank-corr and top-K DEG overlap (single-cell vs metacell).""")

co(r"""def metrics_rows(r, kernel):
    rows = []
    for sp in ["train", "val", "test"]:
        for line in ["ALL"] + LINES:
            cell = np.array([drug_split[d] == sp for d in r["drug"]])
            if line != "ALL":
                cell = cell & (r["cl"] == line)
            if not cell.any():
                continue
            rg = pure_reconstruction(r["true_g"][cell], r["pred_g"][cell])["reconstruction_r2"]
            rx = pure_reconstruction(r["true_x"][cell], r["pred_x"][cell])["reconstruction_r2"]
            mc = r["mc"]; msel = mc["split"].values == sp
            if line != "ALL":
                msel = msel & (mc["cell_line"].values == line)
            pb = float(r2_score(r["true_g"][cell].mean(0), r["mc_profiles"][msel].mean(0)))
            dr2, dpr, spr, jac = [], [], [], []
            conds = [(c, d) for (c, d) in conditions
                     if drug_split[d] == sp and (line == "ALL" or c == line)]
            for (c, d) in conds:
                cc = cell & (r["cl"] == c) & (r["drug"] == d)
                ms = (mc["cell_line"].values == c) & (mc["drug"].values == d)
                if not cc.any() or not ms.any():
                    continue
                sc = r["true_g"][cc].mean(0) - ctrl_mean[c]
                mcd = r["mc_profiles"][ms].mean(0) - ctrl_mean[c]
                if sc.std() == 0 or mcd.std() == 0:
                    continue
                dr2.append(float(r2_score(sc, mcd)))
                dpr.append(float(np.corrcoef(sc, mcd)[0, 1]))
                spr.append(spearmanr(sc, mcd).correlation)
                jac.append(len(set(np.argsort(-np.abs(sc))[:DEG_K]) & set(np.argsort(-np.abs(mcd))[:DEG_K]))
                           / len(set(np.argsort(-np.abs(sc))[:DEG_K]) | set(np.argsort(-np.abs(mcd))[:DEG_K])))
            rows.append(dict(kernel=kernel, split=sp, cell_line=line,
                             recon_r2_gene=rg, recon_r2_xstate=rx, pseudobulk_r2=pb,
                             delta_r2_median=float(np.median(dr2)) if dr2 else np.nan,
                             delta_pearson_median=float(np.median(dpr)) if dpr else np.nan,
                             deg_spearman_median=float(np.median(spr)) if spr else np.nan,
                             deg_jaccard_median=float(np.median(jac)) if jac else np.nan,
                             n_metacells=int(msel.sum()), n_conditions=len(dr2)))
    return rows""")

md("## 4. Build metacells under each kernel (heavy) + compute metrics, freeing per-cell arrays")

co(r"""all_rows, mc_store = [], {}
for k in KERNELS:
    print(f"=== {k} ===", flush=True)
    t = time.time(); r = run_kernel(k)
    all_rows += metrics_rows(r, k)
    mc_store[k] = {"mc": r["mc"], "mc_profiles": r["mc_profiles"], "mc_kernel": r["mc_kernel"],
                   "mc_xstate": r["mc_xstate"]}
    print(f"  -> {len(r['mc'])} metacells from {len(r['true_g']):,} cells "
          f"(mean size {r['mc']['size'].mean():.0f}) in {time.time()-t:.0f}s", flush=True)
    del r
res = pd.DataFrame(all_rows)

# save per-metacell X_state centroids so the decoder (pancellflow env) can produce the
# deployment-relevant gene-space metric: decode(metacell X_state) vs real genes.
import pickle
with open("/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/metacell/decode_inputs.pkl", "wb") as f:
    pickle.dump({k: {"mc": mc_store[k]["mc"], "mc_xstate": mc_store[k]["mc_xstate"]} for k in KERNELS}, f)
print("saved decode_inputs.pkl")
res""")

md("### Overall (ALL lines) and per-cell-line tables")

co(r"""overall = res[res.cell_line == "ALL"].drop(columns="cell_line").reset_index(drop=True)
print("OVERALL"); display(overall.round(3))
print("PER CELL LINE (test split)")
display(res[(res.cell_line != "ALL") & (res.split == "test")]
        .drop(columns=["n_metacells"]).round(3).reset_index(drop=True))""")

md("## 5. Compare kernels (overall)")

co(r"""mets = ["recon_r2_gene", "recon_r2_xstate", "pseudobulk_r2",
        "delta_r2_median", "delta_pearson_median", "deg_jaccard_median"]
fig, axes = plt.subplots(2, 3, figsize=(14, 7)); x = np.arange(len(KERNELS)); w = 0.27
for ax, met in zip(axes.ravel(), mets):
    for i, sp in enumerate(["train", "val", "test"]):
        vals = [overall[(overall.kernel == k) & (overall.split == sp)][met].values[0] for k in KERNELS]
        ax.bar(x + (i - 1) * w, vals, w, label=sp)
    ax.set_xticks(x); ax.set_xticklabels(KERNELS, rotation=15)
    ax.set_title(met); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()""")

md("## 6. Per-cell-line breakdown (test split)")

co(r"""byline = res[(res.cell_line != "ALL") & (res.split == "test")]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, met in zip(axes, ["delta_r2_median", "deg_jaccard_median"]):
    x = np.arange(len(LINES)); w = 0.25
    for i, k in enumerate(KERNELS):
        vals = [byline[(byline.kernel == k) & (byline.cell_line == cl)][met].values[0]
                if len(byline[(byline.kernel == k) & (byline.cell_line == cl)]) else np.nan for cl in LINES]
        ax.bar(x + (i - 1) * w, vals, w, label=k)
    ax.set_xticks(x); ax.set_xticklabels(LINES); ax.set_title(met + " (test)")
    ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()""")

md("""## 7. Example: per-gene Δ scatter (single-cell vs metacell), one test condition per kernel

Cloud hugging the diagonal = aggregation preserved the DEG signature.""")

co(r"""test_conds = [(c, d) for (c, d) in conditions if drug_split[d] == "test"]
cl0, d0 = max(test_conds, key=lambda cd: (pert & (cl_arr == cd[0]) & (drug == cd[1])).sum())
sc = genes[pert & (cl_arr == cl0) & (drug == d0)].mean(0) - ctrl_mean[cl0]
fig, axes = plt.subplots(1, len(KERNELS), figsize=(5 * len(KERNELS), 4.4))
for ax, k in zip(np.atleast_1d(axes), KERNELS):
    mc = mc_store[k]["mc"]; ms = (mc["cell_line"].values == cl0) & (mc["drug"].values == d0)
    mcd = mc_store[k]["mc_profiles"][ms].mean(0) - ctrl_mean[cl0]
    ax.scatter(sc, mcd, s=5, alpha=.4); lim = [min(sc.min(), mcd.min()), max(sc.max(), mcd.max())]
    ax.plot(lim, lim, "r--", lw=1); ax.set_title(f"{k}\n{cl0}/{d0}  Δ-R²={r2_score(sc, mcd):.3f}")
    ax.set_xlabel("single-cell Δ"); ax.set_ylabel("metacell Δ")
plt.tight_layout(); plt.show()""")

md("""## 8. UMAP check — do metacells tile the manifold? (per kernel)

A sample of perturbed **single cells** + that kernel's **metacell centroids** in one UMAP (kernel
space). Top row coloured single-cell (grey) vs metacell (crimson); bottom row single cells by cell
line. Good metacells sit inside the cloud and cover all three lines.""")

co(r"""import scanpy as sc
rng = np.random.default_rng(0); N_SAMP = 9000
fig, axes = plt.subplots(2, len(KERNELS), figsize=(6 * len(KERNELS), 11))
pidx = np.where(pert & np.isin(drug, drugs_use))[0]
for col, k in enumerate(KERNELS):
    samp = rng.choice(pidx, size=min(N_SAMP, len(pidx)), replace=False)
    sc_emb = emb[k][samp]; mc_emb = mc_store[k]["mc_kernel"]
    Z = np.vstack([sc_emb, mc_emb]).astype(np.float32)
    is_mc = np.r_[np.zeros(len(sc_emb), bool), np.ones(len(mc_emb), bool)]
    nc = min(50, Z.shape[1]); rep = PCA(n_components=nc, random_state=0).fit_transform(Z) if Z.shape[1] > nc else Z
    A = ad.AnnData(X=np.zeros((len(Z), 1), np.float32)); A.obsm["rep"] = rep.astype(np.float32)
    sc.pp.neighbors(A, use_rep="rep", n_neighbors=15); sc.tl.umap(A); U = A.obsm["X_umap"]
    ax = axes[0, col]
    ax.scatter(U[~is_mc, 0], U[~is_mc, 1], s=3, alpha=.25, c="lightgray", label="single cells")
    ax.scatter(U[is_mc, 0], U[is_mc, 1], s=14, alpha=.9, c="crimson", label="metacells")
    ax.set_title(f"UMAP on {k} space"); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8)
    ax2 = axes[1, col]; scl = cl_arr[samp]
    for cl in LINES:
        mm = scl == cl; ax2.scatter(U[:len(sc_emb)][mm, 0], U[:len(sc_emb)][mm, 1], s=3, alpha=.4, label=cl)
    ax2.set_title(f"{k}: single cells by cell line"); ax2.set_xticks([]); ax2.set_yticks([]); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()""")

md(r'''## 9. Interpretation (real-gene metacell analysis)

**Setup.** 6,469 metacells from 500,545 perturbed cells (~77 cells/metacell), built per `(cell_line, drug)`
under 3 kernels; every metric here vs single-cell ground truth in **real** gene space (no decoder).

**1. The condition mean (DEGs by fold-change) is NOT lost.** `pseudobulk_r2 ≈ 0.997–1.000` for all
kernels/splits — metacells are means, so the pseudobulk is conserved.

**2. Cell-level resolution: gene_PCA ≳ X_state > AE_10.** Read the **per-cell-line** `recon_r2` — the
pooled "overall" ~0.26 is inflated by between-cell-line variance metacells trivially keep. Per-line test
`recon_r2_gene`: gene_PCA 0.08–0.14 ≥ X_state 0.06–0.12 > AE_10 0.03–0.05 (MCF7 > A549 > K-562). All low
in absolute terms — most single-cell variance is dropout noise a 75-cell mean can't (and shouldn't) keep.

**3. Equal-weight delta ordering — but this is a pooling artifact (see §10).** As shown here, the
*equal-weight* metacell delta orders AE_10 > X_state > gene_PCA (test 0.76 / 0.67 / 0.53). **§10 shows
this collapse — especially the negative K-562/MCF7 — is the equal-weight reweighting bias, NOT lost
content: size-weighting each metacell by its #cells recovers it**, and the kernel then barely matters.

**4. Large per-cell-line differences — A549 alone would have misled.** A549 deltas are strong/robust;
**K-562 (suspension leukemia) and MCF7 are weak/noisy** and degrade most under aggregation. Conclusions
from A549 only would have been over-optimistic — the reason we ran all three lines.

**5. Train/val/test are flat** — metacell building is unsupervised, so no generalization gap (expected).

**Takeaway.** The flow lives in `X_state`, so build metacells on `X_state` (it preserves the DEG
**ranking** best and is the flow's own space; under size-weighting all three kernels reach ~0.69 on the
decoded delta — §10). What metacells sacrifice is **cell-level variance** (pt 2), not the mean/DEGs.''')

md(r'''## 10. Deployment metric — predicted metacell → decode → genes (+ DEG preservation)

A flow-**predicted** metacell is an X_state vector with **no member cells**, so its genes exist only as
`decode(X_state)` (`dec_X_state_opt`). Two rules make the decoded delta meaningful (scripts:
`scripts/metacell/{decoder_ceiling,decode_compare,metacell_weighting,deg_preservation}.py` →
`outputs/metacell/*.csv`):

- **Decode both arms the same way** (perturbed *and* control). The decoder has an input-dependent
  offset that only cancels in the delta when both arms share a decoding mode — comparing a decoded
  perturbed to a *real* control gives **negative** R². (Aside: decoded−**real**-ctrl reproduces
  recon_emb_sanity's 0.664/0.819 exactly; the metacell pipeline must use decoded−**decoded**-ctrl.)
- **Weight each metacell by its size** (≡ "decode each cell by its assigned metacell"). Equal-weight
  pooling biases the pseudobulk and collapses weak lines.

**Delta-R² ladder (decoded, both arms; test; X_state kernel):**

| representation | A549 | K-562 | MCF7 | overall |
|---|---|---|---|---|
| per-cell decode (ceiling — needs the cells) | 0.92 | 0.58 | 0.70 | **0.78** |
| **size-weighted metacell** (decode centroid, weight by k) | 0.88 | 0.47 | 0.48 | **0.69** |
| equal-weight metacell (each metacell = 1 sample) | 0.76 | −0.22 | −0.08 | 0.45 |

→ size-weighting recovers the population delta to **within ~0.09 of decoding every cell**; the
equal-weight collapse was reweighting bias, **not** lost content (real-gene pseudobulk R²≈1.0 already
showed the mean is conserved — this is that, through the decoder). Under size-weighting the **kernel
barely matters** for the decoded population delta (AE_10 0.69 ≈ X_state 0.69 ≈ gene_PCA 0.69) — so
`X_state` (the flow's own space) is the natural choice; the kernel mainly affects *cell-level*
resolution (§3–4).

**Significant-DEG preservation** (DEGs from the true ptb-vs-ctrl test, padj<0.05 & |LFC|≥0.25; size-wt metacell):

| line | #DEGs | magnitude R² (per-cell / metacell) | sign-concord. | DEG recall |
|---|---|---|---|---|
| A549 | 18 | 0.95 / 0.94 | 1.00 | 0.78 |
| K-562 | 20 | 0.84 / 0.80 | 1.00 | 0.72 |
| MCF7 | 37 | 0.96 / 0.94 | 1.00 | 0.89 |
| **overall** | 20 | 0.95 / **0.93** | **1.00** | 0.79 |

→ **on the genes that actually move, magnitude + direction are preserved excellently** (R²≈0.93, sign
100%, ~80% recovered), and the metacell is **≈ per-cell decode**. The low *all-gene* delta/Spearman
were the ~1980 near-zero noise-floor genes, not the biology.

### Bottom line for metacell → flow → decode
1. **Carry each metacell's size** — as a sample weight in training and a pooling weight at decode.
2. **Compute deltas symmetrically** (decode both arms); never decoded-vs-real-control.
3. **Judge on DEGs, not all genes.**
4. Improvable by **fine-tuning the decoder on metacell-mean inputs** and **building control metacells**
   (the builder currently keeps controls raw → a granularity mismatch with the metacell targets).

The metacell representation **preserves the perturbation effect/DEGs** (the marginal that matters for
drug response); what it sacrifices is single-cell variance (§3–4) — fine if the readout is the
delta/DEGs, otherwise restore variance separately.''')

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "seacells", "language": "python", "name": "seacells"}
nb.metadata["language_info"] = {"name": "python"}
OUT = "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/sub/notebooks/metacell_effect.ipynb"
with open(OUT, "w") as f:
    nbf.write(nb, f)
print("wrote", OUT, "|", len(cells), "cells")
