"""Build metacells (X_state grouping, same SEACells params) that carry BOTH an X_state centroid and
an AE_128 centroid, so the decoded deployment metric can be computed two ways:
    X_state pipeline : decode(X_state centroid) via dec_X_state_opt
    AE       pipeline : decode(AE_128 centroid)  via AE_128_opt (autoencoder decoder)
Same grouping for both → isolates the latent+decoder (grouping barely matters for the delta).
Perturbed metacells per (cell_line, drug) + control metacells per cell_line. Runs in `seacells`.
Saves outputs/metacell/{decode_inputs_dual,control_metacells_dual}.pkl.
"""
import pickle, warnings
warnings.filterwarnings("ignore")
import anndata as ad, h5py, numpy as np, pandas as pd
from sklearn.decomposition import PCA
import SEACells

BASE = "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow"
SRC = f"{BASE}/recon_weights/sciplex3_with_emb.h5ad"
CPM, NPCA, MINC, SEED = 75, 50, 50, 42


def labels(e):
    n = e.shape[0]
    if n <= MINC or (n // CPM) < 2:
        return np.zeros(n, int)
    nm = max(2, n // CPM); nc = min(NPCA, n - 1, e.shape[1])
    ep = PCA(n_components=nc, random_state=SEED).fit_transform(e).astype(np.float32)
    sub = ad.AnnData(X=np.zeros((n, 1), np.float32), obs=pd.DataFrame(index=[str(i) for i in range(n)]),
                     obsm={"X_pca": ep})
    m = SEACells.core.SEACells(sub, build_kernel_on="X_pca", n_SEACells=nm, n_neighbors=min(15, n - 1), verbose=False)
    m.construct_kernel_matrix(); m.initialize_archetypes(); m.fit(min_iter=5, max_iter=30)
    return pd.factorize(m.get_hard_assignments()["SEACell"].reindex([str(i) for i in range(n)]).values)[0]


A = ad.read_h5ad(f"{BASE}/outputs/metacell/sciplex_all_for_metacell.h5ad")
Xst = np.asarray(A.obsm["X_state"], np.float32)
cl = A.obs["cell_line"].astype(str).to_numpy(); drug = A.obs["drug"].astype(str).to_numpy()
ctrl = A.obs["control"].to_numpy().astype(bool); split = A.obs["split"].astype(str).to_numpy()
with h5py.File(SRC, "r") as f:                                   # AE_128 aligned to the same cells (prep kept all, in order)
    AE = f["obsm"]["AE_128_opt"][:].astype(np.float32)
assert AE.shape[0] == A.n_obs, (AE.shape, A.n_obs)
pert = ~ctrl; LINES = sorted(set(cl)); drug_split = {d: split[pert & (drug == d)][0] for d in np.unique(drug[pert])}

rows, xc, ac = [], [], []
conds = [(c, d) for c in LINES for d in np.unique(drug[pert & (cl == c)]) if (pert & (cl == c) & (drug == d)).any()]
for j, (c, d) in enumerate(conds):
    idx = np.where(pert & (cl == c) & (drug == d))[0]
    lab = labels(Xst[idx])
    for L in np.unique(lab):
        mem = idx[lab == L]
        rows.append((c, d, drug_split[d], len(mem))); xc.append(Xst[mem].mean(0)); ac.append(AE[mem].mean(0))
    if (j + 1) % 40 == 0:
        print(f"  {j+1}/{len(conds)} conditions", flush=True)
pm = dict(mc=pd.DataFrame(rows, columns=["cell_line", "drug", "split", "size"]),
          mc_xstate=np.stack(xc).astype(np.float32), mc_ae=np.stack(ac).astype(np.float32))
pickle.dump(pm, open(f"{BASE}/outputs/metacell/decode_inputs_dual.pkl", "wb"))
print(f"perturbed: {len(rows)} metacells")

cmc = {}
for c in LINES:
    idx = np.where(ctrl & (cl == c))[0]; lab = labels(Xst[idx])
    xs, ae, sz = [], [], []
    for L in np.unique(lab):
        mem = idx[lab == L]; xs.append(Xst[mem].mean(0)); ae.append(AE[mem].mean(0)); sz.append(len(mem))
    cmc[c] = dict(xstate=np.stack(xs).astype(np.float32), ae=np.stack(ae).astype(np.float32), size=np.array(sz))
    print(f"  control {c}: {len(sz)} metacells", flush=True)
pickle.dump(cmc, open(f"{BASE}/outputs/metacell/control_metacells_dual.pkl", "wb"))
print("saved decode_inputs_dual.pkl + control_metacells_dual.pkl")
