"""Build CONTROL metacells per cell line (X_state kernel, same SEACells params as the perturbed
build) so the decoded-metacell delta has a consistent control arm:
    decoded metacell ptb  -  decoded metacell ctrl
Saves outputs/metacell/control_metacells.pkl = {cell_line: {"xstate": (M,2058), "size": (M,)}}.
Runs in the `seacells` env.
"""
import pickle, sys, warnings
warnings.filterwarnings("ignore")
import anndata as ad, numpy as np, pandas as pd
from sklearn.decomposition import PCA
import SEACells

BASE = "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow"
CELLS_PER_METACELL, N_PCA, MIN_CELLS, SEED = 75, 50, 50, 42


def build_metacell_labels(e):  # identical to the notebook builder
    n = e.shape[0]
    if n <= MIN_CELLS or (n // CELLS_PER_METACELL) < 2:
        return np.zeros(n, dtype=int)
    n_meta = max(2, n // CELLS_PER_METACELL)
    n_comp = min(N_PCA, n - 1, e.shape[1])
    epca = PCA(n_components=n_comp, random_state=SEED).fit_transform(e).astype(np.float32)
    sub = ad.AnnData(X=np.zeros((n, 1), np.float32),
                     obs=pd.DataFrame(index=[str(i) for i in range(n)]), obsm={"X_pca": epca})
    m = SEACells.core.SEACells(sub, build_kernel_on="X_pca", n_SEACells=n_meta,
                               n_neighbors=min(15, n - 1), verbose=False)
    m.construct_kernel_matrix(); m.initialize_archetypes(); m.fit(min_iter=5, max_iter=30)
    return pd.factorize(m.get_hard_assignments()["SEACell"].reindex([str(i) for i in range(n)]).values)[0]


A = ad.read_h5ad(f"{BASE}/outputs/metacell/sciplex_all_for_metacell.h5ad")
Xst = np.asarray(A.obsm["X_state"], np.float32)
cl = A.obs["cell_line"].astype(str).to_numpy(); ctrl = A.obs["control"].to_numpy().astype(bool)
out = {}
for c in sorted(set(cl)):
    idx = np.where(ctrl & (cl == c))[0]
    lab = build_metacell_labels(Xst[idx])
    cents, sizes = [], []
    for L in np.unique(lab):
        mem = idx[lab == L]
        cents.append(Xst[mem].mean(0)); sizes.append(len(mem))
    out[c] = {"xstate": np.stack(cents).astype(np.float32), "size": np.array(sizes)}
    print(f"{c}: {len(idx)} control cells -> {len(sizes)} metacells (mean size {np.mean(sizes):.0f})", flush=True)
pickle.dump(out, open(f"{BASE}/outputs/metacell/control_metacells.pkl", "wb"))
print("saved control_metacells.pkl")
