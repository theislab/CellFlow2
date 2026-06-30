"""Matrix-factorize the PRISM viability matrix into a cell-line table and a drug table.

ONE decomposition of V[cell_line x drug] yields BOTH embedding tables:

    V[i, j]  ~=  mu + a_i + b_j + C[i] . D[j]

  - mu      : global mean viability
  - a_i     : cell-line bias   (marginal sensitivity of line i)
  - b_j     : drug bias        (marginal potency of drug j)
  - C[i]    : cell-line embedding  (row i of the k-dim cell-line table)   <- "select by cell_line"
  - D[j]    : drug embedding       (row j of the k-dim drug table)        <- "select by smiles"

The k-dim factors capture the *interaction* gamma(i, j) that the marginals miss.
Missing entries (2.3% of PRISM) are handled by soft-impute (iterative truncated SVD),
so we never fabricate a fill value into the fitted factors.

Outputs (to analysis/viability_factors/):
    cellline_embeddings.parquet   index=cell_line, cols = ['bias', f0..f{k-1}]
    drug_embeddings.parquet       index=smiles,    cols = ['bias', f0..f{k-1}]
    meta.json                     mu, k, held-out R2, shapes
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASET = "PRISM"
RANK = 32
N_ITERS = 40           # soft-impute iterations
HELDOUT_FRAC = 0.10    # fraction of OBSERVED entries masked for honest R2
SEED = 0
OUT = Path(__file__).parent / "viability_factors"


def build_matrix(parquet, dataset):
    df = pd.read_parquet(parquet)
    df = df[df.dataset == dataset]
    # one viability per (cell_line, smiles): drugs can share a SMILES under different names
    g = df.groupby(["cell_line", "smiles"], as_index=False)["viability"].mean()
    mat = g.pivot(index="cell_line", columns="smiles", values="viability")
    return mat


def double_center(V, mask):
    """mu + per-row + per-col means computed on OBSERVED entries only.

    Returns residual R (NaN where unobserved), and (mu, a, b) to add back.
    """
    mu = np.nanmean(V)
    a = np.nanmean(V - mu, axis=1)                     # cell-line bias
    R = V - mu - a[:, None]
    b = np.nanmean(R, axis=0)                           # drug bias (after row-centering)
    R = R - b[None, :]
    return R, mu, a, b


def soft_impute(R, mask, rank, n_iters):
    """Iterative truncated-SVD imputation of residual R (NaN -> 0 start)."""
    X = np.where(mask, R, 0.0)
    for _ in range(n_iters):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        Uk, sk, Vtk = U[:, :rank], s[:rank], Vt[:rank]
        low = (Uk * sk) @ Vtk
        X = np.where(mask, R, low)                      # keep observed, fill missing with low-rank
    # final factorization of the converged complete matrix
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    Uk, sk, Vtk = U[:, :rank], s[:rank], Vt[:rank]
    sqrt_s = np.sqrt(sk)
    C = Uk * sqrt_s                                     # (n_cl, k)  cell-line factors
    D = (Vtk.T) * sqrt_s                                # (n_dr, k)  drug factors
    return C, D


def main():
    parquet = Path(__file__).parent / "drug_viability_table.parquet"
    mat = build_matrix(parquet, DATASET)
    V = mat.values.astype(np.float64)
    obs = ~np.isnan(V)
    print(f"{DATASET}: {mat.shape[0]} cell_lines x {mat.shape[1]} drugs, "
          f"density={obs.mean():.3%}")

    # ---- held-out mask for honest evaluation ----
    rng = np.random.default_rng(SEED)
    obs_idx = np.argwhere(obs)
    n_held = int(len(obs_idx) * HELDOUT_FRAC)
    held = obs_idx[rng.choice(len(obs_idx), n_held, replace=False)]
    train_mask = obs.copy()
    train_mask[held[:, 0], held[:, 1]] = False
    Vtrain = np.where(train_mask, V, np.nan)

    # ---- decompose on the train mask ----
    R, mu, a, b = double_center(Vtrain, train_mask)
    C, D = soft_impute(R, train_mask, RANK, N_ITERS)

    # ---- evaluate on held-out entries ----
    recon = mu + a[:, None] + b[None, :] + C @ D.T
    yt = V[held[:, 0], held[:, 1]]
    yp = recon[held[:, 0], held[:, 1]]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    # marginal-only baseline (mu + a + b, no factors)
    yp0 = (mu + a[:, None] + b[None, :])[held[:, 0], held[:, 1]]
    r2_0 = 1 - np.sum((yt - yp0) ** 2) / ss_tot
    print(f"held-out R2: marginals-only={r2_0:.3f}   rank-{RANK} factors={r2:.3f}   "
          f"(+{r2 - r2_0:.3f})")

    # ---- refit on ALL observed entries for the final saved embeddings ----
    Rf, muf, af, bf = double_center(np.where(obs, V, np.nan), obs)
    Cf, Df = soft_impute(Rf, obs, RANK, N_ITERS)

    OUT.mkdir(exist_ok=True)
    cols = [f"f{i}" for i in range(RANK)]
    cl = pd.DataFrame(Cf, index=mat.index, columns=cols)
    cl.insert(0, "bias", af)
    cl.to_parquet(OUT / "cellline_embeddings.parquet")
    dr = pd.DataFrame(Df, index=mat.columns, columns=cols)
    dr.insert(0, "bias", bf)
    dr.to_parquet(OUT / "drug_embeddings.parquet")
    json.dump(
        {"dataset": DATASET, "mu": float(muf), "rank": RANK,
         "n_celllines": int(mat.shape[0]), "n_drugs": int(mat.shape[1]),
         "heldout_r2_marginals": float(r2_0), "heldout_r2_factors": float(r2)},
        open(OUT / "meta.json", "w"), indent=2)
    print(f"saved -> {OUT}/  (cellline {cl.shape}, drug {dr.shape})")

    # ---- demo: SELECT the embeddings for one (cell_line, drug) and reconstruct ----
    i, j = 0, 0
    cl_name, dr_name = mat.index[i], mat.columns[j]
    c_vec = cl.loc[cl_name, cols].values
    d_vec = dr.loc[dr_name, cols].values
    pred = muf + cl.loc[cl_name, "bias"] + dr.loc[dr_name, "bias"] + c_vec @ d_vec
    print(f"\nlookup demo: cell_line={cl_name!r}  smiles={dr_name[:30]}...")
    print(f"  cell-line embedding (k={RANK}): {np.round(c_vec[:5], 3)} ...")
    print(f"  drug embedding      (k={RANK}): {np.round(d_vec[:5], 3)} ...")
    print(f"  predicted viability={pred:.3f}   true={V[i, j]:.3f}")


if __name__ == "__main__":
    main()
