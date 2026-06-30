"""Prep a compact h5ad for the metacell-effect notebook (runs in the `pancellflow` env).

Subsets sciplex3_with_emb.h5ad to one cell line and keeps only what the notebook needs:
  X            : log-norm HVG genes (sparse, 2000)         -> ground-truth gene space
  obsm X_state : 2058-d FM latent                          -> SEACells kernel option 1
  obsm AE_10   : 10-d learned AE latent                    -> SEACells kernel option 2
  obs          : drug, control, cell_line, split           -> conditions + held-out-drug split

The held-out-drug split (split/drug.yaml: by=[drug], 0.7/0.2/0.1, rs=42) is computed over the
FULL treated population (all cell lines) so the drug->split map matches train_zarr / recon_emb,
then applied to the subset. Output is small enough for the seacells env's anndata to read fast.
"""
import argparse

import anndata as ad
import h5py
import numpy as np
import pandas as pd

SRC = "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/recon_weights/sciplex3_with_emb.h5ad"


def assign_splits(obs: pd.DataFrame, split_by=("drug",), ratios=(0.7, 0.2, 0.1),
                  random_state: int = 42) -> np.ndarray:
    """Held-out-drug split (controls -> train).

    Verbatim copy of ``experiments/recon_emb.py:assign_splits`` so this prep stays free of the
    jax/scaleflow import. Keep in sync with that file if it changes.
    """
    keys = list(split_by)
    treated = ~np.asarray(obs["control"].values).astype(bool)
    o = obs[keys].astype(str).reset_index(drop=True)
    df_unique = o[treated].drop_duplicates(subset=keys).copy()
    n = len(df_unique)
    train_size = round(ratios[0] * n)
    val_size = round(ratios[1] * n)
    test_size = n - train_size - val_size
    if min(train_size, val_size, test_size) == 0:
        raise ValueError(f"a split is empty for n={n} unique combos and ratios={list(ratios)}")
    df_unique["split"] = "test_val"
    sh = df_unique.sample(frac=1, random_state=random_state).reset_index(drop=True)
    sh.loc[: train_size - 1, "split"] = "train"
    tv = sh[sh["split"] == "test_val"].copy().sample(frac=1, random_state=random_state).reset_index(drop=True)
    tv.loc[: test_size - 1, "split"] = "test"
    tv.loc[test_size:, "split"] = "val"
    combo2split = {tuple(r[k] for k in keys): "train" for _, r in sh[sh["split"] == "train"].iterrows()}
    combo2split.update({tuple(r[k] for k in keys): r["split"] for _, r in tv.iterrows()})
    cell_keys = list(zip(*[o[k].values for k in keys]))
    splits = np.array(["train"] * len(obs), dtype=object)
    splits[treated] = [combo2split[cell_keys[i]] for i in np.where(treated)[0]]
    return splits.astype(str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-line", default="A549")
    ap.add_argument("--out", default="/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/"
                                     "outputs/metacell/sciplex_{cl}_for_metacell.h5ad")
    args = ap.parse_args()
    out = args.out.format(cl=args.cell_line)

    print(f"reading {SRC} …", flush=True)
    with h5py.File(SRC, "r") as f:
        obs = ad.io.read_elem(f["obs"])
        var = ad.io.read_elem(f["var"])
        X = ad.io.read_elem(f["X"])                       # csr (518123, 2000)
        Xstate = f["obsm"]["X_state"][:]                  # (518123, 2058)
        AE10 = f["obsm"]["AE_10"][:]                      # (518123, 10)
    print(f"  loaded: X {X.shape}, X_state {Xstate.shape}, AE_10 {AE10.shape}", flush=True)

    # canonical drug split on the FULL treated population
    obs["split"] = assign_splits(obs)
    n_tr = (obs.split == "train").sum(); n_va = (obs.split == "val").sum(); n_te = (obs.split == "test").sum()
    print(f"  split (all lines): train={n_tr:,} val={n_va:,} test={n_te:,}", flush=True)

    if args.cell_line.lower() == "all":
        mask = np.ones(len(obs), dtype=bool)
    else:
        mask = (obs["cell_line"].astype(str) == args.cell_line).to_numpy()
    print(f"  {args.cell_line}: {mask.sum():,} cells "
          f"({obs.loc[mask, 'cell_line'].astype(str).value_counts().to_dict()})", flush=True)
    sub = ad.AnnData(
        X=X[mask],
        obs=obs.loc[mask, ["cell_line", "drug", "control", "split"]].reset_index(drop=True),
        var=var,
        obsm={"X_state": Xstate[mask].astype(np.float32), "AE_10": AE10[mask].astype(np.float32)},
    )
    sub.obs["drug"] = sub.obs["drug"].astype(str)
    # per-split drug counts among treated
    tre = ~sub.obs.control.to_numpy().astype(bool)
    print("  treated drugs/ split:", {s: int(sub.obs.loc[tre, "drug"][sub.obs.loc[tre, "split"] == s].nunique())
                                       for s in ["train", "val", "test"]}, flush=True)
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sub.write_h5ad(out)
    print(f"wrote {out}  ({sub.shape})", flush=True)


if __name__ == "__main__":
    main()
