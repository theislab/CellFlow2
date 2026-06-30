"""Convert a DOSE-RESOLVED metacell h5ad → GroupedDistribution zarr.

Mirrors prepare_sciplex_prophet.py's dose handling so the output zarr has the
SAME condition structure as sciplex3_X_scconcept (cell_line, drug, dose, prophet):
  - dose = log1p(dose_value), a raw scalar condition (no rep_key)
  - target distributions keyed by (cell_line, drug, dose)
A model trained on this can therefore cross-eval on sciplex3_X_scconcept.

Run in cfp env (zarr 3.x + scaleflow).
"""
import argparse, sys, time, warnings
from pathlib import Path

import anndata as ad
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from scaleflow.data import AnnDataLocation, DataManager

warnings.filterwarnings("ignore")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="dose-resolved metacell h5ad")
    p.add_argument("--output", required=True, help="output zarr path")
    p.add_argument("--obsm",   default="X_scconcept", help="embedding obsm key")
    a = p.parse_args()

    t0 = time.time()
    print(f"loading {a.input} …", flush=True)
    adata = ad.read_h5ad(a.input)
    print(f"  n_obs={adata.n_obs:,}  (ctrl={int(adata.obs['control'].sum()):,}, "
          f"metacells={int((~adata.obs['control']).sum()):,})", flush=True)

    # dose as a raw scalar condition (log1p), exactly like prepare_sciplex_prophet
    adata.obs = adata.obs.rename(columns={"dose_value": "dose"})
    adata.obs["dose"] = np.log1p(adata.obs["dose"].astype("float32"))
    n_bad = int(((~adata.obs["control"]) & adata.obs["dose"].isna()).sum())
    if n_bad:
        print(f"  WARNING: {n_bad:,} perturbed metacells have NaN dose", flush=True)
    print(f"  perturbed doses (log1p): {sorted(adata.obs.loc[~adata.obs['control'],'dose'].unique())}", flush=True)
    print(f"  drugs: {adata.obs.loc[~adata.obs['control'],'drug'].nunique()}  "
          f"cell_lines: {sorted(adata.obs['cell_line'].unique())}", flush=True)

    adl = AnnDataLocation()
    dm = DataManager(
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys=["drug", "dose"],
        rep_keys={"cell_line": "cell_line_ccle_embeddings", "drug": "drug_0_embeddings"},
        data_location=adl.obsm[a.obsm],
        extra_rep_keys={"prophet": ("drug", "prophet_emb")},
    )
    t1 = time.time()
    print("\npreparing GroupedDistribution …", flush=True)
    gd = dm.prepare_data(adata=adata)
    print(f"  prepared in {time.time() - t1:.1f}s", flush=True)

    t2 = time.time()
    print(f"\nwriting {a.output} …", flush=True)
    gd.write_zarr(path=Path(a.output), chunk_size=131072, shard_size=131072 * 8)
    print(f"  wrote in {time.time() - t2:.1f}s; total {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
