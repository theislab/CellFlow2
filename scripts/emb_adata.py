#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
import anndata as ad
import os
import sys
sys.path.append('/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/scripts')

from .reconse import ReconPretrainedStateModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--obsm-key", default="X_state")
    ap.add_argument("--encode-batch-size", type=int, default=256)

    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--protein-embeds", required=True)

    args = ap.parse_args()
    adata = ad.read_h5ad(in_path) 

    emb_model = ReconPretrainedStateModel(
        checkpoint_path=args.checkpoint,
        protein_embeds_path=args.protein_embeds,
        emb_key=args.obsm_key,
        encode_batch_size=args.encode_batch_size,
    )

    print(f"Embedding n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    rep = emb_model.get_latent_representation(adata)
    rep = np.asarray(rep, dtype=np.float32)

    adata.obsm[args.obsm_key] = rep

    adata.write_h5ad(out_path, compression="gzip")
    print(f"Done.")

if __name__ == "__main__":
    main()