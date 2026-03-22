#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import anndata as ad

import os
import sys
sys.path.append('/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/scripts')

from reconse import ReconPretrainedStateModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--obsm-key", default="X_state")
    ap.add_argument("--encode-batch-size", type=int, default=256)

    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--protein-embeds", required=True)

    ap.add_argument("--chunk-id", type=int, default=None)
    ap.add_argument("--n-chunks", type=int, default=None)

    args = ap.parse_args()

    prefix = ""
    if args.chunk_id is not None and args.n_chunks is not None:
        prefix = f"[chunk {args.chunk_id+1}/{args.n_chunks}] "

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{prefix}Loading chunk: {in_path}")
    
    adata = ad.read_h5ad(in_path, backed="r")
    adata_size = adata.n_obs
    chunk_size = (adata_size + args.n_chunks - 1) // args.n_chunks if args.n_chunks else adata_size
    start = args.chunk_id * chunk_size if args.chunk_id is not None else 0
    end = min(start + chunk_size, adata_size) if args.chunk_id is not None else adata_size
    
    processing_adata = adata[start:end].to_memory()

    print(f"{prefix}Loading model")
    emb_model = ReconPretrainedStateModel(
        checkpoint_path=args.checkpoint,
        protein_embeds_path=args.protein_embeds,
        emb_key=args.obsm_key,
        encode_batch_size=args.encode_batch_size,
    )

    print(f"{prefix}Embedding n_obs={processing_adata.n_obs}, n_vars={processing_adata.n_vars}")
    rep = emb_model.get_latent_representation(processing_adata)
    rep = np.asarray(rep, dtype=np.float32)

    print(f"{prefix}Writing: {out_path}")
    adata_out = ad.AnnData(
        X=sp.csr_matrix((rep.shape[0], 1)),
        obsm={args.obsm_key: rep},
    )
    adata_out.var_names = ["_placeholder"]
    adata_out.write_h5ad(out_path, compression="gzip")
    print(f"{prefix}Done.")

if __name__ == "__main__":
    main()