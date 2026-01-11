#!/usr/bin/env python
import argparse
from pathlib import Path
import anndata as ad
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk-size", type=int, default=500_000)
    ap.add_argument("--compression", default="gzip")  # None or "gzip"
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "chunks.txt"

    adata = ad.read_h5ad(args.in_path, backed="r") 
    n = adata.n_obs
    n_chunks = (n + args.chunk_size - 1) // args.chunk_size

    with manifest.open("w") as mf:
        for chunk_id, start in enumerate(tqdm(range(0, n, args.chunk_size), desc="Splitting")):
            stop = min(start + args.chunk_size, n)
            print(f"[chunk {chunk_id+1}/{n_chunks}] rows {start}:{stop}")

            sub = adata[start:stop].to_memory()  
            out_path = out_dir / f"chunk_{chunk_id:05d}.h5ad"
            sub.write_h5ad(out_path, compression=args.compression)  

            mf.write(str(out_path) + "\n")

    print(f"Done. Wrote {n_chunks} chunks.")
    print(f"Manifest: {manifest}")

if __name__ == "__main__":
    main()

# python /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/tests/_processtahoe/01_split_h5ad.py \
#   --in /lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe_a549.h5ad \
#   --out-dir /lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe_a549_chunks \
#   --chunk-size 500000