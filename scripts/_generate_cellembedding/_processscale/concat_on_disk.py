#!/usr/bin/env python
import argparse
from pathlib import Path
import anndata as ad
import os
import glob
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indirec", required=True)
    ap.add_argument("--outh5ad", required=True)
    ap.add_argument("--join", default="exact")  # exact/inner/outer; exact is safest if same var
    args = ap.parse_args()
    file_pattern = args.indirec+'/*.h5ad'
    in_files = sorted(glob.glob(file_pattern))
    print(f"Found {len(in_files)} input files matching pattern: {file_pattern}")
    # Out-of-core concatenation:
    ad.experimental.concat_on_disk(
        in_files=in_files,
        out_file=args.outh5ad,
        axis=0,
        join=args.join,
    )
    print(f"Done: {args.outh5ad}")

if __name__ == "__main__":
    main()