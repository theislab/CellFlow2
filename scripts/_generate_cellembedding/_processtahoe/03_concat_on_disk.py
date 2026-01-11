#!/usr/bin/env python
import argparse
from pathlib import Path
import anndata as ad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-manifest", required=True, help="text file with one .h5ad per line (embedded chunks)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--join", default="exact")  # exact/inner/outer; exact is safest if same var
    args = ap.parse_args()

    in_files = [line.strip() for line in Path(args.in_manifest).read_text().splitlines() if line.strip()]
    print(f"Found {len(in_files)} files.")

    # Out-of-core concatenation:
    ad.experimental.concat_on_disk(
        in_files=in_files,
        out_file=args.out,
        axis=0,
        join=args.join,
    )
    print(f"Done: {args.out}")

if __name__ == "__main__":
    main()