#!/usr/bin/env python
"""
Two-stage concatenation of embedding chunks:

  Stage 1 – batch every N chunk files in memory, write intermediate h5ads
  Stage 2 – concat_on_disk on the (much fewer) intermediate files

Each chunk stores obsm only; we add a placeholder sparse X so that
concat_on_disk can process them.
"""
import argparse
import re
import tempfile
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


def chunk_sort_key(path: Path):
    nums = re.findall(r"\d+", path.stem)
    if nums:
        return tuple(int(x) for x in nums)
    return (path.name,)


def stage1(chunk_files: list[Path], batch_size: int, obsm_key: str,
           tmp_dir: Path) -> list[Path]:
    """Batch-concat every `batch_size` chunks in memory, write intermediates."""
    print(f"tmp_dir: {tmp_dir}")
    intermediates = []
    n = len(chunk_files)
    n_batches = (n + batch_size - 1) // batch_size
    batch_bar = tqdm(range(0, n, batch_size), desc="Batches", unit="batch",
                     total=n_batches)
    for batch_idx, start in enumerate(batch_bar):
        batch = chunk_files[start : start + batch_size]
        embeddings = []
        for fp in tqdm(batch, desc=f"  batch {batch_idx}", unit="chunk",
                       leave=False):
            with h5py.File(fp, "r") as f:
                embeddings.append(f["obsm"][obsm_key][:])

        emb = np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)
        del embeddings

        n_obs = emb.shape[0]
        adata = ad.AnnData(
            X=sp.csr_matrix((n_obs, 1)),
            obsm={obsm_key: emb},
        )
        adata.var_names = ["_placeholder"]

        out = tmp_dir / f"batch_{batch_idx:04d}.h5ad"
        adata.write_h5ad(out, compression="gzip")
        del adata, emb

        batch_bar.set_postfix(rows=f"{n_obs:,}")
        intermediates.append(out)

    return intermediates


def stage2(intermediates: list[Path], out_file: Path) -> None:
    """concat_on_disk on the intermediate files."""
    in_files = [str(p) for p in intermediates]
    ad.experimental.concat_on_disk(
        in_files=in_files,
        out_file=str(out_file),
        axis=0,
        join="inner",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indirec", required=True,
                    help="Directory containing per-chunk h5ad files")
    ap.add_argument("--outh5ad", required=True,
                    help="Final concatenated output h5ad path")
    ap.add_argument("--obsm-key", default="X_state")
    ap.add_argument("--batch-size", type=int, default=20,
                    help="Number of chunks to concat in memory per batch")
    ap.add_argument("--tmp-dir", default='/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe/test_batched',
                    help="Directory for intermediates (default: auto tmpdir)")
    args = ap.parse_args()

    chunks_dir = Path(args.indirec)
    chunk_files = sorted(chunks_dir.glob("*.h5ad"), key=chunk_sort_key)
    print(f"Found {len(chunk_files)} chunks in {chunks_dir}")
    assert len(chunk_files) > 0, "No chunk .h5ad files found."

    out_file = Path(args.outh5ad)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"tmp_dir: {args.tmp_dir}")
    if args.tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="concat_stage1_"))
    else:
        tmp_dir = Path(args.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"tmp_dir: {tmp_dir}")
    print("=== Stage 1: in-memory batch concat ===")
    intermediates = stage1(chunk_files, args.batch_size,
                           args.obsm_key, tmp_dir)
    print(f"\n{len(intermediates)} intermediate files written to {tmp_dir}")

    print("\n=== Stage 2: concat_on_disk ===")
    stage2(intermediates, out_file)
    print(f"\nDone: {out_file}")

    result = ad.read_h5ad(str(out_file), backed="r")
    print(f"Output shape: n_obs={result.n_obs:,}, obsm keys={list(result.obsm.keys())}")




if __name__ == "__main__":
    main()
