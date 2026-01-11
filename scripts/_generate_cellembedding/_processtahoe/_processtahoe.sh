#!/bin/bash
set -euo pipefail

IN_H5AD="/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe_a549.h5ad"
WORKDIR="/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe_a549"
CHUNK_SIZE=100000

SPLIT_PY="/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/tests/_processtahoe/01_split_h5ad.py"
EMBED_PY="/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/tests/_processtahoe/02_embed_chunk.py"
CONCAT_PY="/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/tests/_processtahoe/03_concat_on_disk.py"

CHECKPOINT="/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/sup/SE-600M/se600m_epoch16.ckpt"
PROTEMB="/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/sup/SE-600M/protein_embeddings.pt"
OBSM_KEY="X_state"
ENCODE_BS=256

# Outputs
CHUNK_DIR="${WORKDIR}/chunks"
EMB_DIR="${WORKDIR}/chunks_emb"
MANIFEST="${CHUNK_DIR}/chunks.txt"
EMB_MANIFEST="${EMB_DIR}/embedded_chunks.txt"
OUT_H5AD="${WORKDIR}/tahoe_a549_se.h5ad"

mkdir -p "$CHUNK_DIR" "$EMB_DIR"

# Export to jobs
export IN_H5AD WORKDIR CHUNK_SIZE SPLIT_PY EMBED_PY CONCAT_PY
export CHECKPOINT PROTEMB OBSM_KEY ENCODE_BS
export CHUNK_DIR EMB_DIR MANIFEST EMB_MANIFEST OUT_H5AD

echo "Submitting split job..."
SPLIT_JOBID=$(sbatch --parsable /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/tests/_processtahoe/split.sbatch)
echo "  split job: $SPLIT_JOBID"

echo "Submitting embed array job (depends on split)..."
EMB_JOBID=$(sbatch --parsable --dependency=afterok:${SPLIT_JOBID} /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/tests/_processtahoe/emb_array.sbatch)
echo "  embed job: $EMB_JOBID"

echo "Submitting concat job (depends on embed array completion)..."
CONCAT_JOBID=$(sbatch --parsable --dependency=afterok:${EMB_JOBID} /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/tests/_processtahoe/concat.sbatch)
echo "  concat job: $CONCAT_JOBID"

echo "Submitting embed array job (depends on split)..."
EMB_JOBID=$(sbatch --parsable /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/tests/_processtahoe/emb_array.sbatch)
echo "  embed job: $EMB_JOBID"

echo "Submitting concat job (depends on embed array completion)..."
CONCAT_JOBID=$(sbatch --parsable /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/tests/_processtahoe/concat.sbatch)
echo "  concat job: $CONCAT_JOBID"
echo "Pipeline submitted."
echo "Split:   $SPLIT_JOBID"
echo "Embed:   $EMB_JOBID"
echo "Concat:  $CONCAT_JOBID"