#!/bin/bash
# Cell embedding – CHUNKED mode (SLURM array job for large h5ad files)
#
# Usage:
#   sbatch --array=0-199 embed_adata_chunk.sbatch
#   (or set #SBATCH --array below)
#
# Set IN_H5AD, EMB_DIR, MODEL_CONFIG, and N_CHUNKS before submitting.

#SBATCH --job-name=cell_emb
#SBATCH -o %x_%A_%a_out.out
#SBATCH -e %x_%A_%a_err.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=400G
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_priority
#SBATCH --time=12:00:00
#SBATCH --array=0-9   # adjust range to match N_CHUNKS-1

# ── Edit these ─────────────────────────────────────────────────────────────
N_CHUNKS=10

IN_H5AD="/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe.h5ad"
EMB_DIR="/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe/embeddings"

# Pick one config from scripts/configs/
MODEL_CONFIG="/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/scripts/configs/model_se.yaml"
# MODEL_CONFIG=".../configs/model_scgpt.yaml"
# MODEL_CONFIG=".../configs/model_scimilarity.yaml"
# MODEL_CONFIG=".../configs/model_scconcept.yaml"
# ───────────────────────────────────────────────────────────────────────────

mkdir -p "$EMB_DIR"

source "$HOME/.bashrc"
conda deactivate
conda activate pancellflow

CHUNK_ID="${SLURM_ARRAY_TASK_ID}"
OUT_H5AD="${EMB_DIR}/chunk_$(printf "%05d" "${CHUNK_ID}").h5ad"

echo "MODEL_CONFIG=${MODEL_CONFIG}"
echo "IN_H5AD=${IN_H5AD}"
echo "OUT_H5AD=${OUT_H5AD}"

python -u /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/scripts/embed_adata.py \
  --in        "${IN_H5AD}" \
  --out       "${OUT_H5AD}" \
  --config    "${MODEL_CONFIG}" \
  --chunk-id  "${CHUNK_ID}" \
  --n-chunks  "${N_CHUNKS}"

echo "Done: ${OUT_H5AD}"
