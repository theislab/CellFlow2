#!/bin/bash
#SBATCH --job-name=state_emb
#SBATCH -o %x_%A_out.out
#SBATCH -e %x_%A_err.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=400G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_priority
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_80gb|h100_80gb"
#SBATCH --time=12:00:00
#SBATCH --array=3
N_CHUNKS=200


IN_H5AD="/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe.h5ad"
EMB_DIR="/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe/test"
mkdir -p "$EMB_DIR"

CHECKPOINT="/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/sup/SE-600M/se600m_epoch16.ckpt"
PROTEMB="/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/sup/SE-600M/protein_embeddings.pt"

OBSM_KEY="X_state"
ENCODE_BS=256

source $HOME/.bashrc
conda deactivate 
conda activate pancellflow

CHUNK_ID="${SLURM_ARRAY_TASK_ID}"
OUT_H5AD="${EMB_DIR}/chunk_$(printf "%05d" "${CHUNK_ID}").h5ad"

echo "IN_H5AD=${IN_H5AD}"
echo "OUT_H5AD=${OUT_H5AD}"

python -u /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/scripts/_generate_cellembedding/_processscale/embed_chunk.py \
  --in "${IN_H5AD}" \
  --out "${OUT_H5AD}" \
  --checkpoint "${CHECKPOINT}" \
  --protein-embeds "${PROTEMB}" \
  --obsm-key "${OBSM_KEY}" \
  --encode-batch-size "${ENCODE_BS}" \
  --chunk-id "${CHUNK_ID}" \
  --n-chunks "${N_CHUNKS}"

echo "Done: ${OUT_H5AD}"