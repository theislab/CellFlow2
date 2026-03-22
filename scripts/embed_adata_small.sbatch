#!/bin/bash
# Cell embedding – IN-MEMORY mode (single job for small h5ad files)
#
# The entire adata is loaded into RAM at once; no chunking needed.
# Set IN_H5AD, OUT_H5AD, and MODEL_CONFIG before submitting.

#SBATCH --job-name=cell_emb
#SBATCH -o %x_%A_out.out
#SBATCH -e %x_%A_err.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_80gb|h100_80gb"
#SBATCH --qos=gpu_priority
#SBATCH --time=4:00:00

# ── Edit these ─────────────────────────────────────────────────────────────
IN_H5AD="/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/unipert/tahoe_a549_w_se.h5ad"
OUT_H5AD="/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/unipert/tahoe_a549_w_scconcept.h5ad"

# Pick one config from scripts/configs/
MODEL_CONFIG="/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/scripts/configs/model_scconcept.yaml"
# MODEL_CONFIG=".../configs/model_scgpt.yaml"
# MODEL_CONFIG=".../configs/model_scimilarity.yaml"
# MODEL_CONFIG=".../configs/model_scconcept.yaml"
# ───────────────────────────────────────────────────────────────────────────

source "$HOME/.bashrc"
conda deactivate
conda activate scconcept_env

echo "MODEL_CONFIG=${MODEL_CONFIG}"
echo "IN_H5AD=${IN_H5AD}"
echo "OUT_H5AD=${OUT_H5AD}"

python -u /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/scripts/embed_adata.py \
  --in     "${IN_H5AD}" \
  --out    "${OUT_H5AD}" \
  --config "${MODEL_CONFIG}"

echo "Done: ${OUT_H5AD}"
