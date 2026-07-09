#!/bin/bash
# Train the flow in the AE_128 latent on sciplex3_with_emb.h5ad (dose-resolved), gene-space recon
# metrics → wandb(pancellflow). Pass-through args, e.g.:
#   sbatch .../train_comparison_sciplex3_ae128.sbatch                    # no prophet token
#   sbatch .../train_comparison_sciplex3_ae128.sbatch ablation=prophet   # append prophet embedding
# Requires WANDB_API_KEY in the environment for online logging.
#SBATCH --job-name=cmp_ae128
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nice=0
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --exclude=supergpu32,supergpu17
#SBATCH --output=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/train_comparison/logs/cmp_ae128_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/train_comparison/logs/cmp_ae128_%j.err

set -e
PY=/home/icb/xiaotong.fu/miniconda3/envs/pancellflow/bin/python
cd /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/cellflowtraining
mkdir -p /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/train_comparison/logs
# /storage (default jax cache) is not writable here → use a workspace path
export JAX_COMPILATION_CACHE_DIR=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/train_comparison/jax_cache
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "host=$(hostname)  start=$(date)  args=$*"
nvidia-smi -L || true
$PY experiments/train_comparison.py wandb.enabled=true "$@"
echo "done=$(date)"
