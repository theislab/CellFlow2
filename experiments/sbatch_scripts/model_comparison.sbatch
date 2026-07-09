#!/bin/bash
# Rebuild the RUNS checkpoints in model_comparison.py, predict the sciplex3 held-out test set, and
# draw high-effect-condition UMAP effect panels + per-model effect ratios. Pass-through args, e.g.:
#   sbatch .../model_comparison.sbatch --decode --k 6            # gene space, top-6 conditions
#SBATCH --job-name=model_cmp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nice=0
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --exclude=supergpu32,supergpu17
#SBATCH --output=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/train_comparison/logs/model_cmp_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/train_comparison/logs/model_cmp_%j.err

set -e
PY=/home/icb/xiaotong.fu/miniconda3/envs/pancellflow/bin/python
cd /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/cellflowtraining
mkdir -p /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/train_comparison/logs
export JAX_COMPILATION_CACHE_DIR=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/train_comparison/jax_cache
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "host=$(hostname)  start=$(date)  args=$*"
nvidia-smi -L || true
$PY experiments/model_comparison.py "$@"
echo "done=$(date)"
