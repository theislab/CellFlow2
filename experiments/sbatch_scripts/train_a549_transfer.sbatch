#!/bin/bash
# tahoe_a549 (all-train) + sciplex_a549 (held-out-drug) transfer in X_state, batch covariate, recon on
# sciplex → wandb(pancellflow). Requires WANDB_API_KEY for online logging.
#   sbatch .../train_a549_transfer.sbatch
#SBATCH --job-name=a549_xfer
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nice=0
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --exclude=supergpu32
#SBATCH --output=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/train_comparison/a549_transfer/logs/a549_xfer_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/train_comparison/a549_transfer/logs/a549_xfer_%j.err

set -e
PY=/home/icb/xiaotong.fu/miniconda3/envs/pancellflow/bin/python
cd /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/cellflowtraining
mkdir -p /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/train_comparison/a549_transfer/logs
export JAX_COMPILATION_CACHE_DIR=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/train_comparison/jax_cache
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "host=$(hostname)  start=$(date)  args=$*"
nvidia-smi -L || true
$PY experiments/train_comparison.py --config-name train_a549_transfer wandb.enabled=true "$@"
echo "done=$(date)"
