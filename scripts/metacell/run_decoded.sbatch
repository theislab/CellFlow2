#!/bin/bash
#SBATCH --job-name=metacell_decoded
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nice=0
#SBATCH --output=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/metacell/slurm_decoded_%j.log

set -e
cd /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/sub/notebooks
echo "host=$(hostname) start=$(date)"
~/miniconda3/envs/pancellflow/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=pancellflow \
    --ExecutePreprocessor.timeout=7200 \
    metacell_decoded.ipynb
echo "done=$(date)"
