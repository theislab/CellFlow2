#!/bin/bash
#SBATCH --job-name=metacell_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nice=0
#SBATCH --output=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/metacell/slurm_%j.log

set -e
cd /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/sub/notebooks
echo "host=$(hostname) start=$(date)"
~/miniconda3/envs/seacells/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=seacells \
    --ExecutePreprocessor.timeout=43200 \
    metacell_effect.ipynb
echo "done=$(date)"
