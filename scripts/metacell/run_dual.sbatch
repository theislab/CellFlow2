#!/bin/bash
#SBATCH --job-name=metacell_dual
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nice=0
#SBATCH --output=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/metacell/slurm_dual_%j.log

set -e
echo "host=$(hostname) start=$(date)"
~/miniconda3/envs/seacells/bin/python /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/sub/scripts/metacell/build_metacells_dual.py
echo "done=$(date)"
