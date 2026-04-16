#!/bin/bash

#SBATCH -J qualcheck_ae10
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu_p
#SBATCH --qos=gpu_priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH -t 0-04:00:00
#SBATCH --nice=0

source /home/icb/weixu.wang/.bashrc

# Activate the correct conda environment
conda deactivate
conda activate bio-agent

python -u /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/experiments/qualcheck_ae10_pretrained_umap.py
