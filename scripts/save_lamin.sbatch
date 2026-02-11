#!/bin/bash
#SBATCH --job-name=save_lamin
#SBATCH -o %x_%A_%a.txt
#SBATCH -e %x_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=400G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_80gb|h100_80gb"
#SBATCH --time=24:00:00

source $HOME/.bashrc
conda deactivate 
conda activate pancellflow



python /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/scripts/save_lamin.py 