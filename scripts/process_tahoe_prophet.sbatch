#!/bin/zsh

#SBATCH -o logs/process_tahoe_prophet.out
#SBATCH -e logs/process_tahoe_prophet.err
#SBATCH -J process_tahoe_prophet
#SBATCH --nice=1
#SBATCH --time=23:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=26
#SBATCH --mem=500G

source /home/icb/selman.ozleyen/.zshrc

mamba activate lpert
python /home/icb/selman.ozleyen/projects/CellFlow2/scripts/process_tahoe_prophet.py
