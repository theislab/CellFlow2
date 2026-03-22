#!/bin/bash
#SBATCH --job-name=state_emb
#SBATCH -o %x_%A_out.out
#SBATCH -e %x_%A_err.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=400G
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=24:00:00

source $HOME/.bashrc
conda deactivate 
conda activate pancellflow


python -u /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/scripts/_generate_cellembedding/_processscale/picktahoecellline.py
