#!/bin/bash
#SBATCH --job-name=check_a549
#SBATCH -o %x_%A_out.out
#SBATCH -e %x_%A_err.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=02:00:00

source "$HOME/.bashrc"
conda deactivate
conda activate pancellflow

SCRIPTS_DIR="/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/scripts/_generate_drugembedding"

python -u "${SCRIPTS_DIR}/check_a549_compatibility.py"
