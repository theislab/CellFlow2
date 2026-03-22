#!/bin/bash
#SBATCH --job-name=drug_emb
#SBATCH -o %x_%A_out.out
#SBATCH -e %x_%A_err.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=24:00:00

source "$HOME/.bashrc"
conda deactivate
conda activate pancellflow

SCRIPTS_DIR="/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/scripts/_generate_drugembedding"

python -u "${SCRIPTS_DIR}/add_drug_embeddings.py" \
  --config "${SCRIPTS_DIR}/config_drug_emb.yaml"
