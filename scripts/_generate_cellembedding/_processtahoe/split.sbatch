#!/bin/bash
#SBATCH --job-name=tahoe_split
#SBATCH -o %x_%j.txt
#SBATCH -e %x_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=24:00:00

source $HOME/.bashrc
conda deactivate
conda activate pancellflow

python "$SPLIT_PY" \
  --in "$IN_H5AD" \
  --out-dir "$CHUNK_DIR" \
  --chunk-size "$CHUNK_SIZE" \
  --compression gzip