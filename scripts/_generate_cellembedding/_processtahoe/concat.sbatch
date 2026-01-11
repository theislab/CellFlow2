#!/bin/bash
#SBATCH --job-name=tahoe_concat
#SBATCH -o %x_%j.txt
#SBATCH -e %x_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=24:00:00

source $HOME/.bashrc
conda deactivate
conda activate pancellflow

# Build manifest of embedded chunks
ls -1 "$EMB_DIR"/*_with_state.h5ad > "$EMB_MANIFEST"

python "$CONCAT_PY" \
  --in-manifest "$EMB_MANIFEST" \
  --out "$OUT_H5AD" \
  --join inner