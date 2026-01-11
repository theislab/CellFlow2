#!/bin/bash
#SBATCH --job-name=state_emb
#SBATCH -o %x_%A_%a.txt
#SBATCH -e %x_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=300G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_priority
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_80gb|h100_80gb"
#SBATCH --time=12:00:00

source $HOME/.bashrc
conda deactivate 
conda activate pancellflow

mkdir -p "$EMB_DIR"

# If this is the "parent" submission (no array task id), re-submit as an array now that manifest exists.
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  NCHUNKS=$(wc -l < "$MANIFEST")
  if [ "$NCHUNKS" -le 0 ]; then
    echo "Manifest is empty: $MANIFEST" >&2
    exit 1
  fi
  echo "Resubmitting as array with NCHUNKS=$NCHUNKS"
  sbatch --parsable --array=0-$((NCHUNKS-1)) "$0"
  exit 0
fi

NCHUNKS=$(wc -l < "$MANIFEST")
TASK=${SLURM_ARRAY_TASK_ID}

INFILE=$(sed -n "$((TASK+1))p" "$MANIFEST")
BASENAME=$(basename "$INFILE")
OUTFILE="$EMB_DIR/${BASENAME%.h5ad}_with_state.h5ad"

echo "[chunk $((TASK+1))/$NCHUNKS] $INFILE -> $OUTFILE"

python "$EMBED_PY" \
  --in "$INFILE" \
  --out "$OUTFILE" \
  --checkpoint "$CHECKPOINT" \
  --protein-embeds "$PROTEMB" \
  --obsm-key "$OBSM_KEY" \
  --encode-batch-size "$ENCODE_BS" \
  --chunk-id "$TASK" \
  --n-chunks "$NCHUNKS"