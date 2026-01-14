#!/bin/bash
#SBATCH --job-name=concat_on_disk
#SBATCH -o %x_%j.txt
#SBATCH -e %x_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=300G
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=24:00:00      

source $HOME/.bashrc
conda deactivate 
conda activate pancellflow

INDIR="tahoe_a549/chunks_emb"
OUTH5AD="tahoe_a549_se.h5ad"
JOIN="exact"

mkdir -p "$(dirname "$OUTH5AD")"

echo "INDIR=$INDIR"
echo "OUTH5AD=$OUTH5AD"

python -u concat_on_disk.py \
  --indirec "$INDIR" \
  --outh5ad "$OUTH5AD" \
  --join "$JOIN"

echo "Done: $OUTH5AD"