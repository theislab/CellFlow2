#!/bin/bash
#SBATCH --job-name=concat_on_disk
#SBATCH -o %x_%j.txt
#SBATCH -e %x_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=500G
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=24:00:00      

source $HOME/.bashrc
conda deactivate 
conda activate pancellflow

INDIR="/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe/test"
OUTH5AD="/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe/tahoe_w_se.h5ad"
SCRIPTS="/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/scripts/_generate_cellembedding/_processscale"

# Two-stage concat: batch chunks in memory (reads obsm via h5py), then concat_on_disk
python -u "$SCRIPTS/concat_twostage.py" \
  --indirec "$INDIR" \
  --outh5ad "$OUTH5AD" \
  --obsm-key "X_state" \
  --batch-size 25

echo "Done: $OUTH5AD"