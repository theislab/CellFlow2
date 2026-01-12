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


INFILE="/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe_a549.h5ad"
OUTFILE="/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe_a549_w_se.h5ad"
WORKDIR="/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe_a549"
CHUNK_SIZE=100000



CHECKPOINT="/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/sup/SE-600M/se600m_epoch16.ckpt"
PROTEMB="/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/sup/SE-600M/protein_embeddings.pt"
OBSM_KEY="X_state"
ENCODE_BS=256


python /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/scripts/_generate_cellembedding/emb_adata.py \
  --in "$INFILE" \
  --out "$OUTFILE" \
  --checkpoint "$CHECKPOINT" \
  --protein-embeds "$PROTEMB" \
  --obsm-key "$OBSM_KEY" \
  --encode-batch-size "$ENCODE_BS" 