#!/bin/bash
# Train reconstruction embeddings/decoders on sciplex3 via hydra multirun.
#   - AE mode:         latent 10 / 32 / 128  (writes AE_<latent> embeddings to EMB h5ad)
#   - pretrained mode: decoders for every existing embedding (X_state, scgpt, scimilarity, …)
# Submit:  sbatch experiments/run_recon_emb.sbatch
#SBATCH --job-name=recon_emb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nice=0
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --output=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/recon_weights/logs/recon_emb_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/recon_weights/logs/recon_emb_%j.err

set -e
PY=/home/icb/xiaotong.fu/miniconda3/envs/pancellflow/bin/python
cd /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/sub

WD=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/recon_weights
EMB=$WD/sciplex3_with_emb.h5ad        # writable copy: accumulates the AE_<latent> embeddings

echo "=== AE multirun: latent 10, 32, 128 ==="
$PY experiments/recon_emb.py --multirun \
  mode=ae latent_dim=10,32,128 \
  weights_dir=$WD embedding_out=$EMB

echo "=== Pretrained multirun: X_state, X_scgpt, X_scimilarity, X_scimilarity_correct, X_scconcept ==="
$PY experiments/recon_emb.py --multirun \
  mode=pretrained \
  pretrained_key=X_state,X_scgpt,X_scimilarity,X_scimilarity_correct,X_scconcept \
  weights_dir=$WD

echo "=== DONE ==="
