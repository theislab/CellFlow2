#!/bin/bash

#SBATCH -o train_and_predict_%A_%a.out
#SBATCH -e train_and_predict_%A_%a.err
#SBATCH -J train_predict
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem 200G
#SBATCH --time 24:00:00
#SBATCH --nice=0
#SBATCH --array=0-1
#SBATCH --constraint=h100_80gb

export TMPDIR=/lustre/groups/ml01/workspace/alejandro.tejada/tmp
mkdir -p $TMPDIR

source /home/icb/alejandro.tejada/.bashrc
micromamba activate scaleflow

RANDOM_EMB_FLAGS=("" "--random-embeddings")
RUN_NAMES=("real_embeddings_film_otfm_unipert" "random_embeddings_film_otfm_unipert")

FLAG=${RANDOM_EMB_FLAGS[$SLURM_ARRAY_TASK_ID]}
RUN=${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}
OUTDIR=/lustre/groups/ml01/workspace/alejandro.tejada/train_and_predict/${RUN}

python -u /home/icb/alejandro.tejada/CellFlow2/experiments/train_and_predict.py \
    --condition-fraction 0.5 \
    --cell-fraction 0.5 \
    --experiment-type condition_subsampling \
    --output-dir ${OUTDIR} \
    ${FLAG}
