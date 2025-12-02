#!/bin/bash

#SBATCH -o tahoe_ds_%j.out
#SBATCH -e tahoe_ds_%j.err
#SBATCH -J tahoe
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
##SBATCH --constraint=a100_80gb
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH -t 0-24:00:00
#SBATCH --nice=1
#SBATCH --exclude=gpusrv55

export TMPDIR=/lustre/groups/ml01/workspace/alejandro.tejada/tmp
mkdir -p $TMPDIR

source /home/icb/alejandro.tejada/.bashrc
micromamba activate scaleflow

python -u reproducibility/scripts/train_multidataset.py \
    data=tahoe_sciplex \
    training=standard \
    training.save_model=true \
    training.out_dir="/home/icb/alejandro.tejada/CellFlow2/outputs" \
    training.callbacks.use_wandb=true \
    training.callbacks.wandb_project="cellflow_sciplex_hparam_new_embs"
