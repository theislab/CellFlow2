#!/bin/bash

#SBATCH -o updated_together_%j.out
#SBATCH -e updated_together_%j.err
#SBATCH -J eqm_adaln
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH -t 0-24:00:00
#SBATCH --nice=1
#SBATCH --exclude=gpusrv55

export TMPDIR=/lustre/groups/ml01/workspace/alejandro.tejada/tmp
mkdir -p $TMPDIR

source /home/icb/alejandro.tejada/.bashrc
micromamba activate scaleflow

python -u /home/icb/alejandro.tejada/CellFlow2/experiments/updated_together_training.py

