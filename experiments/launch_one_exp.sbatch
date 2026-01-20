#!/bin/bash

#SBATCH -J launch_one_exp
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1
#SBATCH --exclude=gpusrv55

source $HOME/.bashrc
conda deactivate
conda activate pancellflow
export HYDRA_FULL_ERROR=1

python /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/experiments/train_tahoe_combosciplex.py \
    solver=genot \
    training.num_iterations=200000


