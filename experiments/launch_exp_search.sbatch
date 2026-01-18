#!/bin/bash

#SBATCH -J launch_exp_search
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=20G
#SBATCH -t 0-00:30:00
#SBATCH --nice=1

source $HOME/.bashrc
conda deactivate
conda activate pancellflow
export HYDRA_FULL_ERROR=1

python /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/experiments/train_tahoe_combosciplex.py \
    --multirun \
    solver=eqm,genot,otfm \
    training.num_iterations=100000 \
    +launcher=gpu_normal



