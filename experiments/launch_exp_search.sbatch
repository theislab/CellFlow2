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
    solver=otfm,eqm \
    conditioning=concatenation \
    match_fn.epsilon=10 \
    model.probability_path_kwargs.constant_noise=0.1 \
    training.num_iterations=2000000 \
    'model.decoder_dims="(1024)","(2048)","(2048, 2048, 2048, 2048)"' \
    'model.hidden_dims="(1024)","(2048)","(2048, 2048, 2048, 2048)"' \
    training.valid_freq=500 \
    +launcher=gpu_normal




