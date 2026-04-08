#!/bin/bash

#SBATCH -J test_ccle_cross
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=200G
#SBATCH -t 0-02:00:00
#SBATCH --nice=1

source $HOME/.bashrc
conda deactivate
conda activate pancellflow
export HYDRA_FULL_ERROR=1

python /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/experiments/train_crossdatasets.py \
    solver=otfm \
    selected_datasets=[tahoe_a549,combosciplex] \
    conditioning=concatenation \
    cell_embedding.cell_line_rep_key=cell_line_ccle_embeddings \
    match_fn.epsilon=10 \
    model.probability_path_kwargs.constant_noise=0.1 \
    training.num_iterations=2 \
    training.valid_freq=1 \
    model.decoder_dims='[1024]' \
    model.hidden_dims='[1024]' \
    training.batch_size=256
