#!/bin/bash

#SBATCH -o sanity_cf_tahoe_X_state_%j.out
#SBATCH -e sanity_cf_tahoe_X_state_%j.err
#SBATCH -J sanity_cf_tahoe_X_state
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem 200G
#SBATCH --time 12:00:00
#SBATCH --nice=0
#SBATCH --constraint=h100_80gb

source $HOME/.bashrc
conda deactivate
conda activate pancellflow

SCRIPT=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/experiments/sanity_check_cf_tahoe_X_state.py
BASE=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/sanity_check_cf/tahoe/X_state

# Number of extra drugs to add beyond the top-10 validation drugs
N_EXTRA=80

COMMON_ARGS=(
    --top-n-drugs 10
    --num-iterations 500000
    --valid-freq 20000
    --batch-size 1024
    --epsilon 0.5
    --noise 0.5
    --lr 5e-5
    --seed 42
)

echo "=== top10 + ${N_EXTRA} strong extras ==="
python -u "${SCRIPT}" \
    --output-dir "${BASE}/top10_strong${N_EXTRA}" \
    "${COMMON_ARGS[@]}" \
    --extra-n-drugs ${N_EXTRA} --extra-drug-type strong

echo "=== top10 + ${N_EXTRA} weak extras ==="
python -u "${SCRIPT}" \
    --output-dir "${BASE}/top10_weak${N_EXTRA}" \
    "${COMMON_ARGS[@]}" \
    --extra-n-drugs ${N_EXTRA} --extra-drug-type weak

echo "Done."

