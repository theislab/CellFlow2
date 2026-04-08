#!/bin/bash

#SBATCH -o sanity_cf_tahoe_X_state_%j.out
#SBATCH -e sanity_cf_tahoe_X_state_%j.err
#SBATCH -J sanity_cf_tahoe_X_state
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem 200G
#SBATCH --time 24:00:00
#SBATCH --nice=0
#SBATCH --constraint=h100_80gb

source $HOME/.bashrc
conda deactivate
conda activate pancellflow

SCRIPT=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/experiments/sanity_check_cf_tahoe_X_state.py
BASE=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/sanity_check_cf/tahoe/X_state

# Full grid: 3 capacity levels x 2 drug sets (+80 strong / +80 weak) = 6 runs

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

echo "=== capacity 2x + 80 strong: hidden=4096 decoder=8192 ==="
python -u "${SCRIPT}" \
    --output-dir "${BASE}/top10_capacity2x_strong80" \
    "${COMMON_ARGS[@]}" \
    --hidden-dim 4096 --decoder-dim 8192 \
    --extra-n-drugs 80 --extra-drug-type strong

echo "=== capacity 2x + 80 weak: hidden=4096 decoder=8192 ==="
python -u "${SCRIPT}" \
    --output-dir "${BASE}/top10_capacity2x_weak80" \
    "${COMMON_ARGS[@]}" \
    --hidden-dim 4096 --decoder-dim 8192 \
    --extra-n-drugs 80 --extra-drug-type weak

echo "=== capacity 4x + 80 strong: hidden=8192 decoder=16384 ==="
python -u "${SCRIPT}" \
    --output-dir "${BASE}/top10_capacity4x_strong80" \
    "${COMMON_ARGS[@]}" \
    --hidden-dim 8192 --decoder-dim 16384 \
    --extra-n-drugs 80 --extra-drug-type strong

echo "=== capacity 4x + 80 weak: hidden=8192 decoder=16384 ==="
python -u "${SCRIPT}" \
    --output-dir "${BASE}/top10_capacity4x_weak80" \
    "${COMMON_ARGS[@]}" \
    --hidden-dim 8192 --decoder-dim 16384 \
    --extra-n-drugs 80 --extra-drug-type weak

echo "Done."

