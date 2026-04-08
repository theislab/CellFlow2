#!/bin/bash

#SBATCH -o crosscellline_%A_%a.out
#SBATCH -e crosscellline_%A_%a.err
#SBATCH -J crosscl
#SBATCH -p gpu_p
#SBATCH --qos gpu_priority
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 250G
#SBATCH --time 12:00:00
#SBATCH --nice=0
#SBATCH --array=0-2
#SBATCH --constraint=h100_80gb

source $HOME/.bashrc
conda deactivate
conda activate pancellflow
export HYDRA_FULL_ERROR=1

SCRIPT=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/experiments/train_tahoe_crosscellline.py
OUTDIR=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/crosscellline

# ── Experiment definitions ──────────────────────────────────────────
# A1: Baseline — A549 alone (Set A train, Set B test)
# B1: A549 + SW48 anchor (SW48 most distant cell line)
# B2: A549 + PANC-1 anchor (PANC-1 closest cell line)

case $SLURM_ARRAY_TASK_ID in
    0)
        EXP_ID="A1"
        ANCHOR=""
        EVAL="A549"
        ;;
    1)
        EXP_ID="B1"
        ANCHOR="SW48"
        EVAL="A549"
        ;;
    2)
        EXP_ID="B2"
        ANCHOR="PANC-1"
        EVAL="A549"
        ;;
esac

echo "=========================================="
echo "Experiment: ${EXP_ID}"
echo "Anchor cell lines: ${ANCHOR:-NONE}"
echo "Eval cell lines: ${EVAL}"
echo "SLURM Job ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "=========================================="

# Build command
CMD="python -u ${SCRIPT} \
    --eval-celllines ${EVAL} \
    --experiment-id ${EXP_ID} \
    --output-dir ${OUTDIR} \
    --num-iterations 200000 \
    --batch-size 512 \
    --valid-freq 5000 \
    --max-cells-per-cellline 200000 \
    --wandb-project tahoe-crosscellline"

if [ -n "$ANCHOR" ]; then
    CMD="${CMD} --anchor-celllines ${ANCHOR}"
fi

echo "Running: ${CMD}"
eval ${CMD}
