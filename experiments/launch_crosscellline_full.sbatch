#!/bin/bash

#SBATCH -o crosscellline_full_%A_%a.out
#SBATCH -e crosscellline_full_%A_%a.err
#SBATCH -J crosscl_f
#SBATCH -p gpu_p
#SBATCH --qos gpu_priority
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 350G
#SBATCH --time 12:00:00
#SBATCH --nice=0
#SBATCH --array=0-2
#SBATCH --constraint=h100_80gb

source $HOME/.bashrc
conda deactivate
conda activate pancellflow
export HYDRA_FULL_ERROR=1

SCRIPT=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/experiments/train_tahoe_crosscellline.py
OUTDIR=/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/outputs/crosscellline_full

# Same experiments as before but NO cell subsampling (use all cells)
# A1_full: A549 alone, all cells
# B1_full: SW48 (all) + A549 (all)
# B2_full: PANC-1 (all) + A549 (all)

case $SLURM_ARRAY_TASK_ID in
    0)
        EXP_ID="A1_full"
        ANCHOR=""
        EVAL="A549"
        ;;
    1)
        EXP_ID="B1_full"
        ANCHOR="SW48"
        EVAL="A549"
        ;;
    2)
        EXP_ID="B2_full"
        ANCHOR="PANC-1"
        EVAL="A549"
        ;;
esac

echo "=========================================="
echo "Experiment: ${EXP_ID} (FULL DATA, no subsampling)"
echo "Anchor cell lines: ${ANCHOR:-NONE}"
echo "Eval cell lines: ${EVAL}"
echo "=========================================="

CMD="python -u ${SCRIPT} \
    --eval-celllines ${EVAL} \
    --experiment-id ${EXP_ID} \
    --output-dir ${OUTDIR} \
    --num-iterations 200000 \
    --batch-size 512 \
    --valid-freq 5000 \
    --max-cells-per-cellline 99999999 \
    --wandb-project tahoe-crosscellline"

if [ -n "$ANCHOR" ]; then
    CMD="${CMD} --anchor-celllines ${ANCHOR}"
fi

echo "Running: ${CMD}"
eval ${CMD}
