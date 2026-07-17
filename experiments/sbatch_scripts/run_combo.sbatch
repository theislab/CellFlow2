#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# comboSciplex training on the dedup/reexport-cellflow branch (cellflow API).
# Installs cellflow @ feat/annbatch-loader into a fresh jax_container (the branch
# needs it; main's prepare_data API has drifted), then runs experiments/train_combo.py.
#
#   RUN_NAME=combo_gene_x sbatch experiments/sbatch_scripts/run_combo.sbatch
#   SAMPLE_REP=X_pca NUM_ITER=200000 sbatch experiments/sbatch_scripts/run_combo.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=combo_gene_x
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_p
#SBATCH --nodelist=supergpu[07-33]
#SBATCH --qos=gpu_normal
#SBATCH --output=/lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs/combo_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs/combo_%j.err

EXPERIMENT="${EXPERIMENT:-combosciplex_h5ad}"   # config/experiment/<EXPERIMENT>.yaml (combosciplex_h5ad | sciplex3_h5ad)
RUN_NAME="${RUN_NAME:-${EXPERIMENT}}"
NUM_ITER="${NUM_ITER:-2000000}"
VALID_FREQ="${VALID_FREQ:-200000}"
IMAGE="${IMAGE:-jax_cellflow_hydra.sqsh}"       # baked with hydra-core (compute nodes can't pip-install)
EXTRA="${EXTRA:-}"                      # extra hydra overrides (e.g. ablation=prophet model.epsilon=1 eval_test=true)

# WANDB_API_KEY from ~/.wandb_key (untracked)
[ -f "$HOME/.wandb_key" ] && source "$HOME/.wandb_key"
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in ~/.wandb_key}"

CONTAINER_NAME="jax_combo_${SLURM_JOB_ID}"
rm -rf /localscratch/karthik.viswanathan/enroot/data/${CONTAINER_NAME}/
mkdir -p /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs
cd /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/

# DNS fix so wandb online doesn't hang on nodes with a localhost-only resolver
WANDB_RESOLV="/localscratch/karthik.viswanathan/resolv.conf.wandb"
mkdir -p /localscratch/karthik.viswanathan
printf 'search scidom.de helmholtz-muenchen.de helmholtz-munich.de\nnameserver 146.107.8.88\nnameserver 146.107.1.88\n' > "$WANDB_RESOLV"

# ${IMAGE} = jax_container + cellflow@feat/annbatch-loader + omegaconf + hydra-core pre-installed
# (exported from a verified container; avoids per-job installs, which fail on compute nodes w/o PyPI).
enroot create --name ${CONTAINER_NAME} ${IMAGE}
enroot start --rw \
    --mount /lustre/groups/ml01/workspace/karthik.viswanathan/:/storage \
    --mount ${WANDB_RESOLV}:/etc/resolv.conf \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    ${CONTAINER_NAME} \
    bash -c "cd /storage/pancellflow/CellFlow2 && PYTHONUNBUFFERED=1 python experiments/train_comparison.py \
                 +experiment=${EXPERIMENT} training.num_iterations=${NUM_ITER} training.valid_freq=${VALID_FREQ} \
                 wandb.enabled=true wandb.run_name=${RUN_NAME} ${EXTRA}"

enroot remove ${CONTAINER_NAME} 2>/dev/null || true
