#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Single training run (NOT a wandb sweep). Runs `train_zarr.py +experiment=...`
# directly and logs one wandb run whose name you control.
#
#   sbatch experiments/sbatch_scripts/run_single.sbatch                       # uses default RUN_NAME
#   RUN_NAME=my_cfg_run sbatch experiments/sbatch_scripts/run_single.sbatch   # custom name
#
# Edit EXPERIMENT / RUN_NAME below, or pass RUN_NAME=... on the sbatch line.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=cfg_ae128opt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_p
#SBATCH --nodelist=supergpu[07-33]
#SBATCH --qos=gpu_normal
#SBATCH --output=/lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs/run_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs/run_%j.err

# ── the hydra experiment config (config/experiment/<EXPERIMENT>.yaml) and run name ──
EXPERIMENT="${EXPERIMENT:-cfg_ae128opt}"
RUN_NAME="${RUN_NAME:-cfg_ae128opt_adaln}"
PROJECT="${PROJECT:-sciplex_cfg}"

# WANDB_API_KEY is loaded from ~/.wandb_key (untracked) — never hardcode secrets in the repo.
#   echo "export WANDB_API_KEY='<your_key>'" > ~/.wandb_key && chmod 600 ~/.wandb_key
[ -f "$HOME/.wandb_key" ] && source "$HOME/.wandb_key"
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in ~/.wandb_key}"
CONTAINER_NAME="jax_container_run_${SLURM_JOB_ID}"
rm -rf /localscratch/karthik.viswanathan/enroot/data/${CONTAINER_NAME}/
mkdir -p /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs
cd /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/

# --- DNS fix: pin site nameservers so wandb online doesn't hang on nodes with a
# localhost-only resolver (see run_sweep_template.sbatch). ---
WANDB_RESOLV="/localscratch/karthik.viswanathan/resolv.conf.wandb"
mkdir -p /localscratch/karthik.viswanathan
printf 'search scidom.de helmholtz-muenchen.de helmholtz-munich.de\nnameserver 146.107.8.88\nnameserver 146.107.1.88\n' > "$WANDB_RESOLV"

enroot create --name ${CONTAINER_NAME} jax_container.sqsh
enroot start --rw \
    --mount /lustre/groups/ml01/workspace/karthik.viswanathan/:/storage \
    --mount ${WANDB_RESOLV}:/etc/resolv.conf \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    ${CONTAINER_NAME} \
    bash -c "pip3 install -q hydra-core && cd /storage/pancellflow/CellFlow2 && PYTHONUNBUFFERED=1 python experiments/train_zarr.py +experiment=${EXPERIMENT} wandb.run_name=${RUN_NAME} wandb.project=${PROJECT}"

enroot remove ${CONTAINER_NAME} 2>/dev/null || true

enroot remove ${CONTAINER_NAME}
