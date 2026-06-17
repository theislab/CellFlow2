#!/bin/bash
#SBATCH --job-name=sinkhorn_sweep
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_p
#SBATCH --nodelist=supergpu[14-33]
#SBATCH --qos=gpu_normal
#SBATCH --output=/lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs/sinkhorn_sweep_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs/sinkhorn_sweep_%j.err

# Usage: wandb sweep experiments/sweep_sinkhorn.yaml  →  get <sweep_id>
#        sbatch experiments/sbatch_sinkhorn_sweep.sh <sweep_id>
SWEEP_ID=${1:?Usage: sbatch sbatch_sinkhorn_sweep.sh <sweep_id>}

export WANDB_API_KEY='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'

CONTAINER_NAME="jax_container_sinkhorn_sweep_${SLURM_JOB_ID}"

rm -rf /localscratch/karthik.viswanathan/enroot/data/${CONTAINER_NAME}/
mkdir -p /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs
mkdir -p /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/outputs/sinkhorn
mkdir -p /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/jax_cache

cd /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/
enroot create --name ${CONTAINER_NAME} jax_container.sqsh

enroot start --rw \
    --mount /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/:/storage/pancellflow \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env JAX_COMPILATION_CACHE_DIR=/storage/pancellflow/jax_cache \
    ${CONTAINER_NAME} \
    bash -c "
        pip3 install hydra-core wandb --quiet && \
        cd /storage/pancellflow/CellFlow2 && \
        wandb agent pancellflow/sciplex-sinkhorn/${SWEEP_ID} \
          --count 1
    "

enroot remove -f ${CONTAINER_NAME}
