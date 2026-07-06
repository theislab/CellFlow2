#!/bin/bash
#SBATCH --job-name=cellflow_prophet
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_p
#SBATCH --nodelist=supergpu[07-33]
#SBATCH --qos=gpu_normal
#SBATCH --output=/lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs/sweep_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs/sweep_%j.err
#SBATCH --array=1-<k_sweeps>%4  

export WANDB_API_KEY='<your_api_key>'
CONTAINER_NAME="jax_container_${SLURM_ARRAY_TASK_ID}"
rm -rf /localscratch/karthik.viswanathan/enroot/data/${CONTAINER_NAME}/
mkdir -p /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs
cd /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/

# --- DNS fix: some compute nodes hand the container a localhost-only resolver
# ([::1]:53 / systemd-resolved stub) that enroot cannot reach, so `wandb agent`
# and wandb online uploads fail to resolve api.wandb.ai and retry forever (the run
# hangs at finish). Pin the site nameservers into the container so DNS works on any
# node. Verify with: getent hosts api.wandb.ai
WANDB_RESOLV="/localscratch/karthik.viswanathan/resolv.conf.wandb"
mkdir -p /localscratch/karthik.viswanathan
printf 'search scidom.de helmholtz-muenchen.de helmholtz-munich.de\nnameserver 146.107.8.88\nnameserver 146.107.1.88\n' > "$WANDB_RESOLV"

enroot create --name ${CONTAINER_NAME} jax_container.sqsh
enroot start --rw \
    --mount /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/:/storage/pancellflow \
    --mount ${WANDB_RESOLV}:/etc/resolv.conf \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    ${CONTAINER_NAME} \
    bash -c "pip3 install hydra-core && cd /storage/pancellflow/CellFlow2 && wandb agent <your_sweep_id>"

enroot remove ${CONTAINER_NAME}
