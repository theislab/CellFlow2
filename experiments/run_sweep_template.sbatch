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

export WANDB_API_KEY='<your_api_key>'
CONTAINER_NAME="jax_container_${SLURM_ARRAY_TASK_ID}"
rm -rf /localscratch/karthik.viswanathan/enroot/data/${CONTAINER_NAME}/
mkdir -p /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs
cd /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/
enroot create --name ${CONTAINER_NAME} jax_container.sqsh
enroot start --rw \
    --mount /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/:/storage/pancellflow \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    ${CONTAINER_NAME} \
    bash -c "pip3 install hydra-core && cd /storage/pancellflow/CellFlow2 && wandb agent <your_sweep_id>"

enroot remove ${CONTAINER_NAME}
