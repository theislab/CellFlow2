#!/bin/bash
#SBATCH --job-name=disp_pancellflow
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_p
#SBATCH --nodelist=supergpu[14-33]
#SBATCH --qos=gpu_normal
#SBATCH --output=/lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs/nndisp_dryrun_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs/nndisp_dryrun_%j.err

export WANDB_API_KEY='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'

CONTAINER_NAME="jax_container_sinkhorn_dryrun_${SLURM_JOB_ID}"

rm -rf /localscratch/karthik.viswanathan/enroot/data/${CONTAINER_NAME}/
mkdir -p /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs
mkdir -p /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/outputs/sinkhorn_dryrun
mkdir -p /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/jax_cache

cd /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/
enroot create --name ${CONTAINER_NAME} jax_container.sqsh

enroot start --rw \
    --mount /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/:/storage/pancellflow \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env JAX_COMPILATION_CACHE_DIR=/storage/pancellflow/jax_cache \
    ${CONTAINER_NAME} \
    bash -c "
        pip3 install hydra-core --quiet && \
        cd /storage/pancellflow/CellFlow2 && \
        python experiments/train_zarr.py \
          datasets=zarr_sciplex \
          selected_datasets=[sciplex] \
          ablation.mode=prophet \
          model.hidden_dims=[2048,2048,2048] \
          model.decoder_dims=[2048,2048,2048] \
          model.conditioning_key=concatenation \
          model.condition_embedding_dim=256 \
          model.probability_path_kwargs.constant_noise=0.1 \
          match_fn.epsilon=1 \
          match_fn.sinkhorn_alpha=0 \
          match_fn.debug=true \
          training.peak_lr=1e-4 \
          training.batch_size=512 \
          training.warmup_iterations=10000 \
          training.grad_accumulation=2000000 \
          training.pool_fraction=0.7 \
          training.replacement_prob=0.5 \
          training.num_iterations=50000 \
          training.valid_freq=200000 \
          wandb.enabled=true \
          wandb.project=sciplex-sinkhorn \
          output_dir=/storage/pancellflow/outputs/sinkhorn_dryrun
    "

enroot remove -f ${CONTAINER_NAME}