#!/bin/bash
#SBATCH --job-name=sciplex_fast_runs
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_p
#SBATCH --nodelist=supergpu[14-33]
#SBATCH --qos=gpu_normal
#SBATCH --output=/lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs/sciplex_fast5_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs/sciplex_fast5_%j.err

export WANDB_API_KEY='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'

CONTAINER_NAME="jax_container_sciplex_fast5_${SLURM_JOB_ID}"

# clean up any leftover container and ensure output dirs exist
rm -rf /localscratch/karthik.viswanathan/enroot/data/${CONTAINER_NAME}/
mkdir -p /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/logs
mkdir -p /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/outputs/sciplex_fast
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
          conditioning=adaln_zero \
          model.hidden_dims=[2048,2048,2048] \
          model.decoder_dims=[2048,2048,2048] \
          model.condition_embedding_dim=256 \
          model.probability_path_kwargs.constant_noise=0.1 \
          solver.predict_kwargs.max_steps=32768 \
          match_fn.epsilon=1.0 \
          training.peak_lr=1e-6 \
          training.batch_size=512 \
          training.valid_freq=200000 \
          training.num_iterations=2000000 \
          training.warmup_iterations=10000 \
          training.grad_accumulation=20 \
          training.pool_fraction=0.7 \
          training.replacement_prob=0.5 \
          wandb.enabled=true \
          wandb.project=sciplex-fast-runs \
          output_dir=/storage/pancellflow/outputs/sciplex_fast
    "

enroot remove -f ${CONTAINER_NAME}