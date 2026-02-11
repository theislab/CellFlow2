#!/bin/bash
#SBATCH -J scaling_sweep_exp
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.err
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH -t 0-24:00:00
#SBATCH --nice=1
#SBATCH --exclude=gpusrv55
#SBATCH --array=0-12


source $HOME/.bashrc
conda deactivate
conda activate pancellflow
export HYDRA_FULL_ERROR=1

archs=(
  "[256]"
  "[1024]"
  "[2048]"
  "[4096]"
  "[256,256]"
  "[1024,1024]"
  "[2048,2048]"
  "[4096,4096]"
  "[256,256,256,256]"
  "[1024,1024,1024,1024]"
  "[2048,2048,2048,2048]"
  "[4096,4096,4096,4096]"
)

if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${#archs[@]}" ]]; then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range (max=$(( ${#archs[@]} - 1 )))"
  exit 1
fi

p="${archs[$SLURM_ARRAY_TASK_ID]}"
echo "=== Task ${SLURM_ARRAY_TASK_ID} arch: ${p} ==="

python /lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/experiments/train_tahoe_combosciplex.py \
  solver=otfm \
  conditioning=concatenation \
  match_fn.epsilon=10 \
  model.probability_path_kwargs.constant_noise=0.1 \
  training.num_iterations=2000000 \
  "model.decoder_dims=${p}" \
  "model.hidden_dims=${p}" \
  training.valid_freq=500