## Instructions

- To run an experiment, define a sweep file `sweep_file.yaml`. 
- Login to WANDB using `wandb login`.
- Generate the sweep config `wandb sweep sweep_file.yaml`.
- Paste the sweep ID generated, and your WANDB key in `run_sweep.sbatch` (Copy from `run_sweep_template.sbatch`).
- Update k-sweeps to your sweep limit in `run_sweep.sbatch` : `#SBATCH --array=1-<k_sweeps>%4`.
- Update paths, etc. if needed in SBATCH file.
- Run sbatch.
