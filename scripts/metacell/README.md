# Metacell Construction for Perturbation Datasets

Build [SEACells](https://github.com/dpeerlab/SEACells) metacells from a single-cell perturbation h5ad file. Perturbed cells are aggregated into metacells per condition; control cells are kept as individual cells.

The output h5ad is the expected input format for CellFlow2 cross-dataset responder analysis.

---

## Requirements

**Conda environment:** `bio-agent` (SEACells + zarr 2.x)

```bash
conda activate bio-agent
```

---

## Input h5ad format

The input h5ad must have:

| Slot | Key | Description |
|---|---|---|
| `obs` | `cell_line` | Cell line name (e.g. `"A549"`) |
| `obs` | `drug` | Drug / perturbation name (e.g. `"Trametinib"`; vehicle/DMSO for controls) |
| `obs` | `control` | `bool` — `True` for DMSO/vehicle control cells |
| `obsm` | `X_state` | `(n_cells, D)` float — foundation-model embedding (configurable via `--obsm`) |

---

## Output h5ad format

Same structure as input, restricted to:
- All control cells (kept as-is, `control=True`)
- One metacell row per SEACell per condition (`control=False`)

---

## Usage

### Minimal

```bash
python scripts/metacell/build_perturbation_metacells.py \
    --input  path/to/input.h5ad \
    --output path/to/output_metacell.h5ad
```

### Filter to one cell line

```bash
python scripts/metacell/build_perturbation_metacells.py \
    --input     path/to/input.h5ad \
    --output    path/to/output_metacell.h5ad \
    --cell-line A549
```

### Custom embedding key

```bash
python scripts/metacell/build_perturbation_metacells.py \
    --input  path/to/input.h5ad \
    --output path/to/output_metacell.h5ad \
    --obsm   X_scvi
```

### Rename drugs to match a reference dataset

Supply a two-column TSV (`source_name<TAB>target_name`, no header):

```bash
python scripts/metacell/build_perturbation_metacells.py \
    --input    path/to/input.h5ad \
    --output   path/to/output_metacell.h5ad \
    --drug-map scripts/metacell/sci_to_tahoe_drugs.tsv
```

`sci_to_tahoe_drugs.tsv` is provided as an example mapping Sciplex drug names to Tahoe convention.

### Restrict to a specific set of drugs

```bash
# drugs_of_interest.txt — one drug name per line
python scripts/metacell/build_perturbation_metacells.py \
    --input     path/to/input.h5ad \
    --output    path/to/output_metacell.h5ad \
    --cell-line A549 \
    --drug-list drugs_of_interest.txt
```

---

## SLURM (HPC)

Create a job script adapting the example below:

```bash
#!/bin/zsh
#SBATCH --job-name=build_metacells
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o logs/build_metacells_%j.out
#SBATCH -e logs/build_metacells_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bio-agent

python scripts/metacell/build_perturbation_metacells.py \
    --input     /path/to/input.h5ad \
    --output    /path/to/output_metacell.h5ad \
    --cell-line A549 \
    --obsm      X_state
```

---

## All options

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Path to input h5ad |
| `--output` | required | Path for output h5ad |
| `--obsm` | `X_state` | `obsm` key for the cell embedding |
| `--cell-line` | all lines | Filter to a single cell line |
| `--drug-map` | none | TSV to rename drugs before processing |
| `--drug-list` | all drugs | Text file of drugs to include (one per line) |
| `--checkpoint-dir` | auto | Directory for per-group crash-recovery checkpoints |
| `--ctrl-key` | `control` | `obs` column marking control cells |
| `--drug-key` | `drug` | `obs` column for drug names |
| `--cell-line-key` | `cell_line` | `obs` column for cell line |
| `--cells-per-metacell` | `75` | Target cells aggregated per metacell |
| `--min-cells` | `50` | Groups smaller than this are collapsed to a single mean |
| `--n-pca` | `50` | PCA components for SEACells kernel |
| `--max-kernel-cells` | `20000` | Subsample large groups before kernel construction |
| `--seed` | `42` | Random seed |

---

## Reproduce Sciplex A549 metacells

```bash
python scripts/metacell/build_perturbation_metacells.py \
    --input     /lustre/groups/ml01/workspace/karthik.viswanathan/pancellflow/sciplex3_prophet_filtered.h5ad \
    --output    outputs/sciplex_a549_metacell.h5ad \
    --cell-line A549 \
    --obsm      X_state \
    --drug-map  scripts/metacell/sci_to_tahoe_drugs.tsv
```
