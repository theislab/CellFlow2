# CellFlow Reproducibility

This folder contains configuration files and scripts for reproducing CellFlow experiments with a clean, hierarchical structure using Hydra.

## Directory Structure

```
reproducibility/
├── configs/
│   ├── config.yaml                      # Main config with defaults
│   ├── conditioning/                    # Conditioning methods
│   │   ├── adaln_zero.yaml             # DiT (Diffusion Transformer)
│   │   ├── concatenation.yaml          # Simple concatenation
│   │   ├── film.yaml                   # FiLM conditioning
│   │   └── resnet.yaml                 # ResNet-style conditioning
│   ├── condition_encoder/              # Condition encoder configs
│   │   ├── default.yaml                # Standard encoder with attention pooling
│   │   └── simple.yaml                 # Minimal encoder with mean pooling
│   ├── architecture/                   # Model size configurations
│   │   ├── small.yaml                  # Small model (quick experiments)
│   │   ├── medium.yaml                 # Medium model (default)
│   │   └── large.yaml                  # Large model (best performance)
│   ├── solver/                         # Generative solver configs
│   │   ├── otfm.yaml                   # OT Flow Matching
│   │   └── eqm.yaml                    # Equilibrium Matching
│   └── training/                       # Training configurations
│       ├── quick.yaml                  # Fast training for debugging
│       ├── standard.yaml               # Standard training
│       └── long.yaml                   # Long training run
├── scripts/
│   └── train.py                        # Hydra-based training script
└── README.md                           # This file
```

## Quick Start

### Basic Training

```bash
# Navigate to CellFlow2 root directory
cd /home/icb/alejandro.tejada/CellFlow2

# Train with default configuration (DiT + medium architecture + standard training)
python reproducibility/scripts/train.py \
    data.adata_path=/path/to/your/data.h5ad
```

### Switch Conditioning Methods

```bash
# Train with concatenation conditioning
python reproducibility/scripts/train.py \
    conditioning=concatenation \
    data.adata_path=/path/to/your/data.h5ad

# Train with FiLM conditioning
python reproducibility/scripts/train.py \
    conditioning=film \
    data.adata_path=/path/to/your/data.h5ad

# Train with ResNet conditioning
python reproducibility/scripts/train.py \
    conditioning=resnet \
    data.adata_path=/path/to/your/data.h5ad
```

### Switch Generative Solvers

```bash
# OT Flow Matching (default)
python reproducibility/scripts/train.py \
    solver=otfm \
    data.adata_path=/path/to/your/data.h5ad

# Equilibrium Matching
python reproducibility/scripts/train.py \
    solver=eqm \
    data.adata_path=/path/to/your/data.h5ad
```

### Model Size Variants

```bash
# Small model for quick testing
python reproducibility/scripts/train.py \
    architecture=small \
    training=quick \
    data.adata_path=/path/to/your/data.h5ad

# Large model for best performance
python reproducibility/scripts/train.py \
    architecture=large \
    training=long \
    data.adata_path=/path/to/your/data.h5ad
```

### Combine Multiple Configs

```bash
# DiT conditioning + large architecture + long training
python reproducibility/scripts/train.py \
    conditioning=adaln_zero \
    architecture=large \
    training=long \
    data.adata_path=/path/to/your/data.h5ad

# Concatenation + simple encoder + quick training (for debugging)
python reproducibility/scripts/train.py \
    conditioning=concatenation \
    condition_encoder=simple \
    training=quick \
    data.adata_path=/path/to/your/data.h5ad
```

## Configuration Parameters

### Override Individual Parameters

```bash
# Change learning rate
python reproducibility/scripts/train.py \
    training.optimizer.learning_rate=0.0001 \
    data.adata_path=/path/to/your/data.h5ad

# Change batch size and iterations
python reproducibility/scripts/train.py \
    training.batch_size=256 \
    training.num_iterations=5000 \
    data.adata_path=/path/to/your/data.h5ad

# Change condition embedding dimension
python reproducibility/scripts/train.py \
    condition_encoder.embedding_dim=256 \
    data.adata_path=/path/to/your/data.h5ad

# Change multiple parameters
python reproducibility/scripts/train.py \
    architecture.cell_encoder.dims=[1024,512,256] \
    architecture.decoder.dims=[512,256,128] \
    training.optimizer.learning_rate=0.0001 \
    data.adata_path=/path/to/your/data.h5ad
```

## Hyperparameter Sweeps

Hydra makes it easy to run multiple experiments:

```bash
# Sweep over conditioning methods
python reproducibility/scripts/train.py -m \
    conditioning=adaln_zero,concatenation,film,resnet \
    data.adata_path=/path/to/your/data.h5ad

# Sweep over learning rates
python reproducibility/scripts/train.py -m \
    training.optimizer.learning_rate=0.00001,0.00005,0.0001 \
    data.adata_path=/path/to/your/data.h5ad

# Multi-dimensional sweep
python reproducibility/scripts/train.py -m \
    conditioning=adaln_zero,concatenation \
    architecture=small,medium,large \
    training.optimizer.learning_rate=0.00001,0.0001 \
    data.adata_path=/path/to/your/data.h5ad
```

## Data Configuration

The training script expects certain data fields. You can configure these:

```yaml
# In your config override or custom config file
data:
  adata_path: /path/to/your/data.h5ad
  sample_rep: "X_pca"
  control_key: "control"
  src_dist_keys: ["cell_line"]
  tgt_dist_keys: ["drug", "dosage"]
  perturbation_covariate_reps:
    drug: "drug_embeddings"
    cell_line: "cell_line_embeddings"
  split_by: ["cell_line"]
  split_ratios: [0.7, 0.15, 0.15]
```

## Architecture Details

### Conditioning Methods

1. **adaln_zero** (Default): DiT-style adaptive layer normalization
   - Best for: Complex perturbation responses
   - Parameters: `num_heads`, `qkv_dim`

2. **concatenation**: Simple concatenation of embeddings
   - Best for: Simple baselines
   - Parameters: `layer_norm_before`, `linear_projection_before`

3. **film**: Feature-wise Linear Modulation
   - Best for: Modulating cell features with conditions
   - Parameters: Same as concatenation

4. **resnet**: Residual connections
   - Best for: Preserving cell information
   - Parameters: Same as concatenation

### Model Sizes

| Size   | Cell Encoder | Time Encoder | Time Freqs | Use Case |
|--------|-------------|--------------|------------|----------|
| Small  | [256, 128]  | [64]         | 256        | Quick experiments, debugging |
| Medium | [512, 256]  | [128]        | 512        | Standard training, balanced |
| Large  | [1024, 512] | [256, 256]   | 1024       | Best performance, slow |

### Training Configs

| Config   | Iterations | Batch Size | LR      | Multi-steps | Use Case |
|----------|-----------|------------|---------|-------------|----------|
| Quick    | 1,000     | 256        | 0.0001  | 10          | Debugging |
| Standard | 10,000    | 512        | 0.00005 | 20          | Normal training |
| Long     | 50,000    | 512        | 0.00005 | 20          | Final models |

## Output Directory

By default, Hydra creates timestamped output directories:

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── .hydra/
        │   ├── config.yaml          # Resolved config
        │   ├── hydra.yaml           # Hydra config
        │   └── overrides.yaml       # CLI overrides
        └── train.log                # Training logs
```

For sweeps:

```
multirun/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── 0/                       # First sweep run
        ├── 1/                       # Second sweep run
        └── ...
```

## Tips

1. **Start small**: Use `training=quick architecture=small` for initial experiments
2. **Check shapes**: Ensure `qkv_dim` matches `cell_encoder.dims[-1]` for DiT
3. **Monitor memory**: Large models with big batch sizes may OOM
4. **Use sweeps**: Hydra makes it easy to explore hyperparameters systematically
5. **Save configs**: Hydra automatically saves all configs in `.hydra/` folder

## Troubleshooting

### Shape Mismatch Error

If you see: `mul got incompatible shapes for broadcasting: (1, 1, 256), (1, 1, 512)`

**Solution**: Match the `qkv_dim` in conditioning config to the last dim of `cell_encoder`:

```bash
python reproducibility/scripts/train.py \
    architecture.cell_encoder.dims=[512,256] \
    architecture.conditioning.qkv_dim=256  # Must match 256
```

### Out of Memory

**Solution**: Reduce model size or batch size:

```bash
python reproducibility/scripts/train.py \
    architecture=small \
    training.batch_size=128
```

### Data Loading Issues

Make sure your data has the required structure:
- `adata.obs['control']`: Boolean flag for control samples
- `adata.obsm['X_pca']`: PCA representation (or your sample_rep)
- `adata.uns['drug_embeddings']`: Drug embeddings dictionary
- `adata.uns['cell_line_embeddings']`: Cell line embeddings dictionary

## Example Workflows

### Quick Debug

```bash
python reproducibility/scripts/train.py \
    training=quick \
    architecture=small \
    training.num_iterations=100 \
    data.adata_path=/path/to/data.h5ad
```

### Standard Experiment

```bash
python reproducibility/scripts/train.py \
    conditioning=adaln_zero \
    architecture=medium \
    training=standard \
    data.adata_path=/path/to/data.h5ad
```

### Production Model

```bash
python reproducibility/scripts/train.py \
    conditioning=adaln_zero \
    architecture=large \
    training=long \
    training.num_iterations=100000 \
    data.adata_path=/path/to/data.h5ad
```

### Comparison Study

```bash
python reproducibility/scripts/train.py -m \
    conditioning=adaln_zero,concatenation,film,resnet \
    architecture=medium \
    seed=42,43,44 \
    data.adata_path=/path/to/data.h5ad
```

This will train 12 models (4 conditioning methods × 3 seeds) automatically!

