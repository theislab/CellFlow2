# Memory-Efficient Multi-Dataset Processing

## Problem

The original `prepare_and_split_multiple_datasets` function loads all datasets into memory at once, then processes them all. For large datasets (95M cells), this causes OOM errors:

- Loading 2 datasets: 262 GB
- Processing both into GroupedDistributions: +260 GB
- **Total peak memory: 520+ GB** (exceeds available 400 GB)

## Solution

Modified `train_multidataset.py` to process datasets **one at a time**:

1. **Load dataset → Process → Delete → Repeat**
2. Only keep the final split results in memory
3. Aggressive garbage collection between steps

## Memory Flow (Optimized v2 - Selective Splitting)

```
Step 1: Load dataset_1 (132 GB)
Step 2: Process to GroupedDistribution (+130 GB) = 262 GB
Step 3: Delete raw dataset_1 → 130 GB
Step 4: Split GroupedDistribution (train/val/test)
Step 5: Delete GroupedDistribution, keep ALL splits → ~100 GB
Step 6: Load dataset_2 (132 GB) = 232 GB
Step 7: Process to GroupedDistribution (+100 GB) = 332 GB
Step 8: Delete raw dataset_2 → 200 GB
Step 9: Split GroupedDistribution
Step 10: Keep ONLY test split, discard train/val → ~130 GB
Step 11: Delete GroupedDistribution → ~130 GB

Peak memory: ~332 GB (instead of 520+ GB)
Final memory: ~130 GB (train/val/test from dataset_1 + test from dataset_2)
```

**Key Optimization**: Secondary datasets only keep their test split since you're training on the first dataset only.

## Key Changes

### 1. Load datasets on-demand (not all at once)
```python
# OLD: Load all upfront
datasets = {name: load_adata(path) for name, path in paths.items()}

# NEW: Just store paths, load on-demand
dataset_paths = {"dataset_1": path1, "dataset_2": path2}
```

### 2. Process one at a time with cleanup + selective retention
```python
for i, dataset_name in enumerate(dataset_names):
    is_primary = (i == 0)

    # Load
    adata = load_from_h5ad(dataset_paths[dataset_name])

    # Process
    gd = dm.prepare_data(adata)

    # Delete raw data immediately
    del adata
    gc.collect()

    # Split
    full_splits = splitter.split(gd)

    # Delete intermediate GroupedDistribution
    del gd
    gc.collect()

    # Keep appropriate splits based on dataset role
    if is_primary:
        # First dataset: keep all splits (train/val/test)
        splits_all[dataset_name] = full_splits
    else:
        # Secondary datasets: only keep test split
        splits_all[dataset_name] = {"test": full_splits["test"]}
        del full_splits
        gc.collect()
```

### 3. Memory monitoring throughout
Added `print_memory_usage()` calls at each step to track memory consumption.

## Usage

Run the same way:
```bash
python reproducibility/scripts/train_multidataset.py data=tahoe_sciplex ...
```

But now it will process datasets sequentially with much lower peak memory.

## Future Extension

To use different datasets, modify the `dataset_paths` dictionary:
```python
dataset_paths = {
    "sciplex": "/path/to/sciplex.h5ad",
    "norman": "/path/to/norman.h5ad",
    "replogle": "/path/to/replogle.h5ad",
}
```

