import numpy as np
from pathlib import Path
from scaleflow.data import GroupedDistribution
from scaleflow.data._dataloader import ReservoirSampler

ZARRS = {
    "tahoe":   "/storage/pancellflow/tahoe_updated.zarr",
    "sciplex": "/storage/pancellflow/sciplex3.zarr",
}

def sample_batch(zarr_path, batch_size=512, seed=0):
    gd = GroupedDistribution.read_zarr(Path(zarr_path))
    s = ReservoirSampler(
        gd, np.random.default_rng(seed),
        batch_size=batch_size, pool_fraction=0.7, replacement_prob=0.5,
    )
    s.init_sampler()
    return s.sample()

for name, path in ZARRS.items():
    b   = sample_batch(path)
    src = np.asarray(b["src_cell_data"])
    tgt = np.asarray(b["tgt_cell_data"])

    print(f"=== {name} ===")
    print(f"  dim         : {tgt.shape[-1]}")
    print(f"  tgt var     : {np.var(tgt):.4f}")
    print(f"  src var     : {np.var(src):.4f}")
    print(f"  tgt mean    : {np.mean(tgt):.4f}")
    print(f"  tgt row-norm: {np.linalg.norm(tgt, axis=-1).mean():.3f}   (mean |x|)")
    # how far control -> perturbed moves (the 'distance' the flow must cover)
    print(f"  src→tgt shift: {np.linalg.norm(tgt.mean(0) - src.mean(0)):.4f}")
    print(f"  condition    : {[(k, tuple(v.shape)) for k, v in b['condition'].items()]}")
    print()
