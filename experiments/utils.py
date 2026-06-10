"""
utils.py — shared logic for the unified CellFlow2 training scripts.

Only three things live here:
  • ConditionTransform  — the prophet ablation (default / prophet / random)
  • build_optimizer     — warmup-cosine Adam (+ grad accumulation)
  • run                 — split → samplers → model → train, given pre-loaded `gds`

The two entrypoints (train_zarr.py, train_h5ad.py) differ ONLY in how they build
`gds = {name: GroupedDistribution}`; everything after that is identical and lives
in `run`.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import optax
from omegaconf import DictConfig, OmegaConf

from scaleflow.data import split_datasets
from scaleflow.data._dataloader import CombinedSampler, ReservoirSampler, ValidationSampler
from scaleflow.model import ScaleFlow



# ─────────────────────────────────────────────────────────────────────────────
# Prophet ablation
# ─────────────────────────────────────────────────────────────────────────────
class ConditionTransform:
    """Transforms condition dicts at sample time (operates only on the 'prophet' key).

    mode="default" : drop the 'prophet' key entirely.
    mode="prophet" : no-op (handled by the caller, which passes transform=None).
    mode="random"  : replace 'prophet' values with random vectors.
                     Training (cond_key=None)  → fresh random each call.
                     Validation (cond_key=str) → fixed random per condition.

    For data without a 'prophet' condition key (e.g. the h5ad drug datasets) this
    is inert in every mode, so it is always safe to pass in.
    """

    def __init__(self, mode: str, seed: int = 42):
        self.mode  = mode
        self._rng  = np.random.default_rng(seed)
        self._seed = seed
        self._cache: dict = {}

    def __call__(self, cond: dict, cond_key: str | None = None) -> dict:
        if self.mode == "prophet":
            return cond
        result = {}
        for k, v in cond.items():
            if k != "prophet":
                result[k] = v
            elif self.mode == "random":
                if cond_key is not None:
                    cache_key = (cond_key, k, v.shape)
                    if cache_key not in self._cache:
                        int_seed = abs(hash(cond_key + k + str(self._seed))) % (2 ** 31)
                        self._cache[cache_key] = (
                            np.random.default_rng(int_seed).standard_normal(v.shape).astype(v.dtype)
                        )
                    result[k] = self._cache[cache_key]
                else:
                    result[k] = self._rng.standard_normal(v.shape).astype(v.dtype)
            # mode == "default": drop the key
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer(cfg: DictConfig):
    """Return (optimizer, lr_schedule).

    Warmup-cosine-decay Adam, wrapped in MultiSteps when grad_accumulation > 1.
    Schedule units are optimizer-update steps (training steps / grad_accumulation),
    so the cosine completes over the actual number of updates. The schedule is
    passed to the optimizer (unlike train_crossdatasets.py, which built a schedule
    but then handed a constant adam(1e-4) to the model).
    """
    t        = cfg.training
    accum    = int(t.get("grad_accumulation", 1)) or 1
    num_iter = int(t.num_iterations)

    opt_steps  = max(num_iter // accum, 1)
    warmup_opt = max(min(int(t.warmup_iterations) // accum, opt_steps - 1), 1)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=float(t.get("init_lr", 0.0)),
        peak_value=float(t.peak_lr),
        warmup_steps=warmup_opt,
        decay_steps=opt_steps,
        end_value=float(t.end_lr),
    )
    base = optax.adam(learning_rate=schedule)
    optimizer = optax.MultiSteps(base, accum) if accum > 1 else base
    return optimizer, schedule



