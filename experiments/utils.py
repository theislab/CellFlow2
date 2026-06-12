"""Prophet-ablation condition transform and the optimizer builder."""
from __future__ import annotations

import numpy as np
import optax
from omegaconf import DictConfig


class ConditionTransform:
    """Rewrites the 'prophet' condition at sample time.

    default: drop 'prophet'.  prophet: keep as-is (caller passes None instead).
    random:  replace with random vectors — fresh each train step, fixed per
             condition at validation (keyed by cond_key).
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
                        s = abs(hash(cond_key + k + str(self._seed))) % (2 ** 31)
                        self._cache[cache_key] = np.random.default_rng(s).standard_normal(v.shape).astype(v.dtype)
                    result[k] = self._cache[cache_key]
                else:
                    result[k] = self._rng.standard_normal(v.shape).astype(v.dtype)
        return result


def build_optimizer(cfg: DictConfig):
    """Warmup-cosine Adam wrapped in MultiSteps for gradient accumulation.

    Schedule is in optimizer-update steps (iterations / accumulation).
    Returns (optimizer, schedule).
    """
    t     = cfg.training
    accum = int(t.get("grad_accumulation", 1)) or 1
    steps = max(int(t.num_iterations) // accum, 1)
    warmup = max(min(int(t.warmup_iterations) // accum, steps - 1), 1)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=float(t.get("init_lr", 0.0)),
        peak_value=float(t.peak_lr),
        warmup_steps=warmup,
        decay_steps=steps,
        end_value=float(t.end_lr),
    )
    opt = optax.adam(schedule)
    return (optax.MultiSteps(opt, accum) if accum > 1 else opt), schedule
