"""Fixtures for CellFlowTrainer unit tests.

These are lightweight synthetic samplers (not the annbatch data layer): the trainer is tested
in isolation, so the fixtures only need to honor the batch contract the trainer consumes.
- training sampler: ``sample()`` -> ``{"src_cell_data", "tgt_cell_data", "condition"}``
- validation samplers: ``sample(mode=...)`` -> ``{"source", "condition", "target"}`` keyed by
  a condition key (matching the ValidationSampler format the trainer's _validation_step expects).
"""

import numpy as np
import pytest

# Matches the `cond` used to initialize the solver in test_trainer.py: {"pert1": (1, 2, 3)}.
_COND = {"pert1": np.ones((1, 2, 3), dtype=np.float32)}
_DATA_DIM = 5
_BATCH = 10


class _StubTrainSampler:
    """Yields a fixed-shape (source, target, condition) batch; condition broadcast over cells."""

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self._initialized = True

    def init_sampler(self):
        pass

    @property
    def initialized(self) -> bool:
        return True

    def sample(self) -> dict:
        return {
            "src_cell_data": self._rng.normal(size=(_BATCH, _DATA_DIM)).astype(np.float32),
            "tgt_cell_data": self._rng.normal(size=(_BATCH, _DATA_DIM)).astype(np.float32),
            "condition": {k: v.copy() for k, v in _COND.items()},
        }


class _StubValSampler:
    """Per-condition validation batch in the ValidationSampler format (one condition key)."""

    def __init__(self, seed: int = 1):
        self._rng = np.random.default_rng(seed)
        self._initialized = True

    def init_sampler(self):
        pass

    @property
    def initialized(self) -> bool:
        return True

    def sample(self, mode: str = "on_log_iteration") -> dict:
        key = "my_naming_of_pert"
        return {
            "source": {key: self._rng.normal(size=(_BATCH, _DATA_DIM)).astype(np.float32)},
            "condition": {key: {k: v.copy() for k, v in _COND.items()}},
            "target": {key: self._rng.normal(size=(_BATCH, _DATA_DIM)).astype(np.float32)},
        }


@pytest.fixture
def dataloader():
    return _StubTrainSampler()


@pytest.fixture
def valid_loader():
    return {"val": _StubValSampler()}
