"""Case: ``ScheduledClassSampler`` — the one new piece under the loader.

Directly exercises schedule adherence, chunk coherence, the ``schedule=None`` ≡ ``ClassSampler``
equivalence, and validation — without the DAG loader on top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from annbatch import Loader

from scaleflow.dagloader import ScheduledClassSampler

pytest.importorskip("annbatch")


def _contiguous_classes(n_classes: int = 3, per: int = 50):
    """codes = n_classes contiguous blocks of `per` rows; a 1-feature dataset over the same rows."""
    codes = np.repeat(np.arange(n_classes), per)
    classes = pd.Categorical([str(c) for c in codes], categories=[str(i) for i in range(n_classes)])
    data = np.arange(n_classes * per, dtype=np.float32).reshape(-1, 1)
    return classes, data, per


def _class_of_row(row: int, per: int) -> int:
    return int(row) // per


def test_no_schedule_matches_classsampler():
    """schedule=None ⇒ behaves as ClassSampler: draws ∝ weights, excludes zero-weight classes."""
    classes, data, per = _contiguous_classes()
    smp = ScheduledClassSampler(chunk_size=1, preload_nchunks=10, batch_size=10, classes=classes,
                                num_samples=200, class_weights=np.array([1.0, 0.0, 1.0]),
                                drop_last=True, rng=np.random.default_rng(0))
    assert smp._schedule is None
    loader = Loader(batch_sampler=smp, return_index=True, to_torch=False,
                    preload_to_gpu=False).add_datasets([data])
    seen = {_class_of_row(b["index"][0], per) for b in loader}
    assert seen == {0, 2}  # zero-weight class 1 never sampled


def test_follows_explicit_schedule():
    """A supplied schedule fixes each batch's category exactly (class-coherent, in order)."""
    classes, data, per = _contiguous_classes()
    schedule = np.array([2, 0, 1, 1, 2, 0], dtype=np.int64)
    smp = ScheduledClassSampler(chunk_size=1, preload_nchunks=10, batch_size=10, classes=classes,
                                num_samples=len(schedule) * 10, class_weights=np.ones(3),
                                drop_last=True, rng=np.random.default_rng(0))
    smp.set_schedule(schedule)
    loader = Loader(batch_sampler=smp, return_index=True, to_torch=False,
                    preload_to_gpu=False).add_datasets([data])
    for j, batch in enumerate(loader):
        rows = np.asarray(batch["index"])
        assert len(rows) == 10
        assert {_class_of_row(r, per) for r in rows} == {int(schedule[j])}  # whole batch is the scheduled class


def test_chunk_size_reads_within_class():
    """chunk_size>1 keeps batches class-coherent (reads stay inside the scheduled class's run)."""
    classes, data, per = _contiguous_classes()
    schedule = np.array([0, 2, 1, 0], dtype=np.int64)
    smp = ScheduledClassSampler(chunk_size=5, preload_nchunks=2, batch_size=10, classes=classes,
                                num_samples=len(schedule) * 10, class_weights=np.ones(3),
                                drop_last=True, rng=np.random.default_rng(0))
    smp.set_schedule(schedule)
    loader = Loader(batch_sampler=smp, return_index=True, to_torch=False,
                    preload_to_gpu=False).add_datasets([data])
    for j, batch in enumerate(loader):
        rows = np.asarray(batch["index"])
        assert {_class_of_row(r, per) for r in rows} == {int(schedule[j])}


def test_length_mismatch_raises():
    classes, data, _ = _contiguous_classes()
    smp = ScheduledClassSampler(chunk_size=1, preload_nchunks=10, batch_size=10, classes=classes,
                                num_samples=60, class_weights=np.ones(3), drop_last=True,
                                rng=np.random.default_rng(0))
    smp.set_schedule(np.array([0, 1, 2], dtype=np.int64))  # 3 != n_groups (6)
    loader = Loader(batch_sampler=smp, return_index=True, to_torch=False,
                    preload_to_gpu=False).add_datasets([data])
    with pytest.raises(ValueError, match=r"schedule length 3 != n_groups 6"):
        next(iter(loader))


def test_zero_weight_code_raises():
    classes, data, _ = _contiguous_classes()
    smp = ScheduledClassSampler(chunk_size=1, preload_nchunks=10, batch_size=10, classes=classes,
                                num_samples=20, class_weights=np.array([1.0, 0.0, 1.0]),
                                drop_last=True, rng=np.random.default_rng(0))
    smp.set_schedule(np.array([0, 1], dtype=np.int64))  # class 1 has zero weight
    loader = Loader(batch_sampler=smp, return_index=True, to_torch=False,
                    preload_to_gpu=False).add_datasets([data])
    with pytest.raises(ValueError, match=r"non-sampleable \(zero-weight\) category code"):
        next(iter(loader))
