"""Unit tests for the DAGClassLoader data path (no training loop, no jax).

Covers the two new pieces directly: :class:`ScheduledClassSampler` (schedule adherence, chunk
coherence, validation) and :class:`DAGClassLoader` (batch contract, per-node param validation,
schedule↔row alignment, parent→child conditioning, reproducibility, obsm / on-disk sources).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from annbatch import Loader

from scaleflow.dag_class_loader import (
    Bind,
    DAGClassLoader,
    Node,
    Scheme,
    ScheduledClassSampler,
    perturbation_scheme,
)

from _toydata import DRUG_EMB, EMB, G, PCA_D, shuffled_adata, toy_adata, write_collection

pytest.importorskip("annbatch")


def _cond_fn(cols):
    return lambda leaf: DRUG_EMB[leaf[cols.index("drug")]]


# ───────────────────────────────────────────── ScheduledClassSampler (direct) ─────────────────────────────────────────────
def _contiguous_classes(n_classes: int = 3, per: int = 50):
    """codes = n_classes contiguous blocks of `per` rows; a 1-feature dataset over the same rows."""
    codes = np.repeat(np.arange(n_classes), per)
    classes = pd.Categorical([str(c) for c in codes], categories=[str(i) for i in range(n_classes)])
    data = np.arange(n_classes * per, dtype=np.float32).reshape(-1, 1)
    return classes, data, per


def _class_of_row(row: int, per: int) -> int:
    return int(row) // per


def test_scheduled_sampler_no_schedule_matches_classsampler():
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


def test_scheduled_sampler_follows_explicit_schedule():
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


def test_scheduled_sampler_chunk_size_reads_within_class():
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


def test_scheduled_sampler_length_mismatch_raises():
    classes, data, _ = _contiguous_classes()
    smp = ScheduledClassSampler(chunk_size=1, preload_nchunks=10, batch_size=10, classes=classes,
                                num_samples=60, class_weights=np.ones(3), drop_last=True,
                                rng=np.random.default_rng(0))
    smp.set_schedule(np.array([0, 1, 2], dtype=np.int64))  # 3 != n_groups (6)
    loader = Loader(batch_sampler=smp, return_index=True, to_torch=False,
                    preload_to_gpu=False).add_datasets([data])
    with pytest.raises(ValueError, match=r"schedule length 3 != n_groups 6"):
        next(iter(loader))


def test_scheduled_sampler_zero_weight_code_raises():
    classes, data, _ = _contiguous_classes()
    smp = ScheduledClassSampler(chunk_size=1, preload_nchunks=10, batch_size=10, classes=classes,
                                num_samples=20, class_weights=np.array([1.0, 0.0, 1.0]),
                                drop_last=True, rng=np.random.default_rng(0))
    smp.set_schedule(np.array([0, 1], dtype=np.int64))  # class 1 has zero weight
    loader = Loader(batch_sampler=smp, return_index=True, to_torch=False,
                    preload_to_gpu=False).add_datasets([data])
    with pytest.raises(ValueError, match=r"non-sampleable \(zero-weight\) category code"):
        next(iter(loader))


# ───────────────────────────────────────────── schema validation ─────────────────────────────────────────────
def test_node_structural_validation():
    with pytest.raises(ValueError, match="cols and Node.keys must be non-empty"):
        Node("s", ())
    with pytest.raises(ValueError, match="arity"):
        Node("s", ("a", "b"), weights={("x",): 1.0})       # 1-tuple key vs 2 cols
    with pytest.raises(ValueError, match="non-negative"):
        Node("s", ("a",), weights={("x",): -1.0})
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        Node("s", ("a",), chunk_size=0)


def test_scheme_structural_validation():
    adata = toy_adata()
    src = {"data": adata}
    n = {"pert": Node("data", ("cell_line", "drug")), "ctrl": Node("data", ("cell_line", "drug"))}
    with pytest.raises(ValueError, match="root 'missing' not in nodes"):
        Scheme(src, n, root="missing", n_rows_per_leaf=16, seed=0)
    with pytest.raises(ValueError, match="not bound to the tree"):
        Scheme(src, n, root="pert", n_rows_per_leaf=16, seed=0)  # ctrl unbound
    with pytest.raises(ValueError, match=r"bind.common .* must be ⊆ shared cols"):
        Scheme(src, n, root="pert", n_rows_per_leaf=16, seed=0,
               binds=(Bind("pert", "ctrl", common=("nonshared",)),))
    with pytest.raises(ValueError, match="unknown source"):
        Scheme(src, {"pert": Node("nope", ("cell_line",))}, root="pert", n_rows_per_leaf=16, seed=0)


# ───────────────────────────────────────────── DAGClassLoader contract ─────────────────────────────────────────────
def _scheme(adata, **kw):
    return perturbation_scheme(adata, context=["cell_line"], perturbation=["drug"],
                               control_values={"drug": "control"}, **kw)


def test_yields_matched_batches():
    adata = toy_adata()
    scheme = _scheme(adata, n_rows_per_leaf=16)
    cols = scheme.nodes["pert"].cols
    loader = DAGClassLoader(scheme, condition_fn=_cond_fn(cols))
    b = next(loader)
    assert b["target"].shape == (16, G)
    assert b["source"].shape == (16, G)
    assert b["condition"].shape == (16, EMB)
    assert not np.allclose(b["condition"][0], DRUG_EMB["control"])  # root only samples perturbed leaves


def test_target_rows_match_schedule_leaf():
    """White-box invariant: every streamed root row belongs to the schedule's claimed leaf (any chunk_size)."""
    adata = toy_adata()
    for chunk in (1, 8):  # per-row and chunked reads
        scheme = _scheme(adata, n_rows_per_leaf=64, seed=0, chunk_size=chunk, steps_per_pass=8)
        loader = DAGClassLoader(scheme)
        cols = scheme.nodes["pert"].cols
        st = loader._st["pert"]
        for _ in range(2):  # cross a pass boundary (steps_per_pass=8) to exercise restart
            loader._start_pass()
            for jj in range(loader._n_batches):
                rows = np.asarray(next(loader._iters["pert"])["index"])
                claimed = st["leaves"][int(loader._schedules["pert"][jj])]
                got = {tuple(adata.obs.iloc[int(r)][list(cols)]) for r in rows}
                assert got == {claimed}, f"chunk={chunk} batch {jj}: {got} != {claimed}"


def test_source_matches_target_context():
    """Bound child (control) source is conditioned on the parent's cell_line, not random.

    Toy means: A ≈ +1, B ≈ −1; the single-coordinate drug spike barely moves the mean, so a batch's
    mean sign tracks its cell_line for both target (perturbed) and source (control).
    """
    adata = toy_adata()
    scheme = _scheme(adata, n_rows_per_leaf=64, seed=0)
    loader = DAGClassLoader(scheme, condition_fn=_cond_fn(scheme.nodes["pert"].cols))
    for _ in range(20):
        b = next(loader)
        assert np.sign(b["target"].mean()) == np.sign(b["source"].mean())


def test_chunk_size_must_divide_batch():
    scheme = _scheme(toy_adata(), n_rows_per_leaf=64, chunk_size=3)
    with pytest.raises(ValueError, match=r"node 'pert': chunk_size \(3\) must divide n_rows_per_leaf \(64\)"):
        DAGClassLoader(scheme)


def test_chunk_size_gt1_sorted_ok():
    scheme = _scheme(toy_adata(), n_rows_per_leaf=64, chunk_size=8)  # toy blocks are contiguous
    loader = DAGClassLoader(scheme, condition_fn=_cond_fn(scheme.nodes["pert"].cols))
    root = loader._samplers["pert"]
    assert root._chunk_size == 8 and root._preload_nchunks == 8  # = batch // chunk
    assert next(loader)["target"].shape == (64, G)


def test_chunk_size_gt1_unsorted_raises():
    scheme = _scheme(shuffled_adata(), n_rows_per_leaf=64, chunk_size=8)
    with pytest.raises(ValueError, match=r"node 'pert'.*contiguous run must be at least chunk_size"):
        DAGClassLoader(scheme)


def test_reproducible_from_seed():
    adata = toy_adata()

    def stream(seed):
        s = DAGClassLoader(_scheme(adata, n_rows_per_leaf=32, seed=seed),
                           condition_fn=_cond_fn(("cell_line", "drug")))
        return [next(s) for _ in range(6)]

    a, b, c = stream(0), stream(0), stream(1)
    for x, y in zip(a, b):  # same seed → bit-identical
        assert np.array_equal(x["target"], y["target"]) and np.array_equal(x["source"], y["source"])
        assert np.array_equal(x["condition"], y["condition"])
    assert not all(np.array_equal(x["target"], z["target"]) for x, z in zip(a, c))  # seed 1 diverges


def test_per_node_rngs_are_independent():
    """Root and child get distinct spawned streams (not the same seed → not correlated)."""
    loader = DAGClassLoader(_scheme(toy_adata(), n_rows_per_leaf=16, seed=0))
    r0 = loader._rngs["pert"].integers(0, 1_000_000, size=5)
    r1 = loader._rngs["ctrl"].integers(0, 1_000_000, size=5)
    assert not np.array_equal(r0, r1)


def test_obsm_streaming_shape_in_memory():
    scheme = _scheme(toy_adata(), key="obsm/pca", n_rows_per_leaf=32)
    loader = DAGClassLoader(scheme, condition_fn=_cond_fn(scheme.nodes["pert"].cols))
    b = next(loader)
    assert b["target"].shape == (32, PCA_D) and b["source"].shape == (32, PCA_D)  # streamed obsm, not X (=8)


def test_datasetcollection_source_shape(tmp_path):
    coll = write_collection(toy_adata(), tmp_path)
    scheme = _scheme(coll, n_rows_per_leaf=32, seed=0)
    loader = DAGClassLoader(scheme, condition_fn=_cond_fn(scheme.nodes["pert"].cols))
    b = next(loader)  # out-of-core read
    assert b["target"].shape == (32, G) and b["source"].shape == (32, G)
    assert b["condition"].shape == (32, EMB)
