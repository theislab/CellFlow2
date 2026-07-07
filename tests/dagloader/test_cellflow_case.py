"""Case: cellflow — control → perturbed, matched within a context, one perturbation per batch.

Built with ``perturbation_scheme`` (the cellflow-shaped factory) + a ``SamplerConfig``. Verifies the
matched-batch contract, that the streamed target rows really belong to the scheduled leaf, that the
bound control source shares the target's context (cell line), obsm as the streamed rep, an on-disk
collection, and the ``chunk_size>1`` throughput path.
"""

from __future__ import annotations

import numpy as np
import pytest

from scaleflow.dagloader import DAGClassLoader, SamplerConfig, perturbation_scheme

from _toydata import DRUG_D, DRUG_EMB, G, PCA_D, drug_cond_fn, toy_adata, write_collection

pytest.importorskip("annbatch")


def _scheme(source, **kw):
    return perturbation_scheme(source, context=["cell_line"], perturbation=["drug"],
                               control_values={"drug": "control"}, **kw)


def _cond(scheme):
    return drug_cond_fn(scheme.nodes["pert"].cols)


def test_yields_matched_batches():
    scheme = _scheme(toy_adata())
    loader = DAGClassLoader(scheme, SamplerConfig(batch_size=16), condition_fn=_cond(scheme))
    b = next(loader)
    assert b["target"].shape == (16, G)
    assert b["source"].shape == (16, G)
    assert b["condition"].shape == (16, DRUG_D)
    assert not np.allclose(b["condition"][0], DRUG_EMB["control"])  # root only samples perturbed leaves


def test_target_rows_match_schedule_leaf():
    """White-box invariant: every streamed root row belongs to the schedule's claimed leaf (any chunk)."""
    adata = toy_adata()
    for chunk in (1, 8):  # per-row and chunked reads
        scheme = _scheme(adata, seed=0, steps_per_pass=8)
        loader = DAGClassLoader(scheme, SamplerConfig(batch_size=64, chunk_size=chunk))
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
    scheme = _scheme(toy_adata(), seed=0)
    loader = DAGClassLoader(scheme, SamplerConfig(batch_size=64), condition_fn=_cond(scheme))
    for _ in range(20):
        b = next(loader)
        assert np.sign(b["target"].mean()) == np.sign(b["source"].mean())


def test_obsm_as_streamed_rep():
    scheme = _scheme(toy_adata(), key="obsm/pca")
    loader = DAGClassLoader(scheme, SamplerConfig(batch_size=32), condition_fn=_cond(scheme))
    b = next(loader)
    assert b["target"].shape == (32, PCA_D) and b["source"].shape == (32, PCA_D)  # streamed obsm, not X (=8)


def test_chunk_size_gt1_on_sorted_source():
    """chunk_size>1 (contiguous chunked reads) works on the condition-sorted toy layout."""
    scheme = _scheme(toy_adata())  # toy blocks are contiguous runs
    loader = DAGClassLoader(scheme, SamplerConfig(batch_size=64, chunk_size=8), condition_fn=_cond(scheme))
    root = loader._samplers["pert"]
    assert root._chunk_size == 8 and root._preload_nchunks == 8  # = batch // chunk
    assert next(loader)["target"].shape == (64, G)


def test_datasetcollection_source(tmp_path):
    coll = write_collection(toy_adata(), tmp_path)
    scheme = _scheme(coll, seed=0)
    loader = DAGClassLoader(scheme, SamplerConfig(batch_size=32), condition_fn=_cond(scheme))
    b = next(loader)  # out-of-core read
    assert b["target"].shape == (32, G) and b["source"].shape == (32, G)
    assert b["condition"].shape == (32, DRUG_D)
