"""Case: sampling schemes — weights ARE the selection, and the sampling distribution follows them.

Covers the weight builders (``uniform`` / ``frequency`` / ``inverse_frequency``), the "weight 0 ⇒
excluded" selection semantics, empirical adherence of the drawn schedule to the weights, per-node
independent RNG streams, reproducibility from ``seed``, and the structural validation of ``Node`` /
``Scheme`` / ``SamplerConfig``.
"""

from __future__ import annotations

import numpy as np
import pytest

from scaleflow.dagloader import (
    Bind,
    DAGClassLoader,
    Node,
    SamplerConfig,
    Scheme,
    frequency,
    inverse_frequency,
    perturbation_scheme,
    uniform,
)

from _toydata import drug_cond_fn, toy_adata

pytest.importorskip("annbatch")


# ───────────────────────────────── weight builders (unit) ─────────────────────────────────
def test_weight_builders():
    combos = [("A", "d1"), ("B", "d2")]
    assert uniform(combos) == {("A", "d1"): 1.0, ("B", "d2"): 1.0}
    counts = {("A", "d1"): 10, ("B", "d2"): 40}
    assert frequency(counts) == {("A", "d1"): 10.0, ("B", "d2"): 40.0}
    inv = inverse_frequency(counts)
    assert inv[("A", "d1")] == pytest.approx(0.1) and inv[("B", "d2")] == pytest.approx(0.025)


def _single_root(weights, *, steps_per_pass=6000, seed=0):
    """A one-node scheme (no bind) — enough to inspect the root's drawn schedule distribution."""
    scheme = Scheme(
        sources={"data": toy_adata()},
        nodes={"root": Node("data", ("cell_line", "drug"), "X", weights)},
        root="root",
        seed=seed,
        steps_per_pass=steps_per_pass,
    )
    loader = DAGClassLoader(scheme, SamplerConfig(batch_size=4))
    loader._start_pass()  # draws root schedule from weights; no cells read
    leaves = loader._st["root"]["leaves"]
    codes = loader._schedules["root"]
    freq = {leaves[c]: (codes == c).mean() for c in np.unique(codes)}
    return leaves, freq


def test_zero_weight_is_excluded():
    """A combination absent from the weights (or weight 0) is never sampled — no separate select."""
    weights = {("A", "d1"): 1.0, ("B", "d2"): 1.0}  # only two of the six leaves carry weight
    _, freq = _single_root(weights)
    assert set(freq) == {("A", "d1"), ("B", "d2")}  # the other four leaves never appear


def test_empirical_distribution_follows_weights():
    """The drawn schedule matches the normalized weights (∝ weight)."""
    weights = {("A", "d1"): 3.0, ("A", "d2"): 1.0, ("B", "d1"): 1.0, ("B", "d2"): 3.0}
    _, freq = _single_root(weights)
    total = sum(weights.values())
    for combo, w in weights.items():
        assert freq[combo] == pytest.approx(w / total, abs=0.03)


def test_all_zero_weights_raises():
    with pytest.raises(ValueError, match="all-zero"):
        DAGClassLoader(
            Scheme(
                sources={"data": toy_adata()},
                nodes={"root": Node("data", ("cell_line", "drug"), "X", weights={})},
                root="root", seed=0,
            ),
            SamplerConfig(batch_size=4),
        )


# ───────────────────────────────── RNG / reproducibility ─────────────────────────────────
def _scheme(adata, **kw):
    return perturbation_scheme(adata, context=["cell_line"], perturbation=["drug"],
                               control_values={"drug": "control"}, **kw)


def test_per_node_rngs_are_independent():
    """Root and child get distinct spawned streams (not the same seed → not correlated)."""
    loader = DAGClassLoader(_scheme(toy_adata(), seed=0), SamplerConfig(batch_size=16))
    r0 = loader._rngs["pert"].integers(0, 1_000_000, size=5)
    r1 = loader._rngs["ctrl"].integers(0, 1_000_000, size=5)
    assert not np.array_equal(r0, r1)


def test_reproducible_from_seed():
    adata = toy_adata()

    def stream(seed):
        s = DAGClassLoader(_scheme(adata, seed=seed), SamplerConfig(batch_size=32),
                           condition_fn=drug_cond_fn(("cell_line", "drug")))
        return [next(s) for _ in range(6)]

    a, b, c = stream(0), stream(0), stream(1)
    for x, y in zip(a, b):  # same seed → bit-identical
        assert np.array_equal(x["target"], y["target"]) and np.array_equal(x["source"], y["source"])
        assert np.array_equal(x["condition"], y["condition"])
    assert not all(np.array_equal(x["target"], z["target"]) for x, z in zip(a, c))  # seed 1 diverges


# ───────────────────────────────── schema / config validation ─────────────────────────────────
def test_node_structural_validation():
    with pytest.raises(ValueError, match="Node.cols must be non-empty"):
        Node("s", ())
    with pytest.raises(ValueError, match="Node.key must be a non-empty"):
        Node("s", ("a",), key="")
    with pytest.raises(ValueError, match="arity"):
        Node("s", ("a", "b"), weights={("x",): 1.0})       # 1-tuple key vs 2 cols
    with pytest.raises(ValueError, match="non-negative"):
        Node("s", ("a",), weights={("x",): -1.0})


def test_scheme_structural_validation():
    adata = toy_adata()
    src = {"data": adata}
    n = {"pert": Node("data", ("cell_line", "drug")), "ctrl": Node("data", ("cell_line", "drug"))}
    with pytest.raises(ValueError, match="root 'missing' not in nodes"):
        Scheme(src, n, root="missing", seed=0)
    with pytest.raises(ValueError, match="not bound to the tree"):
        Scheme(src, n, root="pert", seed=0)  # ctrl unbound
    with pytest.raises(ValueError, match=r"bind.common .* must be ⊆ shared cols"):
        Scheme(src, n, root="pert", seed=0, binds=(Bind("pert", "ctrl", common=("nonshared",)),))
    with pytest.raises(ValueError, match="unknown source"):
        Scheme(src, {"pert": Node("nope", ("cell_line",))}, root="pert", seed=0)


def test_sampler_config_validation():
    with pytest.raises(ValueError, match=r"chunk_size \(3\) must divide batch_size \(16\)"):
        SamplerConfig(batch_size=16, chunk_size=3)
    with pytest.raises(ValueError, match="must be a positive multiple of"):
        SamplerConfig(batch_size=16, chunk_size=4, preload_nchunks=3)  # 3 not a multiple of 16//4=4
    assert SamplerConfig(batch_size=16, chunk_size=4).resolved_preload == 4  # default = batch // chunk
