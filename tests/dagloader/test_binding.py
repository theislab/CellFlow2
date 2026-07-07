"""Case: binding — required conditioning, weight-controlled sub-sampling, and hierarchical schemes.

Covers the conditioning contract of :class:`~scaleflow.dagloader.Bind`: a non-empty ``common`` MUST
match (else raise; no silent fallback), ``common=()`` is explicit unconditional sampling, and when the
child partitions on columns beyond ``common`` the extra column is drawn ∝ the child's weights. Also
shows a hierarchical draw ``a ~ P(a)`` then ``c ~ P(c|a)`` expressed by setting leaf weight = P(a)·P(c|a).
"""

from __future__ import annotations

from collections import Counter

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scaleflow.dagloader import Bind, DAGClassLoader, Node, SamplerConfig, Scheme, uniform

pytest.importorskip("annbatch")


def _table(combos, colnames, n_per=40, seed=0):
    """AnnData whose obs has ``colnames``, one contiguous block of ``n_per`` rows per combo (F=4)."""
    rng = np.random.default_rng(seed)
    rows = [dict(zip(colnames, c, strict=True)) for c in combos for _ in range(n_per)]
    obs = pd.DataFrame(rows).astype("category")
    return ad.AnnData(X=rng.normal(size=(len(rows), 4)).astype(np.float32), obs=obs)


def _assert_bind_column(loader, root_idx, child_idx):
    """Every batch: the root leaf's bind column value equals the child leaf's bind column value."""
    child = next(iter(loader._bindmap))
    rl, cl = loader._st[loader.s.root]["leaves"], loader._st[child]["leaves"]
    loader._start_pass()
    for j in range(loader._n_batches):
        rlf = rl[int(loader._schedules[loader.s.root][j])]
        clf = cl[int(loader._schedules[child][j])]
        assert rlf[root_idx] == clf[child_idx]


# ───────────────────────────────── hierarchical sampling via weights ─────────────────────────────────
def test_hierarchical_weights_reproduce_conditional():
    """Users want a ~ P(a) then c ~ P(c|a); we deliver it by setting leaf weight = P(a)·P(c|a)."""
    Pa = {"a0": 0.5, "a1": 0.3, "a2": 0.2}
    Pc_a = {"a0": {"c0": 0.7, "c1": 0.3}, "a1": {"c0": 0.4, "c1": 0.6}, "a2": {"c0": 0.9, "c1": 0.1}}
    combos = [(a, c) for a in Pa for c in Pc_a[a]]
    weights = {(a, c): Pa[a] * Pc_a[a][c] for (a, c) in combos}   # ← the "modify the leaf weights" step
    scheme = Scheme(sources={"data": _table(combos, ("a", "c"))},
                    nodes={"root": Node("data", ("a", "c"), "X", weights)},
                    root="root", seed=0, steps_per_pass=20_000)
    loader = DAGClassLoader(scheme, SamplerConfig(batch_size=8))
    leaves = loader._st["root"]["leaves"]
    counts = np.zeros(len(leaves))
    for _ in range(3):
        loader._start_pass()
        codes, n = np.unique(loader._schedules["root"], return_counts=True)
        counts[codes] += n
    emp = {leaves[i]: counts[i] / counts.sum() for i in range(len(leaves))}
    for (a, c), w in weights.items():                            # joint == P(a)·P(c|a)
        assert abs(emp[(a, c)] - w) < 0.02, f"{(a, c)}: {emp[(a, c)]:.3f} vs {w:.3f}"
    for a, pa in Pa.items():                                     # marginal == P(a); conditional == P(c|a)
        marg = sum(emp[(a, c)] for c in Pc_a[a])
        assert abs(marg - pa) < 0.02
        for c, pca in Pc_a[a].items():
            assert abs(emp[(a, c)] / marg - pca) < 0.03


# ───────────────────────────────── bind orderings ─────────────────────────────────
def test_bind_on_second_col_a_then_b():
    """Sample (a, b) on the root, bind the child on b (the second sampled column)."""
    combos = [(a, b) for a in ["a0", "a1"] for b in ["b0", "b1", "b2"]]
    scheme = Scheme(
        sources={"A": _table(combos, ("a", "b")), "B": _table([(b,) for b in ["b0", "b1", "b2"]], ("b",))},
        nodes={"root": Node("A", ("a", "b"), "X", uniform(combos)),
               "childB": Node("B", ("b",), "X", uniform([(b,) for b in ["b0", "b1", "b2"]]))},
        root="root", seed=0, binds=(Bind("root", "childB", common=("b",)),), steps_per_pass=300,
    )
    _assert_bind_column(DAGClassLoader(scheme, SamplerConfig(batch_size=8)), root_idx=1, child_idx=0)


def test_bind_on_first_col_b_then_a():
    """Reverse the order — sample (b, a) on the root, bind the child on a."""
    combos = [(b, a) for b in ["b0", "b1", "b2"] for a in ["a0", "a1"]]
    scheme = Scheme(
        sources={"A": _table(combos, ("b", "a")), "B": _table([(a,) for a in ["a0", "a1"]], ("a",))},
        nodes={"root": Node("A", ("b", "a"), "X", uniform(combos)),
               "childB": Node("B", ("a",), "X", uniform([(a,) for a in ["a0", "a1"]]))},
        root="root", seed=0, binds=(Bind("root", "childB", common=("a",)),), steps_per_pass=300,
    )
    _assert_bind_column(DAGClassLoader(scheme, SamplerConfig(batch_size=8)), root_idx=1, child_idx=0)


# ───────────────────────────────── bind then sub-sample an extra child column ─────────────────────────────────
def test_bind_then_subsample_child_column():
    """A(a,b,c): sample (b, a); bind B on a; B(a,x,y) then sub-samples x WITHIN the bound a.

    x|a is a real conditional draw over the child's (a,x) leaves, controlled by the child weights
    (uniform within a0, 3:1 within a1) — proving the sub-sample honors weights, not a hardcoded uniform.
    c (in A) and y (in B) are unused metadata.
    """
    A = _table([(a, b, c) for a in ["a0", "a1"] for b in ["b0", "b1"] for c in ["c0", "c1"]], ("a", "b", "c"))
    Px_a = {"a0": {"x0": 0.5, "x1": 0.5}, "a1": {"x0": 0.75, "x1": 0.25}}
    B = _table([(a, x, "y0") for a in ["a0", "a1"] for x in ["x0", "x1"]], ("a", "x", "y"))
    scheme = Scheme(
        sources={"A": A, "B": B},
        nodes={"root": Node("A", ("b", "a"), "X", uniform([(b, a) for b in ["b0", "b1"] for a in ["a0", "a1"]])),
               "childB": Node("B", ("a", "x"), "X", {(a, x): Px_a[a][x] for a in ["a0", "a1"] for x in ["x0", "x1"]})},
        root="root", seed=0, binds=(Bind("root", "childB", common=("a",)),), steps_per_pass=15_000,
    )
    loader = DAGClassLoader(scheme, SamplerConfig(batch_size=8))
    rl, cl = loader._st["root"]["leaves"], loader._st["childB"]["leaves"]
    cnt = Counter()
    for _ in range(2):
        loader._start_pass()
        for j in range(loader._n_batches):
            ra = rl[int(loader._schedules["root"][j])][1]        # root (b,a): a at index 1
            ax = cl[int(loader._schedules["childB"][j])]          # child (a, x)
            assert ax[0] == ra                                    # a-binding holds every batch
            cnt[ax] += 1
    for a in ["a0", "a1"]:                                        # P(x|a) matches child weights
        tot = sum(cnt[(a, x)] for x in ["x0", "x1"])
        for x in ["x0", "x1"]:
            assert abs(cnt[(a, x)] / tot - Px_a[a][x]) < 0.02, f"P({x}|{a})={cnt[(a, x)] / tot:.3f} vs {Px_a[a][x]}"


# ───────────────────────────────── required conditioning ─────────────────────────────────
def _ab_scheme(common):
    A = _table([("a0",), ("a1",), ("a2",)], ("a",))
    B = _table([("a0",), ("a1",)], ("a",))  # a2 has no child leaf
    return Scheme(
        sources={"A": A, "B": B},
        nodes={"root": Node("A", ("a",), "X", uniform([("a0",), ("a1",), ("a2",)])),
               "childB": Node("B", ("a",), "X", uniform([("a0",), ("a1",)]))},
        root="root", seed=0, binds=(Bind("root", "childB", common=common),), steps_per_pass=200,
    )


def test_unmatched_context_raises():
    """A root value with no matching child leaf raises — no silent unconditional fallback."""
    with pytest.raises(ValueError, match=r"no positive-weight child leaf matches common"):
        next(DAGClassLoader(_ab_scheme(common=("a",)), SamplerConfig(batch_size=8)))


def test_empty_common_is_unconditional():
    """Bind(..., common=()) opts into unconditional child sampling even when a value is unmatched."""
    b = next(DAGClassLoader(_ab_scheme(common=()), SamplerConfig(batch_size=8)))
    assert b["target"].shape == (8, 4) and b["source"].shape == (8, 4)
