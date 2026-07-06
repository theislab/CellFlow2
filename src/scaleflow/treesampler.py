"""Declarative, index-free class-mapping sampler over annbatch (prototype).

A rooted tree of `Node`s over named sources; each node is a partition of its source's cells into
leaves (unique column-combinations) with a per-combination weight mapping. The root samples a leaf via
annbatch `ClassSampler` (weight 0 ⇒ excluded ⇒ = selection); bound children are fetched conditioned on
the parent's shared-column values. No row indices are exposed — the scheme is columns / keys / weights.

Scope of this prototype: single-level bind (root + bound children), in-memory `AnnData` sources; the
root streams through annbatch, bound children come from an in-memory cache keyed by the shared columns.
Multi-level trees, `DatasetCollection` sources, and multi-key (obsm) reps are the documented extensions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
from annbatch import DatasetCollection, Loader
from annbatch.abc import Sampler
from annbatch.samplers import ClassSampler

Container = ad.AnnData | DatasetCollection  # a cell source: in-memory AnnData or on-disk DatasetCollection

__all__ = ["Node", "Bind", "Scheme", "TreeSampler", "perturbation_scheme",
           "uniform", "frequency", "inverse_frequency"]

# Weights are just a mapping {combination -> weight}. A combination absent from the mapping (or with
# weight 0) is excluded — that IS the selection, native to annbatch ClassSampler. The "uniform /
# frequency / inverse_frequency" niceties are plain above-layer functions that build such a dict.
Weights = Mapping[tuple, float]


def uniform(combos) -> dict[tuple, float]:
    return {tuple(c): 1.0 for c in combos}


def frequency(counts: Mapping[tuple, int]) -> dict[tuple, float]:
    return {tuple(k): float(c) for k, c in counts.items()}


def inverse_frequency(counts: Mapping[tuple, int]) -> dict[tuple, float]:
    return {tuple(k): 1.0 / c for k, c in counts.items()}


def _weight_vector(weights: Weights, leaves: Sequence[tuple]) -> np.ndarray:
    """Resolve {combo: weight} to normalized per-leaf weights (→ ClassSampler.class_weights)."""
    v = np.array([float(weights.get(tuple(lf), 0.0)) for lf in leaves], dtype=float)
    s = v.sum()
    if s <= 0:
        raise ValueError("weights resolve to all-zero over these leaves — nothing to sample.")
    return v / s


# ───────────────────────────────────────────────────────────────── schema
@dataclass(frozen=True)
class Node:
    source: str  # key into Scheme.sources
    cols: tuple[str, ...]  # tree levels → leaves = unique combinations (over ALL the source's cells)
    keys: tuple[str, ...] = ("X",)  # representation location(s): "X" | "obsm/<k>" | "layers/<k>"
    weights: Weights = field(default_factory=dict)  # {combo: weight}; absent/0 ⇒ excluded (= selection)

    def __post_init__(self) -> None:  # structural checks (data-free)
        if not self.cols or not self.keys:
            raise ValueError("Node.cols and Node.keys must be non-empty.")
        for k in self.weights:
            if len(k) != len(self.cols):
                raise ValueError(f"weight key {k!r} arity != cols {self.cols}.")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("weights must be non-negative.")


@dataclass(frozen=True)
class Bind:
    parent: str
    child: str
    common: tuple[str, ...]  # ⊆ parent.cols ∩ child.cols; child conditions on parent's values


@dataclass(frozen=True)
class Scheme:
    sources: Mapping[str, Container]  # {name: AnnData | DatasetCollection}
    nodes: Mapping[str, Node]
    root: str
    n_rows_per_leaf: int  # cells drawn per node-leaf per batch (with replacement) — no default: decide it
    seed: int  # no default: reproducibility must be explicit
    binds: tuple[Bind, ...] = ()

    def __post_init__(self) -> None:  # structural: rooted tree + references
        if self.root not in self.nodes:
            raise ValueError(f"root {self.root!r} not in nodes.")
        for name, n in self.nodes.items():
            if n.source not in self.sources:
                raise ValueError(f"node {name!r} references unknown source {n.source!r}.")
        parents: dict[str, str] = {}
        for b in self.binds:
            if b.parent not in self.nodes or b.child not in self.nodes:
                raise ValueError("bind references unknown node.")
            if b.child in parents:
                raise ValueError(f"node {b.child!r} has multiple parents — must be a rooted tree.")
            parents[b.child] = b.parent
            shared = set(self.nodes[b.parent].cols) & set(self.nodes[b.child].cols)
            if not set(b.common) <= shared:
                raise ValueError(f"bind.common {b.common} must be ⊆ shared cols of {b.parent}&{b.child} ({shared}).")
        if self.root in parents:
            raise ValueError("root must have no parent.")
        for name in self.nodes:
            if name != self.root and name not in parents:
                raise ValueError(f"non-root node {name!r} is not bound to the tree.")


# ───────────────────────────────────────────────────────────────── helpers
def _resolve(adata: ad.AnnData, loc: str) -> np.ndarray:
    if loc == "X":
        x = adata.X
    elif loc.startswith("obsm/"):
        x = adata.obsm[loc[len("obsm/") :]]
    elif loc.startswith("layers/"):
        x = adata.layers[loc[len("layers/") :]]
    else:
        raise ValueError(f"unknown rep location {loc!r}")
    return np.asarray(x.todense() if hasattr(x, "todense") else x, dtype=np.float32)


def _leaf_codes(obs: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, list[tuple]]:
    """Per-cell leaf code + the ordered leaf combinations. This is the HierarchicalIndexer's job."""
    tuples = [tuple(row) for row in obs[list(cols)].to_numpy()]
    leaves = sorted(set(tuples), key=lambda t: tuple(map(str, t)))
    code_of = {lf: i for i, lf in enumerate(leaves)}
    return np.array([code_of[t] for t in tuples], dtype=np.int64), leaves


def _obs(source: Container, cols: Sequence[str]) -> pd.DataFrame:
    """obs columns from either container (AnnData attr vs DatasetCollection reader) — no cell matrices."""
    if isinstance(source, ad.AnnData):
        return source.obs[list(cols)]
    return source.obs(columns=list(cols))  # DatasetCollection


class _ExplicitRequestSampler(Sampler):
    """Yields one annbatch request for exactly `rows` — used to read specific rows from a collection."""

    def __init__(self, rows: np.ndarray) -> None:
        self._rows = np.asarray(rows, dtype=np.int64)

    @property
    def batch_size(self) -> int | None:
        return None

    @property
    def shuffle(self) -> bool:
        return False

    def n_batches(self, n_obs: int) -> int:  # noqa: ARG002
        return 1

    def validate(self, n_obs: int) -> None:  # noqa: ARG002
        return None

    def _sample(self, n_obs: int):  # noqa: ARG002
        yield {"requests": self._rows, "splits": [np.arange(len(self._rows))]}


def _read_rows(source: Container, loc: str, rows: np.ndarray) -> np.ndarray:
    """Read `rows` of representation `loc` into memory. AnnData → index; DatasetCollection → annbatch."""
    rows = np.asarray(rows, dtype=np.int64)
    if isinstance(source, ad.AnnData):
        return _resolve(source, loc)[rows]
    if loc != "X":  # collections stream X only; obsm/layers reads are the documented gap
        raise NotImplementedError("DatasetCollection reads support loc='X' only (obsm/layers not wired).")
    loader = Loader(batch_sampler=_ExplicitRequestSampler(rows), return_index=False,
                    to_torch=False, preload_to_gpu=False).use_collection(source)
    x = next(iter(loader))["X"]
    return np.asarray(x.todense() if hasattr(x, "todense") else x, dtype=np.float32)


# ───────────────────────────────────────────────────────────────── sampler
class TreeSampler:
    """Yields ``{"source", "target", "condition"}`` batches; the root streams through annbatch."""

    def __init__(self, scheme: Scheme, condition_fn: Callable[[tuple], np.ndarray] | None = None) -> None:
        self.s = scheme
        self._cond_fn = condition_fn
        self._rng = np.random.default_rng(scheme.seed)
        self._children: dict[str, list[Bind]] = {}
        for b in scheme.binds:
            self._children.setdefault(b.parent, []).append(b)

        # per-node leaf partition + weights (obs only — no cell matrices)
        self._st: dict[str, dict] = {}
        for name, node in scheme.nodes.items():
            obs = _obs(scheme.sources[node.source], node.cols)
            codes, leaves = _leaf_codes(obs, node.cols)
            self._st[name] = {"node": node, "codes": codes, "leaves": leaves, "w": _weight_vector(node.weights, leaves)}

        self._build_root_loader()
        self._build_child_caches()

    def _build_root_loader(self) -> None:
        st = self._st[self.s.root]
        node = st["node"]
        src = self.s.sources[node.source]
        B, K = self.s.n_rows_per_leaf, len(st["leaves"])
        # classes = per-cell leaf id; class_weights = the node's weights (0 ⇒ excluded ⇒ selection)
        classes = pd.Categorical([str(c) for c in st["codes"]], categories=[str(i) for i in range(K)])
        sampler = ClassSampler(
            chunk_size=1,  # per-row: no sorting/run-length constraint
            preload_nchunks=B,
            batch_size=B,
            classes=classes,
            num_samples=B * 512,
            class_weights=st["w"],
            drop_last=True,
            rng=np.random.default_rng(self.s.seed),
        )
        loader = Loader(batch_sampler=sampler, return_index=True, to_torch=False, preload_to_gpu=False)
        # annbatch streams the root rep as X: add_adata (in-memory) or use_collection (on-disk)
        if isinstance(src, ad.AnnData):
            stream = src if node.keys[0] == "X" else ad.AnnData(X=_resolve(src, node.keys[0]), obs=src.obs)
            self._root_loader = loader.add_adata(stream)
        else:  # DatasetCollection
            if node.keys[0] != "X":
                raise NotImplementedError("root rep must be 'X' for a DatasetCollection (obsm streaming not wired).")
            self._root_loader = loader.use_collection(src)
        self._root_iter = iter(self._root_loader)

    def _build_child_caches(self) -> None:
        """Bound children: cache their positive-weight cells grouped by the shared-column value."""
        self._caches: dict[str, dict] = {}
        for b in self._children.get(self.s.root, []):
            st = self._st[b.child]
            node = st["node"]
            src = self.s.sources[node.source]
            positive = {int(i) for i in np.flatnonzero(st["w"] > 0)}
            rows: dict[tuple, list[int]] = {}
            for cell, code in enumerate(st["codes"]):
                if int(code) in positive:  # weight 0 ⇒ excluded (e.g. non-control combos)
                    leaf = st["leaves"][code]
                    key = tuple(leaf[node.cols.index(c)] for c in b.common)
                    rows.setdefault(key, []).append(cell)
            # read ONLY the (few, reused) positive-weight cells into memory, grouped by shared value
            cells = {k: _read_rows(src, node.keys[0], np.asarray(v, dtype=np.int64)) for k, v in rows.items()}
            self._caches[b.child] = {"bind": b, "cells": cells}

    def __iter__(self) -> "TreeSampler":
        return self

    def __next__(self) -> dict[str, np.ndarray]:
        try:
            batch = next(self._root_iter)
        except StopIteration:
            self._root_iter = iter(self._root_loader)
            batch = next(self._root_iter)

        st = self._st[self.s.root]
        row0 = int(np.asarray(batch["index"])[0])
        leaf = st["leaves"][int(st["codes"][row0])]  # which condition this class-coherent batch is
        target = np.asarray(batch["X"], dtype=np.float32)
        B = target.shape[0]

        out: dict[str, np.ndarray] = {"target": target}
        if self._cond_fn is not None:
            cond = np.asarray(self._cond_fn(leaf), dtype=np.float32)
            out["condition"] = np.broadcast_to(cond, (B, cond.shape[-1])).copy()

        for b in self._children.get(self.s.root, []):
            c = self._caches[b.child]
            key = tuple(leaf[st["node"].cols.index(col)] for col in b.common)
            pool = c["cells"].get(key)
            if pool is None or len(pool) == 0:  # empty match → unconditional fallback
                pool = np.concatenate(list(c["cells"].values()))
            out["source"] = pool[self._rng.integers(0, len(pool), size=B)]  # with replacement
        return out


# ───────────────────────────────────────────────────────────────── factory (the "above layer")
def perturbation_scheme(
    source: Container,
    *,
    context: Sequence[str],
    perturbation: Sequence[str],
    control_values: Mapping[str, object],
    key: str = "X",
    n_rows_per_leaf: int = 256,
    seed: int = 0,
) -> Scheme:
    """Fill a perturbation Scheme from the obs table: root = perturbed combos, child = control combos.

    ``source`` is an in-memory AnnData or an on-disk DatasetCollection. No `select` — control vs
    perturbed is encoded purely by which combinations carry weight.
    """
    cols = (*context, *perturbation)
    combos = [tuple(r) for r in _obs(source, cols).drop_duplicates().to_numpy()]

    def is_control(combo: tuple) -> bool:
        return all(combo[cols.index(c)] == v for c, v in control_values.items())

    pert = [c for c in combos if not is_control(c)]
    ctrl = [c for c in combos if is_control(c)]
    return Scheme(
        sources={"data": source},
        nodes={
            "pert": Node("data", cols, (key,), uniform(pert)),   # non-control combos weighted; rest excluded
            "ctrl": Node("data", cols, (key,), uniform(ctrl)),   # control combos weighted; rest excluded
        },
        root="pert",
        binds=(Bind("pert", "ctrl", common=tuple(context)),),
        n_rows_per_leaf=n_rows_per_leaf,
        seed=seed,
    )
