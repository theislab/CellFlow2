"""Declarative, index-free class-mapping sampler over annbatch (prototype).

A rooted tree of `Node`s over named sources; each node is a partition of its source's cells into
leaves (unique column-combinations) with a per-combination `Weighting`. The root samples a leaf via
annbatch `ClassSampler` (weight 0 ⇒ excluded ⇒ = selection); bound children are fetched conditioned on
the parent's shared-column values. No row indices are exposed — the scheme is columns / keys / weights.

Scope of this prototype: single-level bind (root + bound children), in-memory `AnnData` sources; the
root streams through annbatch, bound children come from an in-memory cache keyed by the shared columns.
Multi-level trees, `DatasetCollection` sources, and multi-key (obsm) reps are the documented extensions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from annbatch import Loader
from annbatch.samplers import ClassSampler

__all__ = ["Weighting", "Node", "Bind", "Scheme", "TreeSampler", "perturbation_scheme"]


# ───────────────────────────────────────────────────────────────── weighting: one source of truth
@dataclass(frozen=True)
class Weighting:
    """weight(leaf) = weights.get(leaf, default), normalized. `uniform` = just `default`.

    The schema is always this general (weights, default); the policy niceties are constructor-only.
    """

    weights: Mapping[tuple, float] = field(default_factory=dict)
    default: float = 1.0

    @classmethod
    def uniform(cls) -> "Weighting":
        return cls(default=1.0)

    @classmethod
    def explicit(cls, weights: Mapping[tuple, float], default: float = 0.0) -> "Weighting":
        return cls(dict(weights), default)

    @classmethod
    def frequency(cls, counts: Mapping[tuple, int]) -> "Weighting":
        return cls({k: float(c) for k, c in counts.items()}, 0.0)

    @classmethod
    def inverse_frequency(cls, counts: Mapping[tuple, int]) -> "Weighting":
        return cls({k: 1.0 / c for k, c in counts.items()}, 0.0)

    def vector(self, leaves: Sequence[tuple]) -> np.ndarray:
        """Resolve to normalized per-leaf weights (→ ClassSampler.class_weights). The only consumer."""
        v = np.array([self.weights.get(tuple(lf), self.default) for lf in leaves], dtype=float)
        s = v.sum()
        if s <= 0:
            raise ValueError("Weighting resolves to all-zero over these leaves — nothing to sample.")
        return v / s


# ───────────────────────────────────────────────────────────────── schema
@dataclass(frozen=True)
class Node:
    source: str  # key into Scheme.sources
    cols: tuple[str, ...]  # tree levels → leaves = unique combinations (over ALL the source's cells)
    keys: tuple[str, ...] = ("X",)  # representation location(s): "X" | "obsm/<k>" | "layers/<k>"
    weighting: Weighting = field(default_factory=Weighting)

    def __post_init__(self) -> None:  # structural checks (data-free)
        if not self.cols or not self.keys:
            raise ValueError("Node.cols and Node.keys must be non-empty.")
        for k in self.weighting.weights:
            if len(k) != len(self.cols):
                raise ValueError(f"weight key {k!r} arity != cols {self.cols}.")
        if self.weighting.default < 0 or any(w < 0 for w in self.weighting.weights.values()):
            raise ValueError("weights must be non-negative.")


@dataclass(frozen=True)
class Bind:
    parent: str
    child: str
    common: tuple[str, ...]  # ⊆ parent.cols ∩ child.cols; child conditions on parent's values


@dataclass(frozen=True)
class Scheme:
    sources: Mapping[str, Any]  # {name: AnnData}  (DatasetCollection is a documented extension)
    nodes: Mapping[str, Node]
    root: str
    binds: tuple[Bind, ...] = ()
    n_rows_per_leaf: int = 256
    seed: int = 0

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
            obs = scheme.sources[node.source].obs
            codes, leaves = _leaf_codes(obs, node.cols)
            self._st[name] = {"node": node, "codes": codes, "leaves": leaves, "w": node.weighting.vector(leaves)}

        self._build_root_loader()
        self._build_child_caches()

    def _build_root_loader(self) -> None:
        st = self._st[self.s.root]
        node = st["node"]
        adata = self.s.sources[node.source]
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
        # stream the root node's representation as X
        rep = _resolve(adata, node.keys[0])
        stream_adata = adata if node.keys[0] == "X" else ad.AnnData(X=rep, obs=adata.obs)
        loader = Loader(batch_sampler=sampler, return_index=True, to_torch=False, preload_to_gpu=False)
        self._root_loader = loader.add_adata(stream_adata)
        self._root_iter = iter(self._root_loader)

    def _build_child_caches(self) -> None:
        """Bound children: cache their positive-weight cells grouped by the shared-column value."""
        self._caches: dict[str, dict] = {}
        for b in self._children.get(self.s.root, []):
            st = self._st[b.child]
            node = st["node"]
            adata = self.s.sources[node.source]
            rep = _resolve(adata, node.keys[0])
            positive = {int(i) for i in np.flatnonzero(st["w"] > 0)}
            rows: dict[tuple, list[int]] = {}
            for cell, code in enumerate(st["codes"]):
                if int(code) in positive:  # weight 0 ⇒ excluded (e.g. non-control combos)
                    leaf = st["leaves"][code]
                    key = tuple(leaf[node.cols.index(c)] for c in b.common)
                    rows.setdefault(key, []).append(cell)
            self._caches[b.child] = {
                "bind": b,
                "rep": rep,
                "rows": {k: np.asarray(v, dtype=np.int64) for k, v in rows.items()},
            }

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
            rows = c["rows"].get(key)
            if rows is None or len(rows) == 0:  # empty match → unconditional fallback
                rows = np.concatenate(list(c["rows"].values()))
            sel = rows[self._rng.integers(0, len(rows), size=B)]  # with replacement
            out["source"] = c["rep"][sel]
        return out


# ───────────────────────────────────────────────────────────────── factory (the "above layer")
def perturbation_scheme(
    adata: ad.AnnData,
    *,
    context: Sequence[str],
    perturbation: Sequence[str],
    control_values: Mapping[str, Any],
    key: str = "X",
    n_rows_per_leaf: int = 256,
    seed: int = 0,
) -> Scheme:
    """Fill a perturbation Scheme from the obs table: root = perturbed combos, child = control combos.

    No `select` — control vs perturbed is encoded purely by which combinations carry weight.
    """
    cols = (*context, *perturbation)
    combos = [tuple(r) for r in adata.obs[list(cols)].drop_duplicates().to_numpy()]

    def is_control(combo: tuple) -> bool:
        return all(combo[cols.index(c)] == v for c, v in control_values.items())

    pert = [c for c in combos if not is_control(c)]
    ctrl = [c for c in combos if is_control(c)]
    return Scheme(
        sources={"data": adata},
        nodes={
            "pert": Node("data", cols, (key,), Weighting.explicit({c: 1.0 for c in pert}, default=0.0)),
            "ctrl": Node("data", cols, (key,), Weighting.explicit({c: 1.0 for c in ctrl}, default=0.0)),
        },
        root="pert",
        binds=(Bind("pert", "ctrl", common=tuple(context)),),
        n_rows_per_leaf=n_rows_per_leaf,
        seed=seed,
    )
