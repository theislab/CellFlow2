"""Declarative schema for :class:`~scaleflow.dagloader.DAGClassLoader`.

A :class:`Scheme` is a rooted tree of :class:`Node`\\s over named cell *sources* — pure structure
(sources, grouping columns, weights, binds). How each node is *read* (chunk / preload / batch sizes)
lives in a separate :class:`SamplerConfig` passed to the loader, deliberately kept off the ``Node`` so
the same structure can be run with different sampler settings.

Each node partitions its source's cells into **leaves** (unique combinations of ``cols``) with a
per-combination :data:`Weights` mapping. A weight of 0 (or a combination absent from the mapping) is
*excluded* — that IS the selection, native to annbatch's ``ClassSampler``. :class:`Bind` links a
parent to a child on shared columns, so the child is sampled *conditioned* on the parent's values.
See ``README.md`` for the model and the cellflow / sc-flow-tools mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
from annbatch import DatasetCollection

# A cell source: an in-memory AnnData or an out-of-core annbatch DatasetCollection.
Container = ad.AnnData | DatasetCollection

# A sampling scheme is just a mapping {combination -> weight}. A combination absent from the mapping
# (or with weight 0) is excluded — that IS the selection. ``uniform`` / ``frequency`` /
# ``inverse_frequency`` are plain helpers that build such a dict; nothing about them is privileged.
Weights = Mapping[tuple, float]

__all__ = [
    "Bind",
    "Container",
    "Node",
    "SamplerConfig",
    "Scheme",
    "Weights",
    "frequency",
    "inverse_frequency",
    "uniform",
]


def uniform(combos: Sequence[tuple]) -> dict[tuple, float]:
    """Every combination equally likely."""
    return {tuple(c): 1.0 for c in combos}


def frequency(counts: Mapping[tuple, int]) -> dict[tuple, float]:
    """Sample each combination ∝ its cell count (favor abundant conditions)."""
    return {tuple(k): float(c) for k, c in counts.items()}


def inverse_frequency(counts: Mapping[tuple, int]) -> dict[tuple, float]:
    """Sample each combination ∝ 1 / cell count (balance rare vs abundant conditions)."""
    return {tuple(k): 1.0 / c for k, c in counts.items()}


def _weight_vector(weights: Weights, leaves: Sequence[tuple]) -> np.ndarray:
    """Resolve ``{combo: weight}`` to normalized per-leaf weights (→ ``ClassSampler.class_weights``)."""
    v = np.array([float(weights.get(tuple(lf), 0.0)) for lf in leaves], dtype=float)
    s = v.sum()
    if s <= 0:
        raise ValueError("weights resolve to all-zero over these leaves — nothing to sample.")
    return v / s


@dataclass(frozen=True)
class Node:
    """A partition of one source's cells into leaves, with a per-leaf sampling weight.

    Parameters
    ----------
    source
        Key into :attr:`Scheme.sources`.
    cols
        Tree levels → leaves are the unique combinations of these columns (over ALL the source's
        cells). These are the grouping/condition columns (cellflow's ``split_covariates`` +
        ``perturbation_covariates`` columns; sc-flow-tools' grouping keys).
    key
        The representation location to stream: ``"X"`` | ``"obsm/<k>"`` | ``"layers/<k>"``
        (cellflow's ``sample_rep``). One rep per node — add another node for another rep.
    weights
        ``{combo: weight}``; a combination absent or with weight 0 is excluded (= the selection).
    """

    source: str
    cols: tuple[str, ...]
    key: str = "X"
    weights: Weights = field(default_factory=dict)

    def __post_init__(self) -> None:  # structural checks (data-free)
        if not self.cols:
            raise ValueError("Node.cols must be non-empty.")
        if not self.key:
            raise ValueError("Node.key must be a non-empty representation location.")
        for k in self.weights:
            if len(k) != len(self.cols):
                raise ValueError(f"weight key {k!r} arity != cols {self.cols}.")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("weights must be non-negative.")


@dataclass(frozen=True)
class Bind:
    """Condition ``child`` on ``parent``: match on the ``common`` columns (⊆ their shared cols).

    Each batch, the child's sampled leaf is derived from the parent's leaf via the ``common`` values
    (parent leaf → shared-column values → matching child leaf). This is the source↔target matching:
    with ``common`` = the context (e.g. cell line), the child (control) is drawn from the *same*
    context as the parent (perturbed) — cellflow's "control = same group", sc-flow-tools'
    ``control_values_dict`` + default same-context coupling.

    Conditioning is **required**: if a parent value has no matching positive-weight child leaf the
    loader raises (no silent fallback). When several child leaves share the bound value — the child
    partitions on columns beyond ``common`` (e.g. child cols ``(a, x)`` bound on ``a``) — one is drawn
    ∝ the child's leaf weights, so ``P(child extra cols | common)`` is weight-controlled. Pass
    ``common=()`` to opt into unconditional child sampling explicitly.
    """

    parent: str
    child: str
    common: tuple[str, ...]  # ⊆ parent.cols ∩ child.cols; () ⇒ explicit unconditional (see docstring)


@dataclass(frozen=True)
class SamplerConfig:
    """annbatch read parameters for a node's sampler — kept separate from the structural :class:`Node`.

    Passed to :class:`~scaleflow.dagloader.DAGClassLoader` as either one config (applied to every node)
    or a ``{node_name: SamplerConfig}`` mapping (per-node). ``batch_size`` must be equal across nodes —
    a yielded batch has one row count (target rows == source rows == B) — which the loader validates.

    Parameters
    ----------
    batch_size
        Rows per emitted batch (``B``).
    chunk_size
        annbatch read-slice size. ``1`` (default) ⇒ per-row reads (any on-disk layout). ``>1`` ⇒
        contiguous chunked reads (higher throughput on disk), assuming each sampled leaf sits in a
        contiguous run ≥ ``chunk_size``. Must divide ``batch_size`` (one category per batch).
    preload_nchunks
        Chunks per annbatch read window. ``None`` (default) ⇒ ``batch_size // chunk_size`` (one batch
        per window). If given, must be a positive multiple of ``batch_size // chunk_size``.
    """

    batch_size: int
    chunk_size: int = 1
    preload_nchunks: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.chunk_size < 1:
            raise ValueError("batch_size and chunk_size must be >= 1.")
        if self.batch_size % self.chunk_size != 0:  # one category per batch
            raise ValueError(f"chunk_size ({self.chunk_size}) must divide batch_size ({self.batch_size}).")
        if self.preload_nchunks is not None:
            group_chunks = self.batch_size // self.chunk_size
            if self.preload_nchunks < 1 or self.preload_nchunks % group_chunks != 0:
                raise ValueError(
                    f"preload_nchunks ({self.preload_nchunks}) must be a positive multiple of "
                    f"batch_size // chunk_size ({group_chunks})."
                )

    @property
    def resolved_preload(self) -> int:
        """``preload_nchunks`` or its default (``batch_size // chunk_size`` — one batch per window)."""
        return self.preload_nchunks if self.preload_nchunks is not None else self.batch_size // self.chunk_size


@dataclass(frozen=True)
class Scheme:
    """The structural sampling spec: sources, a rooted tree of nodes, and the reproducibility cadence.

    Read parameters (chunk / preload / batch sizes) are NOT here — they are a separate
    :class:`SamplerConfig` given to the loader.

    Parameters
    ----------
    sources
        ``{name: AnnData | DatasetCollection}`` — the cell sources the nodes reference.
    nodes
        ``{name: Node}``. Exactly one is the ``root`` (the streamed target); the rest are bound
        children (sources/controls) via ``binds``.
    root
        Name of the root node (must have no parent).
    seed
        Reproducibility seed. Per-node RNG streams are spawned from one ``SeedSequence(seed)`` so nodes
        do not correlate and the whole stream is reproducible.
    binds
        Parent→child links (see :class:`Bind`). Must form a rooted tree over ``nodes``.
    steps_per_pass
        Batches per with-replacement pass (the loader restarts each pass → effectively infinite with a
        fixed, reproducible restart cadence). Scheme-wide: every node restarts together.
    """

    sources: Mapping[str, Container]
    nodes: Mapping[str, Node]
    root: str
    seed: int
    binds: tuple[Bind, ...] = ()
    steps_per_pass: int = 512

    def __post_init__(self) -> None:  # structural: rooted tree + references
        if self.steps_per_pass < 1:
            raise ValueError("steps_per_pass must be >= 1.")
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
