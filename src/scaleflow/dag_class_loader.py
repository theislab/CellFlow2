"""Declarative, index-free class-mapping sampler over annbatch (prototype).

A rooted tree of `Node`s over named sources; each node is a partition of its source's cells into
leaves (unique column-combinations) with a per-combination weight mapping. Every node streams through
its own annbatch loader; the root samples a leaf per batch (weight 0 ⇒ excluded ⇒ = selection) and
bound children are streamed conditioned on the parent's shared-column values. No row indices are
exposed — the scheme is columns / keys / weights.

## Per-node sampling

Each node runs a :class:`ScheduledClassSampler` — an annbatch ``ClassSampler`` whose per-batch
category sequence can be *supplied* (a "schedule") instead of drawn internally. This keeps annbatch's
chunk math (contiguous ``chunk_size`` reads, run-length encoding, slice sampling) — the throughput
win for big sorted on-disk sources — while letting ``DAGClassLoader`` own *which* category each batch
draws. That ownership is what makes the tree work: the root schedule is drawn from the root's
weights, and each bound child's schedule is *derived* from the parent's (parent leaf → shared-column
value → matching child leaf), so the loaders zip batch-for-batch with no per-step reconfiguration.

Sampler parameters are decided **per node**, not hardcoded:
- ``chunk_size`` (per node, ``Node.chunk_size``; ``None`` ⇒ 1 = per-row, any layout). ``>1`` gives
  contiguous chunked reads but requires the source sorted so each sampled leaf is a run
  ``≥ chunk_size`` (annbatch's run-length rule; validated at build with node context).
- ``batch_size`` == ``Scheme.n_rows_per_leaf`` — **scheme-level**: a yielded batch has one row count
  (target rows == source rows == B), so it cannot vary per node.
- ``preload_nchunks`` — derived ``= batch_size // chunk_size`` (one batch per read window).
- ``num_samples`` — derived ``= steps_per_pass * batch_size`` (``Scheme.steps_per_pass``); the loader
  restarts every pass, so iteration is effectively infinite with a fixed, reproducible restart cadence.
- ``drop_last`` — always ``True`` (a partial batch would misalign root/child and the schedule).

The constraint **``chunk_size`` divides ``batch_size``** is required (validated): it makes each batch
exactly one category, so a schedule is a length-``steps_per_pass`` array of category codes and root and
child align batch-for-batch regardless of their individual ``chunk_size``.

Per-node RNGs are spawned from one ``SeedSequence(seed)`` (one independent stream per node, assigned
by sorted node name) so nodes don't correlate and the whole thing is reproducible from ``seed``.

Scope of this prototype: single-level bind (root + bound children). ``ScheduledClassSampler`` is a
candidate to upstream into annbatch — if ``ClassSampler`` factored its category draw into a
``_group_positions(n_groups)`` hook, this subclass would collapse to overriding that one method
instead of copying ``_iter_requests``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
from annbatch import DatasetCollection, Loader
from annbatch.abc import Sampler
from annbatch.samplers import ClassSampler
from annbatch.utils import split_given_size

Container = ad.AnnData | DatasetCollection  # a cell source: in-memory AnnData or on-disk DatasetCollection

__all__ = ["Node", "Bind", "Scheme", "DAGClassLoader", "ScheduledClassSampler", "perturbation_scheme",
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
    chunk_size: int | None = None  # annbatch read-slice; None ⇒ 1 (per-row). >1 needs a sorted source.

    def __post_init__(self) -> None:  # structural checks (data-free)
        if not self.cols or not self.keys:
            raise ValueError("Node.cols and Node.keys must be non-empty.")
        for k in self.weights:
            if len(k) != len(self.cols):
                raise ValueError(f"weight key {k!r} arity != cols {self.cols}.")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("weights must be non-negative.")
        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError("Node.chunk_size must be >= 1 (or None to default to 1).")


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
    n_rows_per_leaf: int  # cells per node-leaf per batch == batch_size (scheme-level) — no default
    seed: int  # no default: reproducibility must be explicit
    binds: tuple[Bind, ...] = ()
    steps_per_pass: int = 512  # batches per with-replacement pass (loader restart cadence); scheme-level

    def __post_init__(self) -> None:  # structural: rooted tree + references
        if self.n_rows_per_leaf < 1 or self.steps_per_pass < 1:
            raise ValueError("n_rows_per_leaf and steps_per_pass must be >= 1.")
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


# ───────────────────────────────────────────────────────────────── scheduled sampler
class ScheduledClassSampler(ClassSampler):
    """``ClassSampler`` whose per-batch category sequence can be *supplied* instead of drawn.

    ``set_schedule(codes)`` fixes the category (leaf code) of each batch for the next pass; with
    ``schedule=None`` the sampler is identical to ``ClassSampler`` (draws ∝ ``class_weights``). All the
    chunk math — RLE (``_ensure_runs``), run selection, slice sampling, window/split batching — is
    inherited; only the category source changes. The one copied method is ``_iter_requests`` (a hook
    annbatch could expose upstream, collapsing this to a one-method override).

    The schedule is per *group*; with ``chunk_size`` dividing ``batch_size`` there is exactly one group
    (hence one category) per batch, so the schedule is one code per batch — ``len == num_samples // batch_size``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._schedule: np.ndarray | None = None

    def set_schedule(self, category_codes: np.ndarray | None) -> None:
        """Set (or clear) the per-batch category codes used for the next ``iter(loader)`` pass."""
        self._schedule = None if category_codes is None else np.asarray(category_codes, dtype=np.int64)

    def _group_positions(self, n_groups: int) -> np.ndarray:
        """Category (as a position into the sampleable-class table) for each of the ``n_groups`` groups."""
        info = self._per_class_sampling_info
        if self._schedule is None:  # standalone → identical to ClassSampler
            return self._rng.choice(len(info), size=n_groups, p=info["prob"].to_numpy())
        sched = self._schedule
        if sched.shape[0] != n_groups:
            raise ValueError(
                f"schedule length {sched.shape[0]} != n_groups {n_groups}; one category per batch "
                "requires chunk_size to divide batch_size and len(schedule) == num_samples // batch_size."
            )
        # _iter_requests indexes into the sampleable-class table by position; map leaf code → position.
        code_to_pos = {int(c): i for i, c in enumerate(info.index.to_numpy())}
        try:
            return np.array([code_to_pos[int(c)] for c in sched], dtype=np.int64)
        except KeyError as e:
            raise ValueError(f"schedule contains a non-sampleable (zero-weight) category code {e}.") from e

    def _iter_requests(self) -> Iterator[dict]:
        # Copied from annbatch ClassSampler._iter_requests; the ONLY change is sourcing group_classes
        # from the (possibly supplied) schedule via _group_positions instead of an inline rng.choice.
        n_slices, remainder = divmod(self._num_samples, self._chunk_size)
        if remainder > 0:
            n_slices += 1

        group_chunks = self._batch_size // math.gcd(self._chunk_size, self._batch_size)
        n_groups = math.ceil(n_slices / group_chunks)
        group_classes = self._group_positions(n_groups)  # ← the one line that differs from annbatch
        class_of_slice = np.repeat(group_classes, group_chunks)[:n_slices]

        class_n_runs = self._per_class_sampling_info["n_runs"].to_numpy()
        possible_run_pos_within_a_class = self._rng.integers(class_n_runs[class_of_slice])
        first_row_of_class = self._per_class_sampling_info["first_row_in_runs_of_class"].to_numpy()
        chosen = first_row_of_class[class_of_slice] + possible_run_pos_within_a_class
        run_starts = self._class_runs["start"].to_numpy()[chosen]
        run_ends = self._class_runs["end"].to_numpy()[chosen]
        slice_starts = self._rng.integers(run_starts, run_ends - self._chunk_size + 1)

        slices = [slice(int(s), int(s + self._chunk_size)) for s in slice_starts]
        if remainder > 0:
            last = int(slice_starts[-1])
            slices[-1] = slice(last, last + remainder)

        window_size = self._preload_nchunks * self._chunk_size
        full_splits = split_given_size(np.arange(window_size), self._batch_size)
        for window in itertools.batched(slices, self._preload_nchunks):
            n_rows = (len(window) - 1) * self._chunk_size + (window[-1].stop - window[-1].start)
            splits = full_splits if n_rows == window_size else split_given_size(np.arange(n_rows), self._batch_size)
            if self._drop_last and splits[-1].size < self._batch_size:
                splits = splits[:-1]
                if not splits:
                    continue
            for batch in splits:
                self._rng.shuffle(batch)
            yield {"requests": list(window), "splits": splits}


# ───────────────────────────────────────────────────────────────── helpers
def _key_backings(source: Container, loc: str) -> list:
    """The array(s) backing rep `loc` for a source, ready to feed one annbatch Loader.add_datasets.

    annbatch's add_datasets concatenates on the obs axis and needs equal feature dims, so X and obsm
    cannot share a loader — instead each key gets its own loader over its own array(s). For a
    DatasetCollection the per-dataset arrays are gathered in order (matching the global row layout).
    """
    if loc == "X":
        return [source.X] if isinstance(source, ad.AnnData) else [g["X"] for g in source]
    field, sub = loc.split("/", 1)  # "obsm/X_pca" | "layers/log1p"
    if isinstance(source, ad.AnnData):
        return [getattr(source, field)[sub]]
    return [g[field][sub] for g in source]  # DatasetCollection: one zarr array per dataset


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
    """Read `rows` of representation `loc` into memory via one annbatch loader over that key's array(s).

    Uniform across AnnData / DatasetCollection and across X / obsm / layers (via add_datasets).
    """
    rows = np.asarray(rows, dtype=np.int64)
    loader = Loader(batch_sampler=_ExplicitRequestSampler(rows), return_index=False,
                    to_torch=False, preload_to_gpu=False).add_datasets(_key_backings(source, loc))
    x = next(iter(loader))["X"]
    return np.asarray(x.todense() if hasattr(x, "todense") else x, dtype=np.float32)


# ───────────────────────────────────────────────────────────────── sampler
class DAGClassLoader:
    """Yields ``{"source", "target", "condition"}`` batches; every node streams through its own loader."""

    def __init__(self, scheme: Scheme, condition_fn: Callable[[tuple], np.ndarray] | None = None) -> None:
        self.s = scheme
        self._cond_fn = condition_fn
        self._n_batches = scheme.steps_per_pass
        self._children: dict[str, list[Bind]] = {}
        for b in scheme.binds:
            self._children.setdefault(b.parent, []).append(b)

        # per-node independent RNG streams spawned from one seed (assigned by sorted node name)
        seqs = np.random.SeedSequence(scheme.seed).spawn(len(scheme.nodes))
        self._rngs = {name: np.random.default_rng(s) for name, s in zip(sorted(scheme.nodes), seqs)}

        # per-node leaf partition + weights (obs only — no cell matrices)
        self._st: dict[str, dict] = {}
        for name, node in scheme.nodes.items():
            obs = _obs(scheme.sources[node.source], node.cols)
            codes, leaves = _leaf_codes(obs, node.cols)
            self._st[name] = {"node": node, "codes": codes, "leaves": leaves, "w": _weight_vector(node.weights, leaves)}

        self._build_loaders()
        self._build_bind_maps()
        self._iters: dict[str, Iterator[dict]] | None = None
        self._schedules: dict[str, np.ndarray] = {}
        self._pos = 0

    # ── build ────────────────────────────────────────────────────────────
    def _build_loaders(self) -> None:
        """One ScheduledClassSampler + Loader per node, with per-node resolved sampler params."""
        self._samplers: dict[str, ScheduledClassSampler] = {}
        self._loaders: dict[str, Loader] = {}
        self._srcs: dict[str, Container] = {}
        B = self.s.n_rows_per_leaf
        for name, st in self._st.items():
            node = st["node"]
            src = self.s.sources[node.source]
            K = len(st["leaves"])
            chunk = node.chunk_size or 1
            if B % chunk != 0:  # one category per batch ⇒ chunk_size must divide batch_size
                raise ValueError(f"node {name!r}: chunk_size ({chunk}) must divide n_rows_per_leaf ({B}).")
            preload = B // chunk                       # one batch per read window
            num_samples = self._n_batches * B          # steps_per_pass full batches per pass
            classes = pd.Categorical([str(c) for c in st["codes"]], categories=[str(i) for i in range(K)])
            try:  # ClassSampler enforces the run-length rule (chunk_size>1 ⇒ sorted, each leaf run ≥ chunk)
                sampler = ScheduledClassSampler(
                    chunk_size=chunk, preload_nchunks=preload, batch_size=B,
                    classes=classes, num_samples=num_samples, class_weights=st["w"],
                    drop_last=True, rng=self._rngs[name],
                )
            except ValueError as e:
                raise ValueError(f"node {name!r}: {e}") from e
            loader = Loader(batch_sampler=sampler, return_index=(name == self.s.root),
                            to_torch=False, preload_to_gpu=False).add_datasets(_key_backings(src, node.keys[0]))
            self._samplers[name] = sampler
            self._loaders[name] = loader
            self._srcs[name] = src

    def _build_bind_maps(self) -> None:
        """Precompute, per bound child, the maps to turn the parent's schedule into the child's."""
        self._bindmap: dict[str, dict] = {}
        for b in self._children.get(self.s.root, []):
            rst, cst = self._st[b.parent], self._st[b.child]
            rcols, ccols = rst["node"].cols, cst["node"].cols
            # parent leaf code → shared-column value
            root_code_to_cv = {i: tuple(lf[rcols.index(c)] for c in b.common) for i, lf in enumerate(rst["leaves"])}
            # shared-column value → positive-weight child leaf codes carrying it
            common_to_child: dict[tuple, list[int]] = {}
            for code, lf in enumerate(cst["leaves"]):
                if cst["w"][code] > 0:
                    common_to_child.setdefault(tuple(lf[ccols.index(c)] for c in b.common), []).append(code)
            self._bindmap[b.child] = {
                "root_code_to_cv": root_code_to_cv,
                "common_to_child": common_to_child,
                "positive_child_codes": np.flatnonzero(cst["w"] > 0).astype(np.int64),
            }

    # ── per-pass scheduling ────────────────────────────────────────────────
    def _draw_root_schedule(self) -> np.ndarray:
        """Draw ``steps_per_pass`` root leaf codes ∝ the root's weights (the root's own RNG stream)."""
        w = self._st[self.s.root]["w"]
        pos = np.flatnonzero(w > 0)
        return self._rngs[self.s.root].choice(pos, size=self._n_batches, p=w[pos] / w[pos].sum()).astype(np.int64)

    def _derive_child_schedule(self, child: str, root_sched: np.ndarray) -> np.ndarray:
        """Map each root batch's category to a matching child leaf (child's RNG for ties/fallback)."""
        m = self._bindmap[child]
        rng = self._rngs[child]
        out = np.empty(len(root_sched), dtype=np.int64)
        for j, rc in enumerate(root_sched):
            cands = m["common_to_child"].get(m["root_code_to_cv"][int(rc)])
            if not cands:  # no child leaf shares the parent's value → unconditional fallback
                cands = m["positive_child_codes"]
            out[j] = int(cands[0]) if len(cands) == 1 else int(rng.choice(cands))
        return out

    def _start_pass(self) -> None:
        """Draw schedules (root + derived children), push them onto the samplers, rebuild iterators.

        Ordering matters: Loader.__iter__ re-reads sampler.sample() and _iter_requests reads the
        schedule up front, so set_schedule must land before iter(loader).
        """
        root_sched = self._draw_root_schedule()
        self._schedules = {self.s.root: root_sched}
        self._samplers[self.s.root].set_schedule(root_sched)
        for b in self._children.get(self.s.root, []):
            child_sched = self._derive_child_schedule(b.child, root_sched)
            self._schedules[b.child] = child_sched
            self._samplers[b.child].set_schedule(child_sched)
        self._iters = {name: iter(ld) for name, ld in self._loaders.items()}
        self._pos = 0

    # ── iteration ──────────────────────────────────────────────────────────
    def __iter__(self) -> "DAGClassLoader":
        return self

    def __next__(self) -> dict[str, np.ndarray]:
        if self._iters is None or self._pos >= self._n_batches:
            self._start_pass()
        j = self._pos
        self._pos += 1

        st = self._st[self.s.root]
        node = st["node"]
        root_batch = next(self._iters[self.s.root])
        leaf = st["leaves"][int(self._schedules[self.s.root][j])]  # per-batch category — from the schedule
        target = np.asarray(root_batch["X"], dtype=np.float32)     # the primary key (keys[0]) — streamed
        B = target.shape[0]

        out: dict = {"target": target}
        if len(node.keys) > 1:  # extra keys (e.g. obsm reps) via companion reads by the streamed rows
            idx = np.asarray(root_batch["index"])
            out["target_reps"] = {node.keys[0]: target,
                                  **{k: _read_rows(self._srcs[self.s.root], k, idx) for k in node.keys[1:]}}
        if self._cond_fn is not None:
            cond = np.asarray(self._cond_fn(leaf), dtype=np.float32)
            out["condition"] = np.broadcast_to(cond, (B, cond.shape[-1])).copy()

        for b in self._children.get(self.s.root, []):  # bound child source, conditioned via its schedule
            child_batch = next(self._iters[b.child])
            out["source"] = np.asarray(child_batch["X"], dtype=np.float32)
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
    chunk_size: int | None = None,
    steps_per_pass: int = 512,
) -> Scheme:
    """Fill a perturbation Scheme from the obs table: root = perturbed combos, child = control combos.

    ``source`` is an in-memory AnnData or an on-disk DatasetCollection. No `select` — control vs
    perturbed is encoded purely by which combinations carry weight. ``chunk_size`` (root only) opts
    into contiguous chunked reads (needs the source sorted by ``context+perturbation``).
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
            # non-control combos weighted (rest excluded); root opts into chunk_size for throughput
            "pert": Node("data", cols, (key,), uniform(pert), chunk_size=chunk_size),
            "ctrl": Node("data", cols, (key,), uniform(ctrl)),   # control combos weighted; rest excluded
        },
        root="pert",
        binds=(Bind("pert", "ctrl", common=tuple(context)),),
        n_rows_per_leaf=n_rows_per_leaf,
        seed=seed,
        steps_per_pass=steps_per_pass,
    )
