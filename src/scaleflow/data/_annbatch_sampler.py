"""annbatch-backed training sampler for GroupedDistribution.

Streams *target* (perturbed) cells from an :class:`annbatch.DatasetCollection` with
annbatch's :class:`~annbatch.samplers.ClassSampler`, one class-coherent batch per step,
while keeping the *source* (control) populations in an in-memory cache.

Each training step draws a target condition ``c_t ~ Categorical(weights)`` and reads a
``chunk_size``-aligned batch of that condition's cells from disk via ``ClassSampler``;
the matched source population (``tgt -> src`` via ``src_to_tgt_dist_map``) is sampled from
the in-memory :class:`SourceCache`, and ``conditions[c_t]`` is attached. The condition of
each emitted batch is recovered from its global row indices (``return_index=True``) through
a row->tgt lookup -- ``ClassSampler`` batches are single-class, so any one row identifies
the whole batch.

ON-DISK REQUIREMENT: ``ClassSampler`` reads contiguous ``chunk_size`` slices within each
class's run, so the collection must be laid out with **each condition's cells contiguous**
(write it with :func:`write_sorted_collection`), and every *sampled* condition must have at
least ``chunk_size`` cells -- otherwise construction raises (the run-length rule). Condition
pairing itself is recovered from row indices and stays correct regardless of on-disk order;
the sort only governs whether ``ClassSampler`` accepts the layout and read efficiency.

Why asymmetric? Control populations are few and reused by every condition, so caching them
is cheap and avoids re-reading them every step; the many target conditions are streamed.

ASSUMPTION: all source/control cells fit in memory. There is intentionally no fallback;
if they do not fit, this sampler is not the right tool.

Sparsity is preserved end to end: if the collection stores ``X`` as sparse CSR, both the
streamed target batch and the cached source batch come out as ``scipy.sparse`` (nothing
is densified here). Densifying, if needed, is a model-boundary decision.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from annbatch import Loader
from annbatch.abc import Sampler
from annbatch.samplers import ClassSampler

from scaleflow.data._data import GroupedDistribution
from scaleflow.data._dataloader import SamplerABC

__all__ = [
    "GroupedAnnbatchSampler",
    "PredictionSampler",
    "SourceCache",
    "ValidationSampler",
    "write_sorted_collection",
]

# Sentinel category for obs rows that are never sampled (controls + non-split targets).
# It is assigned class weight 0, which ClassSampler excludes and exempts from the
# run-length rule.
_EXCLUDED_CLASS = "__excluded__"


def _open_collection(collection: Any) -> Any:
    """Return a read-mode :class:`annbatch.DatasetCollection` from a path or pass-through."""
    from annbatch import DatasetCollection

    if isinstance(collection, DatasetCollection):
        return collection
    return DatasetCollection(str(collection), mode="r")


class _ExplicitRequestSampler(Sampler):
    """Minimal annbatch sampler that yields caller-provided requests verbatim.

    Each yielded request carries explicit global row indices and a single split, so
    annbatch fetches exactly those rows as one batch (cell selection is fully governed
    by the caller's RNG, not annbatch's internal shuffling).
    """

    def __init__(self, request_iter, *, batch_size: int | None, n_batches: int):
        self._request_iter = request_iter
        self._batch_size = batch_size
        self._n_batches = n_batches

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        return False

    def n_batches(self, n_obs: int) -> int:  # noqa: ARG002
        return self._n_batches

    def validate(self, n_obs: int) -> None:  # noqa: ARG002
        return None

    def _sample(self, n_obs: int):  # noqa: ARG002
        yield from self._request_iter


def _read_distributions(collection: Any, dist_to_rows: dict[int, np.ndarray]) -> dict[int, Any]:
    """Read each distribution's rows from the collection's ``X`` into memory.

    Sparse stays sparse (no densification). Returns ``{dist_idx: cells}``.
    """
    order = sorted(int(k) for k in dist_to_rows.keys())
    if not order:
        return {}
    coll = _open_collection(collection)
    requests = [
        {"requests": np.asarray(dist_to_rows[k], dtype=np.int64), "splits": [np.arange(len(dist_to_rows[k]))]}
        for k in order
    ]
    sampler = _ExplicitRequestSampler(iter(requests), batch_size=None, n_batches=len(requests))
    loader = Loader(batch_sampler=sampler, return_index=False, to_torch=False, preload_to_gpu=False)
    loader = loader.use_collection(coll)
    return {k: batch["X"] for k, batch in zip(order, loader, strict=True)}


def _cond_key(annotation: Any, tgt_idx: int) -> tuple[str, ...]:
    """Condition key (tuple of label strings) for a target distribution index."""
    labels = annotation.tgt_dist_idx_to_labels.get(tgt_idx)
    if labels is None:
        return (str(tgt_idx),)
    if isinstance(labels, (list, tuple, np.ndarray)):
        return tuple(str(x) for x in labels)
    return (str(labels),)


class SourceCache:
    """In-memory cache of source (control) cells, read once from a DatasetCollection.

    Reads each source distribution's rows from the collection's ``X`` (preserving sparse
    vs dense), keyed by ``src_dist_idx``. Sampling returns rows in the stored type.
    """

    def __init__(self, collection: Any, src_dist_to_rows: dict[int, np.ndarray]) -> None:
        self._cache: dict[int, Any] = _read_distributions(collection, src_dist_to_rows)

    @property
    def src_indices(self) -> list[int]:
        return sorted(self._cache.keys())

    def cells(self, src_idx: int) -> Any:
        """Return all cached cells for a source distribution."""
        return self._cache[int(src_idx)]

    def sample(self, src_idx: int, n: int, rng: np.random.Generator) -> Any:
        """Sample ``n`` rows (with replacement) from a cached source distribution."""
        x = self._cache[int(src_idx)]
        sel = rng.integers(0, x.shape[0], size=n)
        return x[sel]


class GroupedAnnbatchSampler(SamplerABC):
    """Training sampler yielding matched (source, target, condition) batches.

    Target cells are streamed with annbatch's :class:`~annbatch.samplers.ClassSampler`:
    each step it draws a target condition ``c_t ~ Categorical(weights)`` and reads a
    ``chunk_size``-aligned batch of that condition's cells from disk. The matched source
    population (``tgt -> src`` via ``src_to_tgt_dist_map``) is sampled from the in-memory
    :class:`SourceCache`, and ``conditions[c_t]`` is attached. The condition of each emitted
    batch is recovered from its global row indices (``return_index=True``) via a row->tgt
    lookup; ClassSampler batches are single-class, so any one row identifies the batch.

    The ClassSampler and source-cell RNGs are independent streams spawned from one seed, so
    the full (source, target) pairing sequence is reproducible.

    On-disk requirement: ClassSampler reads contiguous ``chunk_size`` slices within each
    condition's run, so the collection must be laid out with each condition's cells
    contiguous (use :func:`write_sorted_collection`) and every *sampled* condition must have
    at least ``chunk_size`` cells -- otherwise construction raises.

    Parameters
    ----------
    collection
        An :class:`annbatch.DatasetCollection` or a path to one (cells sorted by condition).
    grouped_distribution
        The :class:`~scaleflow.data.GroupedDistribution` (or a split of one) describing the
        sampleable target conditions and their source mapping / embeddings.
    batch_size
        Number of cells per emitted batch (source and target alike).
    chunk_size
        ClassSampler read-slice size. Must be <= every sampled condition's cell count and
        ``chunk_size * preload_nchunks`` must be divisible by ``batch_size``.
    preload_nchunks
        Number of chunks loaded per ClassSampler window. ``None`` auto-picks a small valid
        value (a multiple of ``batch_size // gcd(chunk_size, batch_size)``).
    seed
        Seed for the (independent) class-selection and source-cell RNG streams.
    weights
        Optional sampling weight per sampleable target condition (in ``_tgt_idx_order``
        order); uniform if ``None``. Non-negative; renormalized by ClassSampler.
    condition_transform
        Optional callable applied to each condition dict before it is returned.

    Yields the standard batch contract: ``{"src_cell_data", "tgt_cell_data", "condition"}``.
    """

    def __init__(
        self,
        collection: Any,
        grouped_distribution: GroupedDistribution,
        *,
        batch_size: int = 1024,
        chunk_size: int,
        preload_nchunks: int | None = None,
        seed: int = 0,
        weights: np.ndarray | None = None,
        condition_transform=None,
        n_batches_per_pass: int = 4096,
    ) -> None:
        self._collection = collection
        self._gd = grouped_distribution
        self.batch_size = batch_size
        self._chunk_size = int(chunk_size)
        self._preload_nchunks = preload_nchunks
        self._seed = seed
        self._condition_transform = condition_transform
        self._n_batches_per_pass = int(n_batches_per_pass)

        data = grouped_distribution.data
        self._conditions = data.conditions
        self._src_dist_to_rows = data.src_dist_to_rows
        self._tgt_dist_to_rows = {int(k): np.asarray(v) for k, v in data.tgt_dist_to_rows.items()}

        # reverse map tgt -> src (each target distribution has exactly one source)
        self._tgt_to_src: dict[int, int] = {}
        for s, tgts in data.src_to_tgt_dist_map.items():
            for t in tgts:
                self._tgt_to_src[int(t)] = int(s)

        # conditions we can actually sample: have a source, rows, and an embedding
        self._tgt_idx_order = sorted(
            t
            for t in self._tgt_dist_to_rows
            if t in self._tgt_to_src and t in self._conditions and len(self._tgt_dist_to_rows[t]) > 0
        )
        if not self._tgt_idx_order:
            raise ValueError("No sampleable target distributions (need a mapped source, rows, and a condition).")

        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.shape[0] != len(self._tgt_idx_order):
                raise ValueError("weights must have one entry per sampleable target distribution.")
            if (weights < 0).any():
                raise ValueError("weights must be non-negative.")
        self._weights = weights

        self._initialized = False

    def _resolve_preload_nchunks(self) -> int:
        """Pick (or validate) a ``preload_nchunks`` that tiles ``batch_size`` cleanly."""
        if self._preload_nchunks is not None:
            return int(self._preload_nchunks)
        # group_chunks = batch_size // gcd(chunk, batch); chunk*group_chunks == lcm, a clean
        # multiple of batch_size. A few groups per window lets ClassSampler pack several
        # classes into one read while keeping divisibility.
        group_chunks = self.batch_size // math.gcd(self._chunk_size, self.batch_size)
        return group_chunks * 4

    def init_sampler(self) -> None:
        if self._initialized:
            raise ValueError("Sampler already initialized. Call init_sampler() only once.")
        coll = _open_collection(self._collection)

        # in-memory source cache (one-time reads; scattered rows are fine here)
        self._source_cache = SourceCache(coll, self._src_dist_to_rows)

        # independent reproducible rng streams: class selection (ClassSampler), source cells
        cls_ss, src_ss = np.random.SeedSequence(self._seed).spawn(2)
        self._src_cell_rng = np.random.default_rng(src_ss)

        # total obs in the collection. old_obs_index spans every collection row and is shared
        # by all splits, so this is the loader's n_obs without touching the store.
        n_obs = int(len(self._gd.annotation.old_obs_index))

        # hard-error: every *sampled* condition (positive weight) must hold at least one full
        # chunk. Zero-weighted conditions are excluded by ClassSampler and exempt from the rule.
        if self._weights is None:
            sampled = set(self._tgt_idx_order)
        else:
            sampled = {t for t, w in zip(self._tgt_idx_order, self._weights, strict=True) if w > 0}
        too_small = [
            t for t in self._tgt_idx_order if t in sampled and len(self._tgt_dist_to_rows[t]) < self._chunk_size
        ]
        if too_small:
            labels = "; ".join("(" + ", ".join(_cond_key(self._gd.annotation, t)) + ")" for t in too_small)
            raise ValueError(
                f"{len(too_small)} target condition(s) have fewer than chunk_size={self._chunk_size} cells, "
                f"which annbatch.ClassSampler cannot sample (run-length rule): {labels}. "
                "Lower chunk_size, or remove these conditions from the GroupedDistribution."
            )

        # classes over all obs: sampleable targets -> their own category, everything else
        # (controls + non-split targets) -> the excluded sentinel (weight 0).
        categories = [str(t) for t in self._tgt_idx_order] + [_EXCLUDED_CLASS]
        excluded_code = len(self._tgt_idx_order)
        codes = np.full(n_obs, excluded_code, dtype=np.int64)
        self._row_to_tgt = np.full(n_obs, -1, dtype=np.int64)
        for code, t in enumerate(self._tgt_idx_order):
            rows = self._tgt_dist_to_rows[t]
            codes[rows] = code
            self._row_to_tgt[rows] = t
        classes = pd.Categorical.from_codes(codes, categories=categories)

        # class weights aligned to categories; sentinel -> 0 (excluded)
        tgt_w = np.ones(len(self._tgt_idx_order), dtype=float) if self._weights is None else self._weights
        class_weights = np.concatenate([np.asarray(tgt_w, dtype=float), [0.0]])

        self._preload_nchunks = self._resolve_preload_nchunks()
        # num_samples bounds each loader pass (ClassSampler precomputes ~num_samples/chunk_size
        # eagerly). sample() restarts the iterator on exhaustion for an effectively infinite,
        # reproducible stream; this is internal chunking and does not change the distribution.
        num_samples = self._n_batches_per_pass * self.batch_size
        class_sampler = ClassSampler(
            chunk_size=self._chunk_size,
            preload_nchunks=self._preload_nchunks,
            batch_size=self.batch_size,
            classes=classes,
            num_samples=num_samples,
            class_weights=class_weights,
            drop_last=True,  # every emitted batch is exactly batch_size (matches source n)
            rng=np.random.default_rng(cls_ss),
        )
        loader = Loader(batch_sampler=class_sampler, return_index=True, to_torch=False, preload_to_gpu=False)
        self._loader = loader.use_collection(coll)
        self._iter = iter(self._loader)
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def sample(self) -> dict[str, Any]:
        # Lazy init: callers never have to remember init_sampler(). The trainer's
        # validation path still calls init_sampler()/initialized explicitly, which
        # stays compatible (init only runs once).
        if not self._initialized:
            self.init_sampler()
        try:
            batch = next(self._iter)
        except StopIteration:
            # one ClassSampler pass exhausted; restart for the next block of batches
            self._iter = iter(self._loader)
            batch = next(self._iter)

        # ClassSampler batches are single-class: any row identifies the whole batch's condition
        c_t = int(self._row_to_tgt[int(np.asarray(batch["index"])[0])])
        src_idx = self._tgt_to_src[c_t]
        src_batch = self._source_cache.sample(src_idx, self.batch_size, self._src_cell_rng)
        cond = self._conditions[c_t]
        if self._condition_transform is not None:
            cond = self._condition_transform(cond)
        return {
            "src_cell_data": src_batch,
            "tgt_cell_data": batch["X"],
            "condition": cond,
        }

    @property
    def data(self) -> GroupedDistribution:
        return self._gd


class ValidationSampler(SamplerABC):
    """Validation sampler returning per-condition source/target cells and conditions.

    Caches the source and target cells of a (validation) :class:`GroupedDistribution`
    from the collection, then yields::

        {"source": {cond_key: cells}, "condition": {cond_key: cond_dict}, "target": {cond_key: cells}}

    keyed by condition label tuple. This matches the trainer's per-condition prediction
    (``jax.tree.map`` over conditions, where each ``cond_dict`` is treated as a leaf).
    """

    def __init__(
        self,
        collection: Any,
        data: GroupedDistribution,
        *,
        n_conditions_on_log_iteration: int | None = None,
        n_conditions_on_train_end: int | None = None,
        seed: int = 0,
        condition_transform=None,
    ) -> None:
        self._collection = collection
        self._data = data
        self._n_log = n_conditions_on_log_iteration
        self._n_end = n_conditions_on_train_end
        self._rng = np.random.default_rng(seed)
        self._condition_transform = condition_transform
        self._initialized = False
        self._tgt_to_src = {int(t): int(s) for s, ts in data.data.src_to_tgt_dist_map.items() for t in ts}

    def init_sampler(self) -> None:
        if self._initialized:
            return
        gdd = self._data.data
        self._src_cells = _read_distributions(self._collection, gdd.src_dist_to_rows)
        self._tgt_cells = _read_distributions(self._collection, gdd.tgt_dist_to_rows)
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def sample(self, mode: str = "on_log_iteration") -> dict[str, dict[tuple[str, ...], Any]]:
        if not self._initialized:
            self.init_sampler()
        candidates = [t for t in self._data.data.conditions if t in self._tgt_to_src and t in self._tgt_cells]
        n_total = len(candidates)
        n = self._n_log if mode == "on_log_iteration" else self._n_end
        n = n_total if n is None else min(int(n), n_total)
        if 0 < n < n_total:
            selected = self._rng.choice(np.asarray(candidates), size=n, replace=False).tolist()
        else:
            selected = candidates

        source: dict[tuple[str, ...], Any] = {}
        condition: dict[tuple[str, ...], Any] = {}
        target: dict[tuple[str, ...], Any] = {}
        for t in selected:
            t = int(t)
            key = _cond_key(self._data.annotation, t)
            source[key] = self._src_cells[self._tgt_to_src[t]]
            target[key] = self._tgt_cells[t]
            cond = self._data.data.conditions[t]
            if self._condition_transform is not None:
                cond = self._condition_transform(cond)
            condition[key] = cond
        return {"source": source, "condition": condition, "target": target}

    @property
    def data(self) -> GroupedDistribution:
        return self._data


class PredictionSampler(SamplerABC):
    """Prediction sampler returning per-condition source cells and conditions (no targets).

    Yields ``{"source": {cond_key: cells}, "condition": {cond_key: cond_dict}}`` for every
    target condition in the :class:`GroupedDistribution` that has a mapped source.
    """

    def __init__(self, collection: Any, data: GroupedDistribution, *, condition_transform=None) -> None:
        self._collection = collection
        self._data = data
        self._condition_transform = condition_transform
        self._initialized = False
        self._tgt_to_src = {int(t): int(s) for s, ts in data.data.src_to_tgt_dist_map.items() for t in ts}

    def init_sampler(self) -> None:
        if self._initialized:
            return
        self._src_cells = _read_distributions(self._collection, self._data.data.src_dist_to_rows)
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def sample(self) -> dict[str, dict[tuple[str, ...], Any]]:
        if not self._initialized:
            self.init_sampler()
        source: dict[tuple[str, ...], Any] = {}
        condition: dict[tuple[str, ...], Any] = {}
        for t in self._data.data.conditions:
            t = int(t)
            if t not in self._tgt_to_src:
                continue
            key = _cond_key(self._data.annotation, t)
            source[key] = self._src_cells[self._tgt_to_src[t]]
            cond = self._data.data.conditions[t]
            if self._condition_transform is not None:
                cond = self._condition_transform(cond)
            condition[key] = cond
        return {"source": source, "condition": condition}

    @property
    def data(self) -> GroupedDistribution:
        return self._data


def write_sorted_collection(
    adata: Any,
    collection_path: str,
    *,
    dist_flag_key: str,
    src_dist_keys: list[str],
    tgt_dist_keys: list[str] | dict[str, list[str]],
    sorted_adata_path: str | None = None,
    **add_adatas_kwargs: Any,
) -> str:
    """Write ``adata`` to an :class:`annbatch.DatasetCollection` sorted by condition.

    Sorts ``adata`` by ``[dist_flag_key, *src_dist_keys, *tgt_dist_keys]`` (the same key
    order :class:`~scaleflow.data.DataManager` uses to group cells into distributions) so
    that every ``(src, tgt)`` condition becomes a single contiguous run on disk, then adds
    it to the collection unshuffled. This is the layout
    :class:`GroupedAnnbatchSampler` / :class:`annbatch.samplers.ClassSampler` require.

    Parameters
    ----------
    adata
        In-memory :class:`anndata.AnnData` to write.
    collection_path
        Path of the :class:`annbatch.DatasetCollection` to create/append to.
    dist_flag_key, src_dist_keys, tgt_dist_keys
        The distribution keys (must match the :class:`~scaleflow.data.DataManager` config).
        ``tgt_dist_keys`` may be the grouped (combination) form ``{group: [cols]}``; it is
        flattened to its columns for sorting.
    sorted_adata_path
        Where to write the intermediate sorted AnnData (a zarr store). Defaults to
        ``f"{collection_path}.sorted_adata.zarr"``.
    add_adatas_kwargs
        Extra keyword arguments forwarded to
        :meth:`annbatch.DatasetCollection.add_adatas` (``shuffle`` is forced to ``False``).

    Returns
    -------
    The ``collection_path`` written to.
    """
    from annbatch import DatasetCollection

    tgt_cols = (
        [c for cols in tgt_dist_keys.values() for c in cols]
        if isinstance(tgt_dist_keys, dict)
        else list(tgt_dist_keys)
    )
    sort_cols = [dist_flag_key, *src_dist_keys, *tgt_cols]
    order = adata.obs.sort_values(sort_cols, kind="stable").index
    adata_sorted = adata[order].copy()

    if sorted_adata_path is None:
        sorted_adata_path = f"{collection_path}.sorted_adata.zarr"
    adata_sorted.write_zarr(str(sorted_adata_path))

    coll = DatasetCollection(str(collection_path), mode="a")
    coll.add_adatas(adata_paths=[str(sorted_adata_path)], **{"shuffle": False, **add_adatas_kwargs})
    return str(collection_path)
