"""annbatch-backed training sampler for GroupedDistribution.

Streams *target* (perturbed) cells from an :class:`annbatch.DatasetCollection`, one
batch per step, while keeping the *source* (control) populations in an in-memory cache.

Why asymmetric? Control populations are few and reused by every condition, so caching
them is cheap and avoids re-reading them from disk every step; the many target
conditions are streamed lazily. This also keeps a single annbatch ``Loader`` emitting
exactly one batch per request (no src/tgt split-pairing).

ASSUMPTION: all source/control cells fit in memory. There is intentionally no fallback;
if they do not fit, this sampler is not the right tool.

Sparsity is preserved end to end: if the collection stores ``X`` as sparse CSR, both the
streamed target batch and the cached source batch come out as ``scipy.sparse`` (nothing
is densified here). Densifying, if needed, is a model-boundary decision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from annbatch import Loader
from annbatch.abc import Sampler

from scaleflow.data._data import GroupedDistribution
from scaleflow.data._dataloader import SamplerABC

__all__ = ["GroupedAnnbatchSampler", "PredictionSampler", "SourceCache", "ValidationSampler"]


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

    Per step: a category RNG picks a target distribution (condition) ``c_t`` (uniform
    over conditions by default); annbatch streams ``batch_size`` target cells for ``c_t``;
    the corresponding source population (``tgt -> src`` via ``src_to_tgt_dist_map``) is
    sampled from the in-memory :class:`SourceCache`; and ``conditions[c_t]`` is attached.

    The category, target-cell, and source-cell RNGs are independent streams spawned from
    one seed, so the full (source, target) pairing sequence is reproducible.

    Yields the standard batch contract: ``{"src_cell_data", "tgt_cell_data", "condition"}``.
    """

    def __init__(
        self,
        collection: Any,
        grouped_distribution: GroupedDistribution,
        *,
        batch_size: int = 1024,
        seed: int = 0,
        weights: np.ndarray | None = None,
        condition_transform=None,
    ) -> None:
        self._collection = collection
        self._gd = grouped_distribution
        self.batch_size = batch_size
        self._seed = seed
        self._condition_transform = condition_transform

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
            weights = weights / weights.sum()
        self._weights = weights

        self._initialized = False
        # the target sampler records the c_t it emitted, consumed in lockstep by sample()
        self._schedule: list[int] = []
        self._step = 0

    def init_sampler(self) -> None:
        if self._initialized:
            raise ValueError("Sampler already initialized. Call init_sampler() only once.")
        coll = _open_collection(self._collection)

        # in-memory source cache
        self._source_cache = SourceCache(coll, self._src_dist_to_rows)

        # independent reproducible rng streams: category, target cells, source cells
        cat_ss, tgt_ss, src_ss = np.random.SeedSequence(self._seed).spawn(3)
        self._cat_rng = np.random.default_rng(cat_ss)
        self._tgt_cell_rng = np.random.default_rng(tgt_ss)
        self._src_cell_rng = np.random.default_rng(src_ss)

        tgt_idxs = np.asarray(self._tgt_idx_order)
        bs = self.batch_size

        def _requests():
            # Infinite stream of target requests. The annbatch Loader processes requests
            # sequentially (one split each), so the t-th yielded batch corresponds to the
            # t-th c_t recorded here — sample() reads self._schedule in lockstep.
            while True:
                c_t = int(self._cat_rng.choice(tgt_idxs, p=self._weights))
                rows = self._tgt_dist_to_rows[c_t]
                sel = self._tgt_cell_rng.integers(0, rows.shape[0], size=bs)
                self._schedule.append(c_t)
                yield {"requests": np.asarray(rows[sel], dtype=np.int64), "splits": [np.arange(bs)]}

        sampler = _ExplicitRequestSampler(_requests(), batch_size=bs, n_batches=2**31 - 1)
        loader = Loader(batch_sampler=sampler, return_index=False, to_torch=False, preload_to_gpu=False)
        loader = loader.use_collection(coll)
        self._loader = loader
        self._iter = iter(loader)
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
        batch = next(self._iter)
        # lockstep with the target request stream (one batch per emitted request)
        c_t = self._schedule[self._step]
        self._step += 1

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
