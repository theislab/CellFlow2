"""Cell-source abstraction over annbatch — the scaleflow-contributed half of the shared layer.

A ``container`` is the only cell store: an in-memory :class:`~anndata.AnnData` (streamed via
``annbatch.Loader.add_adata``) or an on-disk :class:`annbatch.DatasetCollection` (streamed via
``use_collection``). ``use_collection`` is itself sugar over ``add_adatas`` in annbatch, so both
paths share all downstream logic. Reads of explicit row ranges are slice-based: a contiguous
``slice`` is a view (dense) / efficient CSR row-slice in memory, or a single contiguous annbatch
request on disk — no fancy-index copies.

Depends only on annbatch + numpy + anndata (no scaleflow imports) so it is extraction-ready.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from annbatch import Loader
from annbatch.abc import Sampler

__all__ = ["LoaderSource", "as_source", "make_loader", "read_distributions"]


def _open_collection(collection: Any) -> Any:
    from annbatch import DatasetCollection

    if isinstance(collection, DatasetCollection):
        return collection
    return DatasetCollection(str(collection), mode="r")


class LoaderSource:
    """Strategy for populating a fresh annbatch :class:`~annbatch.Loader` and reading rows."""

    #: in-memory sources sample individual rows ⇒ ClassSampler must use ``chunk_size=1``.
    forces_chunk_size_one: bool = False

    def attach(self, loader: Loader) -> Loader:  # pragma: no cover - interface
        raise NotImplementedError

    def read_rows(self, dist_to_rows: dict[int, slice | np.ndarray]) -> dict[int, Any]:  # pragma: no cover
        raise NotImplementedError


class CollectionSource(LoaderSource):
    """On-disk source: an :class:`annbatch.DatasetCollection` (or a path to one)."""

    def __init__(self, collection: Any) -> None:
        self._collection = collection

    def attach(self, loader: Loader) -> Loader:
        return loader.use_collection(_open_collection(self._collection))

    def read_rows(self, dist_to_rows: dict[int, slice | np.ndarray]) -> dict[int, Any]:
        order = sorted(int(k) for k in dist_to_rows)
        if not order:
            return {}

        def _request(rows: slice | np.ndarray) -> dict[str, Any]:
            if isinstance(rows, slice):
                return {"requests": [rows], "splits": [np.arange(rows.stop - rows.start)]}
            rows = np.asarray(rows, dtype=np.int64)
            return {"requests": rows, "splits": [np.arange(len(rows))]}

        sampler = _ExplicitRequestSampler(
            iter([_request(dist_to_rows[k]) for k in order]), batch_size=None, n_batches=len(order)
        )
        loader = make_loader(self, batch_sampler=sampler, return_index=False)
        return {k: batch["X"] for k, batch in zip(order, loader, strict=True)}


class InMemorySource(LoaderSource):
    """In-memory source wrapping a single :class:`~anndata.AnnData` (sorted by condition)."""

    forces_chunk_size_one = True

    def __init__(self, adata: Any) -> None:
        self.adata = adata

    def attach(self, loader: Loader) -> Loader:
        return loader.add_adata(self.adata)

    def read_rows(self, dist_to_rows: dict[int, slice | np.ndarray]) -> dict[int, Any]:
        x = self.adata.X
        return {int(k): x[rows] for k, rows in dist_to_rows.items()}  # slice → view / CSR row-slice


def as_source(container: Any) -> LoaderSource:
    """Normalize a ``container`` into a :class:`LoaderSource` (AnnData → in-memory, else on-disk)."""
    import anndata as ad

    if isinstance(container, LoaderSource):
        return container
    if isinstance(container, ad.AnnData):
        return InMemorySource(container)
    return CollectionSource(container)


def make_loader(source: LoaderSource, *, batch_sampler: Sampler, return_index: bool) -> Loader:
    """Build a fresh annbatch ``Loader`` for ``source`` with the given batch sampler."""
    loader = Loader(batch_sampler=batch_sampler, return_index=return_index, to_torch=False, preload_to_gpu=False)
    return source.attach(loader)


def read_distributions(container: Any, dist_to_rows: dict[int, slice | np.ndarray]) -> dict[int, Any]:
    """Read each distribution's ``X`` rows into memory (sparse stays sparse)."""
    return as_source(container).read_rows(dist_to_rows)


class _ExplicitRequestSampler(Sampler):
    """Minimal annbatch sampler yielding caller-provided requests verbatim (one batch each)."""

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
