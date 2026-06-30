from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from scaleflow.data._anndata_location import AnnDataLocation

__all__ = [
    "GroupedDistribution",
]

# Where the grouped-distribution metadata lives on a prepared AnnData. The per-row dist-id
# columns go in ``obs`` (they ARE the ClassSampler `classes` input); everything else lives in
# ``uns`` so a single AnnData (X + obs + uns) is a self-contained, reloadable prepared dataset.
OBS_TGT_DIST_IDX = "scaleflow_tgt_dist_idx"
OBS_SRC_DIST_IDX = "scaleflow_src_dist_idx"
UNS_KEY = "scaleflow"


def _json_default(o: Any) -> Any:
    """JSON encoder fallback for the numpy scalar/array types in the small metadata."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


@dataclass
class ReturnData:  # TODO: this should rather be a NamedTuple
    split_covariates_mask: np.ndarray | None
    split_idx_to_covariates: dict[int, tuple[Any, ...]]
    perturbation_covariates_mask: np.ndarray | None
    perturbation_idx_to_covariates: dict[int, tuple[Any, ...]]
    perturbation_idx_to_id: dict[int, Any]
    condition_data: dict[str, np.ndarray]
    control_to_perturbation: dict[int, np.ndarray]
    max_combination_length: int


class BaseDataMixin:
    """Base class for data containers."""

    @property
    def n_controls(self) -> int:
        """Returns the number of control covariate values."""
        return len(self.split_idx_to_covariates)  # type: ignore[attr-defined]

    @property
    def n_perturbations(self) -> int:
        """Returns the number of perturbation covariate combinations."""
        return len(self.perturbation_idx_to_covariates)  # type: ignore[attr-defined]

    @property
    def n_perturbation_covariates(self) -> int:
        """Returns the number of perturbation covariates."""
        return len(self.condition_data)  # type: ignore[attr-defined]

    def _format_params(self, fmt: Callable[[Any], str]) -> str:
        params = {
            "n_controls": self.n_controls,
            "n_perturbations": self.n_perturbations,
        }
        return ", ".join(f"{name}={fmt(val)}" for name, val in params.items())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._format_params(repr)}]"


@dataclass(eq=False)
class GroupedDistribution:
    """Typed *view* over a prepared :class:`~anndata.AnnData` — the single source of truth.

    The wrapped AnnData carries the per-row distribution assignment in ``obs``
    (``scaleflow_tgt_dist_idx`` / ``scaleflow_src_dist_idx``) and everything else under
    ``uns["scaleflow"]`` (conditions, source↔target pairing, label maps, keys, ...). Nothing is
    duplicated: the typed metadata is materialized **once** in :meth:`__post_init__` for fast
    repeated access, but the AnnData remains the canonical store and the serialization format.

    Two flavors of the wrapped AnnData:

    - in-memory training: the full sorted dataset (``X`` + obs + uns); the same object also backs
      the in-memory annbatch source, so there is a single object and no copy.
    - on-disk training: a metadata-only AnnData (no ``X``, ``n_obs == collection rows``) while the
      cells stream from the :class:`annbatch.DatasetCollection` passed separately to the samplers.

    ``row_tgt_dist_idx`` is exactly the per-row ``classes`` array
    :class:`~annbatch.samplers.ClassSampler` consumes; per-condition rows for the few explicit-read
    paths (source cache, validation, prediction) are derived on demand via :meth:`rows_for`.
    """

    adata: ad.AnnData

    # materialized once from the AnnData in __post_init__ (not __init__ args)
    row_tgt_dist_idx: np.ndarray = field(init=False, repr=False)
    row_src_dist_idx: np.ndarray = field(init=False, repr=False)
    conditions: dict[int, dict[str, np.ndarray]] = field(init=False, repr=False)
    src_to_tgt_dist_map: dict[int, list[int]] = field(init=False, repr=False)
    src_dist_idx_to_labels: dict[int, tuple] = field(init=False, repr=False)
    tgt_dist_idx_to_labels: dict[int, tuple] = field(init=False, repr=False)
    src_tgt_dist_df: pd.DataFrame = field(init=False, repr=False)
    default_values: dict[str, Any] = field(init=False, repr=False)
    tgt_dist_keys: list[str] = field(init=False, repr=False)
    src_dist_keys: list[str] = field(init=False, repr=False)
    dist_flag_key: str = field(init=False, repr=False)
    data_location: AnnDataLocation | None = field(init=False, repr=False)
    old_obs_index: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if UNS_KEY not in self.adata.uns:
            raise KeyError(
                f"adata.uns has no {UNS_KEY!r} entry; wrap an AnnData produced by "
                "GroupedDistribution.from_parts()/to_adata()."
            )
        obs = self.adata.obs
        store = self.adata.uns[UNS_KEY]
        meta = json.loads(store["meta_json"])

        self.row_tgt_dist_idx = np.asarray(obs[OBS_TGT_DIST_IDX])
        self.row_src_dist_idx = np.asarray(obs[OBS_SRC_DIST_IDX])
        self.conditions = {
            int(t): {c: np.asarray(a) for c, a in cond.items()} for t, cond in store["conditions"].items()
        }
        sd = store["src_tgt_dist_df"]
        self.src_tgt_dist_df = sd if isinstance(sd, pd.DataFrame) else pd.DataFrame(sd)
        self.src_to_tgt_dist_map = {int(k): [int(t) for t in v] for k, v in meta["src_to_tgt_dist_map"].items()}
        self.src_dist_idx_to_labels = {int(k): tuple(v) for k, v in meta["src_dist_idx_to_labels"].items()}
        self.tgt_dist_idx_to_labels = {int(k): tuple(v) for k, v in meta["tgt_dist_idx_to_labels"].items()}
        self.default_values = meta["default_values"]
        self.tgt_dist_keys = list(meta["tgt_dist_keys"])
        self.src_dist_keys = list(meta["src_dist_keys"])
        self.dist_flag_key = meta["dist_flag_key"]
        self.data_location = (
            AnnDataLocation.from_json(meta["data_location"]) if meta.get("data_location") is not None else None
        )
        self.old_obs_index = self.adata.obs_names.to_numpy()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_parts(
        cls,
        adata: ad.AnnData | None,
        *,
        row_tgt_dist_idx: np.ndarray,
        row_src_dist_idx: np.ndarray,
        conditions: dict[int, dict[str, np.ndarray]],
        src_to_tgt_dist_map: dict[int, list[int]],
        src_dist_idx_to_labels: dict[int, Any],
        tgt_dist_idx_to_labels: dict[int, Any],
        src_tgt_dist_df: pd.DataFrame,
        default_values: dict[str, Any],
        tgt_dist_keys: list[str],
        src_dist_keys: list[str],
        dist_flag_key: str,
        data_location: AnnDataLocation | None = None,
        old_obs_index: np.ndarray | None = None,
    ) -> GroupedDistribution:
        """Build a :class:`GroupedDistribution` by writing the metadata onto an AnnData.

        ``adata`` is the prepared (sorted) dataset to attach to (its ``obs``/``uns`` are written
        in place). Pass ``None`` to build a metadata-only AnnData (no ``X``) — e.g. the on-disk
        sidecar or a split — whose obs index is ``old_obs_index``.
        """
        n_obs = int(np.asarray(row_tgt_dist_idx).shape[0])
        if adata is None:
            idx = np.asarray(old_obs_index) if old_obs_index is not None else np.arange(n_obs)
            adata = ad.AnnData(obs=pd.DataFrame(index=idx.astype(str)))
        if adata.n_obs != n_obs:
            raise ValueError(
                f"adata.n_obs ({adata.n_obs}) does not match the per-row dist assignment length ({n_obs})."
            )

        adata.obs[OBS_TGT_DIST_IDX] = np.asarray(row_tgt_dist_idx)
        adata.obs[OBS_SRC_DIST_IDX] = np.asarray(row_src_dist_idx)
        meta = {
            "src_to_tgt_dist_map": {str(k): [int(t) for t in v] for k, v in src_to_tgt_dist_map.items()},
            "src_dist_idx_to_labels": {str(k): list(v) for k, v in src_dist_idx_to_labels.items()},
            "tgt_dist_idx_to_labels": {str(k): list(v) for k, v in tgt_dist_idx_to_labels.items()},
            "default_values": default_values,
            "tgt_dist_keys": list(tgt_dist_keys),
            "src_dist_keys": list(src_dist_keys),
            "dist_flag_key": dist_flag_key,
            "data_location": data_location.to_json() if data_location is not None else None,
        }
        adata.uns[UNS_KEY] = {
            "meta_json": json.dumps(meta, default=_json_default),
            # conditions are arrays -> stored natively (anndata handles nested dict[str, dict[str, array]])
            "conditions": {str(t): {c: np.asarray(a) for c, a in cond.items()} for t, cond in conditions.items()},
            "src_tgt_dist_df": src_tgt_dist_df,
        }
        return cls(adata)

    @staticmethod
    def rows_for(row_dist_idx: np.ndarray) -> dict[int, slice | np.ndarray]:
        """Invert a per-row dist-id column into ``{dist_idx: rows}`` (excludes -1).

        The transient, derive-on-demand replacement for storing ``{dist: [rows]}`` maps; call it
        only where explicit row reads are needed. When a dist's rows are a contiguous run (the
        norm: training data is sorted by condition) it is returned as a :class:`slice` so reads are
        view-based array slices rather than fancy-index copies; otherwise a row-index array.
        """
        col = np.asarray(row_dist_idx)
        out: dict[int, slice | np.ndarray] = {}
        for k in np.unique(col[col >= 0]):
            idx = np.flatnonzero(col == k)
            if idx.size and (idx[-1] - idx[0] + 1) == idx.size:
                out[int(k)] = slice(int(idx[0]), int(idx[-1]) + 1)
            else:
                out[int(k)] = idx
        return out

    # ------------------------------------------------------------------ AnnData (de)serialization
    @classmethod
    def from_adata(cls, adata: ad.AnnData) -> GroupedDistribution:
        """Wrap an AnnData previously enriched by :meth:`from_parts` / :meth:`to_adata`."""
        return cls(adata)

    def to_adata(self) -> ad.AnnData:
        """Return the wrapped (metadata-carrying) AnnData."""
        return self.adata

    def write_zarr(self, path: str, **_legacy: Any) -> None:
        """Persist the wrapped AnnData as a zarr store (legacy chunk/shard kwargs are ignored)."""
        ad.settings.zarr_write_format = 3  # sharding support
        self.adata.write_zarr(path)

    @classmethod
    def read_zarr(cls, path: str, in_memory: bool = False) -> GroupedDistribution:
        """Read a :class:`GroupedDistribution` from an AnnData zarr store written by :meth:`write_zarr`."""
        del in_memory  # metadata is always materialized eagerly
        return cls(ad.read_zarr(path))

    @property
    def is_in_memory(self) -> bool:
        """Always :obj:`True` (cell matrices live in the external store; metadata is in memory)."""
        return True

    def to_memory(self) -> None:
        """No-op, kept for backwards compatibility (cells are streamed externally)."""
        return None
