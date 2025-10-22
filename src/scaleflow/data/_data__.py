from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr
from zarr.storage import LocalStore

from scaleflow._types import ArrayLike
from scaleflow.data._utils import write_sharded

__all__ = [
    "MappedCellData",
    "PairedData",
]



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

@dataclass
class MappedCellData(BaseDataMixin):
    """Lazy, Zarr-backed variant of :class:`TrainingData`.

    Fields mirror those in :class:`TrainingData`, but array-like members are
    Zarr arrays or Zarr-backed mappings. This enables out-of-core training and
    composition without loading everything into memory.

    Use :meth:`read_zarr` to construct from a Zarr v3 group written via
    :meth:`TrainingData.to_zarr`.
    """

    # Note: annotations use Any to allow zarr.Array and zarr groups without
    # importing zarr at module import time.
    src_cell_data: dict[str, Any]
    tgt_cell_data: dict[str, Any]
    src_cell_idx: dict[str, Any]
    tgt_cell_idx: dict[str, Any]
    split_covariates_mask: Any
    perturbation_covariates_mask: Any
    split_idx_to_covariates: dict[int, tuple[Any, ...]]
    perturbation_idx_to_covariates: dict[int, tuple[str, ...]]
    perturbation_idx_to_id: dict[int, Any]
    condition_data: dict[str, Any]
    control_to_perturbation: dict[int, Any]
    max_combination_length: int
    mapping_data_full_cached: bool = False

    def __post_init__(self):
        # load everything except cell_data to memory

        # load masks as numpy arrays
        self.condition_data = {k: np.asarray(v) for k, v in self.condition_data.items()}
        self.control_to_perturbation = {int(k): np.asarray(v) for k, v in self.control_to_perturbation.items()}
        if self.mapping_data_full_cached:
            # used in validation usually
            self.perturbation_idx_to_id = {int(k): np.asarray(v) for k, v in self.perturbation_idx_to_id.items()}
            self.perturbation_idx_to_covariates = {
                int(k): np.asarray(v) for k, v in self.perturbation_idx_to_covariates.items()
            }
            # not used in nested structure
            self.src_cell_idx = self.src_cell_idx[...]
            self.tgt_cell_idx = self.tgt_cell_idx[...]
            self.split_covariates_mask = self.split_covariates_mask[...]
            self.perturbation_covariates_mask = self.perturbation_covariates_mask[...]
            self.split_idx_to_covariates = {int(k): np.asarray(v) for k, v in self.split_idx_to_covariates.items()}

    @staticmethod
    def _get_mapping_data(group: zarr.Group) -> dict[str, Any]:
        return group["mapping_data"]["mapping_data"]

    @staticmethod
    def _read_dict(zgroup: zarr.Group, key: str) -> dict[int, Any]:
        keys = zgroup[key].keys()
        return {k: zgroup[key][k] for k in keys}

    @staticmethod
    def _read_cell_data(zgroup: zarr.Group, key: str) -> dict[int, Any]:
        keys = sorted(zgroup[key].keys())
        data_key = [k for k in keys if not k.endswith("_index")]
        return {int(k): zgroup[key][k] for k in data_key}, {int(k): zgroup[key][f"{k}_index"] for k in data_key}

    @classmethod
    def read_zarr(cls, path: str) -> MappedCellData:
        if isinstance(path, str):
            path = LocalStore(path, read_only=True)
        group = zarr.open_group(path, mode="r")
        max_len_node = group.get("max_combination_length")
        if max_len_node is None:
            max_combination_length = 0
        else:
            try:
                max_combination_length = int(max_len_node[()])
            except Exception:  # noqa: BLE001
                max_combination_length = int(max_len_node)

        mapping_group = cls._get_mapping_data(group)

        src_cell_data, src_cell_idx = cls._read_cell_data(group, "src_cell_data")
        tgt_cell_data, tgt_cell_idx = cls._read_cell_data(group, "tgt_cell_data")
        return cls(
            tgt_cell_data=tgt_cell_data,
            tgt_cell_idx=tgt_cell_idx,
            src_cell_data=src_cell_data,
            src_cell_idx=src_cell_idx,
            split_covariates_mask=mapping_group["split_covariates_mask"],
            perturbation_covariates_mask=mapping_group["perturbation_covariates_mask"],
            split_idx_to_covariates=cls._read_dict(mapping_group, "split_idx_to_covariates"),
            perturbation_idx_to_covariates=cls._read_dict(mapping_group, "perturbation_idx_to_covariates"),
            perturbation_idx_to_id=cls._read_dict(mapping_group, "perturbation_idx_to_id"),
            condition_data=cls._read_dict(mapping_group, "condition_data"),
            control_to_perturbation=cls._read_dict(mapping_group, "control_to_perturbation"),
            max_combination_length=max_combination_length,
        )



@dataclass
class PairedData:
    """Base class for data containers."""

    src_dist_to_tgt_dists: dict[int, list[int]]
    src_idx_to_labels: dict[int, Any]
    tgt_idx_to_labels: dict[int, Any]

    @property
    def n_source_dists(self) -> int:
        """Returns the number of source distributions."""
        return len(self.src_idx_to_labels)  # type: ignore[attr-defined]

    @property
    def n_target_dists(self) -> int:
        """Returns the number of target distributions."""
        return len(self.tgt_idx_to_labels)  # type: ignore[attr-defined]

    def _format_params(self, fmt: Callable[[Any], str]) -> str:
        params = {
            "n_source_dists": self.n_source_dists,
            "n_target_dists": self.n_target_dists,
        }
        return ", ".join(f"{name}={fmt(val)}" for name, val in params.items())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._format_params(repr)}]"
