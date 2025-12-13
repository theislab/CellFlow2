from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import zarr

from scaleflow.data._anndata_location import AnnDataLocation
from scaleflow.data._utils import write_dist_data_threaded, write_nested_dist_data, write_sharded
from scaleflow.data.io import CSRLabelMapping

__all__ = [
    "GroupedDistribution",
    "GroupedDistributionData",
    "GroupedDistributionAnnotation",
]


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


@dataclass
class GroupedDistributionData:
    src_to_tgt_dist_map: dict[int, list[int]]  # (n_src_dists) → (n_tgt_dists_{src_dist_idx})
    src_data: dict[int, np.ndarray]  # (n_src_dists) → (n_cells_{src_dist_idx}, n_features)
    tgt_data: dict[int, np.ndarray]  # (n_tgt_dists) → (n_cells_{tgt_dist_idx}, n_features)
    conditions: dict[int, dict[str, np.ndarray]]  # (n_tgt_dists) → {col_name: (n_rows, *dims)}

    @classmethod
    def read_zarr(
        cls,
        group: zarr.Group,
        in_memory: bool = False,
    ) -> GroupedDistributionData:
        """
        Read the grouped distribution data from a Zarr group.

        Parameters
        ----------
        group
            Zarr group containing the data.
        in_memory
            If True, load all arrays into memory as numpy arrays.
            If False (default), keep arrays as lazy zarr arrays.

        Conditions are stored in CSR-like format:
        - Each column is a contiguous array (all dists concatenated)
        - Metadata contains dist_ids and indptr for each column
        """
        # Read conditions from CSR-like structure (always in memory)
        conditions = {}
        if "conditions" in group:
            cond_group = group["conditions"]

            # Get metadata
            dist_ids = list(cond_group.attrs.get("dist_ids", []))

            if dist_ids:
                # Get column names (everything that's not metadata)
                col_names = list(cond_group.keys())

                # Initialize conditions dict for each dist_id
                for dist_id in dist_ids:
                    conditions[dist_id] = {}

                # Read each column and split by indptr
                for col_name in col_names:
                    indptr = cond_group.attrs.get(f"indptr_{col_name}", [])
                    concatenated = np.asarray(cond_group[col_name])

                    # Split the concatenated array for each distribution
                    for i, dist_id in enumerate(dist_ids):
                        start = indptr[i]
                        end = indptr[i + 1]
                        conditions[dist_id][col_name] = concatenated[start:end]

        # Read src_data and tgt_data (optionally load into memory)
        if in_memory:
            src_data = {int(k): group["src_data"][k][...] for k in group["src_data"].keys()}
            tgt_data = {int(k): group["tgt_data"][k][...] for k in group["tgt_data"].keys()}
        else:
            src_data = {int(k): group["src_data"][k] for k in group["src_data"].keys()}
            tgt_data = {int(k): group["tgt_data"][k] for k in group["tgt_data"].keys()}

        return cls(
            src_to_tgt_dist_map={
                int(k): np.array(group["src_to_tgt_dist_map"][k]) for k in group["src_to_tgt_dist_map"].keys()
            },
            src_data=src_data,
            tgt_data=tgt_data,
            conditions=conditions,
        )

    @property
    def is_in_memory(self) -> bool:
        """Check if all data is loaded in memory (numpy arrays) vs lazy (zarr arrays).

        Returns True if all src_data and tgt_data arrays are numpy arrays.
        Returns False if any are zarr arrays (lazy loading).
        """
        if not self.src_data and not self.tgt_data:
            return True  # Empty data is considered in-memory

        # Check first src_data array
        if self.src_data:
            first_src = next(iter(self.src_data.values()))
            if isinstance(first_src, zarr.Array):
                return False

        # Check first tgt_data array
        if self.tgt_data:
            first_tgt = next(iter(self.tgt_data.values()))
            if isinstance(first_tgt, zarr.Array):
                return False

        return True

    def to_memory(self) -> None:
        """Convert all lazy zarr arrays to in-memory numpy arrays (in-place).

        Does nothing if data is already in memory.
        """
        if self.is_in_memory:
            return None

        self.src_data = {int(k): self.src_data[k][...] for k in self.src_data.keys()}
        self.tgt_data = {int(k): self.tgt_data[k][...] for k in self.tgt_data.keys()}
        return None

    def write_zarr_group(
        self,
        group: zarr.Group,
        chunk_size: int,
        shard_size: int,
        max_workers: int,
    ) -> None:
        """Write the grouped distribution data to a Zarr group."""
        data = group.create_group("data")
        write_sharded(
            group=data,
            name="src_to_tgt_dist_map",
            data={str(k): np.array(v) for k, v in self.src_to_tgt_dist_map.items()},
            chunk_size=chunk_size,
            shard_size=shard_size,
            compressors=None,
        )

        # Write src_data and tgt_data using simple writer
        for key in ["src_data", "tgt_data"]:
            sub_group = data.create_group(key)
            value = getattr(self, key)
            write_dist_data_threaded(
                group=sub_group,
                dist_data=value,
                chunk_size=chunk_size,
                shard_size=shard_size,
                max_workers=max_workers,
            )

        conditions_group = data.create_group("conditions")
        write_nested_dist_data(
            group=conditions_group,
            dist_data=self.conditions,
            chunk_size=chunk_size,
            shard_size=shard_size,
        )

        return None


@dataclass
class GroupedDistributionAnnotation:
    old_obs_index: np.ndarray  # (n_cells,) to be able to map back to the original index

    src_dist_idx_to_labels: dict[int, Any]  # (n_src_dists) → Any (e.g. list of strings)
    tgt_dist_idx_to_labels: dict[int, Any]  # (n_tgt_dists) → Any (e.g. list of strings)
    src_tgt_dist_df: pd.DataFrame

    default_values: dict[str, Any]
    tgt_dist_keys: list[str]
    src_dist_keys: list[str]
    dist_flag_key: str
    data_location: AnnDataLocation | None = None  # The location of the data in the AnnData object

    @classmethod
    def read_zarr(
        cls,
        group: zarr.Group,
    ) -> GroupedDistributionAnnotation:
        """Read the grouped distribution annotation from a Zarr group."""
        # Check if this is the new format (has src_dist_idx_to_labels and tgt_dist_idx_to_labels groups)
        assert "src_dist_idx_to_labels" in group and "tgt_dist_idx_to_labels" in group
        # New CSRLabelMapping format - read and convert to dict
        src_mapping = CSRLabelMapping.read_zarr(group["src_dist_idx_to_labels"])
        tgt_mapping = CSRLabelMapping.read_zarr(group["tgt_dist_idx_to_labels"])

        # Read other elements from the annotation group
        elem = ad.io.read_elem(group["metadata"])

        # Handle data_location - may not exist in older zarr files
        data_location = None
        if "data_location" in elem and elem["data_location"] is not None:
            data_location = AnnDataLocation.from_json(elem["data_location"])

        return cls(
            old_obs_index=elem["old_obs_index"],
            src_dist_idx_to_labels=src_mapping.to_dict(),
            tgt_dist_idx_to_labels=tgt_mapping.to_dict(),
            src_tgt_dist_df=elem["src_tgt_dist_df"],
            default_values=elem["default_values"],
            tgt_dist_keys=np.array(elem["tgt_dist_keys"]).tolist(),
            src_dist_keys=np.array(elem["src_dist_keys"]).tolist(),
            dist_flag_key=elem["dist_flag_key"],
            data_location=data_location,
        )

    def write_zarr_group(
        self,
        group: zarr.Group,
        chunk_size: int,
        shard_size: int,
    ) -> None:
        """Write the grouped distribution annotation to a Zarr group."""
        annotation_group = group.create_group("annotation")

        # Convert dicts to CSRLabelMapping for efficient storage
        src_mapping = CSRLabelMapping.from_dict(self.src_dist_idx_to_labels)
        tgt_mapping = CSRLabelMapping.from_dict(self.tgt_dist_idx_to_labels)

        # Write CSRLabelMapping objects
        src_mapping.write_zarr(annotation_group, "src_dist_idx_to_labels")
        tgt_mapping.write_zarr(annotation_group, "tgt_dist_idx_to_labels")

        # Write other metadata using write_sharded
        to_write = {
            "old_obs_index": self.old_obs_index,
            "src_tgt_dist_df": self.src_tgt_dist_df,
            "default_values": self.default_values,
            "tgt_dist_keys": self.tgt_dist_keys,
            "src_dist_keys": self.src_dist_keys,
            "dist_flag_key": self.dist_flag_key,
            "data_location": self.data_location.to_json() if self.data_location is not None else None,
        }
        write_sharded(
            group=annotation_group,
            name="metadata",
            data=to_write,
            chunk_size=chunk_size,
            shard_size=shard_size,
            compressors=None,
        )
        return None

@dataclass
class GroupedDistribution:
    data: GroupedDistributionData
    annotation: GroupedDistributionAnnotation

    def write_zarr(
        self,
        path: str,
        *,
        chunk_size: int = 4096,
        shard_size: int = 65536,
        max_workers: int = 8,
    ) -> None:
        """Write the grouped distribution to a Zarr group."""
        ad.settings.zarr_write_format = 3  # Needed to support sharding in Zarr

        zgroup = zarr.open_group(path, mode="w")

        self.data.write_zarr_group(
            group=zgroup,
            chunk_size=chunk_size,
            shard_size=shard_size,
            max_workers=max_workers,
        )
        print("writing annotation")
        self.annotation.write_zarr_group(
            group=zgroup,
            chunk_size=chunk_size,
            shard_size=shard_size,
        )
        return None

    @classmethod
    def read_zarr(
        cls,
        path: str,
        in_memory: bool = False,
    ) -> GroupedDistribution:
        """Read the grouped distribution from a Zarr group.

        Parameters
        ----------
        path
            Path to the Zarr store.
        in_memory
            If True, load all arrays into memory as numpy arrays.
            If False (default), keep arrays as lazy zarr arrays.
        """
        zgroup = zarr.open_group(path, mode="r")
        annotation = GroupedDistributionAnnotation.read_zarr(zgroup["annotation"])
        data = GroupedDistributionData.read_zarr(zgroup["data"], in_memory=in_memory)
        return cls(
            annotation=annotation,
            data=data,
        )

    def to_memory(self) -> None:
        """Convert all lazy zarr arrays to in-memory numpy arrays (in-place)."""
        self.data.to_memory()
