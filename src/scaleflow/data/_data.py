from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr
from zarr.storage import LocalStore

import anndata as ad
from scaleflow._types import ArrayLike
from scaleflow.data._utils import write_sharded, write_dist_data_threaded, write_nested_dist_data_threaded

import pandas as pd

__all__ = [
    "BaseDataMixin",
    "ConditionData",
    "PredictionData",
    "TrainingData",
    "ValidationData",
    "MappedCellData",
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
class ConditionData(BaseDataMixin):
    """Data container containing condition embeddings.

    Parameters
    ----------
    condition_data
        Dictionary with embeddings for conditions.
    max_combination_length
        Maximum number of covariates in a combination.
    null_value
        Token to use for masking `null_value`.
    data_manager
        Data manager used to generate the data.
    """

    condition_data: dict[str, np.ndarray]
    max_combination_length: int
    perturbation_idx_to_covariates: dict[int, tuple[str, ...]]
    perturbation_idx_to_id: dict[int, Any]
    null_value: Any
    data_manager: Any


@dataclass
class TrainingData(BaseDataMixin):
    """Training data.

    Parameters
    ----------
    cell_data
        The representation of cell data, e.g. PCA of gene expression data.
    split_covariates_mask
        Mask of the split covariates.
    split_idx_to_covariates
        Dictionary explaining values in ``split_covariates_mask``.
    perturbation_covariates_mask
        Mask of the perturbation covariates.
    perturbation_idx_to_covariates
        Dictionary explaining values in ``perturbation_covariates_mask``.
    condition_data
        Dictionary with embeddings for conditions.
    control_to_perturbation
        Mapping from control index to target distribution indices.
    max_combination_length
        Maximum number of covariates in a combination.
    data_manager
        The data manager
    phenotype_data
        Optional dictionary mapping perturbation_idx to phenotype values.
    """

    cell_data: np.ndarray  # (n_cells, n_features)
    split_covariates_mask: np.ndarray  # (n_cells,), which cell assigned to which source distribution
    split_idx_to_covariates: dict[int, tuple[Any, ...]]  # (n_sources,) dictionary explaining split_covariates_mask
    perturbation_covariates_mask: np.ndarray  # (n_cells,), which cell assigned to which target distribution
    perturbation_idx_to_covariates: dict[
        int, tuple[str, ...]
    ]  # (n_targets,), dictionary explaining perturbation_covariates_mask
    perturbation_idx_to_id: dict[int, Any]
    condition_data: dict[str, np.ndarray]  # (n_targets,) all embeddings for conditions
    control_to_perturbation: dict[int, np.ndarray]  # mapping from control idx to target distribution idcs
    max_combination_length: int
    null_value: Any
    data_manager: Any
    phenotype_data: dict[int, np.ndarray] | None = None  # (n_targets,) phenotype values for each perturbation

    # --- Zarr export helpers -------------------------------------------------
    def write_zarr(
        self,
        path: str,
        *,
        chunk_size: int = 4096,
        shard_size: int = 65536,
        compressors: Any | None = None,
    ) -> None:
        """Write this training data to Zarr v3 with sharded, compressed arrays.

        Parameters
        ----------
        path
            Path to a Zarr group to create or open for writing.
        chunk_size
            Chunk size along the first axis.
        shard_size
            Shard size along the first axis.
        compressors
            Optional list/tuple of Zarr codecs. If ``None``, a sensible default is used.
        """
        # Convert to numpy-backed containers for serialization
        cell_data = np.asarray(self.cell_data)
        split_covariates_mask = np.asarray(self.split_covariates_mask)
        perturbation_covariates_mask = np.asarray(self.perturbation_covariates_mask)
        condition_data = {str(k): np.asarray(v) for k, v in (self.condition_data or {}).items()}
        control_to_perturbation = {str(k): np.asarray(v) for k, v in (self.control_to_perturbation or {}).items()}
        split_idx_to_covariates = {str(k): np.asarray(v) for k, v in (self.split_idx_to_covariates or {}).items()}
        perturbation_idx_to_covariates = {
            str(k): np.asarray(v) for k, v in (self.perturbation_idx_to_covariates or {}).items()
        }
        perturbation_idx_to_id = {str(k): v for k, v in (self.perturbation_idx_to_id or {}).items()}

        train_data_dict: dict[str, Any] = {
            "cell_data": cell_data,
            "split_covariates_mask": split_covariates_mask,
            "perturbation_covariates_mask": perturbation_covariates_mask,
            "split_idx_to_covariates": split_idx_to_covariates,
            "perturbation_idx_to_covariates": perturbation_idx_to_covariates,
            "perturbation_idx_to_id": perturbation_idx_to_id,
            "condition_data": condition_data,
            "control_to_perturbation": control_to_perturbation,
            "max_combination_length": int(self.max_combination_length),
        }

        additional_kwargs = {}
        if compressors is not None:
            additional_kwargs["compressors"] = compressors

        zgroup = zarr.open_group(path, mode="w")
        write_sharded(
            zgroup,
            train_data_dict,
            chunk_size=chunk_size,
            shard_size=shard_size,
            **additional_kwargs,
        )


@dataclass
class ValidationData(BaseDataMixin):
    """Data container for the validation data.

    Parameters
    ----------
    cell_data
        The representation of cell data, e.g. PCA of gene expression data.
    split_covariates_mask
        Mask of the split covariates.
    split_idx_to_covariates
        Dictionary explaining values in ``split_covariates_mask``.
    perturbation_covariates_mask
        Mask of the perturbation covariates.
    perturbation_idx_to_covariates
        Dictionary explaining values in ``perturbation_covariates_mask``.
    condition_data
        Dictionary with embeddings for conditions.
    control_to_perturbation
        Mapping from control index to target distribution indices.
    max_combination_length
        Maximum number of covariates in a combination.
    data_manager
        The data manager
    n_conditions_on_log_iteration
        Number of conditions to use for computation callbacks at each logged iteration.
        If :obj:`None`, use all conditions.
    n_conditions_on_train_end
        Number of conditions to use for computation callbacks at the end of training.
        If :obj:`None`, use all conditions.
    """

    cell_data: np.ndarray  # (n_cells, n_features)
    split_covariates_mask: np.ndarray  # (n_cells,), which cell assigned to which source distribution
    split_idx_to_covariates: dict[int, tuple[Any, ...]]  # (n_sources,) dictionary explaining split_covariates_mask
    perturbation_covariates_mask: np.ndarray  # (n_cells,), which cell assigned to which target distribution
    perturbation_idx_to_covariates: dict[
        int, tuple[str, ...]
    ]  # (n_targets,), dictionary explaining perturbation_covariates_mask
    perturbation_idx_to_id: dict[int, Any]
    condition_data: dict[str, np.ndarray]  # (n_targets,) all embeddings for conditions
    control_to_perturbation: dict[int, np.ndarray]  # mapping from control idx to target distribution idcs
    max_combination_length: int
    null_value: Any
    data_manager: Any
    n_conditions_on_log_iteration: int | None = None
    n_conditions_on_train_end: int | None = None


@dataclass
class PredictionData(BaseDataMixin):
    """Data container to perform prediction.

    Parameters
    ----------
    src_data
        Dictionary with data for source cells.
    condition_data
        Dictionary with embeddings for conditions.
    control_to_perturbation
        Mapping from control index to target distribution indices.
    covariate_encoder
        Encoder for the primary covariate.
    max_combination_length
        Maximum number of covariates in a combination.
    null_value
        Token to use for masking ``null_value``.
    """

    cell_data: ArrayLike  # (n_cells, n_features)
    split_covariates_mask: ArrayLike  # (n_cells,), which cell assigned to which source distribution
    split_idx_to_covariates: dict[int, tuple[Any, ...]]  # (n_sources,) dictionary explaining split_covariates_mask
    perturbation_idx_to_covariates: dict[
        int, tuple[str, ...]
    ]  # (n_targets,), dictionary explaining perturbation_covariates_mask
    perturbation_idx_to_id: dict[int, Any]
    condition_data: dict[str, ArrayLike]  # (n_targets,) all embeddings for conditions
    control_to_perturbation: dict[int, ArrayLike]
    max_combination_length: int
    null_value: Any
    data_manager: Any


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
class GroupedDistributionData:
    src_to_tgt_dist_map: dict[int, list[int]] # (n_src_dists) → (n_tgt_dists_{src_dist_idx})
    src_data: dict[int, np.ndarray] # (n_src_dists) → (n_cells_{src_dist_idx}, n_features)
    tgt_data: dict[int, np.ndarray] # (n_tgt_dists) → (n_cells_{tgt_dist_idx}, n_features)
    conditions: dict[int, dict[str, np.ndarray]] # (n_tgt_dists) → {col: (n_cells_{tgt_dist_idx}, n_cond_features_{col})}


    @classmethod
    def read_zarr(
        cls,
        group: zarr.Group,
    ) -> GroupedDistributionData:
        """
        Read the grouped distribution data from a Zarr group.
        """
        # Read conditions from nested structure with keys and positions stored in attrs
        conditions = {}
        if "conditions" in group:
            cond_group = group["conditions"]
            # Iterate through each distribution group
            for dist_id_str in cond_group.keys():
                dist_id = int(dist_id_str)
                dist_group = cond_group[dist_id_str]
                
                # Get metadata from attrs
                if 'keys' in dist_group.attrs and 'positions' in dist_group.attrs:
                    sorted_keys = dist_group.attrs['keys']
                    positions = dist_group.attrs['positions']
                    
                    # Read the concatenated array
                    concatenated = np.array(dist_group["data"])
                    
                    # Split back into individual arrays
                    conditions[dist_id] = {}
                    for i, col_name in enumerate(sorted_keys):
                        start = positions[i]
                        end = positions[i + 1]
                        conditions[dist_id][col_name] = concatenated[start:end]
        
        return cls(
            src_to_tgt_dist_map={int(k): np.array(group["src_to_tgt_dist_map"][k]) for k in group["src_to_tgt_dist_map"].keys()},
            src_data={int(k): group["src_data"][k] for k in group["src_data"].keys()},
            tgt_data={int(k): group["tgt_data"][k] for k in group["tgt_data"].keys()},
            conditions=conditions,
        )


    def write_zarr_group(
        self,
        group: zarr.Group,
        chunk_size: int,
        shard_size: int,
        max_workers: int,
    ) -> None:
        """
        Write the grouped distribution data to a Zarr group.
        """
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
        
        # Write conditions using nested writer (concatenates arrays per distribution)
        conditions_group = data.create_group("conditions")
        write_nested_dist_data_threaded(
            group=conditions_group,
            dist_data=self.conditions,
            chunk_size=chunk_size,
            shard_size=shard_size,
            max_workers=max_workers,
        )
        
        return None

@dataclass
class GroupedDistributionAnnotation:
    old_obs_index: np.ndarray # (n_cells,) to be able to map back to the original index

    src_dist_idx_to_labels: dict[int, Any] # (n_src_dists) → Any (e.g. list of strings)
    tgt_dist_idx_to_labels: dict[int, Any] # (n_tgt_dists) → Any (e.g. list of strings)
    src_tgt_dist_df: pd.DataFrame


    @classmethod
    def read_zarr(
        cls,
        group: zarr.Group,
    ) -> GroupedDistributionAnnotation:
        """
        Read the grouped distribution annotation from a Zarr group.
        """
        return cls(
            old_obs_index=ad.io.read_elem(group["old_obs_index"]),
            src_dist_idx_to_labels={int(k): np.array(group["src_dist_idx_to_labels"][k]) for k in group["src_dist_idx_to_labels"].keys()},
            tgt_dist_idx_to_labels={int(k): np.array(group["tgt_dist_idx_to_labels"][k]) for k in group["tgt_dist_idx_to_labels"].keys()},
            src_tgt_dist_df=ad.io.read_elem(group["src_tgt_dist_df"])
        )

    def write_zarr_group(
        self,
        group: zarr.Group,
        chunk_size: int,
        shard_size: int,
    ) -> None:
        """
        Write the grouped distribution annotation to a Zarr group.
        """
        to_write = {
            "old_obs_index": self.old_obs_index,
            "src_dist_idx_to_labels": {str(k): np.array(v) for k, v in self.src_dist_idx_to_labels.items()},
            "tgt_dist_idx_to_labels": {str(k): np.array(v) for k, v in self.tgt_dist_idx_to_labels.items()},
            "src_tgt_dist_df": self.src_tgt_dist_df,
        }
        write_sharded(
            group=group,
            name="annotation",
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
        """
        Write the grouped distribution to a Zarr group.
        """
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
    ) -> GroupedDistribution:
        """
        Read the grouped distribution from a Zarr group.
        """
        zgroup = zarr.open_group(path, mode="r")
        annotation = GroupedDistributionAnnotation.read_zarr(zgroup["annotation"])
        data = GroupedDistributionData.read_zarr(zgroup["data"])
        return cls(
            annotation=annotation,
            data=data,
        )

    def split_by_dist_df(self, dist_df: pd.DataFrame, column: str) -> dict[str, GroupedDistributionData]:
        """
        Split the grouped distribution by the given distribution dataframe.
        """
        if column not in dist_df.columns:
            raise ValueError(f"Column {column} not found in dist_df.")
        # assert categorical,boolean, or string
        if not pd.api.types.is_categorical_dtype(dist_df[column]) \
            and not pd.api.types.is_bool_dtype(dist_df[column]) \
            and not pd.api.types.is_string_dtype(dist_df[column]):
            raise ValueError(f"Column {column} must be categorical, boolean, or string.")

        split_values = dist_df[column].unique()
        # get the src_dist_idx and tgt_dist_idx for each value
        split_data = {}
        for value in split_values:
            filtered_df = dist_df.loc[dist_df[column] == value]
            # group by to map src_dist_idx and tgt_dist_idx
            src_tgt_dist_map = filtered_df[['src_dist_idx', 'tgt_dist_idx']].groupby('src_dist_idx')['tgt_dist_idx'].apply(list).to_dict()
            src_data = {int(k): self.data.src_data[k] for k in src_tgt_dist_map.keys()}
            tgt_data = {int(k): self.data.tgt_data[k] for k in src_tgt_dist_map.keys()}
            conditions = {int(k): self.data.conditions[k] for k in src_tgt_dist_map.keys()}
            split_data[value] = GroupedDistributionData(
                src_to_tgt_dist_map=src_tgt_dist_map,
                src_data=src_data,
                tgt_data=tgt_data,
                conditions=conditions,
            )
        return split_data