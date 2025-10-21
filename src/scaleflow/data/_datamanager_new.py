import anndata
from typing import Any
import time

import dask.array as da
import pandas as pd

from dataclasses import dataclass
import time
import numpy as np

from scaleflow.logging import timer

from ._anndata_location import AnnDataLocation

__all__ = ["DataManager", "GroupedDistribution"]



    
@dataclass
class GroupedDistributionData:
    src_to_tgt_dist_map: dict[int, list[int]] # (n_src_dists) → (n_tgt_dists_{src_dist_idx})
    src_data: dict[int, np.ndarray] # (n_src_dists) → (n_cells_{src_dist_idx}, n_features)
    tgt_data: dict[int, np.ndarray] # (n_tgt_dists) → (n_cells_{tgt_dist_idx}, n_features)
    conditions: dict[int, np.ndarray] # (n_tgt_dists) → (n_cond_features_1, n_cond_features_2)


@dataclass
class GroupedDistributionAnnotation:
    old_obs_index: np.ndarray # (n_cells,) to be able to map back to the original index

    src_dist_idx_to_labels: dict[int, Any] # (n_src_dists) → Any (e.g. list of strings)
    tgt_dist_idx_to_labels: dict[int, Any] # (n_tgt_dists) → Any (e.g. list of strings)


@dataclass
class GroupedDistribution:
    data: GroupedDistributionData
    annotation: GroupedDistributionAnnotation


@dataclass
class DataManager:

    dist_flag_key: str
    src_dist_keys: list[str]
    tgt_dist_keys: list[str]
    data_location: AnnDataLocation
    rep_keys: dict[str, str] | None = None

    def __post_init__(
        self,
    ):
        self._verify_dist_keys(self.src_dist_keys)
        self._verify_dist_keys(self.tgt_dist_keys)
        # they shouldn't overlap
        if len(set(self.src_dist_keys) & set(self.tgt_dist_keys)) > 0:
            raise ValueError("Source and target distributions must not overlap.")
        if self.rep_keys is not None:
            if not set(self.rep_keys.keys()).issubset(set(self.src_dist_keys) | set(self.tgt_dist_keys)):
                raise ValueError("Representation locations must be a subset of the source and target distribution keys.")

    @staticmethod
    def _verify_dist_keys(dist_keys: list[str]) -> None:
        if len(dist_keys) == 0:
            raise ValueError("Number of distributions must be greater than 0.")
        # no duplicates
        if len(set(dist_keys)) != len(dist_keys):
            raise ValueError("Distributions must be unique.")

    @staticmethod
    def _verify_rep_keys_exists(rep_keys: dict[str, str], adata: anndata.AnnData) -> None:
        for key, value in rep_keys.items():
            if value not in adata.uns:
                raise ValueError(f"Representation key {value} not found in adata.uns.")


    @staticmethod
    def _col_to_repr(col_to_repr: dict[str, dict[str, np.ndarray]], col: str, label: Any) -> np.ndarray:
        if col not in col_to_repr:
            raise ValueError(f"Column {col} not found in col_to_repr.")
        if label not in col_to_repr[col]:
            raise ValueError(f"Label {label} not found in col_to_repr[{col}].")
        return col_to_repr[col][label]

    def prepare_data(
        self, 
        adata: anndata.AnnData,
        verbose: bool = False,
    ) -> GroupedDistribution:
        """
        Distribution flag key must be a boolean column.
        The src and tgt distribution keys are recommended to be categorical columns otherwise sorting will be slow.
        Resets the index of obs. saves new to old index
        """
        DataManager._verify_rep_keys_exists(self.rep_keys, adata)

        src_tgt_dist_keys = [*self.src_dist_keys, *self.tgt_dist_keys]

        cols = [self.dist_flag_key, *src_tgt_dist_keys]
        obs = adata.obs[cols].copy()
        old_index_mapping = obs.index.to_numpy()
        obs.reset_index(drop=True, inplace=True)
        

        # dtype must be boolean
        assert pd.api.types.is_bool_dtype(obs[self.dist_flag_key]), "Distribution flag key must be a boolean column."
        control_mask = obs[self.dist_flag_key].to_numpy()

        with timer("Sorting values", verbose=verbose):
            obs.sort_values(cols, inplace=True)

        obs["src_dist_idx"] = obs.groupby(self.src_dist_keys, observed=False).ngroup()
        obs["tgt_dist_idx"] = obs.groupby(src_tgt_dist_keys, observed=False).ngroup()
        # Fill NaN indices with a specific value before casting
        obs["src_dist_idx"] = obs["src_dist_idx"].fillna(-1).astype(np.int32)
        obs["tgt_dist_idx"] = obs["tgt_dist_idx"].fillna(-1).astype(np.int32)


        # preparing for src_to_tgt_dist_map
        src_tgt_dist_df = obs.loc[~obs[self.dist_flag_key]]
        src_tgt_dist_df = src_tgt_dist_df[['src_dist_idx', 'tgt_dist_idx']]
        src_tgt_dist_df.drop_duplicates(inplace=True)

        src_to_tgt_dist_map = src_tgt_dist_df.groupby('src_dist_idx')['tgt_dist_idx'].apply(list).to_dict()

        # preparing src_dist_labels
        src_dist_labels = obs.loc[obs[self.dist_flag_key]][[*self.src_dist_keys, 'src_dist_idx']]\
            .drop_duplicates().set_index('src_dist_idx')
        src_dist_labels = dict(zip(src_dist_labels.index, src_dist_labels.itertuples(index=False, name=None)))

        # preparing tgt_dist_labels
        tgt_dist_labels = obs.loc[~obs[self.dist_flag_key]][[*self.tgt_dist_keys, 'tgt_dist_idx']]\
            .drop_duplicates().set_index('tgt_dist_idx')
        tgt_dist_labels = dict(zip(tgt_dist_labels.index, tgt_dist_labels.itertuples(index=False, name=None)))


        arr = self.data_location(adata)
        if isinstance(arr, da.Array):
            arr = arr.compute()
        

        # return obs
        with timer("Getting source and target distribution data", verbose=verbose):
            src_dist_map = obs[control_mask].groupby('src_dist_idx', observed=False).groups
            tgt_dist_map = obs[~control_mask].groupby('tgt_dist_idx', observed=False).groups

            tgt_data = {
                int(k): arr[v.to_numpy()] for k, v in tgt_dist_map.items()
            }
            src_data = {
                int(k): arr[v.to_numpy()] for k, v in src_dist_map.items()
            }
        
        col_to_repr = {
            key: adata.uns[self.rep_keys[key]] for key in self.rep_keys.keys()
        }

        with timer("Getting conditions", verbose=verbose):
            conditions = {}
            for src_dist_idx, tgt_dist_idxs in src_to_tgt_dist_map.items():
                src_label = src_dist_labels[src_dist_idx]
                src_repr = [
                    DataManager._col_to_repr(col_to_repr, col, label) 
                    for col, label in zip(self.src_dist_keys, src_label)
                ]
                for tgt_dist_idx in tgt_dist_idxs:
                    tgt_label = tgt_dist_labels[tgt_dist_idx]
                    tgt_repr = [
                        DataManager._col_to_repr(col_to_repr, col, label) 
                        for col, label in zip(self.tgt_dist_keys, tgt_label)
                    ]
                    conditions[tgt_dist_idx] = np.concatenate([*src_repr, *tgt_repr])

            
        return GroupedDistribution(
            data=GroupedDistributionData(
                src_to_tgt_dist_map=src_to_tgt_dist_map,
                src_data=src_data,
                tgt_data=tgt_data,
                conditions=conditions,
            ),
            annotation=GroupedDistributionAnnotation(
                old_obs_index=old_index_mapping,
                src_dist_idx_to_labels=src_dist_labels,
                tgt_dist_idx_to_labels=tgt_dist_labels,
            ),
        )
    