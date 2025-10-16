import anndata
from typing import Any
import time

import dask.array as da
import pandas as pd

from dataclasses import dataclass
import time
import numpy as np
from contextlib import contextmanager

__all__ = ["DataManager"]

@contextmanager
def timer(description: str, verbose: bool = True):
    """Context manager for timing operations with optional verbose output."""
    if verbose:
        start_time = time.time()
        print(f"{description}...")
    
    yield
    
    if verbose:
        end_time = time.time()
        print(f"{description} took {end_time - start_time:.2f} seconds.")


class AnnDataLocation:
    """
    An object that stores a sequence of access operations (attributes and keys)
    and can be called on an AnnData object to execute them.
    """
    def __init__(self, path=None):
        # The path is a list of tuples, e.g., [('getattr', 'obsm'), ('getitem', 's')]
        self._path = path if path is not None else []

    def __getattr__(self, name):
        """
        Handles attribute access, like .obs or .X.
        It returns a new AnnDataLocation with the attribute access added to the path.
        """
        if name.startswith('__') and name.endswith('__'):
            # Avoid interfering with special methods
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
        new_path = self._path + [('getattr', name)]
        return AnnDataLocation(new_path)

    def __getitem__(self, key):
        """
        Handles item access, like ['my_key'].
        It returns a new AnnDataLocation with the item access added to the path.
        """
        new_path = self._path + [('getitem', key)]
        return AnnDataLocation(new_path)

    def __call__(self, adata: anndata.AnnData):
        """
        Executes the stored path of operations on the provided AnnData object.
        """
        target = adata
        try:
            for op_type, op_arg in self._path:
                if op_type == 'getattr':
                    target = getattr(target, op_arg)
                elif op_type == 'getitem':
                    target = target[op_arg]
            return target
        except (AttributeError, KeyError) as e:
            raise type(e)(f"Failed to resolve location {self!r} on the AnnData object. Reason: {e}") from e

    def __repr__(self):
        """Provides a user-friendly string representation of the stored path."""
        representation = "AnnDataAccessor()"
        for op_type, op_arg in self._path:
            if op_type == 'getattr':
                representation += f'.{op_arg}'
            elif op_type == 'getitem':
                # Use repr() to correctly handle string keys with quotes
                representation += f'[{repr(op_arg)}]'
        return f"<AnnDataLocation: {representation}>"


    
@dataclass
class MappedData:
    old_obs_index: np.ndarray
    src_to_tgt_dist_map: dict[int, list[int]]
    src_data: dict[int, np.ndarray]
    tgt_data: dict[int, np.ndarray]



class DataManager:

    def __init__(
        self,
        dist_flag_key: str,
        src_dist_keys: list[str],
        tgt_dist_keys: list[str],
    ):
        self._dist_flag_key = dist_flag_key
        self._verify_dist_keys(src_dist_keys)
        self._verify_dist_keys(tgt_dist_keys)
        # they shouldn't overlap
        assert len(set(src_dist_keys) & set(tgt_dist_keys)) == 0, "Source and target distributions must not overlap."
        self._tgt_dist_keys = tgt_dist_keys
        self._src_dist_keys = src_dist_keys

    @staticmethod
    def _verify_dist_keys(dist_keys: list[str]) -> list[str]:
        assert len(dist_keys) > 0, "Number of distributions must be greater than 0."
        # no duplicates
        assert len(set(dist_keys)) == len(dist_keys), "Distributions must be unique."


    def prepare_mapped_data(
        self, 
        adata: anndata.AnnData,
        data_location: AnnDataLocation,
        verbose: bool = False,
    ) -> MappedData:
        """
        Distribution flag key must be a boolean column.
        The src and tgt distribution keys are recommended to be categorical columns otherwise sorting will be slow.
        Resets the index of obs. saves new to old index
        """
        cols = [self._dist_flag_key, *self._src_dist_keys, *self._tgt_dist_keys]
        obs = adata.obs[cols].copy()
        old_index_mapping = obs.index.to_numpy()
        obs.reset_index(drop=True, inplace=True)
        

        # dtype must be boolean
        assert pd.api.types.is_bool_dtype(obs[self._dist_flag_key]), "Distribution flag key must be a boolean column."
        control_mask = obs[self._dist_flag_key].to_numpy()

        with timer("Sorting values", verbose=verbose):
            obs.sort_values(cols, inplace=True)

        obs["src_dist_idx"] = obs.groupby(self._src_dist_keys, observed=False).ngroup()
        obs["tgt_dist_idx"] = obs.groupby([*self._src_dist_keys, *self._tgt_dist_keys], observed=False).ngroup()
        # Fill NaN indices with a specific value before casting
        obs["src_dist_idx"] = obs["src_dist_idx"].fillna(-1).astype(np.int32)
        obs["tgt_dist_idx"] = obs["tgt_dist_idx"].fillna(-1).astype(np.int32)


        src_tgt_dist_df = obs.loc[~obs[self._dist_flag_key]]
        src_tgt_dist_df = src_tgt_dist_df[['src_dist_idx', 'tgt_dist_idx']]
        src_tgt_dist_df.drop_duplicates(inplace=True)

        src_to_tgt_dist_map = src_tgt_dist_df.groupby('src_dist_idx')['tgt_dist_idx'].apply(list).to_dict()
        
        arr = data_location(adata)
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
        

        return MappedData(
            old_obs_index=old_index_mapping,
            src_to_tgt_dist_map=src_to_tgt_dist_map,
            src_data=src_data,
            tgt_data=tgt_data,
        )
    