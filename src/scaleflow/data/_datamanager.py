from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from scaleflow.data._data import (
    GroupedDistribution,
    GroupedDistributionAnnotation,
    GroupedDistributionData,
)
from scaleflow.logging import timer

from ._anndata_location import AnnDataLocation

__all__ = ["DataManager"]

if TYPE_CHECKING:
    import anndata


@dataclass
class DataManager:
    dist_flag_key: str
    src_dist_keys: list[str]
    # Target perturbation covariates. Either a flat ``list[str]`` (each column is its own
    # single-value condition key) or a grouped ``dict[str, list[str]]`` mapping a covariate
    # name to a list of obs columns that form a *combination* (a set of perturbations of that
    # type, e.g. ``{"drug": ["drug_1", "drug_2"]}``). Grouped columns are stacked into a set
    # of length K (= number of columns) and pooled at the model level; no padding is used.
    tgt_dist_keys: list[str] | dict[str, list[str]]
    data_location: AnnDataLocation
    rep_keys: dict[str, str] | None = None
    # Optional extra embeddings keyed by a new condition name.
    # Format: {new_condition_key: (obs_col_used_for_lookup, adata_uns_key)}
    # Example: {"prophet": ("drug", "prophet_emb")}
    # If the uns_key is absent at prepare_data time the entry is silently skipped,
    # so the same DataManager definition works with and without the embedding.
    extra_rep_keys: dict[str, tuple[str, str]] | None = None

    def __post_init__(
        self,
    ):
        self._verify_dist_keys(self.src_dist_keys)
        # Normalize tgt_dist_keys to grouped form: {group_name: [obs_col, ...]}.
        # A flat list maps each column to its own singleton group (length-1 set).
        if isinstance(self.tgt_dist_keys, dict):
            self._tgt_groups: dict[str, list[str]] = {g: list(cols) for g, cols in self.tgt_dist_keys.items()}
        else:
            self._tgt_groups = {c: [c] for c in self.tgt_dist_keys}
        if len(self._tgt_groups) == 0:
            raise ValueError("Number of distributions must be greater than 0.")
        for group, cols in self._tgt_groups.items():
            if len(cols) == 0:
                raise ValueError(f"tgt_dist_keys group '{group}' must have at least one column.")
        # flattened obs columns used for grouping cells into distributions
        self._tgt_cols: list[str] = [c for cols in self._tgt_groups.values() for c in cols]
        self._verify_dist_keys(self._tgt_cols)
        # all multi-column (combination) groups must share the same width K (position-aligned)
        pooled_lengths = {len(cols) for cols in self._tgt_groups.values() if len(cols) > 1}
        if len(pooled_lengths) > 1:
            raise ValueError(
                f"All multi-column tgt_dist_keys groups must share the same combination width, got {pooled_lengths}."
            )
        # source columns and target columns must not overlap
        if set(self.src_dist_keys) & set(self._tgt_cols):
            raise ValueError("Source and target distributions must not overlap.")
        # rep_keys / extra_rep_keys are keyed by covariate *name*: a src column or a tgt group name
        cond_keys = set(self.src_dist_keys) | set(self._tgt_groups.keys())
        if self.rep_keys is not None:
            if not set(self.rep_keys.keys()).issubset(cond_keys):
                raise ValueError(
                    "Representation locations must be a subset of the source and target distribution keys."
                )
        if self.extra_rep_keys is not None:
            obs_cols = set(self.src_dist_keys) | set(self._tgt_cols)
            for new_key, (obs_col, _uns_key) in self.extra_rep_keys.items():
                if obs_col not in obs_cols:
                    raise ValueError(
                        f"extra_rep_keys: obs_col '{obs_col}' must be in src_dist_keys or tgt_dist_keys."
                    )
                if new_key in cond_keys:
                    raise ValueError(
                        f"extra_rep_keys: new_key '{new_key}' conflicts with existing dist key '{new_key}'."
                    )

    def _get_label_for_col(
        self,
        obs_col: str,
        src_label: tuple,
        tgt_label: tuple,
    ):
        """Return the label value for ``obs_col`` from the current src/tgt label tuples.

        Parameters
        ----------
        obs_col
            The obs column whose label we want (must be in src_dist_keys or tgt_dist_keys).
        src_label
            Tuple of label values aligned with src_dist_keys.
        tgt_label
            Tuple of label values aligned with tgt_dist_keys.

        Returns
        -------
        The label value, or :obj:`None` if ``obs_col`` is not found.
        """
        if obs_col in self.src_dist_keys:
            return src_label[self.src_dist_keys.index(obs_col)]
        if obs_col in self._tgt_cols:
            return tgt_label[self._tgt_cols.index(obs_col)]
        return None

    def _prepare_annotation(
        self,
        obs_df: pd.DataFrame,
        verbose: bool = False,
    ) -> tuple[pd.DataFrame, GroupedDistributionAnnotation, dict[int, list[int]], dict[int, tuple], dict[int, tuple]]:
        """
        Prepare annotation data from an observation DataFrame.

        Parameters
        ----------
        obs_df
            The observation DataFrame (e.g. ``adata.obs`` or
            :meth:`annbatch.DatasetCollection.obs`). The cell matrix is never needed
            here; grouping into distributions is computed from obs alone.

        Returns
        -------
        obs
            Processed observation DataFrame with distribution indices.
        annotation
            GroupedDistributionAnnotation object containing metadata.
        src_to_tgt_dist_map
            Mapping from source distribution indices to target distribution indices.
        src_dist_labels
            Mapping from source distribution indices to their labels.
        tgt_dist_labels
            Mapping from target distribution indices to their labels.
        """
        # grouping/labels use the flattened target columns (a combination group contributes
        # all of its columns); the grouped structure only matters when building embeddings.
        src_tgt_dist_keys = [*self.src_dist_keys, *self._tgt_cols]

        cols = [self.dist_flag_key, *src_tgt_dist_keys]
        obs = obs_df[cols].copy()
        old_index_mapping = obs.index.to_numpy()
        obs.reset_index(drop=True, inplace=True)

        # dtype must be boolean
        assert pd.api.types.is_bool_dtype(obs[self.dist_flag_key]), "Distribution flag key must be a boolean column."

        with timer("Sorting values", verbose=verbose):
            obs.sort_values(cols, inplace=True)

        obs["src_dist_idx"] = obs.groupby(self.src_dist_keys, observed=False).ngroup()
        dist_mask = ~obs[self.dist_flag_key]
        obs.loc[dist_mask, "tgt_dist_idx"] = obs.loc[dist_mask].groupby(src_tgt_dist_keys, observed=False).ngroup()

        # Fill NaN indices with a specific value before casting
        obs["src_dist_idx"] = obs["src_dist_idx"].fillna(-1).astype(np.int32)
        obs["tgt_dist_idx"] = obs["tgt_dist_idx"].fillna(-1).astype(np.int32)

        # prepare src_tgt_dist_df
        src_tgt_dist_df = obs.loc[~obs[self.dist_flag_key]]
        src_tgt_dist_df = src_tgt_dist_df[["src_dist_idx", "tgt_dist_idx", *src_tgt_dist_keys]]
        src_tgt_dist_df.drop_duplicates(inplace=True)

        # prepare default_values
        temp_df = obs.loc[obs[self.dist_flag_key]][self._tgt_cols].drop_duplicates()
        if len(temp_df) != 1:
            raise ValueError("There should be exactly one control value.")
        default_values = temp_df.iloc[0].to_dict()

        # prepare src_to_tgt_dist_map
        src_to_tgt_dist_map = (
            src_tgt_dist_df[["src_dist_idx", "tgt_dist_idx"]]
            .groupby("src_dist_idx")["tgt_dist_idx"]
            .apply(list)
            .to_dict()
        )

        # prepare src_dist_labels
        src_dist_labels = (
            obs.loc[obs[self.dist_flag_key]][[*self.src_dist_keys, "src_dist_idx"]]
            .drop_duplicates()
            .set_index("src_dist_idx")
        )
        src_dist_labels = dict(
            zip(src_dist_labels.index, src_dist_labels.itertuples(index=False, name=None), strict=True)
        )

        # prepare tgt_dist_labels — one value per flattened target column, used by
        # _prepare_data for embedding lookups (label order matches self._tgt_cols).
        tgt_dist_labels = (
            obs.loc[~obs[self.dist_flag_key]][[*self._tgt_cols, "tgt_dist_idx"]]
            .drop_duplicates()
            .set_index("tgt_dist_idx")
        )
        tgt_dist_labels = dict(
            zip(tgt_dist_labels.index, tgt_dist_labels.itertuples(index=False, name=None), strict=True)
        )

        # prepare tgt_dist_labels_for_annotation — includes src_dist_keys so
        # that each (cell_line, drug) pair gets a unique label in the zarr.
        # This fixes ValidationSampler collapsing 252 conditions → 36 by
        # ensuring _get_key() returns unique (cell_line, drug) cond_keys.
        tgt_dist_labels_annotation = (
            obs.loc[~obs[self.dist_flag_key]][[*src_tgt_dist_keys, "tgt_dist_idx"]]
            .drop_duplicates()
            .set_index("tgt_dist_idx")
        )
        tgt_dist_labels_annotation = dict(
            zip(tgt_dist_labels_annotation.index,
                tgt_dist_labels_annotation.itertuples(index=False, name=None), strict=True)
        )

        annotation = GroupedDistributionAnnotation(
            src_tgt_dist_df=src_tgt_dist_df,
            old_obs_index=old_index_mapping,
            tgt_dist_keys=self._tgt_cols,
            src_dist_keys=self.src_dist_keys,
            dist_flag_key=self.dist_flag_key,
            src_dist_idx_to_labels=src_dist_labels,
            tgt_dist_idx_to_labels=tgt_dist_labels_annotation,
            default_values=default_values,
            data_location=self.data_location,
        )

        return obs, annotation, src_to_tgt_dist_map, src_dist_labels, tgt_dist_labels

    def _prepare_data(
        self,
        obs: pd.DataFrame,
        rep_dict: Mapping[str, Any],
        src_to_tgt_dist_map: dict[int, list[int]],
        src_dist_labels: dict[int, tuple],
        tgt_dist_labels: dict[int, tuple],
        verbose: bool = False,
    ) -> GroupedDistributionData:
        """
        Prepare the per-distribution row indices and conditions.

        Parameters
        ----------
        obs
            Processed observation DataFrame with distribution indices.
        rep_dict
            Mapping of representation keys to embedding dictionaries, e.g.
            ``adata.uns`` or a dict read from a separate ``uns`` store. Used only to
            build the condition embeddings; the cell matrix is never read here.
        src_to_tgt_dist_map
            Mapping from source distribution indices to target distribution indices.
        src_dist_labels
            Mapping from source distribution indices to their labels.
        tgt_dist_labels
            Mapping from target distribution indices to their labels.
        verbose
            Whether to print timing information.

        Returns
        -------
        GroupedDistributionData containing src_data, tgt_data, conditions, and the mapping.
        """
        # prepare conditions as nested dicts: {tgt_dist_idx: {col_name: array}}
        col_to_repr = {key: rep_dict[self.rep_keys[key]] for key in self.rep_keys.keys()}

        with timer("Getting conditions", verbose=verbose):
            conditions = {}
            for src_dist_idx, tgt_dist_idxs in src_to_tgt_dist_map.items():
                src_label = src_dist_labels[src_dist_idx]
                for tgt_dist_idx in tgt_dist_idxs:
                    tgt_label = tgt_dist_labels[tgt_dist_idx]
                    conditions[tgt_dist_idx] = self._build_condition(src_label, tgt_label, col_to_repr, rep_dict)

        # prepare row indices per source/target distribution.
        # We no longer materialize the cell matrix here: the actual ``X`` rows are
        # streamed on demand by annbatch using ``annotation.data_location``. We only
        # record the positional obs row indices (into ``adata``'s obs order) that make
        # up each source/target distribution.
        with timer("Getting source and target distribution row indices", verbose=verbose):
            src_dist_map = obs[obs[self.dist_flag_key]].groupby("src_dist_idx", observed=False).groups
            tgt_dist_map = obs[~obs[self.dist_flag_key]].groupby("tgt_dist_idx", observed=False).groups

            tgt_dist_to_rows = {int(k): np.asarray(v.to_numpy()) for k, v in tgt_dist_map.items()}
            src_dist_to_rows = {int(k): np.asarray(v.to_numpy()) for k, v in src_dist_map.items()}

        return GroupedDistributionData(
            src_to_tgt_dist_map=src_to_tgt_dist_map,
            src_dist_to_rows=src_dist_to_rows,
            tgt_dist_to_rows=tgt_dist_to_rows,
            conditions=conditions,
        )

    def prepare_data(
        self,
        adata: "anndata.AnnData",
        verbose: bool = False,
    ) -> GroupedDistribution:
        """
        Prepare grouped distribution metadata from an in-memory AnnData object.

        Only ``adata.obs`` (for grouping into distributions) and ``adata.uns`` (for
        condition embeddings) are read; the cell matrix is never materialized here.
        The resulting :class:`GroupedDistribution` stores per-distribution obs row
        indices, and the cells are streamed on demand (e.g. via annbatch).

        Distribution flag key must be a boolean column.
        The src and tgt distribution keys are recommended to be categorical columns otherwise sorting will be slow.

        Parameters
        ----------
        adata
            The AnnData object containing the data.
        verbose
            Whether to print timing information.

        Returns
        -------
        GroupedDistribution containing metadata and annotation.
        """
        return self._prepare(obs_df=adata.obs, rep_dict=adata.uns, verbose=verbose)

    def prepare_data_from_collection(
        self,
        collection: "Any",
        *,
        rep_path: str | None = None,
        rep_dict: Mapping[str, Any] | None = None,
        verbose: bool = False,
    ) -> GroupedDistribution:
        """Prepare grouped distribution metadata from an on-disk store, by path only.

        No in-memory AnnData is required and the cell matrix ``X`` is never loaded:
        only the obs columns needed for grouping are read from the collection, and the
        condition embeddings are read from a *separate* ``uns`` store (since
        :class:`annbatch.DatasetCollection` does not round-trip ``uns``).

        The per-distribution row indices recorded here index into the collection's
        global row order (the dataset-order concatenation), so they can be used
        verbatim as ``annbatch`` ``LoadRequest`` indices.

        Parameters
        ----------
        collection
            An :class:`annbatch.DatasetCollection` or a path to one.
        rep_path
            Path to a zarr/h5 store whose top-level keys are the representation keys
            referenced by ``rep_keys``/``extra_rep_keys`` (i.e. an ``uns``-shaped store
            mapping each key to a ``{label: embedding}`` dict). Read lazily via
            :func:`anndata.io.read_elem`. Ignored if ``rep_dict`` is given.
        rep_dict
            An already-loaded mapping of representations (same shape as ``adata.uns``),
            as an alternative to ``rep_path``.
        verbose
            Whether to print timing information.

        Returns
        -------
        GroupedDistribution containing metadata and annotation.
        """
        coll = self._open_collection(collection)
        needed_cols = [self.dist_flag_key, *self.src_dist_keys, *self._tgt_cols]
        obs_df = coll.obs(columns=needed_cols)
        if rep_dict is None:
            if rep_path is None:
                raise ValueError("Either `rep_path` or `rep_dict` must be provided when reading from a collection.")
            rep_dict = self._load_rep_dict(rep_path)
        return self._prepare(obs_df=obs_df, rep_dict=rep_dict, verbose=verbose)

    def _prepare(
        self,
        obs_df: pd.DataFrame,
        rep_dict: Mapping[str, Any],
        verbose: bool = False,
    ) -> GroupedDistribution:
        """Shared preparation from an obs DataFrame and a representations mapping."""
        DataManager._verify_rep_keys_exists(self.rep_keys, rep_dict)

        obs, annotation, src_to_tgt_dist_map, src_dist_labels, tgt_dist_labels = self._prepare_annotation(
            obs_df, verbose=verbose
        )
        data = self._prepare_data(
            obs, rep_dict, src_to_tgt_dist_map, src_dist_labels, tgt_dist_labels, verbose=verbose
        )

        return GroupedDistribution(data=data, annotation=annotation)

    def _build_condition(
        self,
        src_label: tuple,
        tgt_label: tuple,
        col_to_repr: dict[str, Any],
        rep_dict: Mapping[str, Any],
    ) -> dict[str, np.ndarray]:
        """Build a single condition dict ``{key: (1, K, emb_dim)}`` from labels.

        Shared by ``_prepare_data`` (conditions from obs) and ``get_condition_data``
        (conditions for arbitrary covariate combinations).

        Source covariates and single-column target groups have set length ``K = 1``
        (``(1, 1, emb_dim)``); a multi-column target group is a *combination*: its columns'
        embeddings are stacked into a set of length ``K = len(columns)`` (``(1, K, emb_dim)``)
        and pooled at the model level. No padding is introduced -- ``K`` is the true number of
        columns in the group. ``tgt_label`` is ordered to match :attr:`_tgt_cols`.
        """
        cond_dict: dict[str, np.ndarray] = {}
        for col, label in zip(self.src_dist_keys, src_label, strict=True):
            emb = DataManager._col_to_repr(col_to_repr, col, label)
            cond_dict[col] = emb[None, None, :]
        label_by_col = dict(zip(self._tgt_cols, tgt_label, strict=True))
        for group_name, cols in self._tgt_groups.items():
            embs = [DataManager._col_to_repr(col_to_repr, group_name, label_by_col[c]) for c in cols]
            cond_dict[group_name] = np.stack(embs, axis=0)[None, ...]  # (1, K, emb_dim)
        if self.extra_rep_keys is not None:
            for new_key, (obs_col, uns_key) in self.extra_rep_keys.items():
                if uns_key not in rep_dict:
                    continue
                label = self._get_label_for_col(obs_col, src_label, tgt_label)
                if label is None:
                    continue
                extra_repr = rep_dict[uns_key]
                if label in extra_repr:
                    cond_dict[new_key] = np.array(extra_repr[label])[None, None, :]
        return cond_dict

    def get_condition_data(
        self,
        covariate_data: "pd.DataFrame",
        *,
        rep_dict: Mapping[str, Any] | None = None,
        rep_path: str | None = None,
        condition_id_key: str | None = None,
    ) -> dict[Any, dict[str, np.ndarray]]:
        """Build condition embeddings for arbitrary covariate combinations.

        Unlike :meth:`prepare_data`, this does not require the combinations to appear in
        any observed data — it builds a condition dict for each unique row of
        ``covariate_data`` (over ``src_dist_keys + tgt_dist_keys``).

        Parameters
        ----------
        covariate_data
            DataFrame whose columns include ``src_dist_keys`` and ``tgt_dist_keys``.
        rep_dict / rep_path
            Representations (embeddings), as in :meth:`prepare_data_from_collection`.
        condition_id_key
            If given, use this column's value as the condition key; otherwise the key is
            the tuple of label values ``(src_dist_keys + tgt_dist_keys)``.

        Returns
        -------
        Mapping ``{condition_key: {col_name: (1, 1, emb_dim)}}``.
        """
        if rep_dict is None:
            rep_dict = self._load_rep_dict(rep_path) if rep_path is not None else {}
        DataManager._verify_rep_keys_exists(self.rep_keys, rep_dict)
        col_to_repr = {key: rep_dict[self.rep_keys[key]] for key in self.rep_keys.keys()}

        keys = [*self.src_dist_keys, *self._tgt_cols]
        select = keys + ([condition_id_key] if condition_id_key is not None else [])
        df = covariate_data[select].drop_duplicates()

        conditions: dict[Any, dict[str, np.ndarray]] = {}
        for _, row in df.iterrows():
            src_label = tuple(row[k] for k in self.src_dist_keys)
            tgt_label = tuple(row[k] for k in self._tgt_cols)
            cond_key = row[condition_id_key] if condition_id_key is not None else tuple(str(row[k]) for k in keys)
            conditions[cond_key] = self._build_condition(src_label, tgt_label, col_to_repr, rep_dict)
        return conditions

    @staticmethod
    def _open_collection(collection: "Any") -> "Any":
        """Return a read-mode :class:`annbatch.DatasetCollection` from a path or pass-through."""
        from annbatch import DatasetCollection

        if isinstance(collection, DatasetCollection):
            return collection
        return DatasetCollection(str(collection), mode="r")

    @staticmethod
    def _load_rep_dict(rep_path: str) -> dict[str, Any]:
        """Read an ``uns``-shaped representation store (zarr or h5) into a dict.

        The store's top-level keys are treated as the representation keys; each is read
        with :func:`anndata.io.read_elem` (so a ``{label: embedding}`` dict round-trips).
        """
        import anndata as ad

        path = str(rep_path)
        if path.endswith((".h5", ".h5ad", ".hdf5")):
            import h5py

            with h5py.File(path, "r") as f:
                return {k: ad.io.read_elem(f[k]) for k in f.keys()}
        import zarr

        group = zarr.open_group(path, mode="r")
        return {k: ad.io.read_elem(group[k]) for k in group.keys()}

    @staticmethod
    def _verify_dist_keys(dist_keys: list[str]) -> None:
        if len(dist_keys) == 0:
            raise ValueError("Number of distributions must be greater than 0.")
        # no duplicates
        if len(set(dist_keys)) != len(dist_keys):
            raise ValueError("Distributions must be unique.")

    @staticmethod
    def _verify_rep_keys_exists(rep_keys: dict[str, str], rep_dict: Mapping[str, Any]) -> None:
        for _, value in rep_keys.items():
            if value not in rep_dict:
                raise ValueError(f"Representation key {value} not found in the provided representations (`uns`).")

    @staticmethod
    def _col_to_repr(col_to_repr: dict[str, dict[str, np.ndarray]], col: str, label: Any) -> np.ndarray:
        if col not in col_to_repr:
            # for example in case of dosage, we have a float label
            if isinstance(label, float):
                return np.array([label])
            raise ValueError(f"Column {col} not found in col_to_repr.")
        if label not in col_to_repr[col]:
            raise ValueError(f"Label {label} not found in col_to_repr[{col}].")
        return col_to_repr[col][label]