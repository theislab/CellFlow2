"""Data splitter for creating train/validation/test splits from GroupedDistribution objects."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from scaleflow.data._data import GroupedDistribution, GroupedDistributionAnnotation

logger = logging.getLogger(__name__)


@dataclass
class GroupedDistributionSplitter:
    gd: GroupedDistribution
    holdout_combinations: bool
    split_by: list[str]
    split_key: str
    force_training_values: dict[str, Any]
    ratios: list[float]
    random_state: int
    _computed_split_df: pd.DataFrame | None = None

    def __post_init__(self):
        if len(self.split_by) == 0:
            raise ValueError("split_by must be a non-empty list")
        if len(self.ratios) != 3:
            raise ValueError("ratios must be a list of 3 values [train, val, test]")
        if not np.isclose(sum(self.ratios), 1.0):
            raise ValueError("ratios must sum to 1.0")
        if not all(ratio > 0 and ratio < 1.0 for ratio in self.ratios):
            raise ValueError("ratios must be between 0.0 and 1.0")
        if not set(self.force_training_values.keys()).issubset(self.split_by):
            raise ValueError(
                f"force_training_values keys must be a subset of split_by: {self.force_training_values.keys()}"
            )
        self.train_ratio = self.ratios[0]
        self.val_ratio = self.ratios[1]
        self.test_ratio = self.ratios[2]

    @property
    def annotation(self) -> GroupedDistributionAnnotation:
        """Convenience property to access the annotation from the GroupedDistribution."""
        return self.gd.annotation

    def _calculate_split_sizes(self, total_combinations: int) -> tuple[int, int, int]:
        train_size = round(self.train_ratio * total_combinations)
        val_size = round(self.val_ratio * total_combinations)
        test_size_temp = round(self.test_ratio * total_combinations)
        if test_size_temp == 0 or val_size == 0 or train_size == 0:
            raise ValueError(
                "When multiplying the ratio by the total number of combinations, "
                "0 was encountered for one of the splits."
                "Either modify the ratios or the total number of combinations."
                f"The ratios are {self.train_ratio}, {self.val_ratio}, {self.test_ratio} and the total number of combinations is {total_combinations}."
                f"The calculated sizes are {train_size}, {val_size}, {test_size_temp}."
            )
        test_size = total_combinations - train_size - val_size
        return train_size, val_size, test_size

    @staticmethod
    def _check_df_unique_columns(df_unique: pd.DataFrame, combination_keys: list[str]) -> None:
        if not all(key in df_unique.columns for key in combination_keys):
            raise ValueError(f"combination_keys must be in df.columns: {combination_keys}")

    @staticmethod
    def _split_two(
        df_unique: pd.DataFrame,
        train_size: int,
        split_key: str,
        is_in_training_key: str | None = None,
        not_in_training_key: str | None = None,
        random_state: int = 42,
        overwrite: bool = False,
    ):
        if is_in_training_key is not None:
            if is_in_training_key not in df_unique.columns:
                raise ValueError(f"{is_in_training_key} must be in df.columns: {is_in_training_key}")
        if not_in_training_key is not None:
            if not_in_training_key not in df_unique.columns:
                raise ValueError(f"{not_in_training_key} must be in df.columns: {not_in_training_key}")
        if split_key in df_unique.columns and not overwrite:
            raise ValueError(f"{split_key} already in df.columns: {split_key} and overwrite is False")
        df_unique.loc[:, split_key] = "test_val"

        # remove the forced training combinations from the unique combinations
        if is_in_training_key is not None:
            df_unique = df_unique[df_unique[is_in_training_key]]
        if not_in_training_key is not None:
            df_unique = df_unique[~df_unique[not_in_training_key]]
        if len(df_unique) == 0:
            raise ValueError(
                "There must be at least one unique combination of split_by keys after removing forced training combinations"
            )
        # shuffle the unique combinations
        df_unique = df_unique.sample(frac=1, random_state=random_state).reset_index(drop=True)

        df_unique.loc[: train_size - 1, split_key] = "train"

        return df_unique

    def _split_df(self):
        # calculate the sizes of the splits
        is_in_training_key = None
        df_unique = self.annotation.src_tgt_dist_df.drop_duplicates(subset=self.split_by).copy()
        GroupedDistributionSplitter._check_df_unique_columns(df_unique, self.split_by)
        total_combinations = len(df_unique)
        train_size, val_size, test_size = self._calculate_split_sizes(
            total_combinations=total_combinations,
        )

        if self.holdout_combinations:
            is_in_training_key = "forced_in_train"
            GroupedDistributionSplitter._contains_value(
                df_unique=df_unique,
                combination_keys=self.split_by,
                values=self.annotation.default_values,
                column_key=is_in_training_key,
                accept_nan=True,
            )
        if len(self.force_training_values) > 0:
            is_in_training_key = "forced_in_train"
            GroupedDistributionSplitter._contains_value(
                df_unique=df_unique,
                combination_keys=self.split_by,
                values=self.force_training_values,
                column_key=is_in_training_key,
                accept_nan=False,
            )
        df_unique = self._split_two(
            df_unique=df_unique,
            train_size=train_size,
            random_state=self.random_state,
            is_in_training_key=is_in_training_key,
            not_in_training_key=None,
            split_key=self.split_key,
        )

        # Shuffle test_val and split into test and val
        test_val_mask = df_unique[self.split_key] == "test_val"
        test_val_df = df_unique[test_val_mask].copy()
        test_val_df = test_val_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # Split test_val into test and val
        test_val_df.loc[: test_size - 1, self.split_key] = "test"
        test_val_df.loc[test_size:, self.split_key] = "val"

        # Update the original dataframe with the new split labels
        df_unique.loc[test_val_mask, self.split_key] = test_val_df[self.split_key].values
        return df_unique

    @staticmethod
    def _contains_value(
        df_unique: pd.DataFrame,
        combination_keys: list[str],
        values: dict[str, Any],
        column_key: str,
        accept_nan: bool,
    ):
        """
        Check if the values are in the dataframe, then write it to the column_key.

        If the column_key is already there then just use an OR operator to add the values.
        If column_key is not there then initialize with False

        Parameters
        ----------
        df_unique: pd.DataFrame
            The dataframe to check the values in.
        combination_keys: list[str]
            The keys to check the values in.
        values: dict[str, Any]
            The values to check the values in.
        column_key: str
            The column to write the values to.
        accept_nan: bool
            Whether to accept NaN values.
        """
        GroupedDistributionSplitter._check_df_unique_columns(
            df_unique=df_unique,
            combination_keys=combination_keys,
        )

        # Initialize column_key if it doesn't exist
        if column_key not in df_unique.columns:
            df_unique[column_key] = False

        # For each key-value pair in values dict
        for key, value in values.items():
            if key not in combination_keys:
                continue

            # Check if the row matches the value or is NaN (if accept_nan is True)
            if accept_nan:
                mask = (df_unique[key] == value) | (pd.isna(df_unique[key]))
            else:
                mask = df_unique[key] == value

            # Use OR operator to update column_key
            df_unique[column_key] = df_unique[column_key] | mask

    def _split_src_tgt_dist_df(self) -> pd.DataFrame:
        """Split the src_tgt_dist_df into train, val, and test dataframes."""
        df_split = self._split_df()

        # Convert split column to categorical for faster operations
        df_split[self.split_key] = pd.Categorical(
            df_split[self.split_key],
            categories=["train", "val", "test"],
        )

        # Get the full dataframe and merge on the unique key (src_dist_idx, tgt_dist_idx)
        full_df = self.annotation.src_tgt_dist_df.copy()

        # The unique combination is (src_dist_idx, tgt_dist_idx)
        # We need to merge the split assignments back to the full dataframe
        # df_split contains split_by columns + split_key, we need to merge on split_by
        merge_cols = self.split_by + [self.split_key]
        df_merged = full_df.merge(
            df_split[merge_cols],
            on=self.split_by,
            how="left",
        )

        # Convert split column to categorical in merged df
        df_merged[self.split_key] = pd.Categorical(
            df_merged[self.split_key],
            categories=["train", "val", "test"],
        )
        df_merged.sort_values(by=self.split_key, inplace=True)
        return df_merged

    def split_annotation(self) -> dict[str, GroupedDistributionAnnotation]:
        """
        Split the annotation into train, val, and test GroupedDistributionAnnotations.

        Returns
        -------
        dict[str, GroupedDistributionAnnotation]
            Dictionary with keys 'train', 'val', 'test', each containing a
            GroupedDistributionAnnotation with filtered src_tgt_dist_df and
            corresponding label dictionaries.
        """
        # Get the merged dataframe with split assignments
        df_merged = self._split_src_tgt_dist_df()

        result = {}
        for split_name in ["train", "val", "test"]:
            # Filter the dataframe for this split
            split_df = df_merged[df_merged[self.split_key] == split_name].copy()

            # Drop the split column from the resulting dataframe
            split_df = split_df.drop(columns=[self.split_key])

            # Get the unique src_dist_idx and tgt_dist_idx for this split
            split_tgt_idxs = set(split_df["tgt_dist_idx"].unique())
            split_src_idxs = set(split_df["src_dist_idx"].unique())

            # Filter the label dictionaries to only include relevant indices
            filtered_src_labels = {
                src_idx: self.annotation.src_dist_idx_to_labels[src_idx]
                for src_idx in split_src_idxs
                if src_idx in self.annotation.src_dist_idx_to_labels
            }
            filtered_tgt_labels = {
                tgt_idx: self.annotation.tgt_dist_idx_to_labels[tgt_idx]
                for tgt_idx in split_tgt_idxs
                if tgt_idx in self.annotation.tgt_dist_idx_to_labels
            }

            # Create a new GroupedDistributionAnnotation for this split
            result[split_name] = GroupedDistributionAnnotation(
                old_obs_index=self.annotation.old_obs_index,  # Shared across all splits
                src_dist_idx_to_labels=filtered_src_labels,
                tgt_dist_idx_to_labels=filtered_tgt_labels,
                src_tgt_dist_df=split_df,
                default_values=self.annotation.default_values,
                tgt_dist_keys=self.annotation.tgt_dist_keys,
                src_dist_keys=self.annotation.src_dist_keys,
                dist_flag_key=self.annotation.dist_flag_key,
            )

        return result

    def split(self) -> dict[str, "GroupedDistribution"]:
        """
        Split the GroupedDistribution into train, val, and test GroupedDistributions.

        Returns
        -------
        dict[str, GroupedDistribution]
            Dictionary with keys 'train', 'val', 'test', each containing a
            GroupedDistribution with filtered data and annotation.
        """
        from scaleflow.data._data import GroupedDistribution, GroupedDistributionData

        # Get split annotations first
        split_annotations = self.split_annotation()

        result = {}
        for split_name, split_annotation in split_annotations.items():
            # Get the unique indices for this split
            split_tgt_idxs = set(split_annotation.src_tgt_dist_df["tgt_dist_idx"].unique())
            split_src_idxs = set(split_annotation.src_tgt_dist_df["src_dist_idx"].unique())

            # Filter tgt_data, conditions to only include relevant tgt_dist_idx
            filtered_tgt_data = {
                tgt_idx: self.gd.data.tgt_data[tgt_idx]
                for tgt_idx in split_tgt_idxs
                if tgt_idx in self.gd.data.tgt_data
            }
            filtered_conditions = {
                tgt_idx: self.gd.data.conditions[tgt_idx]
                for tgt_idx in split_tgt_idxs
                if tgt_idx in self.gd.data.conditions
            }

            # Filter src_data to only include relevant src_dist_idx
            filtered_src_data = {
                src_idx: self.gd.data.src_data[src_idx]
                for src_idx in split_src_idxs
                if src_idx in self.gd.data.src_data
            }

            # Build filtered src_to_tgt_dist_map
            # Only include mappings where both src and tgt are in this split
            filtered_src_to_tgt_map = {}
            for src_idx in split_src_idxs:
                if src_idx in self.gd.data.src_to_tgt_dist_map:
                    # Filter to only targets that are in this split
                    filtered_tgts = [
                        tgt_idx for tgt_idx in self.gd.data.src_to_tgt_dist_map[src_idx] if tgt_idx in split_tgt_idxs
                    ]
                    if filtered_tgts:
                        filtered_src_to_tgt_map[src_idx] = filtered_tgts

            # Create the filtered GroupedDistributionData
            filtered_data = GroupedDistributionData(
                src_to_tgt_dist_map=filtered_src_to_tgt_map,
                src_data=filtered_src_data,
                tgt_data=filtered_tgt_data,
                conditions=filtered_conditions,
            )

            # Create the GroupedDistribution for this split
            result[split_name] = GroupedDistribution(
                data=filtered_data,
                annotation=split_annotation,
            )

        return result

    def __repr__(self) -> str:
        """Show the split dataframe."""
        res = ""
        if self._computed_split_df is None:
            self._computed_split_df = self._split_src_tgt_dist_df()
        tmp_dict = dict(tuple(self._computed_split_df.groupby(self.split_key, observed=False)))
        for split_name, df in tmp_dict.items():
            res += f"Split {split_name}:\n"
            res += df.drop(columns=[self.split_key]).to_string(index=False)
            res += "\n"
        return res
