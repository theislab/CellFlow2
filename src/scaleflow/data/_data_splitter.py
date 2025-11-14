"""Data splitter for creating train/validation/test splits from TrainingData objects."""

import logging
from typing import Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

from scaleflow.data._data import GroupedDistributionAnnotation

logger = logging.getLogger(__name__)




@dataclass
class AnnotationSplitter:
    annotation: GroupedDistributionAnnotation
    holdout_combinations: bool
    split_keys: list[str]
    force_training_values: dict[str, Any]
    ratios: list[float]
    random_state: int

    def __post_init__(self):
        if len(self.split_keys) == 0:
            raise ValueError("split_keys must be a non-empty list")
        if len(self.ratios) != 3:
            raise ValueError("ratios must be a list of 3 values [train, val, test]")
        if not np.isclose(sum(self.ratios), 1.0):
            raise ValueError("ratios must sum to 1.0")
        if not all(ratio > 0 and ratio < 1.0 for ratio in self.ratios):
            raise ValueError("ratios must be between 0.0 and 1.0")
        if not set(self.force_training_values.keys()).issubset(self.split_keys):
            raise ValueError(f"force_training_values keys must be a subset of split_keys: {self.force_training_values.keys()}")
        self.train_ratio = self.ratios[0]
        self.val_ratio = self.ratios[1]
        self.test_ratio = self.ratios[2]

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
                f"The calculated sizes are {train_size}, {val_size}, {test_size_temp}.")
        test_size = total_combinations - train_size - val_size
        return train_size, val_size, test_size

    def _check_df_unique_columns(self, df_unique: pd.DataFrame) -> None:
        if not all(key in df_unique.columns for key in self.split_keys):
            raise ValueError(f"split_keys must be in df.columns: {self.split_keys}")


    @staticmethod
    def _split_two(
        df_unique: pd.DataFrame,
        train_size: int,
        is_in_training_key: str | None = None,
        not_in_training_key: str | None = None,
        random_state: int = 42,
    ):
        if is_in_training_key is not None:
            if is_in_training_key not in df_unique.columns:
                raise ValueError(f"{is_in_training_key} must be in df.columns: {is_in_training_key}")
        if not_in_training_key is not None:
            if not_in_training_key not in df_unique.columns:
                raise ValueError(f"{not_in_training_key} must be in df.columns: {not_in_training_key}")


        # remove the forced training combinations from the unique combinations
        if is_in_training_key is not None:
            df_unique = df_unique[df_unique[is_in_training_key] == is_in_training_key]
        if not_in_training_key is not None:
            df_unique = df_unique[df_unique[not_in_training_key] != not_in_training_key]
        if len(df_unique) == 0:
            raise ValueError("There must be at least one unique combination of split_by keys after removing forced training combinations")
        # shuffle the unique combinations
        df_unique = df_unique.sample(frac=1, random_state=random_state).reset_index(drop=True)

        train_combinations = df_unique[:train_size]
        test_val_combinations = df_unique[train_size:]

        return train_combinations, test_val_combinations

    def split(self, annotation: GroupedDistributionAnnotation):
        # calculate the sizes of the splits
        is_in_training_key = None
        df = annotation.src_tgt_dist_df.copy()
        df_unique = df.drop_duplicates(subset=self.split_keys)
        self._check_df_unique_columns(df_unique)
        total_combinations = len(df_unique)
        train_size, val_size, test_size = self._calculate_split_sizes(total_combinations)

        if self.holdout_combinations:
            is_in_training_key = "forced_in_train"
            _contains_value(
                df_unique=df_unique,
                combination_keys=self.split_keys,
                values=self.annotation.default_values,
                column_key=is_in_training_key,
                accept_nan=True,
            )
        if len(self.force_training_values) > 0:
            is_in_training_key = "forced_in_train"
            _contains_value(
                df_unique=df_unique,
                combination_keys=self.split_keys,
                values=self.force_training_values,
                column_key=is_in_training_key,
                accept_nan=False,
            )
        train, test_val = self._split_two(
            df_unique=df_unique,
            train_size=train_size,
            random_state=self.random_state,
            is_in_training_key=is_in_training_key,
            not_in_training_key=None,
        )
        # shuffle test_val
        test_val = test_val.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        test = test_val[:test_size]
        val = test_val[test_size:]
        return train, val, test