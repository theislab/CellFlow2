"""Data splitter for creating train/validation/test splits from GroupedDistribution objects."""

import logging
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scaleflow.data._data import GroupedDistribution, GroupedDistributionAnnotation

logger = logging.getLogger(__name__)

SplitType = Literal["holdout_groups", "holdout_combinations", "random", "stratified"]


class DataSplitter:
    """
    A lightweight class for creating train/validation/test splits from GroupedDistribution objects.

    This class works on GroupedDistributionAnnotation objects, making it lightweight and easy to
    inspect splits before applying them to the actual data. Splits are performed at the
    distribution level (tgt_dist_idx) rather than at the cell level.

    Supports various splitting strategies:
    - holdout_groups: Hold out specific groups (drugs, cell lines, donors, etc.) for validation/test
    - holdout_combinations: Keep single treatments in training, hold out combination treatments for validation/test
    - random: Random split of distributions
    - stratified: Stratified split maintaining source distribution proportions

    Parameters
    ----------
    annotations : list[GroupedDistributionAnnotation]
        List of GroupedDistributionAnnotation objects to split
    dataset_names : list[str]
        List of names for each dataset (for saving/loading)
    split_ratios : list[list[float]]
        List of triples, each indicating [train, validation, test] ratios for each dataset.
        Each triple must sum to 1.0. Length must match annotations.
    split_type : SplitType
        Type of split to perform
    split_key : str | list[str] | None
        Column name(s) from the src_tgt_dist_df to use for splitting (required for holdout_groups and holdout_combinations).
        Can be a single column or list of columns for combination treatments.
    force_training_values : list[str] | None
        Values that should be forced to appear only in training (e.g., ['control', 'dmso']).
        These values will never appear in validation or test sets.
    control_value : str | list[str] | None
        Value(s) that represent control/untreated condition (e.g., 'control' or ['control', 'dmso']).
        Required for holdout_combinations split type.
    hard_test_split : bool
        If True, validation and test get completely different distributions (no overlap).
        If False, validation and test can share distributions, split at cell level.
    random_state : int
        Random seed for reproducible splits
    test_random_state : int | None
        Random seed specifically for selecting which conditions go to the test set.
        If None, uses random_state as fallback.
    val_random_state : int | None
        Random seed specifically for selecting which conditions go to the validation set.
        If None, uses random_state as fallback.

    Examples
    --------
    >>> # Example: Hold out specific drugs
    >>> splitter = DataSplitter(
    ...     annotations=[gd.annotation],
    ...     dataset_names=["experiment1"],
    ...     split_ratios=[[0.7, 0.15, 0.15]],
    ...     split_type="holdout_groups",
    ...     split_key="drug",  # Column from src_tgt_dist_df
    ...     force_training_values=["control"],
    ... )
    >>> split_annotations = splitter.split_all()
    >>> # Inspect splits before applying to data
    >>> print(split_annotations["experiment1"]["train"].src_tgt_dist_df)
    >>> # Apply to full GroupedDistribution
    >>> train_gd = gd.filter_by_tgt_dist_indices(
    ...     split_annotations["experiment1"]["train"].src_tgt_dist_df["tgt_dist_idx"].tolist()
    ... )
    """

    def __init__(
        self,
        annotations: list[GroupedDistributionAnnotation],
        dataset_names: list[str],
        split_ratios: list[list[float]],
        split_type: SplitType = "random",
        split_key: str | list[str] | None = None,
        force_training_values: list[str] | None = None,
        control_value: str | list[str] | None = None,
        hard_test_split: bool = True,
        random_state: int = 42,
        test_random_state: int | None = None,
        val_random_state: int | None = None,
    ):
        self.annotations = annotations
        self.dataset_names = dataset_names
        self.split_ratios = split_ratios
        self.split_type = split_type
        self.split_key = [split_key] if isinstance(split_key, str) else split_key
        self.force_training_values = force_training_values or []
        self.control_value = [control_value] if isinstance(control_value, str) else control_value
        self.hard_test_split = hard_test_split
        self.random_state = random_state
        self.test_random_state = test_random_state if test_random_state is not None else random_state
        self.val_random_state = val_random_state if val_random_state is not None else random_state

        self._validate_inputs()

        self.split_results: dict[str, dict] = {}

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if len(self.annotations) != len(self.dataset_names):
            raise ValueError(
                f"annotations length ({len(self.annotations)}) must match "
                f"dataset_names length ({len(self.dataset_names)})"
            )

        if not isinstance(self.split_ratios, list):
            raise ValueError("split_ratios must be a list of lists")

        if len(self.split_ratios) != len(self.annotations):
            raise ValueError(
                f"split_ratios length ({len(self.split_ratios)}) must match "
                f"annotations length ({len(self.annotations)})"
            )

        # Check each split ratio
        for i, ratios in enumerate(self.split_ratios):
            if not isinstance(ratios, list) or len(ratios) != 3:
                raise ValueError(f"split_ratios[{i}] must be a list of 3 values [train, val, test]")

            if not np.isclose(sum(ratios), 1.0):
                raise ValueError(f"split_ratios[{i}] must sum to 1.0, got {sum(ratios)}")

            if any(ratio < 0 for ratio in ratios):
                raise ValueError(f"All values in split_ratios[{i}] must be non-negative")

        # Check split key requirement
        if self.split_type in ["holdout_groups", "holdout_combinations"] and self.split_key is None:
            raise ValueError(f"split_key must be provided for split_type '{self.split_type}'")

        # Check control_value requirement for holdout_combinations
        if self.split_type == "holdout_combinations" and self.control_value is None:
            raise ValueError("control_value must be provided for split_type 'holdout_combinations'")

        for i, ann in enumerate(self.annotations):
            if not isinstance(ann, GroupedDistributionAnnotation):
                raise ValueError(f"annotations[{i}] must be a GroupedDistributionAnnotation object")

    def _split_random(
        self,
        dist_indices: list[int],
        split_ratios: list[float]
    ) -> dict[str, list[int]]:
        """Perform random split of distributions."""
        train_ratio, val_ratio, test_ratio = split_ratios
        n_dists = len(dist_indices)

        # Shuffle distributions
        np.random.seed(self.random_state)
        shuffled = np.random.permutation(dist_indices)

        if self.hard_test_split:
            # HARD: Val and test are completely separate
            train_end = int(train_ratio * n_dists)
            val_end = train_end + int(val_ratio * n_dists)

            train_dists = shuffled[:train_end].tolist()
            val_dists = shuffled[train_end:val_end].tolist() if val_ratio > 0 else []
            test_dists = shuffled[val_end:].tolist() if test_ratio > 0 else []

            logger.info("HARD RANDOM SPLIT: Completely separate val/test distributions")
        else:
            # SOFT: Val and test can share distributions (split cells within distributions)
            train_end = int(train_ratio * n_dists)
            train_dists = shuffled[:train_end].tolist()
            val_test_dists = shuffled[train_end:].tolist()

            # Split val+test distributions according to val/test ratios
            if len(val_test_dists) > 0 and val_ratio + test_ratio > 0:
                val_size = val_ratio / (val_ratio + test_ratio)
                np.random.seed(self.random_state + 1)
                val_dists, test_dists = train_test_split(
                    val_test_dists, train_size=val_size, random_state=self.random_state + 1
                )
            else:
                val_dists = []
                test_dists = []

            logger.info("SOFT RANDOM SPLIT: Val/test can share distributions")

        return {"train": train_dists, "val": val_dists, "test": test_dists}

    def _split_holdout_groups(
        self,
        dist_df: pd.DataFrame,
        split_ratios: list[float],
    ) -> dict[str, list[int]]:
        """Split by holding out specific condition groups."""
        if self.split_key is None:
            raise ValueError("split_key must be provided for holdout_groups splitting")

        # Verify split_key columns exist in dist_df
        for key in self.split_key:
            if key not in dist_df.columns:
                raise ValueError(f"split_key '{key}' not found in dist_df columns: {dist_df.columns.tolist()}")

        # Get all unique values from the split_key columns
        unique_values = set()
        for key in self.split_key:
            unique_values.update(dist_df[key].unique())
        unique_values = list(unique_values)

        # Remove forced training values
        available_values = [v for v in unique_values if v not in self.force_training_values]
        forced_train_values = [v for v in unique_values if v in self.force_training_values]

        logger.info(f"Total unique values: {len(unique_values)}")
        logger.info(f"Forced training values: {forced_train_values}")
        logger.info(f"Available for val/test: {len(available_values)}")

        if len(available_values) < 3:
            warnings.warn(
                f"Only {len(available_values)} unique values available for splitting. "
                "Consider using random split instead.",
                stacklevel=2,
            )

        # Split values according to ratios
        train_ratio, val_ratio, test_ratio = split_ratios
        n_values = len(available_values)

        # Calculate number of values for each split
        n_test = int(test_ratio * n_values)
        n_val = int(val_ratio * n_values)
        n_train = n_values - n_test - n_val

        if train_ratio > 0 and n_train == 0:
            n_train = 1
            n_test = max(0, n_test - 1)

        # Step 1: Select test values using test_random_state
        np.random.seed(self.test_random_state)
        shuffled_for_test = np.random.permutation(available_values)
        test_values = shuffled_for_test[-n_test:].tolist() if n_test > 0 else []
        remaining_after_test = shuffled_for_test[:-n_test].tolist() if n_test > 0 else shuffled_for_test.tolist()

        # Step 2: Select val values from remaining using val_random_state
        np.random.seed(self.val_random_state)
        shuffled_for_val = np.random.permutation(remaining_after_test)
        val_values = shuffled_for_val[-n_val:].tolist() if n_val > 0 else []
        train_values_random = shuffled_for_val[:-n_val].tolist() if n_val > 0 else shuffled_for_val.tolist()

        # Step 3: Combine forced training values with randomly assigned training values
        train_values = list(train_values_random) + forced_train_values

        logger.info(f"Split values - Train: {len(train_values)}, Val: {len(val_values)}, Test: {len(test_values)}")

        # Get distributions that contain each value
        def _get_dists_with_values(values_set):
            if len(values_set) == 0:
                return []
            matching_dist_indices = []
            for _, row in dist_df.iterrows():
                # Check if any split_key column contains any of the values
                if any(row[key] in values_set for key in self.split_key):
                    matching_dist_indices.append(row["tgt_dist_idx"])
            return list(set(matching_dist_indices))

        if self.hard_test_split:
            # HARD: Val and test get different values
            train_dists = _get_dists_with_values(train_values)
            val_dists = _get_dists_with_values(val_values)
            test_dists = _get_dists_with_values(test_values)

            logger.info("HARD HOLDOUT GROUPS: Val and test get different values")
        else:
            # SOFT: Val and test can share values
            val_test_values = list(val_values) + list(test_values)
            train_dists = _get_dists_with_values(train_values)
            val_test_dists = _get_dists_with_values(val_test_values)

            # Split val+test distributions at distribution level
            if len(val_test_dists) > 0 and val_ratio + test_ratio > 0:
                val_size = val_ratio / (val_ratio + test_ratio)
                np.random.seed(self.random_state + 1)
                val_dists, test_dists = train_test_split(
                    val_test_dists, train_size=val_size, random_state=self.random_state + 1
                )
                val_dists = val_dists.tolist()
                test_dists = test_dists.tolist()
            else:
                val_dists = []
                test_dists = []

            logger.info("SOFT HOLDOUT GROUPS: Val/test can share values")

        return {
            "train": train_dists,
            "val": val_dists,
            "test": test_dists,
            "train_values": train_values,
            "val_values": val_values,
            "test_values": test_values,
        }

    def _split_holdout_combinations(
        self,
        dist_df: pd.DataFrame,
        split_ratios: list[float],
    ) -> dict[str, list[int]]:
        """Split by keeping single conditions in training and holding out combinations for val/test."""
        if self.split_key is None:
            raise ValueError("split_key must be provided for holdout_combinations splitting")
        if self.control_value is None:
            raise ValueError("control_value must be provided for holdout_combinations splitting")

        logger.info("Identifying combinations vs singletons from distribution labels")
        logger.info(f"Control value(s): {self.control_value}")

        # Classify each distribution as control, singleton, or combination
        control_dists = []
        singleton_dists = []
        combination_dists = []

        for _, row in dist_df.iterrows():
            tgt_dist_idx = row["tgt_dist_idx"]
            # Count non-control values in split_key columns
            non_control_values = [
                row[key] for key in self.split_key 
                if row[key] not in self.control_value
            ]
            n_non_control = len(non_control_values)

            if n_non_control == 0:
                control_dists.append(tgt_dist_idx)
            elif n_non_control == 1:
                singleton_dists.append(tgt_dist_idx)
            else:
                combination_dists.append(tgt_dist_idx)

        # Remove duplicates
        control_dists = list(set(control_dists))
        singleton_dists = list(set(singleton_dists))
        combination_dists = list(set(combination_dists))

        logger.info(f"Found {len(combination_dists)} combination treatment distributions")
        logger.info(f"Found {len(singleton_dists)} singleton treatment distributions")
        logger.info(f"Found {len(control_dists)} control treatment distributions")

        if len(combination_dists) == 0:
            warnings.warn(
                "No combination treatments found. Consider using 'holdout_groups' instead.",
                stacklevel=2
            )

        # All singletons and controls go to training
        train_dists = singleton_dists + control_dists

        # Split combinations according to the provided ratios
        train_ratio, val_ratio, test_ratio = split_ratios

        if len(combination_dists) > 0:
            if self.hard_test_split:
                # HARD: Val and test get completely different combinations
                n_combos = len(combination_dists)
                n_test = int(test_ratio * n_combos)
                n_val = int(val_ratio * n_combos)

                # Step 1: Select test combinations using test_random_state
                np.random.seed(self.test_random_state)
                shuffled_for_test = np.random.permutation(combination_dists)
                test_combo_dists = shuffled_for_test[-n_test:].tolist() if n_test > 0 else []
                remaining_after_test = shuffled_for_test[:-n_test].tolist() if n_test > 0 else shuffled_for_test.tolist()

                # Step 2: Select val combinations from remaining using val_random_state
                np.random.seed(self.val_random_state)
                shuffled_for_val = np.random.permutation(remaining_after_test)
                val_combo_dists = shuffled_for_val[-n_val:].tolist() if n_val > 0 else []
                train_combo_dists = shuffled_for_val[:-n_val].tolist() if n_val > 0 else shuffled_for_val.tolist()

                train_dists.extend(train_combo_dists)
                val_dists = val_combo_dists
                test_dists = test_combo_dists

                logger.info(f"HARD TEST SPLIT - Combinations: Train={len(train_combo_dists)}, Val={len(val_combo_dists)}, Test={len(test_combo_dists)}")
            else:
                # SOFT: Val and test can share combinations
                n_combos = len(combination_dists)
                n_train_combos = int(train_ratio * n_combos)

                # Split combinations into train vs (val+test)
                np.random.seed(self.test_random_state)
                shuffled_combos = np.random.permutation(combination_dists)
                train_combo_dists = shuffled_combos[:n_train_combos].tolist()
                val_test_combo_dists = shuffled_combos[n_train_combos:].tolist()

                train_dists.extend(train_combo_dists)

                # Split val+test distributions
                if len(val_test_combo_dists) > 0 and val_ratio + test_ratio > 0:
                    val_size = val_ratio / (val_ratio + test_ratio)
                    np.random.seed(self.random_state + 1)
                    val_dists, test_dists = train_test_split(
                        val_test_combo_dists, train_size=val_size, random_state=self.random_state + 1
                    )
                    val_dists = val_dists.tolist()
                    test_dists = test_dists.tolist()
                else:
                    val_dists = []
                    test_dists = []

                logger.info(f"SOFT TEST SPLIT - Combinations: Train={len(train_combo_dists)}, Val+Test={len(val_test_combo_dists)}")
        else:
            val_dists = []
            test_dists = []

        logger.info(f"Final split - Train: {len(train_dists)}, Val: {len(val_dists)}, Test: {len(test_dists)} distributions")

        return {"train": train_dists, "val": val_dists, "test": test_dists}

    def _split_stratified(
        self,
        dist_df: pd.DataFrame,
        split_ratios: list[float],
    ) -> dict[str, list[int]]:
        """Perform stratified split maintaining source distribution proportions."""
        train_ratio, val_ratio, test_ratio = split_ratios

        # Use src_dist_idx as stratification labels
        dist_indices = dist_df["tgt_dist_idx"].values
        src_labels = dist_df["src_dist_idx"].values

        if self.hard_test_split:
            # HARD: Val and test get different stratification groups
            if val_ratio + test_ratio > 0:
                train_idx, temp_idx = train_test_split(
                    dist_indices, train_size=train_ratio, stratify=src_labels, random_state=self.random_state
                )

                if val_ratio > 0 and test_ratio > 0:
                    temp_src_labels = dist_df[dist_df["tgt_dist_idx"].isin(temp_idx)]["src_dist_idx"].values
                    val_size = val_ratio / (val_ratio + test_ratio)
                    val_idx, test_idx = train_test_split(
                        temp_idx, train_size=val_size, stratify=temp_src_labels, random_state=self.random_state
                    )
                elif val_ratio > 0:
                    val_idx = temp_idx
                    test_idx = np.array([])
                else:
                    val_idx = np.array([])
                    test_idx = temp_idx
            else:
                train_idx = dist_indices
                val_idx = np.array([])
                test_idx = np.array([])

            logger.info("HARD STRATIFIED SPLIT: Val and test get different strata")
        else:
            # SOFT: Val and test can share stratification groups
            if val_ratio + test_ratio > 0:
                train_idx, val_test_idx = train_test_split(
                    dist_indices, train_size=train_ratio, stratify=src_labels, random_state=self.random_state
                )

                # Split val+test distributions (not stratified)
                if len(val_test_idx) > 0 and val_ratio + test_ratio > 0:
                    val_size = val_ratio / (val_ratio + test_ratio)
                    val_idx, test_idx = train_test_split(
                        val_test_idx, train_size=val_size, random_state=self.random_state + 1
                    )
                else:
                    val_idx = np.array([])
                    test_idx = np.array([])
            else:
                train_idx = dist_indices
                val_idx = np.array([])
                test_idx = np.array([])

            logger.info("SOFT STRATIFIED SPLIT: Val/test can share strata")

        return {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist()
        }

    def split_single(
        self, 
        annotation: GroupedDistributionAnnotation, 
        dataset_index: int
    ) -> dict[str, GroupedDistributionAnnotation]:
        """
        Split a single GroupedDistributionAnnotation according to the specified strategy.

        Parameters
        ----------
        annotation : GroupedDistributionAnnotation
            Annotation object to split
        dataset_index : int
            Index of the dataset to get the correct split ratios

        Returns
        -------
        dict[str, GroupedDistributionAnnotation]
            Dictionary with "train", "val", "test" keys mapping to split annotations
        """
        dist_df = annotation.src_tgt_dist_df
        current_split_ratios = self.split_ratios[dataset_index]

        logger.info(f"Splitting {len(dist_df)} distributions")
        logger.info(f"Unique source distributions: {dist_df['src_dist_idx'].nunique()}")

        # Perform split based on strategy
        if self.split_type == "random":
            split_result = self._split_random(
                dist_df["tgt_dist_idx"].tolist(),
                current_split_ratios
            )
        elif self.split_type == "holdout_groups":
            split_result = self._split_holdout_groups(dist_df, current_split_ratios)
        elif self.split_type == "holdout_combinations":
            split_result = self._split_holdout_combinations(dist_df, current_split_ratios)
        elif self.split_type == "stratified":
            split_result = self._split_stratified(dist_df, current_split_ratios)
        else:
            raise ValueError(f"Unknown split_type: {self.split_type}")

        # Create split annotations
        split_annotations = {}
        for split_name in ["train", "val", "test"]:
            dist_indices = split_result[split_name]
            if len(dist_indices) > 0:
                split_annotations[split_name] = annotation.filter_by_tgt_dist_indices(dist_indices)
            else:
                # Create empty annotation
                split_annotations[split_name] = GroupedDistributionAnnotation(
                    old_obs_index=annotation.old_obs_index,
                    src_dist_idx_to_labels={},
                    tgt_dist_idx_to_labels={},
                    src_tgt_dist_df=pd.DataFrame(columns=dist_df.columns),
                )

        # Store metadata
        metadata = {
            "split_type": self.split_type,
            "split_key": self.split_key,
            "split_ratios": current_split_ratios,
            "random_state": self.random_state,
            "test_random_state": self.test_random_state,
            "val_random_state": self.val_random_state,
            "hard_test_split": self.hard_test_split,
            "train_distributions": len(split_result["train"]),
            "val_distributions": len(split_result["val"]),
            "test_distributions": len(split_result["test"]),
        }

        if self.force_training_values:
            metadata["force_training_values"] = self.force_training_values
        if self.control_value:
            metadata["control_value"] = self.control_value

        # Add split values if available
        if "train_values" in split_result:
            metadata["train_values"] = split_result["train_values"]
            metadata["val_values"] = split_result["val_values"]
            metadata["test_values"] = split_result["test_values"]

        split_annotations["metadata"] = metadata

        # Log split statistics
        logger.info(f"Split results for {self.dataset_names[dataset_index]}:")
        for split_name in ["train", "val", "test"]:
            n_dists = len(split_result[split_name])
            logger.info(f"  {split_name}: {n_dists} distributions")

        return split_annotations

    def split_all(self) -> dict[str, dict[str, GroupedDistributionAnnotation]]:
        """
        Split all GroupedDistributionAnnotation objects according to the specified strategy.

        Returns
        -------
        dict[str, dict[str, GroupedDistributionAnnotation]]
            Nested dictionary with dataset names as keys and split annotations as values.
            Each split contains "train", "val", "test" annotation objects and "metadata".
        """
        logger.info(f"Starting data splitting with strategy: {self.split_type}")
        logger.info(f"Number of datasets: {len(self.annotations)}")

        for i, (annotation, dataset_name) in enumerate(zip(self.annotations, self.dataset_names, strict=True)):
            logger.info(f"\nProcessing dataset {i}: {dataset_name}")
            logger.info(f"Using split ratios: {self.split_ratios[i]}")

            split_result = self.split_single(annotation, i)
            self.split_results[dataset_name] = split_result

        logger.info(f"\nCompleted splitting {len(self.annotations)} datasets")
        return self.split_results

    def save_splits(self, output_dir: str | Path) -> None:
        """
        Save all split information to the specified directory.

        Parameters
        ----------
        output_dir : str | Path
            Directory to save the split information
        """
        import json
        import pickle

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving splits to: {output_dir}")

        for dataset_name, split_info in self.split_results.items():
            dataset_dir = output_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Save each split annotation
            for split_name in ["train", "val", "test"]:
                annotation = split_info[split_name]
                
                # Save as pickle
                annotation_file = dataset_dir / f"{split_name}_annotation.pkl"
                with open(annotation_file, "wb") as f:
                    pickle.dump(annotation, f)
                logger.info(f"Saved {split_name} annotation -> {annotation_file}")

                # Save distribution indices as numpy array for convenience
                if len(annotation.src_tgt_dist_df) > 0:
                    indices_file = dataset_dir / f"{split_name}_dist_indices.npy"
                    np.save(indices_file, annotation.src_tgt_dist_df["tgt_dist_idx"].values)

            # Save metadata as JSON
            metadata_file = dataset_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                # Convert to JSON-serializable format
                metadata_json = split_info["metadata"].copy()
                json.dump(metadata_json, f, indent=2)
            logger.info(f"Saved metadata -> {metadata_file}")

        logger.info("All splits saved successfully")

    @staticmethod
    def load_split_annotations(split_dir: str | Path, dataset_name: str) -> dict[str, GroupedDistributionAnnotation]:
        """
        Load split annotations from disk.

        Parameters
        ----------
        split_dir : str | Path
            Directory containing saved splits
        dataset_name : str
            Name of the dataset

        Returns
        -------
        dict[str, GroupedDistributionAnnotation]
            Dictionary containing train/val/test annotations and metadata
        """
        import pickle
        import json

        split_dir = Path(split_dir)
        dataset_dir = split_dir / dataset_name

        if not dataset_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {dataset_dir}")

        result = {}

        # Load annotations
        for split_name in ["train", "val", "test"]:
            annotation_file = dataset_dir / f"{split_name}_annotation.pkl"
            if annotation_file.exists():
                with open(annotation_file, "rb") as f:
                    result[split_name] = pickle.load(f)
            else:
                raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        # Load metadata
        metadata_file = dataset_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                result["metadata"] = json.load(f)

        return result


def apply_split_to_grouped_distribution(
    gd: GroupedDistribution,
    split_annotations: dict[str, GroupedDistributionAnnotation]
) -> dict[str, GroupedDistribution]:
    """
    Apply split annotations to a GroupedDistribution to get train/val/test GroupedDistributions.

    Parameters
    ----------
    gd : GroupedDistribution
        Full GroupedDistribution to split
    split_annotations : dict[str, GroupedDistributionAnnotation]
        Dictionary with "train", "val", "test" annotations (from DataSplitter.split_single)

    Returns
    -------
    dict[str, GroupedDistribution]
        Dictionary with "train", "val", "test" GroupedDistribution objects

    Examples
    --------
    >>> splitter = DataSplitter(...)
    >>> split_annotations = splitter.split_all()
    >>> train_val_test = apply_split_to_grouped_distribution(
    ...     gd, split_annotations["dataset1"]
    ... )
    >>> train_gd = train_val_test["train"]
    >>> val_gd = train_val_test["val"]
    >>> test_gd = train_val_test["test"]
    """
    result = {}
    for split_name in ["train", "val", "test"]:
        annotation = split_annotations[split_name]
        if len(annotation.src_tgt_dist_df) > 0:
            tgt_dist_indices = annotation.src_tgt_dist_df["tgt_dist_idx"].tolist()
            result[split_name] = gd.filter_by_tgt_dist_indices(tgt_dist_indices)
        else:
            # Create empty GroupedDistribution
            result[split_name] = GroupedDistribution(
                data=gd.data.__class__(
                    src_to_tgt_dist_map={},
                    src_data={},
                    tgt_data={},
                    conditions={},
                ),
                annotation=annotation,
            )
    return result
