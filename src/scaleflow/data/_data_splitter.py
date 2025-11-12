"""Data splitter for creating train/validation/test splits from GroupedDistribution objects."""

import logging
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scaleflow.data._data import GroupedDistribution, GroupedDistributionAnnotation, GroupedDistributionData

logger = logging.getLogger(__name__)

SplitType = Literal["holdout_groups", "holdout_combinations", "random", "stratified"]


class AnnotationSplitter:
    """
    A lightweight class for splitting a single GroupedDistributionAnnotation.

    Parameters
    ----------
    annotation : GroupedDistributionAnnotation
        Annotation object to split
    split_ratios : dict[str, float]
        Dictionary of split names to ratios (e.g., {"train": 0.7, "val": 0.15, "test": 0.15}). Must sum to 1.0.
    split_type : SplitType
        Type of split to perform
    split_key : str | list[str] | None
        Column name(s) from src_tgt_dist_df for splitting (required for holdout strategies)
    force_values : dict[str, list[str]] | None
        Dictionary mapping split names to lists of values that must appear in those splits.
        For example, {"train": ["control"], "val": ["drug_A"]} forces "control" to train and "drug_A" to val.
    control_value : str | list[str] | None
        Control/untreated condition values (required for holdout_combinations)
    hard_test_split : bool
        If True, val and test get completely different distributions
    random_state : int
        Random seed for reproducible splits
    error_on_empty : bool
        Whether to allow empty splits
    """

    def __init__(
        self,
        annotation: GroupedDistributionAnnotation,
        split_ratios: dict[str, float],
        split_type: SplitType = "random",
        split_key: str | list[str] | None = None,
        force_values: dict[str, list[str]] | None = None,
        control_value: str | list[str] | None = None,
        hard_test_split: bool = True,
        random_state: int = 42,
        error_on_empty: bool = True,
    ):
        self.annotation = annotation
        self.split_ratios = split_ratios
        self.split_type = split_type
        self.split_key = [split_key] if isinstance(split_key, str) else split_key
        self.force_values = force_values or {}
        self.control_value = [control_value] if isinstance(control_value, str) else control_value
        self.hard_test_split = hard_test_split
        self.random_state = random_state
        self.error_on_empty = error_on_empty

        self._validate_inputs()
        self._split_result: dict | None = None
        self.split_names = sorted(self.split_ratios.keys())

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not isinstance(self.annotation, GroupedDistributionAnnotation):
            raise ValueError("annotation must be a GroupedDistributionAnnotation object")

        if not isinstance(self.split_ratios, dict) or len(self.split_ratios) < 2:
            raise ValueError("split_ratios must be a dict with at least 2 split names")

        if not np.isclose(sum(self.split_ratios.values()), 1.0):
            raise ValueError(f"split_ratios must sum to 1.0, got {sum(self.split_ratios.values())}")

        if any(ratio < 0 for ratio in self.split_ratios.values()):
            raise ValueError("All values in split_ratios must be non-negative")

        if self.split_type in ["holdout_groups", "holdout_combinations"] and self.split_key is None:
            raise ValueError(f"split_key must be provided for split_type '{self.split_type}'")

        if self.split_type == "holdout_combinations" and self.control_value is None:
            raise ValueError("control_value must be provided for split_type 'holdout_combinations'")

        # Validate force_values keys are valid split names
        if self.force_values:
            invalid_keys = set(self.force_values.keys()) - set(self.split_ratios.keys())
            if invalid_keys:
                raise ValueError(
                    f"force_values contains invalid split names: {invalid_keys}. "
                    f"Valid split names are: {self.split_names}"
                )

    def _split_random(self, dist_indices: list[int]) -> dict[str, list[int]]:
        """Perform random split of distributions."""
        n_dists = len(dist_indices)
        np.random.seed(self.random_state)
        shuffled = np.random.permutation(dist_indices)

        result = {}
        start_idx = 0
        for name in self.split_names:
            ratio = self.split_ratios[name]
            if ratio > 0:
                end_idx = start_idx + int(ratio * n_dists)
                result[name] = shuffled[start_idx:end_idx].tolist()
                start_idx = end_idx
            else:
                result[name] = []

        # Check if any split got 0 distributions due to rounding
        empty_splits = [name for name in self.split_names if len(result[name]) == 0 and self.split_ratios[name] > 0]
        if empty_splits:
            msg = (
                f"Split ratios {self.split_ratios} result in empty splits {empty_splits} "
                f"when applied to {n_dists} distributions. "
                f"Consider adjusting ratios (e.g., use larger ratios or fewer splits)."
            )
            if self.error_on_empty:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)

        logger.info(f"Random split: {', '.join(f'{k}={len(v)}' for k, v in result.items())}")
        return result

    def _split_holdout_groups(self, dist_df: pd.DataFrame) -> dict[str, list[int]]:
        """Split by holding out specific condition groups."""
        if self.split_key is None:
            raise ValueError("split_key must be provided for holdout_groups splitting")

        for key in self.split_key:
            if key not in dist_df.columns:
                raise ValueError(f"split_key '{key}' not found in dist_df columns: {dist_df.columns.tolist()}")

        unique_values = set()
        for key in self.split_key:
            unique_values.update(dist_df[key].unique())
        unique_values = list(unique_values)

        # Collect all forced values from all splits
        all_forced_values = set()
        for forced_list in self.force_values.values():
            all_forced_values.update(forced_list)

        # Separate available values (not forced) from forced values
        available_values = [v for v in unique_values if v not in all_forced_values]

        logger.info(f"Total unique values: {len(unique_values)}, Available for splitting: {len(available_values)}")

        n_values = len(available_values)
        n_splits = len(self.split_names)

        # Check if we have enough values for the number of splits
        if n_values < n_splits:
            msg = f"Only {n_values} unique values available for {n_splits} splits."
            if self.error_on_empty:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)

        def _get_dists_with_values(values_set):
            if len(values_set) == 0:
                return []
            matching = []
            for _, row in dist_df.iterrows():
                if any(row[key] in values_set for key in self.split_key):
                    matching.append(row["tgt_dist_idx"])
            return list(set(matching))

        np.random.seed(self.random_state)
        shuffled = np.random.permutation(available_values)

        result = {}
        value_splits = {}
        start_idx = 0

        for name in self.split_names:
            ratio = self.split_ratios[name]
            if ratio > 0:
                n_split = int(ratio * n_values)
                end_idx = start_idx + n_split
                value_splits[name] = shuffled[start_idx:end_idx].tolist()
                start_idx = end_idx
            else:
                value_splits[name] = []

        # Check if any split got 0 values due to rounding
        empty_splits = [name for name in self.split_names if len(value_splits[name]) == 0 and self.split_ratios[name] > 0]
        if empty_splits:
            msg = (
                f"Split ratios {self.split_ratios} result in empty splits {empty_splits} "
                f"when applied to {n_values} unique values. "
                f"Consider adjusting ratios (e.g., use larger ratios or fewer splits)."
            )
            if self.error_on_empty:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)

        # Add forced values to their respective splits
        for split_name, forced_list in self.force_values.items():
            if split_name in value_splits:
                value_splits[split_name].extend(forced_list)

        for name in self.split_names:
            result[name] = _get_dists_with_values(value_splits[name])
            result[f"{name}_values"] = value_splits[name]

        logger.info(
            f"Holdout groups split: {', '.join(f'{k}={len(v)}' for k, v in result.items() if not k.endswith('_values'))}"
        )
        return result

    def _split_holdout_combinations(self, dist_df: pd.DataFrame) -> dict[str, list[int]]:
        """Split by keeping single conditions in training and holding out combinations."""
        if self.split_key is None or self.control_value is None:
            raise ValueError("split_key and control_value must be provided for holdout_combinations splitting")

        control_dists = []
        singleton_dists = []
        combination_dists = []

        for _, row in dist_df.iterrows():
            tgt_dist_idx = row["tgt_dist_idx"]
            non_control = [row[key] for key in self.split_key if row[key] not in self.control_value]
            if len(non_control) == 0:
                control_dists.append(tgt_dist_idx)
            elif len(non_control) == 1:
                singleton_dists.append(tgt_dist_idx)
            else:
                combination_dists.append(tgt_dist_idx)

        control_dists = list(set(control_dists))
        singleton_dists = list(set(singleton_dists))
        combination_dists = list(set(combination_dists))

        logger.info(
            f"Found {len(combination_dists)} combinations, {len(singleton_dists)} singletons, {len(control_dists)} controls"
        )

        if len(combination_dists) == 0:
            warnings.warn("No combination treatments found. Consider using 'holdout_groups' instead.", stacklevel=2)

        # First split gets singletons + controls
        result = {}
        first_name = self.split_names[0]
        result[first_name] = singleton_dists + control_dists

        if len(combination_dists) > 0:
            n_combos = len(combination_dists)
            np.random.seed(self.random_state)
            shuffled = np.random.permutation(combination_dists)

            start_idx = 0
            for name in self.split_names[1:]:
                ratio = self.split_ratios[name]
                if ratio > 0:
                    n_split = int(ratio * n_combos)
                    end_idx = start_idx + n_split
                    result[name] = shuffled[start_idx:end_idx].tolist()
                    start_idx = end_idx
                else:
                    result[name] = []

            # Add remaining combos to first split
            if start_idx < len(shuffled):
                result[first_name].extend(shuffled[start_idx:].tolist())
        else:
            for name in self.split_names[1:]:
                result[name] = []

        logger.info(f"Holdout combinations split: {', '.join(f'{k}={len(v)}' for k, v in result.items())}")
        return result

    def _split_stratified(self, dist_df: pd.DataFrame) -> dict[str, list[int]]:
        """Perform stratified split maintaining source distribution proportions."""
        dist_indices = dist_df["tgt_dist_idx"].values
        src_labels = dist_df["src_dist_idx"].values

        result = {}
        remaining_indices = dist_indices
        remaining_labels = src_labels

        for i, name in enumerate(self.split_names):
            ratio = self.split_ratios[name]
            if ratio > 0 and len(remaining_indices) > 0:
                if i == len(self.split_names) - 1:  # Last split gets remainder
                    result[name] = remaining_indices.tolist()
                else:
                    # Calculate cumulative ratio up to this point
                    cum_ratio = sum(self.split_ratios[n] for n in self.split_names[: i + 1])
                    train_size = cum_ratio / (1 - sum(self.split_ratios[n] for n in self.split_names[:i]))

                    split_idx, remaining_indices = train_test_split(
                        remaining_indices,
                        train_size=ratio / (1 - sum(self.split_ratios[n] for n in self.split_names[:i])),
                        stratify=remaining_labels,
                        random_state=self.random_state + i,
                    )
                    remaining_labels = dist_df[dist_df["tgt_dist_idx"].isin(remaining_indices)]["src_dist_idx"].values
                    result[name] = split_idx.tolist()
            else:
                result[name] = []

        logger.info(f"Stratified split: {', '.join(f'{k}={len(v)}' for k, v in result.items())}")
        return result

    def split(self) -> dict[str, GroupedDistributionAnnotation]:
        """Perform the split and return split annotations."""
        dist_df = self.annotation.src_tgt_dist_df
        logger.info(f"Splitting {len(dist_df)} distributions")

        split_methods = {
            "random": lambda: self._split_random(dist_df["tgt_dist_idx"].tolist()),
            "holdout_groups": lambda: self._split_holdout_groups(dist_df),
            "holdout_combinations": lambda: self._split_holdout_combinations(dist_df),
            "stratified": lambda: self._split_stratified(dist_df),
        }

        if self.split_type not in split_methods:
            raise ValueError(f"Unknown split_type: {self.split_type}")

        split_result = split_methods[self.split_type]()
        self._split_result = split_result

        # Create split annotations
        split_annotations = {}
        for split_name in self.split_names:
            dist_indices = split_result.get(split_name, [])
            if len(dist_indices) > 0:
                split_annotations[split_name] = self.annotation.filter_by_tgt_dist_indices(dist_indices)
            else:
                split_annotations[split_name] = GroupedDistributionAnnotation(
                    old_obs_index=self.annotation.old_obs_index,
                    src_dist_idx_to_labels={},
                    tgt_dist_idx_to_labels={},
                    src_tgt_dist_df=pd.DataFrame(columns=dist_df.columns),
                )

        # Store metadata
        metadata = {
            "split_type": self.split_type,
            "split_key": self.split_key,
            "split_ratios": self.split_ratios,
            "random_state": self.random_state,
            "hard_test_split": self.hard_test_split,
            **{f"{name}_distributions": len(split_result.get(name, [])) for name in self.split_names},
        }

        if self.force_values:
            metadata["force_values"] = self.force_values
        if self.control_value:
            metadata["control_value"] = self.control_value

        # Add value splits if present (from holdout_groups)
        for key in split_result:
            if key.endswith("_values"):
                metadata[key] = split_result[key]

        split_annotations["metadata"] = metadata

        # Validate non-empty splits
        if self.error_on_empty:
            for split_name in self.split_names:
                if len(split_annotations[split_name].src_tgt_dist_df) == 0:
                    raise ValueError(f"Split {split_name} is empty. Set error_on_empty=False to allow empty splits.")

        return split_annotations


class DataSplitter:
    """
    Orchestrator class for splitting multiple GroupedDistribution datasets.

    Parameters
    ----------
    annotations : list[GroupedDistributionAnnotation]
        List of annotations to split
    dataset_names : list[str]
        Names for each dataset
    split_ratios : dict[str, float] | list[dict[str, float]]
        Single dict of split ratios (applied to all) or list of dicts (one per dataset)
    split_type : SplitType
        Type of split (applies to all datasets)
    split_key : str | list[str] | None
        Column name(s) for splitting (applies to all datasets)
    force_values : dict[str, list[str]] | None
        Dictionary mapping split names to lists of values that must appear in those splits.
        For example, {"train": ["control"], "val": ["drug_A"]} forces "control" to train and "drug_A" to val.
    control_value : str | list[str] | None
        Control condition values
    hard_test_split : bool
        Whether val/test get separate distributions
    random_state : int
        Base random seed
    error_on_empty : bool
        Whether to allow empty splits
    """

    def __init__(
        self,
        annotations: list[GroupedDistributionAnnotation],
        dataset_names: list[str],
        split_ratios: dict[str, float] | list[dict[str, float]],
        split_type: SplitType = "random",
        split_key: str | list[str] | None = None,
        force_values: dict[str, list[str]] | None = None,
        control_value: str | list[str] | None = None,
        hard_test_split: bool = True,
        random_state: int = 42,
        error_on_empty: bool = True,
    ):
        self.annotations = annotations
        self.dataset_names = dataset_names

        # Normalize split_ratios to list of dicts
        if isinstance(split_ratios, dict):
            self.split_ratios = [split_ratios] * len(annotations)
        else:
            self.split_ratios = split_ratios

        self.split_type = split_type
        self.split_key = split_key
        self.force_values = force_values
        self.control_value = control_value
        self.hard_test_split = hard_test_split
        self.random_state = random_state
        self.error_on_empty = error_on_empty

        self._validate_inputs()
        self.splitters = [
            AnnotationSplitter(
                annotation=ann,
                split_ratios=ratios,
                split_type=self.split_type,
                split_key=self.split_key,
                force_values=self.force_values,
                control_value=self.control_value,
                hard_test_split=self.hard_test_split,
                random_state=self.random_state,
                error_on_empty=error_on_empty
            )
            for ann, ratios in zip(self.annotations, self.split_ratios, strict=True)
        ]
        self.split_results: dict[str, dict] = {}

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if len(self.annotations) != len(self.dataset_names):
            raise ValueError(
                f"annotations length ({len(self.annotations)}) must match dataset_names length ({len(self.dataset_names)})"
            )

        if isinstance(self.split_ratios, dict):
            ratios_list = [self.split_ratios] * len(self.annotations)
        else:
            ratios_list = self.split_ratios

        if len(ratios_list) != len(self.annotations):
            raise ValueError(
                f"split_ratios length ({len(ratios_list)}) must match annotations length ({len(self.annotations)})"
            )

        for i, ratios in enumerate(ratios_list):
            if not isinstance(ratios, dict) or len(ratios) < 2:
                raise ValueError(f"split_ratios[{i}] must be a dict with at least 2 split names")
            if not np.isclose(sum(ratios.values()), 1.0):
                raise ValueError(f"split_ratios[{i}] must sum to 1.0, got {sum(ratios.values())}")
            if any(r < 0 for r in ratios.values()):
                raise ValueError(f"All values in split_ratios[{i}] must be non-negative")

        if self.split_type in ["holdout_groups", "holdout_combinations"] and self.split_key is None:
            raise ValueError(f"split_key must be provided for split_type '{self.split_type}'")
        if self.split_type == "holdout_combinations" and self.control_value is None:
            raise ValueError("control_value must be provided for split_type 'holdout_combinations'")

    def split_all(self) -> dict[str, dict[str, GroupedDistributionAnnotation]]:
        """Split all annotations using their respective AnnotationSplitters."""
        logger.info(f"Splitting {len(self.annotations)} datasets with strategy: {self.split_type}")

        for splitter, dataset_name in zip(self.splitters, self.dataset_names, strict=True):
            self.split_results[dataset_name] = splitter.split()

        return self.split_results

    def apply_to_grouped_distribution(
        self, gd: GroupedDistribution, dataset_name: str
    ) -> dict[str, GroupedDistribution]:
        """Apply split annotations to a GroupedDistribution."""
        if dataset_name not in self.split_results:
            raise ValueError(f"Dataset '{dataset_name}' has not been split yet. Call split_all() first.")
        return apply_split_to_grouped_distribution(gd, self.split_results[dataset_name])

    def save_splits(self, output_dir: str | Path) -> None:
        """Save all split information to disk."""
        import json
        import pickle

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving splits to: {output_dir}")

        for dataset_name, split_info in self.split_results.items():
            dataset_dir = output_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            split_names = [k for k in split_info.keys() if k != "metadata"]
            for split_name in split_names:
                annotation = split_info[split_name]
                annotation_file = dataset_dir / f"{split_name}_annotation.pkl"
                with open(annotation_file, "wb") as f:
                    pickle.dump(annotation, f)

                if len(annotation.src_tgt_dist_df) > 0:
                    indices_file = dataset_dir / f"{split_name}_dist_indices.npy"
                    np.save(indices_file, annotation.src_tgt_dist_df["tgt_dist_idx"].values)

            metadata_file = dataset_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(split_info["metadata"], f, indent=2)

        logger.info("All splits saved successfully")

    @staticmethod
    def load_split_annotations(split_dir: str | Path, dataset_name: str) -> dict[str, GroupedDistributionAnnotation]:
        """Load split annotations from disk."""
        import pickle
        import json

        split_dir = Path(split_dir)
        dataset_dir = split_dir / dataset_name

        if not dataset_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {dataset_dir}")

        result = {}
        for file in dataset_dir.glob("*_annotation.pkl"):
            split_name = file.stem.replace("_annotation", "")
            with open(file, "rb") as f:
                result[split_name] = pickle.load(f)

        metadata_file = dataset_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                result["metadata"] = json.load(f)

        return result


def apply_split_to_grouped_distribution(
    gd: GroupedDistribution, split_annotations: dict[str, GroupedDistributionAnnotation]
) -> dict[str, GroupedDistribution]:
    """Apply split annotations to a GroupedDistribution to get split GroupedDistributions."""
    result = {}
    split_names = [k for k in split_annotations.keys() if k != "metadata"]

    for split_name in split_names:
        annotation = split_annotations[split_name]
        if len(annotation.src_tgt_dist_df) > 0:
            tgt_dist_indices = annotation.src_tgt_dist_df["tgt_dist_idx"].tolist()
            result[split_name] = gd.filter_by_tgt_dist_indices(tgt_dist_indices)
        else:
            result[split_name] = GroupedDistribution(
                data=GroupedDistributionData(
                    src_to_tgt_dist_map={},
                    src_data={},
                    tgt_data={},
                    conditions={},
                ),
                annotation=annotation,
            )
    return result
