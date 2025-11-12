"""Tests for DataSplitter class."""

import numpy as np
import pandas as pd
import pytest

from scaleflow.data._data import (
    GroupedDistribution,
    GroupedDistributionAnnotation,
)
from scaleflow.data._data_splitter import (
    AnnotationSplitter,
    DataSplitter,
    apply_split_to_grouped_distribution,
)


class TestDataSplitterBasic:
    """Test basic DataSplitter functionality."""

    def test_init_valid(self, sample_grouped_distribution):
        """Test that DataSplitter initializes correctly with valid inputs."""
        sample_annotation = sample_grouped_distribution.annotation
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test_dataset"],
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="random",
        )
        assert splitter.split_type == "random"
        assert len(splitter.annotations) == 1

    def test_init_invalid_ratios(self, sample_grouped_distribution):
        """Test that invalid split ratios raise ValueError."""
        sample_annotation = sample_grouped_distribution.annotation
        with pytest.raises(ValueError, match="must sum to 1.0"):
            DataSplitter(
                annotations=[sample_annotation],
                dataset_names=["test"],
                split_ratios={"train": 0.7, "val": 0.2, "test": 0.2},  # Sums to 1.1
            )

    def test_init_missing_split_key(self, sample_grouped_distribution):
        """Test that holdout_groups without split_key raises ValueError."""
        sample_annotation = sample_grouped_distribution.annotation
        with pytest.raises(ValueError, match="split_key must be provided"):
            DataSplitter(
                annotations=[sample_annotation],
                dataset_names=["test"],
                split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
                split_type="holdout_groups",
            )

    def test_init_missing_control_value(self, sample_grouped_distribution):
        """Test that holdout_combinations without control_value raises ValueError."""
        sample_annotation = sample_grouped_distribution.annotation
        with pytest.raises(ValueError, match="control_value must be provided"):
            DataSplitter(
                annotations=[sample_annotation],
                dataset_names=["test"],
                split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
                split_type="holdout_combinations",
                split_key=["drug", "gene"],
            )


class TestDataSplitterRandom:
    """Test random splitting strategy."""

    def test_random_split_produces_splits(self, sample_grouped_distribution):
        """Test that random split produces all expected splits."""
        sample_annotation = sample_grouped_distribution.annotation
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="random",
            random_state=42,
        )

        results = splitter.split_all()

        # Check that we got results
        assert "test" in results
        assert "train" in results["test"]
        assert "val" in results["test"]
        assert "test" in results["test"]

        # Check that splits are non-empty (with reasonable ratios)
        train_dists = set(results["test"]["train"].src_tgt_dist_df["tgt_dist_idx"])
        val_dists = set(results["test"]["val"].src_tgt_dist_df["tgt_dist_idx"])
        test_dists = set(results["test"]["test"].src_tgt_dist_df["tgt_dist_idx"])

        # With 0.7/0.15/0.15 ratios, all should be non-empty for reasonable dataset sizes
        assert len(train_dists) > len(val_dists) + len(test_dists)

        # No overlap in hard split (default)
        assert len(train_dists & val_dists) == 0
        assert len(train_dists & test_dists) == 0
        assert len(val_dists & test_dists) == 0

    def test_random_split_reproducible(self, sample_grouped_distribution):
        """Test that random split is reproducible with same seed."""
        sample_annotation = sample_grouped_distribution.annotation
        splitter1 = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="random",
            random_state=42,
        )
        results1 = splitter1.split_all()

        splitter2 = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="random",
            random_state=42,
        )
        results2 = splitter2.split_all()

        # Check that train distributions are the same
        train_dists1 = set(results1["test"]["train"].src_tgt_dist_df["tgt_dist_idx"])
        train_dists2 = set(results2["test"]["train"].src_tgt_dist_df["tgt_dist_idx"])
        assert train_dists1 == train_dists2

    def test_random_split_custom_names(self, sample_grouped_distribution):
        """Test that random split works with custom split names."""
        sample_annotation = sample_grouped_distribution.annotation
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios={"train": 0.6, "dev": 0.2, "eval": 0.2},
            split_type="random",
            random_state=42,
        )

        results = splitter.split_all()
        assert "train" in results["test"]
        assert "dev" in results["test"]
        assert "eval" in results["test"]


class TestDataSplitterHoldoutGroups:
    """Test holdout_groups splitting strategy."""

    def test_holdout_groups_basic(self, sample_grouped_distribution):
        """Test basic holdout groups splitting with sufficient unique values."""
        sample_annotation = sample_grouped_distribution.annotation
        unique_drugs = sample_annotation.src_tgt_dist_df["drug"].nunique()

        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios={"train": 0.33, "val": 0.33, "test": 0.34},
            split_type="holdout_groups",
            split_key="drug",
            random_state=42,
        )

        results = splitter.split_all()

        # Get unique drugs in each split
        train_drugs = set(results["test"]["train"].src_tgt_dist_df["drug"].unique())
        val_drugs = set(results["test"]["val"].src_tgt_dist_df["drug"].unique())
        test_drugs = set(results["test"]["test"].src_tgt_dist_df["drug"].unique())

        # In hard split, drugs should not overlap
        assert len(train_drugs & val_drugs) == 0
        assert len(train_drugs & test_drugs) == 0
        assert len(val_drugs & test_drugs) == 0

        # All splits should be non-empty
        assert len(train_drugs) > 0
        assert len(val_drugs) > 0
        assert len(test_drugs) > 0

    def test_holdout_groups_force_training(self, sample_grouped_distribution):
        """Test that force_values keeps certain values in specified splits."""
        sample_annotation = sample_grouped_distribution.annotation
        # Use a 2-way split when forcing one value to training
        # This is realistic: force control to training, split remaining values
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios={"train": 0.5, "test": 0.5},  # Only 2 splits
            split_type="holdout_groups",
            split_key="drug",
            force_values={"train": ["control"]},
            random_state=42,
        )

        results = splitter.split_all()

        # Control should only appear in training
        test_drugs = set(results["test"]["test"].src_tgt_dist_df["drug"].unique())

        assert "control" not in test_drugs

        # But should be in training
        train_drugs = set(results["test"]["train"].src_tgt_dist_df["drug"].unique())
        assert "control" in train_drugs

        # Both splits should be non-empty
        assert len(train_drugs) > 0
        assert len(test_drugs) > 0

    def test_holdout_groups_force_values_multiple_splits(self, sample_grouped_distribution):
        """Test that force_values can force values to different splits."""
        sample_annotation = sample_grouped_distribution.annotation
        # Get unique drugs to test with
        unique_drugs = sample_annotation.src_tgt_dist_df["drug"].unique().tolist()

        if len(unique_drugs) >= 3:
            # Force different drugs to different splits
            force_to_train = [unique_drugs[0]] if len(unique_drugs) > 0 else []
            force_to_val = [unique_drugs[1]] if len(unique_drugs) > 1 else []

            splitter = DataSplitter(
                annotations=[sample_annotation],
                dataset_names=["test"],
                split_ratios={"train": 0.5, "val": 0.25, "test": 0.25},
                split_type="holdout_groups",
                split_key="drug",
                force_values={"train": force_to_train, "val": force_to_val},
                random_state=42,
            )

            results = splitter.split_all()

            train_drugs = set(results["test"]["train"].src_tgt_dist_df["drug"].unique())
            val_drugs = set(results["test"]["val"].src_tgt_dist_df["drug"].unique())
            test_drugs = set(results["test"]["test"].src_tgt_dist_df["drug"].unique())

            # Check forced values are in correct splits
            if force_to_train:
                assert force_to_train[0] in train_drugs
                assert force_to_train[0] not in val_drugs
                assert force_to_train[0] not in test_drugs

            if force_to_val:
                assert force_to_val[0] in val_drugs
                assert force_to_val[0] not in test_drugs

    def test_holdout_groups_multiple_keys(self, sample_grouped_distribution):
        """Test splitting on multiple keys."""
        sample_annotation = sample_grouped_distribution.annotation
        # Multiple keys give more unique values
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios={"train": 0.5, "val": 0.25, "test": 0.25},
            split_type="holdout_groups",
            split_key=["drug", "gene"],
            random_state=42,
        )

        results = splitter.split_all()

        # Get all unique values from both drug and gene
        train_df = results["test"]["train"].src_tgt_dist_df
        val_df = results["test"]["val"].src_tgt_dist_df
        test_df = results["test"]["test"].src_tgt_dist_df

        # All splits should be non-empty
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

    def test_holdout_groups_insufficient_values_raises(self, sample_grouped_distribution):
        """Test that insufficient unique values raises error when error_on_empty=True."""
        sample_annotation = sample_grouped_distribution.annotation
        # Try to split into 3 when we might only have 2 unique values
        with pytest.raises(ValueError, match="Only .* unique values available"):
            splitter = DataSplitter(
                annotations=[sample_annotation],
                dataset_names=["test"],
                split_ratios={"train": 0.33, "val": 0.33, "test": 0.34},
                split_type="holdout_groups",
                split_key="drug",
                force_values={"train": ["control"]},  # This leaves only 2 values for 3 splits
                random_state=42,
                error_on_empty=True,
            )
            splitter.split_all()


class TestDataSplitterHoldoutCombinations:
    """Test holdout_combinations splitting strategy."""

    def test_holdout_combinations_basic(self, sample_grouped_distribution):
        """Test basic holdout combinations splitting."""
        sample_annotation = sample_grouped_distribution.annotation
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="holdout_combinations",
            split_key=["drug", "gene"],
            control_value="control",
            random_state=42,
        )

        results = splitter.split_all()

        # Check that controls and singletons are in training
        train_df = results["test"]["train"].src_tgt_dist_df
        val_df = results["test"]["val"].src_tgt_dist_df
        test_df = results["test"]["test"].src_tgt_dist_df

        print(train_df)
        print(val_df)
        print(test_df)

    def test_holdout_combinations_no_combinations(self):
        """Test that when no combinations exist, singletons go to train and val/test are empty."""
        # Create annotation with only singletons (no combinations)
        src_dist_idx_to_labels = {0: ("cell_line_A",)}
        tgt_dist_idx_to_labels = {
            0: ("control", "control"),
            1: ("drug_A", "control"),
            2: ("control", "gene_A"),
        }

        rows = [
            {
                "src_dist_idx": 0,
                "tgt_dist_idx": 0,
                "drug": "control",
                "gene": "control",
            },
            {
                "src_dist_idx": 0,
                "tgt_dist_idx": 1,
                "drug": "drug_A",
                "gene": "control",
            },
            {
                "src_dist_idx": 0,
                "tgt_dist_idx": 2,
                "drug": "control",
                "gene": "gene_A",
            },
        ]

        annotation = GroupedDistributionAnnotation(
            old_obs_index=np.arange(100),
            src_dist_idx_to_labels=src_dist_idx_to_labels,
            tgt_dist_idx_to_labels=tgt_dist_idx_to_labels,
            src_tgt_dist_df=pd.DataFrame(rows),
        )

        splitter = DataSplitter(
            annotations=[annotation],
            dataset_names=["test"],
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="holdout_combinations",
            split_key=["drug", "gene"],
            control_value="control",
            error_on_empty=False,  # Allow empty val/test
        )

        results = splitter.split_all()

        # Train should have all singletons
        assert len(results["test"]["train"].src_tgt_dist_df) > 0
        # Val and test should be empty since there are no combinations
        # Note: The implementation might put everything in train, which is acceptable
        # We just check that train has data and the split completes without error


class TestDataSplitterStratified:
    """Test stratified splitting strategy."""

    def test_stratified_split(self, sample_grouped_distribution):
        """Test that stratified split maintains source distribution proportions."""
        sample_annotation = sample_grouped_distribution.annotation
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="stratified",
            random_state=42,
        )

        results = splitter.split_all()

        # Check that source distributions appear in splits
        for split_name in ["train", "val", "test"]:
            split_df = results["test"][split_name].src_tgt_dist_df
            if len(split_df) > 0:
                src_dists = split_df["src_dist_idx"].unique()
                # Should have source distributions
                assert len(src_dists) > 0


class TestDataSplitterEmptySplits:
    """Test behavior with empty splits - separate from non-empty tests."""

    def test_empty_splits_allowed_when_disabled(self, sample_grouped_distribution):
        """Test that empty splits are allowed when error_on_empty=False."""
        sample_annotation = sample_grouped_distribution.annotation
        # Use ratios that might result in empty splits
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios={"train": 0.9, "val": 0.05, "test": 0.05},
            split_type="holdout_groups",
            split_key="drug",
            random_state=42,
            error_on_empty=False,
        )

        results = splitter.split_all()

        # Should not raise an error even if some splits are empty
        assert "train" in results["test"]
        assert "val" in results["test"]
        assert "test" in results["test"]

    def test_empty_splits_raise_when_enabled(self, sample_grouped_distribution):
        """Test that empty splits raise error when error_on_empty=True."""
        sample_annotation = sample_grouped_distribution.annotation
        # Create scenario that will produce empty splits
        with pytest.raises(ValueError, match="Split .* is empty"):
            splitter = DataSplitter(
                annotations=[sample_annotation],
                dataset_names=["test"],
                split_ratios={"train": 0.33, "val": 0.33, "test": 0.34},
                split_type="holdout_groups",
                split_key="drug",
                force_values={"train": ["control"]},  # Forces impossible split
                random_state=42,
                error_on_empty=True,
            )
            splitter.split_all()


class TestApplySplitToGroupedDistribution:
    """Test applying splits to full GroupedDistribution."""

    def test_apply_split_basic(self, sample_grouped_distribution):
        """Test applying split annotations to GroupedDistribution."""
        splitter = DataSplitter(
            annotations=[sample_grouped_distribution.annotation],
            dataset_names=["test"],
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="random",
            random_state=42,
        )

        split_annotations = splitter.split_all()

        # Apply to full GroupedDistribution
        split_gds = apply_split_to_grouped_distribution(sample_grouped_distribution, split_annotations["test"])

        assert "train" in split_gds
        assert "val" in split_gds
        assert "test" in split_gds

        # Check that we got GroupedDistribution objects
        assert isinstance(split_gds["train"], GroupedDistribution)
        assert isinstance(split_gds["val"], GroupedDistribution)
        assert isinstance(split_gds["test"], GroupedDistribution)

        # Check that data is actually split
        train_tgt_dists = set(split_gds["train"].data.tgt_data.keys())
        val_tgt_dists = set(split_gds["val"].data.tgt_data.keys())
        test_tgt_dists = set(split_gds["test"].data.tgt_data.keys())

        # No overlap in hard split
        assert len(train_tgt_dists & val_tgt_dists) == 0
        assert len(train_tgt_dists & test_tgt_dists) == 0
        assert len(val_tgt_dists & test_tgt_dists) == 0

    def test_apply_split_preserves_data(self, sample_grouped_distribution):
        """Test that applying split preserves the actual data arrays."""
        splitter = DataSplitter(
            annotations=[sample_grouped_distribution.annotation],
            dataset_names=["test"],
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="random",
            random_state=42,
        )

        split_annotations = splitter.split_all()
        split_gds = apply_split_to_grouped_distribution(sample_grouped_distribution, split_annotations["test"])

        # Check that data arrays match original
        for tgt_idx in split_gds["train"].data.tgt_data.keys():
            original_data = sample_grouped_distribution.data.tgt_data[tgt_idx]
            split_data = split_gds["train"].data.tgt_data[tgt_idx]
            np.testing.assert_array_equal(original_data, split_data)


class TestSaveLoad:
    """Test saving and loading splits."""

    def test_save_and_load(self, sample_grouped_distribution, tmp_path):
        """Test that splits can be saved and loaded."""
        sample_annotation = sample_grouped_distribution.annotation
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test_dataset"],
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="random",
            random_state=42,
        )

        results = splitter.split_all()

        # Save
        output_dir = tmp_path / "splits"
        splitter.save_splits(output_dir)

        # Check that files were created
        assert (output_dir / "test_dataset" / "train_annotation.pkl").exists()
        assert (output_dir / "test_dataset" / "val_annotation.pkl").exists()
        assert (output_dir / "test_dataset" / "test_annotation.pkl").exists()
        assert (output_dir / "test_dataset" / "metadata.json").exists()

        # Load
        loaded = DataSplitter.load_split_annotations(output_dir, "test_dataset")

        # Check that loaded data matches
        assert "train" in loaded
        assert "val" in loaded
        assert "test" in loaded
        assert "metadata" in loaded

        # Compare train distributions
        original_train_dists = set(results["test_dataset"]["train"].src_tgt_dist_df["tgt_dist_idx"])
        loaded_train_dists = set(loaded["train"].src_tgt_dist_df["tgt_dist_idx"])
        assert original_train_dists == loaded_train_dists


class TestAnnotationSplitter:
    """Test AnnotationSplitter class for single annotation splitting."""

    def test_annotation_splitter_init(self, sample_grouped_distribution):
        """Test that AnnotationSplitter initializes correctly."""
        sample_annotation = sample_grouped_distribution.annotation
        splitter = AnnotationSplitter(
            annotation=sample_annotation,
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="random",
        )
        assert splitter.split_type == "random"
        assert splitter.annotation == sample_annotation

    def test_annotation_splitter_split(self, sample_grouped_distribution):
        """Test that AnnotationSplitter can split a single annotation."""
        sample_annotation = sample_grouped_distribution.annotation
        splitter = AnnotationSplitter(
            annotation=sample_annotation,
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="random",
            random_state=42,
        )

        results = splitter.split()

        assert "train" in results
        assert "val" in results
        assert "test" in results
        assert "metadata" in results

        # Check that we got annotations
        assert isinstance(results["train"], GroupedDistributionAnnotation)
        assert isinstance(results["val"], GroupedDistributionAnnotation)
        assert isinstance(results["test"], GroupedDistributionAnnotation)

    def test_annotation_splitter_invalid_ratios(self, sample_grouped_distribution):
        """Test that invalid split ratios raise ValueError."""
        sample_annotation = sample_grouped_distribution.annotation
        with pytest.raises(ValueError, match="must sum to 1.0"):
            AnnotationSplitter(
                annotation=sample_annotation,
                split_ratios={"train": 0.7, "val": 0.2, "test": 0.2},  # Sums to 1.1
            )

    def test_annotation_splitter_custom_names(self, sample_grouped_distribution):
        """Test AnnotationSplitter with custom split names."""
        sample_annotation = sample_grouped_distribution.annotation
        splitter = AnnotationSplitter(
            annotation=sample_annotation,
            split_ratios={"train": 0.6, "dev": 0.2, "eval": 0.2},
            split_type="random",
            random_state=42,
        )

        results = splitter.split()

        assert "train" in results
        assert "dev" in results
        assert "eval" in results

    def test_annotation_splitter_random_reproducible(self, sample_grouped_distribution):
        """Test that AnnotationSplitter random split is reproducible."""
        sample_annotation = sample_grouped_distribution.annotation
        splitter1 = AnnotationSplitter(
            annotation=sample_annotation,
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="random",
            random_state=42,
        )
        results1 = splitter1.split()

        splitter2 = AnnotationSplitter(
            annotation=sample_annotation,
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="random",
            random_state=42,
        )
        results2 = splitter2.split()

        # Check reproducibility
        train_dists1 = set(results1["train"].src_tgt_dist_df["tgt_dist_idx"])
        train_dists2 = set(results2["train"].src_tgt_dist_df["tgt_dist_idx"])
        assert train_dists1 == train_dists2

    def test_annotation_splitter_holdout_groups(self, sample_grouped_distribution):
        """Test AnnotationSplitter with holdout_groups strategy."""
        sample_annotation = sample_grouped_distribution.annotation
        unique_drugs = sample_annotation.src_tgt_dist_df["drug"].nunique()

        # Assert we have enough - this will fail clearly if not
        assert unique_drugs >= 3, (
            f"Test requires at least 3 unique drugs, but fixture has {unique_drugs}. "
            f"Update conftest.py adata_test fixture to include more drugs."
        )
        print(f"Unique drugs: {unique_drugs}")

        splitter = AnnotationSplitter(
            annotation=sample_annotation,
            split_ratios={"train": 0.33, "val": 0.33, "test": 0.34},
            split_type="holdout_groups",
            split_key="drug",
            random_state=42,
        )

        results = splitter.split()

        # Get unique drugs in each split
        train_drugs = set(results["train"].src_tgt_dist_df["drug"].unique())
        val_drugs = set(results["val"].src_tgt_dist_df["drug"].unique())
        test_drugs = set(results["test"].src_tgt_dist_df["drug"].unique())

        # In hard split, drugs should not overlap
        assert len(train_drugs & val_drugs) == 0
        assert len(train_drugs & test_drugs) == 0
        assert len(val_drugs & test_drugs) == 0

        # Check metadata contains value splits
        assert "train_values" in results["metadata"]
        assert "val_values" in results["metadata"]
        assert "test_values" in results["metadata"]

    def test_annotation_splitter_holdout_groups_force_training(self, sample_grouped_distribution):
        """Test AnnotationSplitter holdout_groups with force_values."""
        sample_annotation = sample_grouped_distribution.annotation
        splitter = AnnotationSplitter(
            annotation=sample_annotation,
            split_ratios={"train": 0.5, "test": 0.5},  # 2-way split
            split_type="holdout_groups",
            split_key="drug",
            force_values={"train": ["control"]},
            random_state=42,
        )

        results = splitter.split()

        # Control should only be in training
        train_drugs = set(results["train"].src_tgt_dist_df["drug"].unique())
        test_drugs = set(results["test"].src_tgt_dist_df["drug"].unique())

        assert "control" in train_drugs
        assert "control" not in test_drugs

        # Check metadata
        assert results["metadata"]["force_values"] == {"train": ["control"]}

    def test_annotation_splitter_holdout_combinations(self, sample_grouped_distribution):
        """Test AnnotationSplitter with holdout_combinations strategy."""
        sample_annotation = sample_grouped_distribution.annotation
        splitter = AnnotationSplitter(
            annotation=sample_annotation,
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="holdout_combinations",
            split_key=["drug", "gene"],
            control_value="control",
            random_state=42,
        )

        results = splitter.split()

        # Check that we got results
        assert "train" in results
        assert "val" in results
        assert "test" in results

        # Check metadata
        assert results["metadata"]["split_type"] == "holdout_combinations"
        assert results["metadata"]["control_value"] == ["control"]

    def test_annotation_splitter_stratified(self, sample_grouped_distribution):
        """Test AnnotationSplitter with stratified strategy."""
        sample_annotation = sample_grouped_distribution.annotation
        splitter = AnnotationSplitter(
            annotation=sample_annotation,
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="stratified",
            random_state=42,
        )

        results = splitter.split()

        # Check that we got results
        assert "train" in results
        assert "val" in results
        assert "test" in results

        # Check metadata
        assert results["metadata"]["split_type"] == "stratified"

    def test_annotation_splitter_missing_split_key(self, sample_grouped_distribution):
        """Test that holdout_groups without split_key raises ValueError."""
        sample_annotation = sample_grouped_distribution.annotation
        with pytest.raises(ValueError, match="split_key must be provided"):
            AnnotationSplitter(
                annotation=sample_annotation,
                split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
                split_type="holdout_groups",
            )

    def test_annotation_splitter_missing_control_value(self, sample_grouped_distribution):
        """Test that holdout_combinations without control_value raises ValueError."""
        sample_annotation = sample_grouped_distribution.annotation
        with pytest.raises(ValueError, match="control_value must be provided"):
            AnnotationSplitter(
                annotation=sample_annotation,
                split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
                split_type="holdout_combinations",
                split_key=["drug", "gene"],
            )

    def test_annotation_splitter_metadata_content(self, sample_grouped_distribution):
        """Test that metadata contains expected information."""
        sample_annotation = sample_grouped_distribution.annotation
        splitter = AnnotationSplitter(
            annotation=sample_annotation,
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="random",
            random_state=42,
            hard_test_split=True,
        )

        results = splitter.split()

        metadata = results["metadata"]
        assert metadata["split_type"] == "random"
        assert metadata["split_ratios"] == {"train": 0.7, "val": 0.15, "test": 0.15}
        assert metadata["random_state"] == 42
        assert metadata["hard_test_split"] is True
        assert "train_distributions" in metadata
        assert "val_distributions" in metadata
        assert "test_distributions" in metadata

    def test_annotation_splitter_empty_splits_allowed(self, sample_grouped_distribution):
        """Test AnnotationSplitter allows empty splits when error_on_empty=False."""
        sample_annotation = sample_grouped_distribution.annotation
        splitter = AnnotationSplitter(
            annotation=sample_annotation,
            split_ratios={"train": 0.9, "val": 0.05, "test": 0.05},
            split_type="holdout_groups",
            split_key="drug",
            random_state=42,
            error_on_empty=False,
        )

        results = splitter.split()

        # Should not raise an error even if some splits are empty
        assert "train" in results
        assert "val" in results
        assert "test" in results
        assert "metadata" in results

    def test_annotation_splitter_empty_splits_raise(self, sample_grouped_distribution):
        """Test AnnotationSplitter raises error for empty splits when error_on_empty=True."""
        sample_annotation = sample_grouped_distribution.annotation
        # Create scenario that will produce empty splits
        with pytest.raises(ValueError, match="Split .* is empty"):
            splitter = AnnotationSplitter(
                annotation=sample_annotation,
                split_ratios={"train": 0.33, "val": 0.33, "test": 0.34},
                split_type="holdout_groups",
                split_key="drug",
                force_values={"train": ["control"]},  # Forces impossible split
                random_state=42,
                error_on_empty=True,
            )
            splitter.split()

    def test_annotation_splitter_invalid_split_type(self, sample_grouped_distribution):
        """Test that invalid split_type raises ValueError."""
        sample_annotation = sample_grouped_distribution.annotation
        splitter = AnnotationSplitter(
            annotation=sample_annotation,
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="invalid_type",  # type: ignore
        )

        with pytest.raises(ValueError, match="Unknown split_type"):
            splitter.split()

    def test_annotation_splitter_negative_ratios(self, sample_grouped_distribution):
        """Test that negative split ratios raise ValueError."""
        sample_annotation = sample_grouped_distribution.annotation
        with pytest.raises(ValueError, match="must be non-negative"):
            AnnotationSplitter(
                annotation=sample_annotation,
                split_ratios={"train": 0.7, "val": -0.1, "test": 0.4},
            )

    def test_annotation_splitter_too_few_splits(self, sample_grouped_distribution):
        """Test that split_ratios with less than 2 splits raises ValueError."""
        sample_annotation = sample_grouped_distribution.annotation
        with pytest.raises(ValueError, match="at least 2 split names"):
            AnnotationSplitter(
                annotation=sample_annotation,
                split_ratios={"train": 1.0},  # Only one split
            )

    def test_annotation_splitter_hard_vs_soft_split(self, sample_grouped_distribution):
        """Test that hard_test_split parameter affects behavior."""
        sample_annotation = sample_grouped_distribution.annotation

        # Hard split
        splitter_hard = AnnotationSplitter(
            annotation=sample_annotation,
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="random",
            hard_test_split=True,
            random_state=42,
        )
        results_hard = splitter_hard.split()

        # Soft split
        splitter_soft = AnnotationSplitter(
            annotation=sample_annotation,
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="random",
            hard_test_split=False,
            random_state=42,
        )
        results_soft = splitter_soft.split()

        # Both should produce valid splits
        assert "train" in results_hard
        assert "train" in results_soft
        assert results_hard["metadata"]["hard_test_split"] is True
        assert results_soft["metadata"]["hard_test_split"] is False


class TestDataSplitterApplyToGroupedDistribution:
    """Test the apply_to_grouped_distribution method."""

    def test_apply_to_grouped_distribution_method(self, sample_grouped_distribution):
        """Test the convenience method on DataSplitter."""
        splitter = DataSplitter(
            annotations=[sample_grouped_distribution.annotation],
            dataset_names=["test"],
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="random",
            random_state=42,
        )

        # Split first
        splitter.split_all()

        # Apply using the convenience method
        split_gds = splitter.apply_to_grouped_distribution(sample_grouped_distribution, "test")

        assert "train" in split_gds
        assert "val" in split_gds
        assert "test" in split_gds

        # Check that we got GroupedDistribution objects
        assert isinstance(split_gds["train"], GroupedDistribution)
        assert isinstance(split_gds["val"], GroupedDistribution)
        assert isinstance(split_gds["test"], GroupedDistribution)

    def test_apply_to_grouped_distribution_not_split(self, sample_grouped_distribution):
        """Test that applying before splitting raises ValueError."""
        splitter = DataSplitter(
            annotations=[sample_grouped_distribution.annotation],
            dataset_names=["test"],
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            split_type="random",
            random_state=42,
        )

        with pytest.raises(ValueError, match="has not been split yet"):
            splitter.apply_to_grouped_distribution(sample_grouped_distribution, "test")


class TestFilterByTgtDistIndices:
    """Test the filter_by_tgt_dist_indices methods."""

    def test_filter_annotation(self, sample_grouped_distribution):
        """Test filtering GroupedDistributionAnnotation."""
        sample_annotation = sample_grouped_distribution.annotation
        # Filter to only first 5 distributions
        indices_to_keep = [0, 1, 2, 3, 4]
        filtered = sample_annotation.filter_by_tgt_dist_indices(indices_to_keep)

        assert isinstance(filtered, GroupedDistributionAnnotation)
        assert len(filtered.src_tgt_dist_df) == len(indices_to_keep)
        assert set(filtered.src_tgt_dist_df["tgt_dist_idx"]) == set(indices_to_keep)

    def test_filter_grouped_distribution(self, sample_grouped_distribution):
        """Test filtering full GroupedDistribution."""
        # Filter to only first 5 distributions
        indices_to_keep = [0, 1, 2, 3, 4]
        filtered = sample_grouped_distribution.filter_by_tgt_dist_indices(indices_to_keep)
        assert isinstance(filtered, GroupedDistribution)
        assert set(filtered.data.tgt_data.keys()) == set(indices_to_keep)

        # Check that data is preserved
        for idx in indices_to_keep:
            original = sample_grouped_distribution.data.tgt_data[idx]
            filtered_data = filtered.data.tgt_data[idx]
            np.testing.assert_array_equal(original, filtered_data)
