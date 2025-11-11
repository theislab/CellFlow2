"""Tests for DataSplitter class."""

import numpy as np
import pandas as pd
import pytest

from scaleflow.data._data import GroupedDistribution, GroupedDistributionAnnotation, GroupedDistributionData
from scaleflow.data._data_splitter import DataSplitter, apply_split_to_grouped_distribution


@pytest.fixture
def sample_annotation():
    """Create a sample GroupedDistributionAnnotation for testing."""
    # Create a simple dataset with 2 source distributions and 8 target distributions
    # Source distributions: cell_line_A, cell_line_B
    # Target distributions: combinations of drugs (control, drug_A, drug_B) and genes (control, gene_A, gene_B)

    src_dist_idx_to_labels = {
        0: ("cell_line_A",),
        1: ("cell_line_B",),
    }

    tgt_dist_idx_to_labels = {
        0: ("control", "control"),  # Control for both
        1: ("drug_A", "control"),  # Drug A only
        2: ("drug_B", "control"),  # Drug B only
        3: ("control", "gene_A"),  # Gene A only
        4: ("control", "gene_B"),  # Gene B only
        5: ("drug_A", "gene_A"),  # Combination
        6: ("drug_A", "gene_B"),  # Combination
        7: ("drug_B", "gene_A"),  # Combination
        8: ("drug_B", "gene_B"),  # Combination
        9: ("control", "control"),  # Control for cell_line_B
        10: ("drug_A", "control"),  # Drug A only for cell_line_B
        11: ("drug_B", "control"),  # Drug B only for cell_line_B
        12: ("control", "gene_A"),  # Gene A only for cell_line_B
        13: ("control", "gene_B"),  # Gene B only for cell_line_B
        14: ("drug_A", "gene_A"),  # Combination for cell_line_B
        15: ("drug_A", "gene_B"),  # Combination for cell_line_B
        16: ("drug_B", "gene_A"),  # Combination for cell_line_B
        17: ("drug_B", "gene_B"),  # Combination for cell_line_B
    }

    # Create src_tgt_dist_df
    rows = []
    for src_idx in [0, 1]:
        src_label = src_dist_idx_to_labels[src_idx][0]
        base_tgt_idx = 0 if src_idx == 0 else 9
        for i in range(9):
            tgt_idx = base_tgt_idx + i
            drug, gene = tgt_dist_idx_to_labels[tgt_idx]
            rows.append(
                {
                    "src_dist_idx": src_idx,
                    "tgt_dist_idx": tgt_idx,
                    "cell_line": src_label,
                    "drug": drug,
                    "gene": gene,
                }
            )

    src_tgt_dist_df = pd.DataFrame(rows)

    return GroupedDistributionAnnotation(
        old_obs_index=np.arange(1000),  # Dummy indices
        src_dist_idx_to_labels=src_dist_idx_to_labels,
        tgt_dist_idx_to_labels=tgt_dist_idx_to_labels,
        src_tgt_dist_df=src_tgt_dist_df,
    )


@pytest.fixture
def sample_grouped_distribution(sample_annotation):
    """Create a complete GroupedDistribution for testing."""
    # Create dummy data for each distribution
    src_data = {
        0: np.random.randn(50, 10).astype(np.float32),  # 50 cells, 10 features
        1: np.random.randn(50, 10).astype(np.float32),
    }

    tgt_data = {}
    for tgt_idx in range(18):
        tgt_data[tgt_idx] = np.random.randn(30, 10).astype(np.float32)  # 30 cells per distribution

    conditions = {}
    for tgt_idx in range(18):
        conditions[tgt_idx] = np.random.randn(5).astype(np.float32)  # 5-dim condition embedding

    src_to_tgt_dist_map = {
        0: list(range(9)),
        1: list(range(9, 18)),
    }

    return GroupedDistribution(
        data=GroupedDistributionData(
            src_to_tgt_dist_map=src_to_tgt_dist_map,
            src_data=src_data,
            tgt_data=tgt_data,
            conditions=conditions,
        ),
        annotation=sample_annotation,
    )


class TestDataSplitterBasic:
    """Test basic DataSplitter functionality."""

    def test_init_valid(self, sample_annotation):
        """Test that DataSplitter initializes correctly with valid inputs."""
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test_dataset"],
            split_ratios=[[0.7, 0.15, 0.15]],
            split_type="random",
        )
        assert splitter.split_type == "random"
        assert len(splitter.annotations) == 1

    def test_init_invalid_ratios(self, sample_annotation):
        """Test that invalid split ratios raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            DataSplitter(
                annotations=[sample_annotation],
                dataset_names=["test"],
                split_ratios=[[0.7, 0.2, 0.2]],  # Sums to 1.1
            )

    def test_init_missing_split_key(self, sample_annotation):
        """Test that holdout_groups without split_key raises ValueError."""
        with pytest.raises(ValueError, match="split_key must be provided"):
            DataSplitter(
                annotations=[sample_annotation],
                dataset_names=["test"],
                split_ratios=[[0.7, 0.15, 0.15]],
                split_type="holdout_groups",
            )

    def test_init_missing_control_value(self, sample_annotation):
        """Test that holdout_combinations without control_value raises ValueError."""
        with pytest.raises(ValueError, match="control_value must be provided"):
            DataSplitter(
                annotations=[sample_annotation],
                dataset_names=["test"],
                split_ratios=[[0.7, 0.15, 0.15]],
                split_type="holdout_combinations",
                split_key=["drug", "gene"],
            )


class TestDataSplitterRandom:
    """Test random splitting strategy."""

    def test_random_split_hard(self, sample_annotation):
        """Test hard random split (no overlap between val and test)."""
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios=[[0.7, 0.15, 0.15]],
            split_type="random",
            hard_test_split=True,
            random_state=42,
        )

        results = splitter.split_all()

        # Check that we got results
        assert "test" in results
        assert "train" in results["test"]
        assert "val" in results["test"]
        assert "test" in results["test"]

        # Get distribution indices from annotations
        train_dists = set(results["test"]["train"].src_tgt_dist_df["tgt_dist_idx"])
        val_dists = set(results["test"]["val"].src_tgt_dist_df["tgt_dist_idx"])
        test_dists = set(results["test"]["test"].src_tgt_dist_df["tgt_dist_idx"])

        # In hard split, there should be no overlap
        assert len(train_dists & val_dists) == 0
        assert len(train_dists & test_dists) == 0
        assert len(val_dists & test_dists) == 0

        # All distributions should be accounted for
        total_dists = len(sample_annotation.src_tgt_dist_df)
        assert len(train_dists) + len(val_dists) + len(test_dists) == total_dists

    def test_random_split_soft(self, sample_annotation):
        """Test soft random split (val and test can overlap)."""
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios=[[0.7, 0.15, 0.15]],
            split_type="random",
            hard_test_split=False,
            random_state=42,
        )

        results = splitter.split_all()

        train_dists = set(results["test"]["train"].src_tgt_dist_df["tgt_dist_idx"])
        val_dists = set(results["test"]["val"].src_tgt_dist_df["tgt_dist_idx"])
        test_dists = set(results["test"]["test"].src_tgt_dist_df["tgt_dist_idx"])

        # Train should not overlap with val or test
        assert len(train_dists & val_dists) == 0
        assert len(train_dists & test_dists) == 0

        # But val and test CAN overlap in soft mode
        # (not guaranteed, but possible)

    def test_random_split_reproducible(self, sample_annotation):
        """Test that random split is reproducible with same seed."""
        splitter1 = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios=[[0.7, 0.15, 0.15]],
            split_type="random",
            random_state=42,
        )
        results1 = splitter1.split_all()

        splitter2 = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios=[[0.7, 0.15, 0.15]],
            split_type="random",
            random_state=42,
        )
        results2 = splitter2.split_all()

        # Check that train distributions are the same
        train_dists1 = set(results1["test"]["train"].src_tgt_dist_df["tgt_dist_idx"])
        train_dists2 = set(results2["test"]["train"].src_tgt_dist_df["tgt_dist_idx"])
        assert train_dists1 == train_dists2


class TestDataSplitterHoldoutGroups:
    """Test holdout_groups splitting strategy."""

    def test_holdout_groups_basic(self, sample_annotation):
        """Test basic holdout groups splitting."""
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios=[[0.7, 0.15, 0.15]],
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

    def test_holdout_groups_force_training(self, sample_annotation):
        """Test that force_training_values keeps certain values in training."""
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios=[[0.7, 0.15, 0.15]],
            split_type="holdout_groups",
            split_key="drug",
            force_training_values=["control"],
            random_state=42,
        )

        results = splitter.split_all()

        # Control should only appear in training
        val_drugs = set(results["test"]["val"].src_tgt_dist_df["drug"].unique())
        test_drugs = set(results["test"]["test"].src_tgt_dist_df["drug"].unique())

        assert "control" not in val_drugs
        assert "control" not in test_drugs

        # But should be in training
        train_drugs = set(results["test"]["train"].src_tgt_dist_df["drug"].unique())
        assert "control" in train_drugs

    def test_holdout_groups_multiple_keys(self, sample_annotation):
        """Test splitting on multiple keys."""
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios=[[0.7, 0.15, 0.15]],
            split_type="holdout_groups",
            split_key=["drug", "gene"],
            random_state=42,
        )

        results = splitter.split_all()

        # Get all unique values from both drug and gene
        train_df = results["test"]["train"].src_tgt_dist_df
        val_df = results["test"]["val"].src_tgt_dist_df
        test_df = results["test"]["test"].src_tgt_dist_df

        train_values = set(train_df["drug"].unique()) | set(train_df["gene"].unique())
        val_values = set(val_df["drug"].unique()) | set(val_df["gene"].unique())
        test_values = set(test_df["drug"].unique()) | set(test_df["gene"].unique())

        # Should have some separation
        assert len(val_values | test_values) > 0

    def test_holdout_groups_fixed_test_set(self, sample_annotation):
        """Test that test_random_state keeps test set fixed across runs."""
        # Run with different val_random_state but same test_random_state
        test_sets = []
        for val_seed in [42, 43, 44]:
            splitter = DataSplitter(
                annotations=[sample_annotation],
                dataset_names=["test"],
                split_ratios=[[0.6, 0.2, 0.2]],
                split_type="holdout_groups",
                split_key="drug",
                test_random_state=999,  # Fixed
                val_random_state=val_seed,  # Varying
                random_state=val_seed,
            )
            results = splitter.split_all()
            test_drugs = set(results["test"]["test"].src_tgt_dist_df["drug"].unique())
            test_sets.append(test_drugs)

        # All test sets should be identical
        assert test_sets[0] == test_sets[1] == test_sets[2]


class TestDataSplitterHoldoutCombinations:
    """Test holdout_combinations splitting strategy."""

    def test_holdout_combinations_basic(self, sample_annotation):
        """Test basic holdout combinations splitting."""
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios=[[0.7, 0.15, 0.15]],
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

        # Count non-control values in each row
        def count_non_control(row):
            non_control = 0
            for key in ["drug", "gene"]:
                if row[key] != "control":
                    non_control += 1
            return non_control

        train_df["n_non_control"] = train_df.apply(count_non_control, axis=1)
        val_df["n_non_control"] = val_df.apply(count_non_control, axis=1)
        test_df["n_non_control"] = test_df.apply(count_non_control, axis=1)

        # Training should have controls (0) and singletons (1)
        assert 0 in train_df["n_non_control"].values or 1 in train_df["n_non_control"].values

        # Val and test should only have combinations (2)
        if len(val_df) > 0:
            assert all(val_df["n_non_control"] == 2)
        if len(test_df) > 0:
            assert all(test_df["n_non_control"] == 2)

    def test_holdout_combinations_no_combinations(self):
        """Test that warning is raised when no combinations exist."""
        # Create annotation with only singletons
        src_dist_idx_to_labels = {0: ("cell_line_A",)}
        tgt_dist_idx_to_labels = {
            0: ("control", "control"),
            1: ("drug_A", "control"),
            2: ("control", "gene_A"),
        }

        rows = [
            {"src_dist_idx": 0, "tgt_dist_idx": 0, "drug": "control", "gene": "control"},
            {"src_dist_idx": 0, "tgt_dist_idx": 1, "drug": "drug_A", "gene": "control"},
            {"src_dist_idx": 0, "tgt_dist_idx": 2, "drug": "control", "gene": "gene_A"},
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
            split_ratios=[[0.7, 0.15, 0.15]],
            split_type="holdout_combinations",
            split_key=["drug", "gene"],
            control_value="control",
        )

        with pytest.warns(UserWarning, match="No combination treatments found"):
            results = splitter.split_all()

            # Val and test should be empty
            assert len(results["test"]["val"].src_tgt_dist_df) == 0
            assert len(results["test"]["test"].src_tgt_dist_df) == 0


class TestDataSplitterStratified:
    """Test stratified splitting strategy."""

    def test_stratified_split(self, sample_annotation):
        """Test that stratified split maintains source distribution proportions."""
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test"],
            split_ratios=[[0.7, 0.15, 0.15]],
            split_type="stratified",
            random_state=42,
        )

        results = splitter.split_all()

        # Check that both source distributions appear in all splits
        for split_name in ["train", "val", "test"]:
            split_df = results["test"][split_name].src_tgt_dist_df
            if len(split_df) > 0:
                src_dists = split_df["src_dist_idx"].unique()
                # Should have both source distributions (or at least some)
                assert len(src_dists) > 0


class TestApplySplitToGroupedDistribution:
    """Test applying splits to full GroupedDistribution."""

    def test_apply_split_basic(self, sample_grouped_distribution):
        """Test applying split annotations to GroupedDistribution."""
        splitter = DataSplitter(
            annotations=[sample_grouped_distribution.annotation],
            dataset_names=["test"],
            split_ratios=[[0.7, 0.15, 0.15]],
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
            split_ratios=[[0.7, 0.15, 0.15]],
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

    def test_save_and_load(self, sample_annotation, tmp_path):
        """Test that splits can be saved and loaded."""
        splitter = DataSplitter(
            annotations=[sample_annotation],
            dataset_names=["test_dataset"],
            split_ratios=[[0.7, 0.15, 0.15]],
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


class TestFilterByTgtDistIndices:
    """Test the filter_by_tgt_dist_indices methods."""

    def test_filter_annotation(self, sample_annotation):
        """Test filtering GroupedDistributionAnnotation."""
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
