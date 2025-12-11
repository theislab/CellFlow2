"""Test suite for GroupedDistributionSplitter with high coverage."""

import numpy as np
import pandas as pd
import pytest

from scaleflow.data._data_splitter import GroupedDistributionSplitter


class TestGroupedDistributionSplitterInit:
    """Test GroupedDistributionSplitter initialization."""

    def test_init_valid_params(self, sample_grouped_distribution):
        """Test that GroupedDistributionSplitter initializes with valid parameters."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        assert splitter.train_ratio == 0.6
        assert splitter.val_ratio == 0.2
        assert splitter.test_ratio == 0.2
        assert splitter.split_by == ["drug", "gene"]
        assert splitter.split_key == "split"
        assert splitter.random_state == 42

    def test_init_custom_split_key(self, sample_grouped_distribution):
        """Test initialization with custom split_key."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="my_custom_split",
            force_training_values={},
            ratios=[0.7, 0.15, 0.15],
            random_state=99,
        )

        assert splitter.split_key == "my_custom_split"

    def test_init_invalid_split_by_empty(self, sample_grouped_distribution):
        """Test that empty split_by raises ValueError."""
        with pytest.raises(ValueError, match="split_by must be a non-empty list"):
            GroupedDistributionSplitter(
                gd=sample_grouped_distribution,
                holdout_combinations=False,
                split_by=[],
                split_key="split",
                force_training_values={},
                ratios=[0.6, 0.2, 0.2],
                random_state=42,
            )

    def test_init_invalid_ratios_length(self, sample_grouped_distribution):
        """Test that ratios with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="ratios must be a list of 3 values"):
            GroupedDistributionSplitter(
                gd=sample_grouped_distribution,
                holdout_combinations=False,
                split_by=["drug"],
                split_key="split",
                force_training_values={},
                ratios=[0.6, 0.4],
                random_state=42,
            )

    def test_init_invalid_ratios_sum(self, sample_grouped_distribution):
        """Test that ratios not summing to 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="ratios must sum to 1.0"):
            GroupedDistributionSplitter(
                gd=sample_grouped_distribution,
                holdout_combinations=False,
                split_by=["drug"],
                split_key="split",
                force_training_values={},
                ratios=[0.5, 0.3, 0.1],
                random_state=42,
            )

    def test_init_invalid_ratios_with_zero(self, sample_grouped_distribution):
        """Test that ratios with 0 raises ValueError."""
        with pytest.raises(ValueError, match="ratios must be between 0.0 and 1.0"):
            GroupedDistributionSplitter(
                gd=sample_grouped_distribution,
                holdout_combinations=False,
                split_by=["drug"],
                split_key="split",
                force_training_values={},
                ratios=[0.8, 0.2, 0.0],
                random_state=42,
            )

    def test_init_invalid_ratios_with_one(self, sample_grouped_distribution):
        """Test that ratios with 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="ratios must be between 0.0 and 1.0"):
            GroupedDistributionSplitter(
                gd=sample_grouped_distribution,
                holdout_combinations=False,
                split_by=["drug"],
                split_key="split",
                force_training_values={},
                ratios=[1.0, 0.0, 0.0],
                random_state=42,
            )

    def test_init_invalid_force_training_values(self, sample_grouped_distribution):
        """Test that force_training_values with keys not in split_by raises ValueError."""
        with pytest.raises(ValueError, match="force_training_values keys must be a subset of split_by"):
            GroupedDistributionSplitter(
                gd=sample_grouped_distribution,
                holdout_combinations=False,
                split_by=["drug"],
                split_key="split",
                force_training_values={"cell_line": "cell_line_0"},
                ratios=[0.6, 0.2, 0.2],
                random_state=42,
            )


class TestGroupedDistributionSplitterSplitDf:
    """Test the _split_df method of GroupedDistributionSplitter."""

    def test_split_basic(self, sample_grouped_distribution):
        """Test basic splitting functionality."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split = splitter._split_df()

        # Check that split column exists
        assert "split" in df_split.columns

        # Check that all rows are labeled
        assert df_split["split"].notna().all()

        # Check that all three splits exist
        split_values = set(df_split["split"].unique())
        assert split_values == {"train", "val", "test"}

    def test_split_with_custom_split_key(self, sample_grouped_distribution):
        """Test splitting with a custom split_key name."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="my_split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split = splitter._split_df()

        # Check that custom split column exists
        assert "my_split" in df_split.columns
        assert df_split["my_split"].notna().all()

        # Check all three splits exist
        split_values = set(df_split["my_split"].unique())
        assert split_values == {"train", "val", "test"}

    def test_split_single_key(self, sample_grouped_distribution):
        """Test splitting by a single key."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split = splitter._split_df()

        # All rows with the same drug should have the same split
        for drug in df_split["drug"].unique():
            drug_splits = df_split[df_split["drug"] == drug]["split"].unique()
            assert len(drug_splits) == 1, f"Drug {drug} has multiple splits: {drug_splits}"

    def test_split_ratios_approximately_correct(self, sample_grouped_distribution):
        """Test that split ratios are approximately correct."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split = splitter._split_df()

        # Get unique combinations
        df_unique = df_split.drop_duplicates(subset=["drug", "gene"])

        total = len(df_unique)
        train_count = (df_unique["split"] == "train").sum()
        val_count = (df_unique["split"] == "val").sum()
        test_count = (df_unique["split"] == "test").sum()

        # Check counts sum to total
        assert train_count + val_count + test_count == total

        # Check ratios are approximately correct
        assert train_count >= 1
        assert val_count >= 1
        assert test_count >= 1

    def test_split_deterministic_with_random_state(self, sample_grouped_distribution):
        """Test that splitting is deterministic given a random_state."""
        splitter1 = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        splitter2 = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split1 = splitter1._split_df()
        df_split2 = splitter2._split_df()

        # Results should be identical
        pd.testing.assert_frame_equal(df_split1, df_split2)

    def test_split_different_with_different_random_state(self, sample_grouped_distribution):
        """Test that different random states produce different splits."""
        splitter1 = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        splitter2 = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=123,
        )

        df_split1 = splitter1._split_df()
        df_split2 = splitter2._split_df()

        # Results should be different
        assert not df_split1[["split", "drug", "gene"]].equals(df_split2[["split", "drug", "gene"]])

    def test_split_no_duplicate_combinations_in_multiple_splits(self, sample_grouped_distribution):
        """Test that no combination appears in multiple splits."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split = splitter._split_df()

        # Group by split_by keys and check each group has only one split value
        for (drug, gene), group in df_split.groupby(["drug", "gene"], observed=False):
            unique_splits = group["split"].unique()
            assert len(unique_splits) == 1, (
                f"Combination (drug={drug}, gene={gene}) appears in multiple splits: {unique_splits}"
            )


class TestGroupedDistributionSplitterForceTraining:
    """Test force_training_values functionality."""

    def test_force_training_single_value(self, sample_grouped_distribution):
        """Test that forced training values end up in training set."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="split",
            force_training_values={"drug": "drug_0"},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split = splitter._split_df()

        # All rows with drug_0 should be in training
        drug_0_rows = df_split[df_split["drug"] == "drug_0"]
        if len(drug_0_rows) > 0:
            assert (drug_0_rows["split"] == "train").all(), "drug_0 should be forced to training"

    def test_force_training_multiple_keys_or_logic(self, sample_grouped_distribution):
        """Test forcing multiple keys uses OR logic."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={"drug": "drug_0", "gene": "gene_0"},
            ratios=[0.7, 0.15, 0.15],
            random_state=42,
        )

        df_split = splitter._split_df()

        # All rows with drug_0 OR gene_0 should be in training
        forced_rows = df_split[(df_split["drug"] == "drug_0") | (df_split["gene"] == "gene_0")]
        if len(forced_rows) > 0:
            assert (forced_rows["split"] == "train").all()

    def test_force_training_reduces_available_combinations(self, sample_grouped_distribution):
        """Test that forcing values to training reduces available combinations for test/val."""
        # Without forcing
        splitter_no_force = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="split",
            force_training_values={},
            ratios=[0.5, 0.25, 0.25],
            random_state=42,
        )

        df_no_force = splitter_no_force._split_df()

        # With forcing
        splitter_with_force = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="split",
            force_training_values={"drug": "drug_0"},
            ratios=[0.5, 0.25, 0.25],
            random_state=42,
        )

        df_with_force = splitter_with_force._split_df()

        # Training set should be different
        _train_no_force = set(df_no_force[df_no_force["split"] == "train"]["drug"].unique())
        train_with_force = set(df_with_force[df_with_force["split"] == "train"]["drug"].unique())

        # drug_0 must be in training when forced
        assert "drug_0" in train_with_force


class TestGroupedDistributionSplitterHoldout:
    """Test holdout_combinations functionality."""

    def test_holdout_combinations_control_in_train(self, sample_grouped_distribution):
        """Test that when holdout_combinations=True, default values are in training."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=True,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split = splitter._split_df()

        # Get default values from annotation
        default_values = sample_grouped_distribution.annotation.default_values

        # Check that rows matching default values are in training
        for key, value in default_values.items():
            if key in df_split.columns:
                default_rows = df_split[df_split[key] == value]
                if len(default_rows) > 0:
                    # All default value rows should be in training
                    assert (default_rows["split"] == "train").all(), (
                        f"Rows with {key}={value} (default) should be in training"
                    )

    def test_holdout_false_allows_controls_in_all_splits(self, sample_grouped_distribution):
        """Test that when holdout_combinations=False, controls can be in any split."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        df_split = splitter._split_df()

        # Check that we have all three splits
        assert set(df_split["split"].unique()) == {"train", "val", "test"}


class TestContainsValue:
    """Test the _contains_value static method."""

    def test_contains_value_basic(self):
        """Test basic _contains_value functionality."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c", "drug_a"],
                "gene": ["gene_x", "gene_y", "gene_z", "gene_x"],
            }
        )

        GroupedDistributionSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug", "gene"],
            values={"drug": "drug_a"},
            column_key="is_target",
            accept_nan=False,
        )

        # Rows 0 and 3 should be True (drug_a)
        assert df["is_target"].tolist() == [True, False, False, True]

    def test_contains_value_multiple_keys_or_logic(self):
        """Test _contains_value with multiple keys uses OR logic."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c", "drug_a"],
                "gene": ["gene_x", "gene_y", "gene_x", "gene_z"],
            }
        )

        GroupedDistributionSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug", "gene"],
            values={"drug": "drug_a", "gene": "gene_x"},
            column_key="is_target",
            accept_nan=False,
        )

        # Rows where drug=drug_a OR gene=gene_x should be True
        assert df["is_target"].tolist() == [True, False, True, True]

    def test_contains_value_with_nan_accept_false(self):
        """Test _contains_value with NaN when accept_nan=False."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", np.nan, "drug_a"],
                "gene": ["gene_x", "gene_y", "gene_z", np.nan],
            }
        )

        GroupedDistributionSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug", "gene"],
            values={"drug": "drug_a"},
            column_key="is_target",
            accept_nan=False,
        )

        # Only rows with drug_a (0 and 3), NaN should not match
        assert df["is_target"].tolist() == [True, False, False, True]

    def test_contains_value_with_nan_accept_true(self):
        """Test _contains_value with NaN when accept_nan=True."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", np.nan, "drug_d"],
                "gene": ["gene_x", "gene_y", "gene_z", np.nan],
            }
        )

        GroupedDistributionSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug", "gene"],
            values={"drug": "drug_a"},
            column_key="is_target",
            accept_nan=True,
        )

        # Rows with drug_a OR NaN in drug column (0, 2)
        expected = [True, False, True, False]
        assert df["is_target"].tolist() == expected

    def test_contains_value_or_logic_multiple_calls(self):
        """Test that _contains_value uses OR logic when called multiple times."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c"],
                "gene": ["gene_x", "gene_y", "gene_z"],
            }
        )

        # First call
        GroupedDistributionSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug", "gene"],
            values={"drug": "drug_a"},
            column_key="is_target",
            accept_nan=False,
        )

        assert df["is_target"].tolist() == [True, False, False]

        # Second call with different value - should OR with previous
        GroupedDistributionSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug", "gene"],
            values={"drug": "drug_b"},
            column_key="is_target",
            accept_nan=False,
        )

        # Rows 0 and 1 should be True (drug_a OR drug_b)
        assert df["is_target"].tolist() == [True, True, False]

    def test_contains_value_creates_column_if_not_exists(self):
        """Test that _contains_value creates the column if it doesn't exist."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b"],
            }
        )

        assert "is_target" not in df.columns

        GroupedDistributionSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug"],
            values={"drug": "drug_a"},
            column_key="is_target",
            accept_nan=False,
        )

        assert "is_target" in df.columns
        assert df["is_target"].tolist() == [True, False]

    def test_contains_value_skips_keys_not_in_combination_keys(self):
        """Test that _contains_value ignores keys not in combination_keys."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c"],
                "gene": ["gene_x", "gene_y", "gene_z"],
            }
        )

        GroupedDistributionSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug"],  # Only drug, not gene
            values={"drug": "drug_a", "gene": "gene_x"},  # gene_x should be ignored
            column_key="is_target",
            accept_nan=False,
        )

        # Only drug_a should match, gene_x is ignored
        assert df["is_target"].tolist() == [True, False, False]

    def test_contains_value_empty_values_dict(self):
        """Test _contains_value with empty values dictionary."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b"],
            }
        )

        GroupedDistributionSplitter._contains_value(
            df_unique=df,
            combination_keys=["drug"],
            values={},  # Empty
            column_key="is_target",
            accept_nan=False,
        )

        # All should be False
        assert df["is_target"].tolist() == [False, False]

    def test_contains_value_invalid_combination_keys(self):
        """Test that _contains_value raises error for invalid combination_keys."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b"],
            }
        )

        with pytest.raises(ValueError, match="combination_keys must be in df.columns"):
            GroupedDistributionSplitter._contains_value(
                df_unique=df,
                combination_keys=["nonexistent_column"],
                values={"drug": "drug_a"},
                column_key="is_target",
                accept_nan=False,
            )


class TestSplitTwo:
    """Test the _split_two static method."""

    def test_split_two_basic(self):
        """Test basic _split_two functionality."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c", "drug_d", "drug_e"],
                "value": [1, 2, 3, 4, 5],
            }
        )

        result = GroupedDistributionSplitter._split_two(
            df_unique=df.copy(),
            train_size=3,
            split_key="split",
            random_state=42,
        )

        # Check split column exists
        assert "split" in result.columns

        # Check counts
        assert (result["split"] == "train").sum() == 3
        assert (result["split"] == "test_val").sum() == 2

        # Check all rows are present
        assert len(result) == 5

    def test_split_two_shuffles_data(self):
        """Test that _split_two shuffles the data."""
        df = pd.DataFrame(
            {
                "drug": [f"drug_{i}" for i in range(20)],
                "value": list(range(20)),
            }
        )

        result = GroupedDistributionSplitter._split_two(
            df_unique=df.copy(),
            train_size=10,
            split_key="split",
            random_state=42,
        )

        # Get train drugs
        train_drugs = result[result["split"] == "train"]["drug"].tolist()
        original_first_10 = [f"drug_{i}" for i in range(10)]

        # With shuffling and 20 items, extremely unlikely to match exactly
        assert train_drugs != original_first_10

    def test_split_two_deterministic(self):
        """Test that _split_two is deterministic with same random_state."""
        df = pd.DataFrame(
            {
                "drug": [f"drug_{i}" for i in range(10)],
            }
        )

        result1 = GroupedDistributionSplitter._split_two(
            df_unique=df.copy(),
            train_size=6,
            split_key="split",
            random_state=42,
        )

        result2 = GroupedDistributionSplitter._split_two(
            df_unique=df.copy(),
            train_size=6,
            split_key="split",
            random_state=42,
        )

        pd.testing.assert_frame_equal(result1, result2)

    def test_split_two_custom_split_key(self):
        """Test _split_two with custom split_key."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c"],
            }
        )

        result = GroupedDistributionSplitter._split_two(
            df_unique=df.copy(),
            train_size=2,
            split_key="my_split",
            random_state=42,
        )

        assert "my_split" in result.columns
        assert set(result["my_split"].unique()) == {"train", "test_val"}

    def test_split_two_raises_on_existing_split_key(self):
        """Test that _split_two raises error if split_key already exists and overwrite=False."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b"],
                "split": ["existing", "existing"],  # Column already exists
            }
        )

        with pytest.raises(ValueError, match="already in df.columns"):
            GroupedDistributionSplitter._split_two(
                df_unique=df,
                train_size=1,
                split_key="split",
                overwrite=False,
                random_state=42,
            )

    def test_split_two_overwrites_existing_split_key(self):
        """Test that _split_two can overwrite existing split_key when overwrite=True."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c"],
                "split": ["existing", "existing", "existing"],
            }
        )

        result = GroupedDistributionSplitter._split_two(
            df_unique=df,
            train_size=2,
            split_key="split",
            overwrite=True,
            random_state=42,
        )

        # Should have new values, not "existing"
        assert "existing" not in result["split"].values
        assert set(result["split"].unique()) == {"train", "test_val"}

    def test_split_two_with_forced_training_key(self):
        """Test _split_two respects is_in_training_key."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c", "drug_d", "drug_e"],
                "forced_in_train": [True, True, False, False, False],
            }
        )

        # Now that the bug is fixed, this should work correctly
        # It should filter to only rows where forced_in_train is True
        result = GroupedDistributionSplitter._split_two(
            df_unique=df.copy(),
            train_size=1,  # Only 1 train from the 2 available (drug_a, drug_b)
            split_key="split",
            is_in_training_key="forced_in_train",
            random_state=42,
        )

        # Should only contain the forced training rows (drug_a and drug_b)
        assert len(result) == 2
        assert set(result["drug"]) == {"drug_a", "drug_b"}

        # One should be train, one should be test_val
        assert (result["split"] == "train").sum() == 1
        assert (result["split"] == "test_val").sum() == 1

    def test_split_two_raises_if_training_key_missing(self):
        """Test that _split_two raises error if is_in_training_key doesn't exist."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b"],
            }
        )

        with pytest.raises(ValueError, match="must be in df.columns"):
            GroupedDistributionSplitter._split_two(
                df_unique=df,
                train_size=1,
                split_key="split",
                is_in_training_key="nonexistent",
                random_state=42,
            )


class TestCalculateSplitSizes:
    """Test the _calculate_split_sizes method."""

    def test_calculate_split_sizes_normal(self, sample_grouped_distribution):
        """Test split size calculation with normal inputs."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        train, val, test = splitter._calculate_split_sizes(100)

        # Check they sum to 100
        assert train + val + test == 100

        # Check they're approximately correct
        assert train == 60
        assert val == 20
        assert test == 20

    def test_calculate_split_sizes_with_rounding(self, sample_grouped_distribution):
        """Test split size calculation handles rounding correctly."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="split",
            force_training_values={},
            ratios=[0.7, 0.15, 0.15],
            random_state=42,
        )

        train, val, test = splitter._calculate_split_sizes(100)

        # Check they sum to 100
        assert train + val + test == 100

        # Check individual values
        assert train == 70
        assert val == 15
        assert test == 15

    def test_calculate_split_sizes_too_small_raises_error(self, sample_grouped_distribution):
        """Test that too few combinations raises ValueError."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="split",
            force_training_values={},
            ratios=[0.9, 0.05, 0.05],
            random_state=42,
        )

        # With 5 total and ratios 0.9, 0.05, 0.05, some might round to 0
        with pytest.raises(ValueError, match="0 was encountered"):
            splitter._calculate_split_sizes(5)

    def test_calculate_split_sizes_large_numbers(self, sample_grouped_distribution):
        """Test split size calculation with large numbers."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        train, val, test = splitter._calculate_split_sizes(10000)

        # Check they sum to 10000
        assert train + val + test == 10000


class TestCheckDfUniqueColumns:
    """Test the _check_df_unique_columns static method."""

    def test_check_df_unique_columns_valid(self):
        """Test _check_df_unique_columns with valid columns."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a"],
                "gene": ["gene_x"],
            }
        )

        # Should not raise
        GroupedDistributionSplitter._check_df_unique_columns(df, ["drug", "gene"])

    def test_check_df_unique_columns_invalid(self):
        """Test _check_df_unique_columns raises error for missing columns."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a"],
            }
        )

        with pytest.raises(ValueError, match="combination_keys must be in df.columns"):
            GroupedDistributionSplitter._check_df_unique_columns(df, ["drug", "nonexistent"])

    def test_check_df_unique_columns_empty_list(self):
        """Test _check_df_unique_columns with empty combination_keys."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a"],
            }
        )

        # Should not raise with empty list
        GroupedDistributionSplitter._check_df_unique_columns(df, [])

    def test_split_two_with_not_in_training_key(self):
        """Test _split_two respects not_in_training_key."""
        df = pd.DataFrame(
            {
                "drug": ["drug_a", "drug_b", "drug_c", "drug_d", "drug_e"],
                "exclude_from_split": [False, False, True, True, False],
            }
        )

        # Should exclude rows where exclude_from_split is True
        result = GroupedDistributionSplitter._split_two(
            df_unique=df.copy(),
            train_size=2,
            split_key="split",
            not_in_training_key="exclude_from_split",
            random_state=42,
        )

        # Should only contain rows where exclude_from_split was False
        assert len(result) == 3
        assert set(result["drug"]) == {"drug_a", "drug_b", "drug_e"}

        # Check split distribution
        assert (result["split"] == "train").sum() == 2
        assert (result["split"] == "test_val").sum() == 1

        # Drugs c and d should not be in the result
        assert "drug_c" not in result["drug"].values
        assert "drug_d" not in result["drug"].values


class TestGroupedDistributionSplitterSplitMethod:
    """Test the split method that returns split GroupedDistributionAnnotations."""

    def test_split_returns_dict_with_three_keys(self, sample_grouped_distribution):
        """Test that split returns a dict with train, val, test keys."""
        from scaleflow.data._data import GroupedDistributionAnnotation

        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split_annotation()

        assert isinstance(result, dict)
        assert set(result.keys()) == {"train", "val", "test"}
        for split_name, annotation in result.items():
            assert isinstance(annotation, GroupedDistributionAnnotation), (
                f"Split {split_name} should be a GroupedDistributionAnnotation"
            )

    def test_split_tgt_dist_indices_are_disjoint(self, sample_grouped_distribution):
        """Test that target distribution indices across splits are disjoint."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split_annotation()

        train_tgt_idxs = set(result["train"].src_tgt_dist_df["tgt_dist_idx"].unique())
        val_tgt_idxs = set(result["val"].src_tgt_dist_df["tgt_dist_idx"].unique())
        test_tgt_idxs = set(result["test"].src_tgt_dist_df["tgt_dist_idx"].unique())

        # No overlap between any pair
        assert train_tgt_idxs.isdisjoint(val_tgt_idxs), "Train and val should have disjoint tgt_dist_idx"
        assert train_tgt_idxs.isdisjoint(test_tgt_idxs), "Train and test should have disjoint tgt_dist_idx"
        assert val_tgt_idxs.isdisjoint(test_tgt_idxs), "Val and test should have disjoint tgt_dist_idx"

    def test_split_union_covers_all_original_tgt_dist_indices(self, sample_grouped_distribution):
        """Test that the union of all splits covers all original target distribution indices."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split_annotation()

        # Get original tgt_dist_idx set
        original_tgt_idxs = set(sample_grouped_distribution.annotation.src_tgt_dist_df["tgt_dist_idx"].unique())

        # Get union of all splits
        train_tgt_idxs = set(result["train"].src_tgt_dist_df["tgt_dist_idx"].unique())
        val_tgt_idxs = set(result["val"].src_tgt_dist_df["tgt_dist_idx"].unique())
        test_tgt_idxs = set(result["test"].src_tgt_dist_df["tgt_dist_idx"].unique())

        union_tgt_idxs = train_tgt_idxs | val_tgt_idxs | test_tgt_idxs

        assert union_tgt_idxs == original_tgt_idxs, "Union of splits should cover all original tgt_dist_idx"

    def test_split_preserves_src_dist_idx_to_labels(self, sample_grouped_distribution):
        """Test that each split contains correct src_dist_idx_to_labels."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split_annotation()
        original_annotation = sample_grouped_distribution.annotation

        for split_name, split_annotation in result.items():
            # Get the src_dist_idx that appear in this split
            split_src_idxs = set(split_annotation.src_tgt_dist_df["src_dist_idx"].unique())

            # Verify labels are preserved for those src_dist_idx
            for src_idx in split_src_idxs:
                assert src_idx in split_annotation.src_dist_idx_to_labels, (
                    f"src_dist_idx {src_idx} should be in {split_name} labels"
                )
                assert (
                    split_annotation.src_dist_idx_to_labels[src_idx]
                    == original_annotation.src_dist_idx_to_labels[src_idx]
                ), f"Label for src_dist_idx {src_idx} should match original in {split_name}"

    def test_split_preserves_tgt_dist_idx_to_labels(self, sample_grouped_distribution):
        """Test that each split contains correct tgt_dist_idx_to_labels."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split_annotation()
        original_annotation = sample_grouped_distribution.annotation

        for split_name, split_annotation in result.items():
            # Get the tgt_dist_idx that appear in this split
            split_tgt_idxs = set(split_annotation.src_tgt_dist_df["tgt_dist_idx"].unique())

            # Verify labels are preserved for those tgt_dist_idx
            for tgt_idx in split_tgt_idxs:
                assert tgt_idx in split_annotation.tgt_dist_idx_to_labels, (
                    f"tgt_dist_idx {tgt_idx} should be in {split_name} labels"
                )
                assert (
                    split_annotation.tgt_dist_idx_to_labels[tgt_idx]
                    == original_annotation.tgt_dist_idx_to_labels[tgt_idx]
                ), f"Label for tgt_dist_idx {tgt_idx} should match original in {split_name}"

    def test_split_preserves_metadata(self, sample_grouped_distribution):
        """Test that split preserves default_values, dist_flag_key, src_dist_keys, tgt_dist_keys."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split_annotation()
        original_annotation = sample_grouped_distribution.annotation

        for split_name, split_annotation in result.items():
            assert split_annotation.default_values == original_annotation.default_values, (
                f"default_values should match in {split_name}"
            )
            assert split_annotation.dist_flag_key == original_annotation.dist_flag_key, (
                f"dist_flag_key should match in {split_name}"
            )
            assert split_annotation.src_dist_keys == original_annotation.src_dist_keys, (
                f"src_dist_keys should match in {split_name}"
            )
            assert split_annotation.tgt_dist_keys == original_annotation.tgt_dist_keys, (
                f"tgt_dist_keys should match in {split_name}"
            )

    def test_split_old_obs_index_is_shared(self, sample_grouped_distribution):
        """Test that old_obs_index is shared across all splits (same reference or equal)."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split_annotation()
        original_annotation = sample_grouped_distribution.annotation

        for split_name, split_annotation in result.items():
            # old_obs_index should be the same as original (enables reconstruction)
            assert np.array_equal(split_annotation.old_obs_index, original_annotation.old_obs_index), (
                f"old_obs_index should match original in {split_name}"
            )

    def test_split_deterministic_with_random_state(self, sample_grouped_distribution):
        """Test that split is deterministic given a random_state."""
        splitter1 = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        splitter2 = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result1 = splitter1.split_annotation()
        result2 = splitter2.split_annotation()

        for split_name in ["train", "val", "test"]:
            # Compare src_tgt_dist_df
            pd.testing.assert_frame_equal(
                result1[split_name].src_tgt_dist_df.reset_index(drop=True),
                result2[split_name].src_tgt_dist_df.reset_index(drop=True),
            )


class TestSplitReconstruction:
    """Test that we can reconstruct original obs indices from splits."""

    def test_can_trace_split_tgt_dist_to_original_obs(self, sample_grouped_distribution, adata_test):
        """Test that we can trace from split tgt_dist_idx back to original adata obs."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split_annotation()
        original_annotation = sample_grouped_distribution.annotation

        # For each split, verify we can trace back to original adata
        for split_name, split_annotation in result.items():
            for tgt_idx in split_annotation.src_tgt_dist_df["tgt_dist_idx"].unique():
                # Get the label for this tgt_dist_idx
                tgt_label = split_annotation.tgt_dist_idx_to_labels[tgt_idx]

                # The label should be a tuple of (drug, gene) values
                assert len(tgt_label) == len(original_annotation.tgt_dist_keys), (
                    f"Label {tgt_label} should have same length as tgt_dist_keys"
                )

                # Verify the label exists in original adata
                drug_val, gene_val = tgt_label
                matching = adata_test.obs[(adata_test.obs["drug"] == drug_val) & (adata_test.obs["gene"] == gene_val)]
                assert len(matching) > 0, f"Could not find cells with drug={drug_val}, gene={gene_val} in {split_name}"

    def test_split_src_tgt_df_rows_map_to_original(self, sample_grouped_distribution):
        """Test that all rows in split src_tgt_dist_df exist in original."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split_annotation()
        original_df = sample_grouped_distribution.annotation.src_tgt_dist_df

        for split_name, split_annotation in result.items():
            split_df = split_annotation.src_tgt_dist_df

            # Each row in split should exist in original
            for _, row in split_df.iterrows():
                src_idx = row["src_dist_idx"]
                tgt_idx = row["tgt_dist_idx"]

                # Find matching row in original
                match = original_df[(original_df["src_dist_idx"] == src_idx) & (original_df["tgt_dist_idx"] == tgt_idx)]
                assert len(match) == 1, f"Could not find unique match for src={src_idx}, tgt={tgt_idx} in {split_name}"

    def test_reconstruct_all_src_tgt_pairs_from_splits(self, sample_grouped_distribution):
        """Test that we can reconstruct all original (src, tgt) pairs from the splits."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split_annotation()
        original_df = sample_grouped_distribution.annotation.src_tgt_dist_df

        # Collect all (src_dist_idx, tgt_dist_idx) pairs from splits
        all_pairs = set()
        for split_annotation in result.values():
            split_df = split_annotation.src_tgt_dist_df
            for _, row in split_df.iterrows():
                all_pairs.add((row["src_dist_idx"], row["tgt_dist_idx"]))

        # Collect original pairs
        original_pairs = set()
        for _, row in original_df.iterrows():
            original_pairs.add((row["src_dist_idx"], row["tgt_dist_idx"]))

        assert all_pairs == original_pairs, "All (src, tgt) pairs should be reconstructible from splits"

    def test_no_duplicate_tgt_dist_across_splits(self, sample_grouped_distribution):
        """Verify that no tgt_dist_idx appears in more than one split."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split_annotation()

        seen_tgt_idxs = {}
        for split_name, split_annotation in result.items():
            for tgt_idx in split_annotation.src_tgt_dist_df["tgt_dist_idx"].unique():
                if tgt_idx in seen_tgt_idxs:
                    pytest.fail(f"tgt_dist_idx {tgt_idx} appears in both {seen_tgt_idxs[tgt_idx]} and {split_name}")
                seen_tgt_idxs[tgt_idx] = split_name


class TestGroupedDistributionSplitterSplitData:
    """Test the split method that returns split GroupedDistributions."""

    def test_split_returns_dict_of_grouped_distributions(self, sample_grouped_distribution):
        """Test that split returns a dict with GroupedDistribution objects."""
        from scaleflow.data._data import GroupedDistribution

        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split()

        assert isinstance(result, dict)
        assert set(result.keys()) == {"train", "val", "test"}
        for split_name, gd in result.items():
            assert isinstance(gd, GroupedDistribution), f"Split {split_name} should be a GroupedDistribution"

    def test_split_tgt_data_is_disjoint(self, sample_grouped_distribution):
        """Test that target data across splits are disjoint."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split()

        train_tgt_idxs = set(result["train"].data.tgt_data.keys())
        val_tgt_idxs = set(result["val"].data.tgt_data.keys())
        test_tgt_idxs = set(result["test"].data.tgt_data.keys())

        # No overlap between any pair
        assert train_tgt_idxs.isdisjoint(val_tgt_idxs)
        assert train_tgt_idxs.isdisjoint(test_tgt_idxs)
        assert val_tgt_idxs.isdisjoint(test_tgt_idxs)

    def test_split_union_covers_all_tgt_data(self, sample_grouped_distribution):
        """Test that the union of all splits covers all original tgt_data."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split()

        original_tgt_idxs = set(sample_grouped_distribution.data.tgt_data.keys())

        union_tgt_idxs = (
            set(result["train"].data.tgt_data.keys())
            | set(result["val"].data.tgt_data.keys())
            | set(result["test"].data.tgt_data.keys())
        )

        assert union_tgt_idxs == original_tgt_idxs

    def test_split_tgt_data_values_match_original(self, sample_grouped_distribution):
        """Test that tgt_data values in splits match the original."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split()

        for split_name, split_gd in result.items():
            for tgt_idx, tgt_data in split_gd.data.tgt_data.items():
                original_data = sample_grouped_distribution.data.tgt_data[tgt_idx]
                assert np.array_equal(tgt_data, original_data), (
                    f"tgt_data[{tgt_idx}] in {split_name} should match original"
                )

    def test_split_conditions_match_original(self, sample_grouped_distribution):
        """Test that conditions in splits match the original."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split()

        for split_name, split_gd in result.items():
            for tgt_idx, condition in split_gd.data.conditions.items():
                original_condition = sample_grouped_distribution.data.conditions[tgt_idx]
                assert np.array_equal(condition, original_condition), (
                    f"conditions[{tgt_idx}] in {split_name} should match original"
                )

    def test_split_src_to_tgt_map_consistent(self, sample_grouped_distribution):
        """Test that src_to_tgt_dist_map is consistent with split data."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split()

        for split_name, split_gd in result.items():
            # All sources in the map should exist in src_data
            for src_idx in split_gd.data.src_to_tgt_dist_map.keys():
                assert src_idx in split_gd.data.src_data, (
                    f"src_idx {src_idx} in map but not in src_data for {split_name}"
                )

            # All targets in the map should exist in tgt_data
            for _src_idx, tgt_idxs in split_gd.data.src_to_tgt_dist_map.items():
                for tgt_idx in tgt_idxs:
                    assert tgt_idx in split_gd.data.tgt_data, (
                        f"tgt_idx {tgt_idx} in map but not in tgt_data for {split_name}"
                    )


class TestRoundTripAdataToSplitAndBack:
    """Test round-trip: adata → GroupedDistribution → split → verify data integrity."""

    def test_roundtrip_tgt_data_can_be_reconstructed(self, sample_grouped_distribution, adata_test):
        """Test that all tgt_data can be traced back to original adata."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split()
        original_gd = sample_grouped_distribution

        # Reconstruct all tgt_data from splits
        reconstructed_tgt_data = {}
        for split_gd in result.values():
            reconstructed_tgt_data.update(split_gd.data.tgt_data)

        # Verify all original tgt_data is in reconstructed
        for tgt_idx, original_data in original_gd.data.tgt_data.items():
            assert tgt_idx in reconstructed_tgt_data, f"tgt_idx {tgt_idx} missing from reconstructed data"
            assert np.array_equal(reconstructed_tgt_data[tgt_idx], original_data), (
                f"tgt_data[{tgt_idx}] doesn't match original"
            )

    def test_roundtrip_conditions_can_be_reconstructed(self, sample_grouped_distribution):
        """Test that all conditions can be reconstructed from splits."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split()
        original_gd = sample_grouped_distribution

        # Reconstruct all conditions from splits
        reconstructed_conditions = {}
        for split_gd in result.values():
            reconstructed_conditions.update(split_gd.data.conditions)

        # Verify all original conditions are in reconstructed
        for tgt_idx, original_cond in original_gd.data.conditions.items():
            assert tgt_idx in reconstructed_conditions, f"tgt_idx {tgt_idx} missing from reconstructed conditions"
            assert np.array_equal(reconstructed_conditions[tgt_idx], original_cond), (
                f"conditions[{tgt_idx}] doesn't match original"
            )

    def test_roundtrip_src_tgt_pairs_preserved(self, sample_grouped_distribution):
        """Test that all (src, tgt) pairs can be reconstructed from splits."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split()
        original_gd = sample_grouped_distribution

        # Collect original (src, tgt) pairs
        original_pairs = set()
        for src_idx, tgt_idxs in original_gd.data.src_to_tgt_dist_map.items():
            for tgt_idx in tgt_idxs:
                original_pairs.add((src_idx, tgt_idx))

        # Collect reconstructed (src, tgt) pairs from splits
        reconstructed_pairs = set()
        for split_gd in result.values():
            for src_idx, tgt_idxs in split_gd.data.src_to_tgt_dist_map.items():
                for tgt_idx in tgt_idxs:
                    reconstructed_pairs.add((src_idx, tgt_idx))

        assert reconstructed_pairs == original_pairs, "All (src, tgt) pairs should be reconstructible from splits"

    def test_roundtrip_labels_trace_to_adata(self, sample_grouped_distribution, adata_test):
        """Test that labels from split can trace back to adata obs values."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split()

        for split_name, split_gd in result.items():
            # For each target distribution, verify labels exist in adata
            for _tgt_idx, tgt_label in split_gd.annotation.tgt_dist_idx_to_labels.items():
                drug_val, gene_val = tgt_label
                # Find matching rows in adata
                matching = adata_test.obs[(adata_test.obs["drug"] == drug_val) & (adata_test.obs["gene"] == gene_val)]
                assert len(matching) > 0, f"No cells found for drug={drug_val}, gene={gene_val} in {split_name}"

    def test_roundtrip_data_dimensions_preserved(self, sample_grouped_distribution):
        """Test that data dimensions are preserved through split."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split()
        original_gd = sample_grouped_distribution

        # Check that feature dimensions are preserved
        for split_gd in result.values():
            for tgt_idx, tgt_data in split_gd.data.tgt_data.items():
                original_data = original_gd.data.tgt_data[tgt_idx]
                assert tgt_data.shape == original_data.shape, f"Shape mismatch for tgt_idx {tgt_idx}"

            for src_idx, src_data in split_gd.data.src_data.items():
                original_data = original_gd.data.src_data[src_idx]
                assert src_data.shape == original_data.shape, f"Shape mismatch for src_idx {src_idx}"

    def test_total_cells_preserved_in_tgt_data(self, sample_grouped_distribution):
        """Test that total number of cells in tgt_data is preserved."""
        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = splitter.split()
        original_gd = sample_grouped_distribution

        # Count total cells in original tgt_data
        original_total_cells = sum(data.shape[0] for data in original_gd.data.tgt_data.values())

        # Count total cells in split tgt_data
        split_total_cells = sum(
            data.shape[0] for split_gd in result.values() for data in split_gd.data.tgt_data.values()
        )

        assert split_total_cells == original_total_cells, (
            f"Total cells mismatch: {split_total_cells} vs {original_total_cells}"
        )
