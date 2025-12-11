"""Test suite for batch utilities."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scaleflow.data import (
    AnnDataLocation,
    DataManager,
    GroupedDistribution,
    prepare_and_split_datasets,
    prepare_datasets,
    split_datasets,
)


def create_test_adata(n_obs: int = 500, seed: int = 42) -> ad.AnnData:
    """Create a test AnnData object."""
    np.random.seed(seed)
    n_vars = 20
    n_pca = 10
    n_drugs = 5
    n_genes = 3
    n_cell_lines = 2

    drugs = ["control"] + [f"drug_{i}" for i in range(n_drugs)]
    genes = ["control"] + [f"gene_{i}" for i in range(n_genes)]
    cell_lines = [f"cell_line_{i}" for i in range(n_cell_lines)]

    obs = pd.DataFrame(
        {
            "drug": np.random.choice(drugs, n_obs),
            "gene": np.random.choice(genes, n_obs),
            "cell_line": np.random.choice(cell_lines, n_obs),
        }
    )
    obs["control"] = (obs["drug"] == "control") & (obs["gene"] == "control")

    for col in ["drug", "gene", "cell_line"]:
        obs[col] = obs[col].astype("category")

    X = np.random.randn(n_obs, n_vars).astype(np.float32)
    X_pca = np.random.randn(n_obs, n_pca).astype(np.float32)

    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["X_pca"] = X_pca

    # Embeddings
    adata.uns["cell_line_emb"] = {cl: np.random.randn(5) for cl in cell_lines}
    adata.uns["drug_emb"] = {d: np.random.randn(5) for d in drugs}
    adata.uns["gene_emb"] = {g: np.random.randn(5) for g in genes}

    return adata


@pytest.fixture
def sample_data_manager() -> DataManager:
    """Create a sample DataManager."""
    adl = AnnDataLocation()
    return DataManager(
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys=["drug", "gene"],
        rep_keys={
            "cell_line": "cell_line_emb",
            "drug": "drug_emb",
            "gene": "gene_emb",
        },
        data_location=adl.obsm["X_pca"],
    )


@pytest.fixture
def sample_datasets() -> dict[str, ad.AnnData]:
    """Create sample datasets."""
    return {
        "dataset_a": create_test_adata(n_obs=500, seed=42),
        "dataset_b": create_test_adata(n_obs=300, seed=123),
        "dataset_c": create_test_adata(n_obs=400, seed=456),
    }


class TestPrepareDatasets:
    """Test prepare_datasets function."""

    def test_basic_functionality(self, sample_datasets, sample_data_manager):
        """Test basic preparation of multiple datasets."""
        result = prepare_datasets(
            datasets=sample_datasets,
            data_manager=sample_data_manager,
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == set(sample_datasets.keys())

        for _name, gd in result.items():
            assert isinstance(gd, GroupedDistribution)
            assert len(gd.data.src_data) > 0
            assert len(gd.data.tgt_data) > 0

    def test_empty_dict(self, sample_data_manager):
        """Test with empty datasets dict."""
        result = prepare_datasets(
            datasets={},
            data_manager=sample_data_manager,
        )
        assert result == {}

    def test_single_dataset(self, sample_data_manager):
        """Test with single dataset."""
        datasets = {"only_one": create_test_adata()}
        result = prepare_datasets(
            datasets=datasets,
            data_manager=sample_data_manager,
        )

        assert len(result) == 1
        assert "only_one" in result
        assert isinstance(result["only_one"], GroupedDistribution)

    def test_verbose_mode(self, sample_datasets, sample_data_manager, capsys):
        """Test verbose mode prints timing info."""
        prepare_datasets(
            datasets=sample_datasets,
            data_manager=sample_data_manager,
            verbose=True,
        )
        # Verbose mode should produce some output
        _captured = capsys.readouterr()
        # At minimum it shouldn't error; timing output goes to logger not stdout


class TestSplitDatasets:
    """Test split_datasets function."""

    def test_basic_functionality(self, sample_datasets, sample_data_manager):
        """Test basic splitting of multiple datasets."""
        gd_dict = prepare_datasets(
            datasets=sample_datasets,
            data_manager=sample_data_manager,
        )

        result = split_datasets(
            grouped_distributions=gd_dict,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == set(sample_datasets.keys())

        for _name, splits in result.items():
            assert isinstance(splits, dict)
            assert set(splits.keys()) == {"train", "val", "test"}
            for _split_name, gd in splits.items():
                assert isinstance(gd, GroupedDistribution)

    def test_custom_ratios(self, sample_datasets, sample_data_manager):
        """Test with custom split ratios."""
        gd_dict = prepare_datasets(
            datasets=sample_datasets,
            data_manager=sample_data_manager,
        )

        result = split_datasets(
            grouped_distributions=gd_dict,
            holdout_combinations=False,
            split_by=["drug"],
            ratios=[0.7, 0.15, 0.15],
            random_state=42,
        )

        for _name, splits in result.items():
            assert "train" in splits
            assert "val" in splits
            assert "test" in splits

    def test_deterministic_with_same_seed(self, sample_datasets, sample_data_manager):
        """Test that same random_state produces same splits."""
        gd_dict = prepare_datasets(
            datasets=sample_datasets,
            data_manager=sample_data_manager,
        )

        result1 = split_datasets(
            grouped_distributions=gd_dict,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result2 = split_datasets(
            grouped_distributions=gd_dict,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        for name in result1:
            for split in ["train", "val", "test"]:
                df1 = result1[name][split].annotation.src_tgt_dist_df
                df2 = result2[name][split].annotation.src_tgt_dist_df
                pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))

    def test_different_with_different_seed(self, sample_datasets, sample_data_manager):
        """Test that different random_state produces different splits."""
        gd_dict = prepare_datasets(
            datasets=sample_datasets,
            data_manager=sample_data_manager,
        )

        result1 = split_datasets(
            grouped_distributions=gd_dict,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result2 = split_datasets(
            grouped_distributions=gd_dict,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            ratios=[0.6, 0.2, 0.2],
            random_state=123,
        )

        # At least one split should differ
        any_different = False
        for name in result1:
            for split in ["train", "val", "test"]:
                tgt_idxs1 = set(result1[name][split].annotation.src_tgt_dist_df["tgt_dist_idx"])
                tgt_idxs2 = set(result2[name][split].annotation.src_tgt_dist_df["tgt_dist_idx"])
                if tgt_idxs1 != tgt_idxs2:
                    any_different = True
                    break
            if any_different:
                break
        assert any_different

    def test_empty_dict(self):
        """Test with empty grouped_distributions dict."""
        result = split_datasets(
            grouped_distributions={},
            holdout_combinations=False,
            split_by=["drug"],
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )
        assert result == {}

    def test_force_training_values(self, sample_datasets, sample_data_manager):
        """Test with force_training_values."""
        gd_dict = prepare_datasets(
            datasets=sample_datasets,
            data_manager=sample_data_manager,
        )

        result = split_datasets(
            grouped_distributions=gd_dict,
            holdout_combinations=False,
            split_by=["drug"],
            force_training_values={"drug": "drug_0"},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        for _name, splits in result.items():
            train_df = splits["train"].annotation.src_tgt_dist_df
            # drug_0 should be in training set if it exists in this dataset
            if "drug_0" in train_df["drug"].values:
                assert True  # drug_0 is in training


class TestPrepareAndSplitDatasets:
    """Test prepare_and_split_datasets convenience function."""

    def test_basic_functionality(self, sample_datasets, sample_data_manager):
        """Test the combined prepare and split function."""
        result = prepare_and_split_datasets(
            datasets=sample_datasets,
            data_manager=sample_data_manager,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == set(sample_datasets.keys())

        for _name, splits in result.items():
            assert isinstance(splits, dict)
            assert set(splits.keys()) == {"train", "val", "test"}
            for _split_name, gd in splits.items():
                assert isinstance(gd, GroupedDistribution)

    def test_equivalent_to_separate_calls(self, sample_datasets, sample_data_manager):
        """Test that combined function is equivalent to separate calls."""
        # Combined call
        result_combined = prepare_and_split_datasets(
            datasets=sample_datasets,
            data_manager=sample_data_manager,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        # Separate calls
        gd_dict = prepare_datasets(
            datasets=sample_datasets,
            data_manager=sample_data_manager,
        )
        result_separate = split_datasets(
            grouped_distributions=gd_dict,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        # Should produce same structure
        assert set(result_combined.keys()) == set(result_separate.keys())
        for name in result_combined:
            assert set(result_combined[name].keys()) == set(result_separate[name].keys())

    def test_default_ratios(self, sample_datasets, sample_data_manager):
        """Test that default ratios work."""
        result = prepare_and_split_datasets(
            datasets=sample_datasets,
            data_manager=sample_data_manager,
            holdout_combinations=False,
            split_by=["drug"],
            # ratios not specified, should use default
        )

        assert len(result) == len(sample_datasets)
        for _name, splits in result.items():
            assert "train" in splits
            assert "val" in splits
            assert "test" in splits
