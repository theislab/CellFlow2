from pathlib import Path

import numpy as np
import pytest

from scaleflow.data import DataManager
from scaleflow.data._data_splitter import DataSplitter


class TestDataSplitterValidation:
    def test_mismatched_datasets_and_names(self, adata_perturbation):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        with pytest.raises(ValueError, match="training_datasets length.*must match.*dataset_names length"):
            DataSplitter(
                training_datasets=[train_data],
                dataset_names=["dataset1", "dataset2"],
                split_ratios=[[0.8, 0.1, 0.1]],
            )

    def test_mismatched_datasets_and_ratios(self, adata_perturbation):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        with pytest.raises(ValueError, match="split_ratios length.*must match.*training_datasets length"):
            DataSplitter(
                training_datasets=[train_data],
                dataset_names=["dataset1"],
                split_ratios=[[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]],
            )

    def test_invalid_split_ratios_format(self, adata_perturbation):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        with pytest.raises(ValueError, match="must be a list of 3 values"):
            DataSplitter(
                training_datasets=[train_data],
                dataset_names=["dataset1"],
                split_ratios=[[0.8, 0.2]],
            )

    def test_split_ratios_dont_sum_to_one(self, adata_perturbation):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        with pytest.raises(ValueError, match="must sum to 1.0"):
            DataSplitter(
                training_datasets=[train_data],
                dataset_names=["dataset1"],
                split_ratios=[[0.8, 0.1, 0.2]],
            )

    def test_negative_split_ratios(self, adata_perturbation):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        with pytest.raises(ValueError, match="must be non-negative"):
            DataSplitter(
                training_datasets=[train_data],
                dataset_names=["dataset1"],
                split_ratios=[[0.9, 0.2, -0.1]],
            )

    def test_holdout_groups_requires_split_key(self, adata_perturbation):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        with pytest.raises(ValueError, match="split_key must be provided"):
            DataSplitter(
                training_datasets=[train_data],
                dataset_names=["dataset1"],
                split_ratios=[[0.8, 0.1, 0.1]],
                split_type="holdout_groups",
            )

    def test_holdout_combinations_requires_control_value(self, adata_perturbation):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1", "drug2"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        with pytest.raises(ValueError, match="control_value must be provided"):
            DataSplitter(
                training_datasets=[train_data],
                dataset_names=["dataset1"],
                split_ratios=[[0.8, 0.1, 0.1]],
                split_type="holdout_combinations",
                split_key="drug",
            )


class TestRandomSplit:
    @pytest.mark.parametrize("hard_test_split", [True, False])
    @pytest.mark.parametrize("split_ratios", [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [1.0, 0.0, 0.0]])
    def test_random_split_ratios(self, adata_perturbation, hard_test_split, split_ratios):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        splitter = DataSplitter(
            training_datasets=[train_data],
            dataset_names=["dataset1"],
            split_ratios=[split_ratios],
            split_type="random",
            hard_test_split=hard_test_split,
            random_state=42,
        )

        results = splitter.split_all_datasets()

        assert "dataset1" in results
        indices = results["dataset1"]["indices"]

        n_cells = train_data.perturbation_covariates_mask.shape[0]
        total_assigned = len(indices["train"]) + len(indices["val"]) + len(indices["test"])
        assert total_assigned == n_cells

        train_ratio, val_ratio, test_ratio = split_ratios
        assert len(indices["train"]) == pytest.approx(train_ratio * n_cells, abs=1)
        if val_ratio > 0:
            assert len(indices["val"]) > 0
        if test_ratio > 0:
            assert len(indices["test"]) > 0

    def test_random_split_reproducibility(self, adata_perturbation):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        splitter1 = DataSplitter(
            training_datasets=[train_data],
            dataset_names=["dataset1"],
            split_ratios=[[0.8, 0.1, 0.1]],
            split_type="random",
            random_state=42,
        )
        results1 = splitter1.split_all_datasets()

        splitter2 = DataSplitter(
            training_datasets=[train_data],
            dataset_names=["dataset1"],
            split_ratios=[[0.8, 0.1, 0.1]],
            split_type="random",
            random_state=42,
        )
        results2 = splitter2.split_all_datasets()

        assert np.array_equal(results1["dataset1"]["indices"]["train"], results2["dataset1"]["indices"]["train"])
        assert np.array_equal(results1["dataset1"]["indices"]["val"], results2["dataset1"]["indices"]["val"])
        assert np.array_equal(results1["dataset1"]["indices"]["test"], results2["dataset1"]["indices"]["test"])

    def test_random_split_no_overlap_hard(self, adata_perturbation):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        splitter = DataSplitter(
            training_datasets=[train_data],
            dataset_names=["dataset1"],
            split_ratios=[[0.7, 0.2, 0.1]],
            split_type="random",
            hard_test_split=True,
            random_state=42,
        )
        results = splitter.split_all_datasets()
        indices = results["dataset1"]["indices"]

        train_set = set(indices["train"])
        val_set = set(indices["val"])
        test_set = set(indices["test"])

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0


class TestHoldoutGroups:
    @pytest.mark.parametrize("hard_test_split", [True, False])
    def test_holdout_groups_basic(self, adata_perturbation, hard_test_split):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        splitter = DataSplitter(
            training_datasets=[train_data],
            dataset_names=["dataset1"],
            split_ratios=[[0.6, 0.2, 0.2]],
            split_type="holdout_groups",
            split_key="drug",
            hard_test_split=hard_test_split,
            random_state=42,
        )

        results = splitter.split_all_datasets()

        assert "dataset1" in results
        assert "split_values" in results["dataset1"]

        split_values = results["dataset1"]["split_values"]
        assert "train" in split_values
        assert "val" in split_values
        assert "test" in split_values

    def test_holdout_groups_force_training_values(self, adata_perturbation):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=[],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        # Get available perturbation values (not control)
        unique_values = set()
        for covariates in train_data.perturbation_idx_to_covariates.values():
            unique_values.update(covariates)

        # Use "drug_a" instead of "control" since control cells are filtered out
        force_value = "drug_a"
        if force_value not in unique_values:
            pytest.skip("drug_a not in perturbation values")

        splitter = DataSplitter(
            training_datasets=[train_data],
            dataset_names=["dataset1"],
            split_ratios=[[0.6, 0.2, 0.2]],
            split_type="holdout_groups",
            split_key="drug",
            force_training_values=[force_value],
            random_state=42,
        )

        results = splitter.split_all_datasets()
        split_values = results["dataset1"]["split_values"]

        assert force_value in split_values["train"]
        assert force_value not in split_values["val"]
        assert force_value not in split_values["test"]

    def test_holdout_groups_fixed_test_seed(self, adata_perturbation):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        results_list = []
        for seed in [42, 43, 44]:
            splitter = DataSplitter(
                training_datasets=[train_data],
                dataset_names=["dataset1"],
                split_ratios=[[0.6, 0.2, 0.2]],
                split_type="holdout_groups",
                split_key="drug",
                test_random_state=999,
                val_random_state=seed,
                random_state=seed,
            )
            results = splitter.split_all_datasets()
            results_list.append(results)

        test_values_1 = set(results_list[0]["dataset1"]["split_values"]["test"])
        test_values_2 = set(results_list[1]["dataset1"]["split_values"]["test"])
        test_values_3 = set(results_list[2]["dataset1"]["split_values"]["test"])

        assert test_values_1 == test_values_2 == test_values_3

        val_values_1 = set(results_list[0]["dataset1"]["split_values"]["val"])
        val_values_2 = set(results_list[1]["dataset1"]["split_values"]["val"])

        if len(val_values_1) > 0 and len(val_values_2) > 0:
            assert val_values_1 != val_values_2


class TestHoldoutCombinations:
    @pytest.mark.parametrize("hard_test_split", [True, False])
    def test_holdout_combinations_basic(self, adata_perturbation, hard_test_split):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1", "drug2"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        splitter = DataSplitter(
            training_datasets=[train_data],
            dataset_names=["dataset1"],
            split_ratios=[[0.6, 0.2, 0.2]],
            split_type="holdout_combinations",
            split_key=["drug1", "drug2"],
            control_value="control",
            hard_test_split=hard_test_split,
            random_state=42,
        )

        results = splitter.split_all_datasets()

        assert "dataset1" in results
        indices = results["dataset1"]["indices"]

        assert len(indices["train"]) > 0
        assert len(indices["val"]) >= 0
        assert len(indices["test"]) >= 0

    def test_holdout_combinations_singletons_in_train(self):
        # Create test data with a good number of combinations
        import anndata as ad

        n_obs = 1000  # Increased to accommodate more combinations
        n_vars = 50
        n_pca = 10

        X_data = np.random.rand(n_obs, n_vars)
        my_counts = np.random.rand(n_obs, n_vars)
        X_pca = np.random.rand(n_obs, n_pca)

        # Use 5 drugs to get 20 unique combinations (5 * 4)
        drugs = ["drug_a", "drug_b", "drug_c", "drug_d", "drug_e"]
        cell_lines = ["cell_line_a", "cell_line_b", "cell_line_c"]

        # Create structured data with known combinations
        drug1_list = []
        drug2_list = []

        # Control cells (100)
        drug1_list.extend(["control"] * 100)
        drug2_list.extend(["control"] * 100)

        # Singleton on drug1 (250 cells: 50 per drug)
        for drug in drugs:
            drug1_list.extend([drug] * 50)
            drug2_list.extend(["control"] * 50)

        # Singleton on drug2 (250 cells: 50 per drug)
        for drug in drugs:
            drug1_list.extend(["control"] * 50)
            drug2_list.extend([drug] * 50)

        # Combinations (400 cells distributed across 20 combinations = 20 cells each)
        # Create all possible non-control combinations
        combinations = []
        for d1 in drugs:
            for d2 in drugs:
                if d1 != d2:  # Different drugs (true combinations)
                    combinations.append((d1, d2))

        # Distribute 400 cells evenly across combinations (20 cells per combination)
        cells_per_combo = 400 // len(combinations)

        for d1, d2 in combinations:
            drug1_list.extend([d1] * cells_per_combo)
            drug2_list.extend([d2] * cells_per_combo)

        # Create cell line assignments
        import pandas as pd
        cell_type_list = np.random.choice(cell_lines, n_obs)
        dosages = np.random.choice([10.0, 100.0, 1000.0], n_obs)

        obs_data = pd.DataFrame({
            "cell_type": cell_type_list,
            "dosage": dosages,
            "drug1": drug1_list,
            "drug2": drug2_list,
            "drug3": ["control"] * n_obs,
            "dosage_a": np.random.choice([10.0, 100.0, 1000.0], n_obs),
            "dosage_b": np.random.choice([10.0, 100.0, 1000.0], n_obs),
            "dosage_c": np.random.choice([10.0, 100.0, 1000.0], n_obs),
        })

        # Create an AnnData object
        adata_combinations = ad.AnnData(X=X_data, obs=obs_data)
        adata_combinations.layers["my_counts"] = my_counts
        adata_combinations.obsm["X_pca"] = X_pca

        # Add boolean columns for each drug
        for drug in drugs:
            adata_combinations.obs[drug] = (
                (adata_combinations.obs["drug1"] == drug) |
                (adata_combinations.obs["drug2"] == drug) |
                (adata_combinations.obs["drug3"] == drug)
            )

        adata_combinations.obs["control"] = (
            (adata_combinations.obs["drug1"] == "control") &
            (adata_combinations.obs["drug2"] == "control")
        )

        # Convert to categorical EXCEPT for control and boolean drug columns
        for col in adata_combinations.obs.columns:
            if col not in ["control"] + drugs:
                adata_combinations.obs[col] = adata_combinations.obs[col].astype("category")

        # Add embeddings
        drug_emb = {}
        for drug in adata_combinations.obs["drug1"].cat.categories:
            drug_emb[drug] = np.random.randn(5, 1)
        adata_combinations.uns["drug"] = drug_emb

        cell_type_emb = {}
        for cell_type in adata_combinations.obs["cell_type"].cat.categories:
            cell_type_emb[cell_type] = np.random.randn(3, 1)
        adata_combinations.uns["cell_type"] = cell_type_emb

        # Now run the actual test
        dm = DataManager(
            adata_combinations,
            sample_rep="X",
            split_covariates=[],
            control_key="control",
            perturbation_covariates={"drug": ["drug1", "drug2"]},
        )
        train_data = dm.get_train_data(adata_combinations)

        splitter = DataSplitter(
            training_datasets=[train_data],
            dataset_names=["dataset1"],
            split_ratios=[[0.6, 0.2, 0.2]],
            split_type="holdout_combinations",
            split_key=["drug1", "drug2"],
            control_value="control",
            random_state=42,
        )

        results = splitter.split_all_datasets()

        perturbation_covariates_mask = train_data.perturbation_covariates_mask
        perturbation_idx_to_covariates = train_data.perturbation_idx_to_covariates

        train_indices = results["dataset1"]["indices"]["train"]
        val_indices = results["dataset1"]["indices"]["val"]
        test_indices = results["dataset1"]["indices"]["test"]

        # Verify that ALL singletons and controls are in training
        all_singletons = []
        all_combinations = []

        for idx in range(len(perturbation_covariates_mask)):
            pert_idx = perturbation_covariates_mask[idx]
            if pert_idx >= 0:
                covariates = perturbation_idx_to_covariates[pert_idx]
                non_control_count = sum(1 for c in covariates if c != "control")
                if non_control_count == 1:
                    all_singletons.append(idx)
                elif non_control_count > 1:
                    all_combinations.append(idx)

        train_set = set(train_indices)

        # All singletons should be in training
        for singleton_idx in all_singletons:
            assert singleton_idx in train_set, "All singleton perturbations should be in training"

        # Some (but not all) combinations should be in training according to split_ratios
        combinations_in_train = [idx for idx in all_combinations if idx in train_set]
        combinations_in_val = [idx for idx in all_combinations if idx in set(val_indices)]
        combinations_in_test = [idx for idx in all_combinations if idx in set(test_indices)]

        # With enough combinations, we should see proper distribution
        assert len(all_combinations) > 0, "Test data should have combination perturbations"

        train_combo_ratio = len(combinations_in_train) / len(all_combinations)
        val_combo_ratio = len(combinations_in_val) / len(all_combinations)
        test_combo_ratio = len(combinations_in_test) / len(all_combinations)

        # With 0.6, 0.2, 0.2 ratios, allow some tolerance
        assert 0.4 < train_combo_ratio < 0.8, f"Expected ~60% of combinations in training, got {train_combo_ratio:.2%}"
        assert 0.05 < val_combo_ratio < 0.35, f"Expected ~20% of combinations in val, got {val_combo_ratio:.2%}"
        assert 0.05 < test_combo_ratio < 0.35, f"Expected ~20% of combinations in test, got {test_combo_ratio:.2%}"


class TestStratifiedSplit:
    @pytest.mark.parametrize("hard_test_split", [True, False])
    def test_stratified_split_basic(self, adata_perturbation, hard_test_split):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        splitter = DataSplitter(
            training_datasets=[train_data],
            dataset_names=["dataset1"],
            split_ratios=[[0.8, 0.1, 0.1]],
            split_type="stratified",
            split_key="drug",
            hard_test_split=hard_test_split,
            random_state=42,
        )

        results = splitter.split_all_datasets()

        assert "dataset1" in results
        indices = results["dataset1"]["indices"]

        n_cells = train_data.perturbation_covariates_mask.shape[0]
        total_assigned = len(indices["train"]) + len(indices["val"]) + len(indices["test"])
        assert total_assigned == n_cells


class TestMultipleDatasets:
    def test_multiple_datasets_different_ratios(self, adata_perturbation):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data1 = dm.get_train_data(adata_perturbation)
        train_data2 = dm.get_train_data(adata_perturbation)

        splitter = DataSplitter(
            training_datasets=[train_data1, train_data2],
            dataset_names=["dataset1", "dataset2"],
            split_ratios=[[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]],
            split_type="random",
            random_state=42,
        )

        results = splitter.split_all_datasets()

        assert "dataset1" in results
        assert "dataset2" in results

        n_cells = train_data1.perturbation_covariates_mask.shape[0]

        assert len(results["dataset1"]["indices"]["train"]) == pytest.approx(0.8 * n_cells, abs=1)
        assert len(results["dataset2"]["indices"]["train"]) == pytest.approx(0.7 * n_cells, abs=1)


class TestSaveAndLoad:
    def test_save_and_load_splits(self, adata_perturbation, tmp_path):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        splitter = DataSplitter(
            training_datasets=[train_data],
            dataset_names=["dataset1"],
            split_ratios=[[0.8, 0.1, 0.1]],
            split_type="holdout_groups",
            split_key="drug",
            random_state=42,
        )

        results = splitter.split_all_datasets()
        splitter.save_splits(tmp_path / "splits")

        assert (tmp_path / "splits" / "split_summary.json").exists()
        assert (tmp_path / "splits" / "dataset1" / "metadata.json").exists()
        assert (tmp_path / "splits" / "dataset1" / "split_info.pkl").exists()

        loaded_info = DataSplitter.load_split_info(tmp_path / "splits", "dataset1")

        assert "indices" in loaded_info
        assert "metadata" in loaded_info

        assert np.array_equal(loaded_info["indices"]["train"], results["dataset1"]["indices"]["train"])
        assert np.array_equal(loaded_info["indices"]["val"], results["dataset1"]["indices"]["val"])
        assert np.array_equal(loaded_info["indices"]["test"], results["dataset1"]["indices"]["test"])

    def test_load_nonexistent_split(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DataSplitter.load_split_info(tmp_path / "nonexistent", "dataset1")


class TestSplitSummary:
    def test_generate_split_summary(self, adata_perturbation):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        splitter = DataSplitter(
            training_datasets=[train_data],
            dataset_names=["dataset1"],
            split_ratios=[[0.8, 0.1, 0.1]],
            split_type="holdout_groups",
            split_key="drug",
            random_state=42,
        )

        splitter.split_all_datasets()
        summary = splitter.generate_split_summary()

        assert "dataset1" in summary
        assert "configuration" in summary["dataset1"]
        assert "statistics" in summary["dataset1"]
        assert "observations_per_condition" in summary["dataset1"]

        config = summary["dataset1"]["configuration"]
        assert config["split_type"] == "holdout_groups"
        assert config["random_state"] == 42

        stats = summary["dataset1"]["statistics"]
        assert "total_observations" in stats
        assert "train_observations" in stats
        assert "val_observations" in stats
        assert "test_observations" in stats

    def test_summary_before_split_raises(self, adata_perturbation):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        splitter = DataSplitter(
            training_datasets=[train_data],
            dataset_names=["dataset1"],
            split_ratios=[[0.8, 0.1, 0.1]],
            split_type="random",
            random_state=42,
        )

        with pytest.raises(ValueError, match="No split results available"):
            splitter.generate_split_summary()


class TestExtractPerturbationInfo:
    def test_extract_perturbation_info(self, adata_perturbation):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )
        train_data = dm.get_train_data(adata_perturbation)

        splitter = DataSplitter(
            training_datasets=[train_data],
            dataset_names=["dataset1"],
            split_ratios=[[0.8, 0.1, 0.1]],
            split_type="random",
        )

        pert_info = splitter.extract_perturbation_info(train_data)

        assert "perturbation_covariates_mask" in pert_info
        assert "perturbation_idx_to_covariates" in pert_info
        assert "n_cells" in pert_info

        assert isinstance(pert_info["perturbation_covariates_mask"], np.ndarray)
        assert isinstance(pert_info["perturbation_idx_to_covariates"], dict)
        assert pert_info["n_cells"] == len(train_data.perturbation_covariates_mask)
