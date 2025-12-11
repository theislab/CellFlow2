import anndata as ad
import numpy as np
import pandas as pd

from scaleflow.data import AnnDataLocation, DataManager, GroupedDistribution


# hardcoded adata for testing the DataManager
def adata_test_basic() -> ad.AnnData:
    """Create an AnnData object for testing the DataManager."""
    drugs = ["control", "drug_A", "drug_B"]
    genes = ["control", "gene_A", "gene_B"]
    cell_lines = ["cell_line_A", "cell_line_B"]
    batches = ["batch_1", "batch_2", "batch_3"]
    plates = ["plate_1", "plate_2", "plate_3"]
    days = ["day_1", "day_2", "day_3"]
    doses = [1.0, 10.0, 100.0]

    rows = []
    for drug in drugs:
        for gene in genes:
            for cell_line in cell_lines:
                for batch in batches:
                    for plate in plates:
                        for day in days:
                            if drug != "control":
                                for dose in doses:
                                    rows.append(
                                        {
                                            "drug": drug,
                                            "gene": gene,
                                            "cell_line": cell_line,
                                            "batch": batch,
                                            "plate": plate,
                                            "day": day,
                                            "dose": dose,
                                            "control": False,
                                        }
                                    )
                            else:
                                rows.append(
                                    {
                                        "drug": drug,
                                        "gene": gene,
                                        "cell_line": cell_line,
                                        "batch": batch,
                                        "plate": plate,
                                        "day": day,
                                        "dose": 0.0,
                                        "control": gene == "control" and drug == "control",
                                    }
                                )

    n_obs = len(rows)
    n_vars = 20
    n_pca = 10

    obs = pd.DataFrame(rows)

    # Convert to categorical
    for col in ["cell_line", "drug", "gene", "batch", "plate", "day"]:
        obs[col] = obs[col].astype("category")

    # Simple X matrix (not really used in tests, just needs to exist)
    X = np.random.randn(n_obs, n_vars).astype(np.float32)

    # X_pca: Put cell index at position [idx, 0] for easy tracing
    X_pca = np.zeros((n_obs, n_pca), dtype=np.float32)
    for i in range(n_obs):
        X_pca[i, 0] = float(i)  # Cell 0 has value 0, cell 1 has value 1, etc.

    # Create AnnData
    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["X_pca"] = X_pca

    # Simple embeddings
    adata.uns["cell_line_embeddings"] = {
        "cell_line_A": np.array([1.0, 0.0], dtype=np.float32),
        "cell_line_B": np.array([0.0, 1.0], dtype=np.float32),
    }

    adata.uns["drug_embeddings"] = {
        "drug_A": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "drug_B": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "control": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    }

    adata.uns["gene_embeddings"] = {
        "gene_A": np.array([1.0, 0.0], dtype=np.float32),
        "gene_B": np.array([0.0, 1.0], dtype=np.float32),
        "control": np.array([0.0, 0.0], dtype=np.float32),
    }

    return adata


class TestDataManagerBasic:
    """Test basic DataManager functionality."""

    def test_prepare_data_basic(self):
        """Test that prepare_data works and returns correct structure."""
        adl = AnnDataLocation()

        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_embeddings",
                "drug": "drug_embeddings",
                "gene": "gene_embeddings",
            },
            data_location=adl.obsm["X_pca"],
        )

        gd = dm.prepare_data(adata_test_basic())

        assert isinstance(gd, GroupedDistribution)

        # src_dist_idx	tgt_dist_idx	cell_line	drug	gene
        #   0	0	cell_line_A	control	gene_A
        #   0	1	cell_line_A	control	gene_B
        #   0	2	cell_line_A	drug_A	control
        #   0	3	cell_line_A	drug_A	gene_A
        #   0	4	cell_line_A	drug_A	gene_B
        #   0	5	cell_line_A	drug_B	control
        #   0	6	cell_line_A	drug_B	gene_A
        #   0	7	cell_line_A	drug_B	gene_B
        #   1	8	cell_line_B	control	gene_A
        #   1	9	cell_line_B	control	gene_B
        #   1	10	cell_line_B	drug_A	control
        #   1	11	cell_line_B	drug_A	gene_A
        #   1	12	cell_line_B	drug_A	gene_B
        #   1	13	cell_line_B	drug_B	control
        #   1	14	cell_line_B	drug_B	gene_A
        #   1	15	cell_line_B	drug_B	gene_B

        expected_src_data = {0: ("cell_line_A",), 1: ("cell_line_B",)}
        expected_tgt_data = {
            0: ("drug_A", "control"),
            1: ("drug_B", "control"),
            2: ("gene_A", "control"),
            3: ("gene_B", "control"),
            4: ("drug_A", "gene_A"),
            5: ("drug_A", "gene_B"),
            6: ("drug_B", "gene_A"),
            7: ("drug_B", "gene_B"),
            8: ("drug_A", "control"),
            9: ("drug_B", "control"),
            10: ("gene_A", "control"),
            11: ("gene_B", "control"),
            12: ("drug_A", "gene_A"),
            13: ("drug_A", "gene_B"),
            14: ("drug_B", "gene_A"),
            15: ("drug_B", "gene_B"),
        }
        expected_mapping = {
            0: {0, 1, 2, 3, 4, 5, 6, 7},
            1: {8, 9, 10, 11, 12, 13, 14, 15},
        }

        assert len(gd.data.src_data) == len(expected_src_data)
        assert len(gd.data.tgt_data) == len(expected_tgt_data)

        # Test target mapping correctness
        # Verify that src_to_tgt_dist_map exists for each source
        assert len(gd.data.src_to_tgt_dist_map) == len(expected_src_data)

        # sum of the values in src_to_tgt_dist_map should be equal to the number of target distributions
        assert sum(len(v) for v in gd.data.src_to_tgt_dist_map.values()) == len(expected_tgt_data)

        # Each source should have at least one target
        for src_idx, tgt_indices in gd.data.src_to_tgt_dist_map.items():
            assert len(tgt_indices) > 0, f"Source {src_idx} has no targets"
            # All target indices should exist in tgt_data
            for tgt_idx in tgt_indices:
                assert tgt_idx in gd.data.tgt_data, f"Target {tgt_idx} not in tgt_data"

        # Verify that targets are correctly mapped to their source cell_lines
        # using the src_tgt_dist_df
        src_tgt_df = gd.annotation.src_tgt_dist_df

        # For each target, verify it belongs to the correct source
        for _, row in src_tgt_df.iterrows():
            src_idx = row["src_dist_idx"]
            tgt_idx = row["tgt_dist_idx"]
            # Target should be in the source's target list
            assert tgt_idx in gd.data.src_to_tgt_dist_map[src_idx], (
                f"Target {tgt_idx} not found in source {src_idx}'s mapping"
            )
            assert tgt_idx in expected_mapping[src_idx], f"Target {tgt_idx} not found in source {src_idx}'s mapping"

    def test_ordering_reconstruction_after_shuffle(self):
        """Test that we can reconstruct original ordering after shuffling."""
        # Store original order information
        adata_test = adata_test_basic()  # again hardcoded
        # because the data should be a bit small otherwise this
        # test will be too slow
        original_index = adata_test.obs.index.to_numpy().copy()
        original_X_pca = adata_test.obsm["X_pca"].copy()

        # Shuffle the adata
        shuffle_idx = np.random.permutation(len(adata_test))
        adata_shuffled = adata_test[shuffle_idx].copy()

        # Verify it's actually shuffled (should not be identical for reasonable dataset sizes)
        assert not np.array_equal(adata_shuffled.obs.index.to_numpy(), original_index), "Data should be shuffled"

        # Create DataManager and prepare data
        adl = AnnDataLocation()
        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_embeddings",
                "drug": "drug_embeddings",
                "gene": "gene_embeddings",
            },
            data_location=adl.obsm["X_pca"],
        )

        gd = dm.prepare_data(adata_shuffled)

        # Test 1: old_obs_index should map from sorted order back to shuffled AnnData index
        assert len(gd.annotation.old_obs_index) == len(adata_shuffled)
        assert np.all(np.isin(gd.annotation.old_obs_index, adata_shuffled.obs.index.to_numpy())), (
            "old_obs_index should contain valid indices from shuffled adata"
        )

        # Verify: For each position in the sorted data, the old_obs_index tells us
        # which original (shuffled adata) index it came from
        for _, old_idx in enumerate(gd.annotation.old_obs_index):
            # Find the corresponding cell in the shuffled adata
            shuffled_pos = np.where(adata_shuffled.obs.index == old_idx)[0][0]

            # The first element of X_pca contains the original cell index from adata_test
            # (this was set up in the fixture)
            original_cell_id = adata_shuffled.obsm["X_pca"][shuffled_pos, 0]

            # This should match the original unshuffled data
            assert original_cell_id == original_X_pca[np.where(original_index == old_idx)[0][0], 0]

        # Test 3: Verify we can fully reconstruct the mapping
        # Create inverse mapping: from old_obs_index position -> shuffled adata position
        old_idx_to_shuffled_pos = {
            old_idx: np.where(adata_shuffled.obs.index == old_idx)[0][0] for old_idx in gd.annotation.old_obs_index
        }

        # This should cover all cells in the shuffled adata
        assert len(old_idx_to_shuffled_pos) == len(adata_shuffled)


class TestDataManagerSplitIntegration:
    """Test DataManager integration with GroupedDistributionSplitter."""

    def test_split_annotation_preserves_data_mapping(self):
        """Test that split annotations can still be used to access correct data."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter

        adata = adata_test_basic()
        adl = AnnDataLocation()

        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_embeddings",
                "drug": "drug_embeddings",
                "gene": "gene_embeddings",
            },
            data_location=adl.obsm["X_pca"],
        )

        gd = dm.prepare_data(adata)

        splitter = GroupedDistributionSplitter(
            gd=gd,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        split_result = splitter.split_annotation()

        # Verify that for each split, the tgt_dist_idx in the annotation
        # corresponds to valid keys in gd.data.tgt_data
        for split_name, split_annotation in split_result.items():
            for tgt_idx in split_annotation.src_tgt_dist_df["tgt_dist_idx"].unique():
                assert tgt_idx in gd.data.tgt_data, (
                    f"tgt_dist_idx {tgt_idx} from {split_name} should exist in gd.data.tgt_data"
                )
                assert tgt_idx in gd.data.conditions, (
                    f"tgt_dist_idx {tgt_idx} from {split_name} should exist in gd.data.conditions"
                )

    def test_split_annotation_preserves_src_data_mapping(self):
        """Test that split annotations maintain valid src_dist_idx references."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter

        adata = adata_test_basic()
        adl = AnnDataLocation()

        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_embeddings",
                "drug": "drug_embeddings",
                "gene": "gene_embeddings",
            },
            data_location=adl.obsm["X_pca"],
        )

        gd = dm.prepare_data(adata)

        splitter = GroupedDistributionSplitter(
            gd=gd,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        split_result = splitter.split_annotation()

        # Verify that for each split, the src_dist_idx in the annotation
        # corresponds to valid keys in gd.data.src_data
        for split_name, split_annotation in split_result.items():
            for src_idx in split_annotation.src_tgt_dist_df["src_dist_idx"].unique():
                assert src_idx in gd.data.src_data, (
                    f"src_dist_idx {src_idx} from {split_name} should exist in gd.data.src_data"
                )

    def test_old_obs_index_consistency_after_split(self):
        """Test that old_obs_index enables reconstruction to original adata."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter

        adata = adata_test_basic()
        original_obs_index = adata.obs.index.to_numpy().copy()

        adl = AnnDataLocation()
        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_embeddings",
                "drug": "drug_embeddings",
                "gene": "gene_embeddings",
            },
            data_location=adl.obsm["X_pca"],
        )

        gd = dm.prepare_data(adata)

        splitter = GroupedDistributionSplitter(
            gd=gd,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        split_result = splitter.split_annotation()

        # All splits should have the same old_obs_index that maps to original adata
        for split_annotation in split_result.values():
            # old_obs_index should contain valid indices from original adata
            assert np.all(np.isin(split_annotation.old_obs_index, original_obs_index))


class TestFullRoundTrip:
    """Test full round-trip: adata → DataManager → GroupedDistribution → split → verify."""

    def test_full_roundtrip_preserves_data_integrity(self):
        """Test complete round-trip from adata to split and back."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter

        adata = adata_test_basic()
        adl = AnnDataLocation()

        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_embeddings",
                "drug": "drug_embeddings",
                "gene": "gene_embeddings",
            },
            data_location=adl.obsm["X_pca"],
        )

        gd = dm.prepare_data(adata)

        splitter = GroupedDistributionSplitter(
            gd=gd,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        split_result = splitter.split()

        # Verify we can reconstruct all tgt_data
        reconstructed_tgt_data = {}
        for split_gd in split_result.values():
            reconstructed_tgt_data.update(split_gd.data.tgt_data)

        # All original tgt distributions should be present
        assert set(reconstructed_tgt_data.keys()) == set(gd.data.tgt_data.keys())

        # Data should match exactly
        for tgt_idx in gd.data.tgt_data:
            assert np.array_equal(reconstructed_tgt_data[tgt_idx], gd.data.tgt_data[tgt_idx])

    def test_split_gd_can_access_original_adata_cells(self):
        """Test that split GD can trace cells back to original adata."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter

        adata = adata_test_basic()
        adl = AnnDataLocation()

        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_embeddings",
                "drug": "drug_embeddings",
                "gene": "gene_embeddings",
            },
            data_location=adl.obsm["X_pca"],
        )

        gd = dm.prepare_data(adata)

        splitter = GroupedDistributionSplitter(
            gd=gd,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        split_result = splitter.split()

        # Each split should have old_obs_index that maps to original adata
        original_obs_index = adata.obs.index.to_numpy()
        for split_name, split_gd in split_result.items():
            # old_obs_index should contain valid indices from original adata
            assert np.all(np.isin(split_gd.annotation.old_obs_index, original_obs_index)), (
                f"Invalid old_obs_index in {split_name}"
            )

    def test_split_preserves_embedding_conditions(self):
        """Test that conditions (embeddings) are preserved after split."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter

        adata = adata_test_basic()
        adl = AnnDataLocation()

        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_embeddings",
                "drug": "drug_embeddings",
                "gene": "gene_embeddings",
            },
            data_location=adl.obsm["X_pca"],
        )

        gd = dm.prepare_data(adata)

        splitter = GroupedDistributionSplitter(
            gd=gd,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        split_result = splitter.split()

        # Verify conditions are preserved
        for split_gd in split_result.values():
            for tgt_idx, condition in split_gd.data.conditions.items():
                original_condition = gd.data.conditions[tgt_idx]
                assert np.array_equal(condition, original_condition), (
                    f"Condition for tgt_idx {tgt_idx} doesn't match original"
                )


class TestShuffleReconstructionWithSplit:
    """Test that order can be reconstructed after shuffling, through the split process."""

    def test_split_preserves_order_reconstruction_after_shuffle(self):
        """Test that we can reconstruct original order after shuffle and split."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter

        # Store original order information
        adata = adata_test_basic()
        original_index = adata.obs.index.to_numpy().copy()
        _original_X_pca = adata.obsm["X_pca"].copy()

        # Shuffle the adata
        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(adata))
        adata_shuffled = adata[shuffle_idx].copy()

        # Verify it's actually shuffled
        assert not np.array_equal(adata_shuffled.obs.index.to_numpy(), original_index), "Data should be shuffled"

        # Create GroupedDistribution from shuffled data
        adl = AnnDataLocation()
        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_embeddings",
                "drug": "drug_embeddings",
                "gene": "gene_embeddings",
            },
            data_location=adl.obsm["X_pca"],
        )

        gd = dm.prepare_data(adata_shuffled)

        # Split the data
        splitter = GroupedDistributionSplitter(
            gd=gd,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        split_result = splitter.split()

        # Each split should preserve the old_obs_index for reconstruction
        for split_name, split_gd in split_result.items():
            # old_obs_index should still map to valid shuffled adata indices
            assert np.all(
                np.isin(
                    split_gd.annotation.old_obs_index,
                    adata_shuffled.obs.index.to_numpy(),
                )
            ), f"old_obs_index in {split_name} should map to shuffled adata"

            # old_obs_index should match the original gd's old_obs_index
            assert np.array_equal(split_gd.annotation.old_obs_index, gd.annotation.old_obs_index), (
                f"old_obs_index in {split_name} should match original gd"
            )

    def test_split_tgt_data_traceable_to_original_cells_after_shuffle(self):
        """Test that tgt_data in splits can be traced back to original adata cells."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter

        # Store original order information
        adata = adata_test_basic()
        original_X_pca = adata.obsm["X_pca"].copy()

        # Shuffle the adata
        np.random.seed(123)
        shuffle_idx = np.random.permutation(len(adata))
        adata_shuffled = adata[shuffle_idx].copy()

        # Create GroupedDistribution from shuffled data
        adl = AnnDataLocation()
        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_embeddings",
                "drug": "drug_embeddings",
                "gene": "gene_embeddings",
            },
            data_location=adl.obsm["X_pca"],
        )

        gd = dm.prepare_data(adata_shuffled)

        # Split the data
        splitter = GroupedDistributionSplitter(
            gd=gd,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        split_result = splitter.split()

        # For each split, verify tgt_data matches original gd's tgt_data
        for split_name, split_gd in split_result.items():
            for tgt_idx, tgt_data in split_gd.data.tgt_data.items():
                # tgt_data should match original gd's tgt_data
                assert np.array_equal(tgt_data, gd.data.tgt_data[tgt_idx]), (
                    f"tgt_data[{tgt_idx}] in {split_name} doesn't match original"
                )

                # The first column of tgt_data contains original cell indices
                # (this is how the fixture was set up)
                for cell_row in tgt_data:
                    original_cell_id = int(cell_row[0])
                    # This ID should exist in the original X_pca
                    assert original_cell_id in original_X_pca[:, 0], (
                        f"Cell ID {original_cell_id} not found in original data"
                    )

    def test_split_src_data_traceable_to_original_cells_after_shuffle(self):
        """Test that src_data in splits can be traced back to original adata cells."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter

        # Store original order information
        adata = adata_test_basic()
        original_X_pca = adata.obsm["X_pca"].copy()

        # Shuffle the adata
        np.random.seed(456)
        shuffle_idx = np.random.permutation(len(adata))
        adata_shuffled = adata[shuffle_idx].copy()

        # Create GroupedDistribution from shuffled data
        adl = AnnDataLocation()
        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_embeddings",
                "drug": "drug_embeddings",
                "gene": "gene_embeddings",
            },
            data_location=adl.obsm["X_pca"],
        )

        gd = dm.prepare_data(adata_shuffled)

        # Split the data
        splitter = GroupedDistributionSplitter(
            gd=gd,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        split_result = splitter.split()

        # For each split, verify src_data matches original gd's src_data
        for split_name, split_gd in split_result.items():
            for src_idx, src_data in split_gd.data.src_data.items():
                # src_data should match original gd's src_data
                assert np.array_equal(src_data, gd.data.src_data[src_idx]), (
                    f"src_data[{src_idx}] in {split_name} doesn't match original"
                )

                # The first column of src_data contains original cell indices
                for cell_row in src_data:
                    original_cell_id = int(cell_row[0])
                    # This ID should exist in the original X_pca
                    assert original_cell_id in original_X_pca[:, 0], (
                        f"Cell ID {original_cell_id} not found in original data"
                    )

    def test_reconstruct_original_cell_indices_from_split_after_shuffle(self):
        """Test complete reconstruction of original cell indices after shuffle and split."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter

        # Store original order information
        adata = adata_test_basic()
        original_X_pca = adata.obsm["X_pca"].copy()
        # Cell IDs are stored in X_pca[:, 0]
        original_cell_ids = set(original_X_pca[:, 0].astype(int))

        # Shuffle the adata
        np.random.seed(789)
        shuffle_idx = np.random.permutation(len(adata))
        adata_shuffled = adata[shuffle_idx].copy()

        # Create GroupedDistribution from shuffled data
        adl = AnnDataLocation()
        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_embeddings",
                "drug": "drug_embeddings",
                "gene": "gene_embeddings",
            },
            data_location=adl.obsm["X_pca"],
        )

        gd = dm.prepare_data(adata_shuffled)

        # Split the data
        splitter = GroupedDistributionSplitter(
            gd=gd,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        split_result = splitter.split()

        # Collect all cell IDs from tgt_data across all splits
        reconstructed_tgt_cell_ids = set()
        for split_gd in split_result.values():
            for tgt_data in split_gd.data.tgt_data.values():
                cell_ids = set(tgt_data[:, 0].astype(int))
                reconstructed_tgt_cell_ids.update(cell_ids)

        # Collect all cell IDs from src_data across all splits
        # (src_data may overlap across splits since same source can have multiple targets)
        reconstructed_src_cell_ids = set()
        for split_gd in split_result.values():
            for src_data in split_gd.data.src_data.values():
                cell_ids = set(src_data[:, 0].astype(int))
                reconstructed_src_cell_ids.update(cell_ids)

        # All reconstructed cell IDs should be subset of original
        assert reconstructed_tgt_cell_ids.issubset(original_cell_ids), "All tgt cell IDs should exist in original data"
        assert reconstructed_src_cell_ids.issubset(original_cell_ids), "All src cell IDs should exist in original data"

        # Union of src and tgt should cover all original cells
        all_reconstructed = reconstructed_src_cell_ids | reconstructed_tgt_cell_ids
        assert all_reconstructed == original_cell_ids, "All original cells should be in either src or tgt data"

    def test_multiple_shuffles_produce_same_splits_with_same_seed(self):
        """Test that the same random seed produces identical splits after different shuffles."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter

        adata = adata_test_basic()

        # First shuffle
        np.random.seed(111)
        shuffle_idx1 = np.random.permutation(len(adata))
        adata_shuffled1 = adata[shuffle_idx1].copy()

        # Second shuffle (different)
        np.random.seed(222)
        shuffle_idx2 = np.random.permutation(len(adata))
        adata_shuffled2 = adata[shuffle_idx2].copy()

        # Create GroupedDistributions from both shuffles
        adl = AnnDataLocation()
        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_embeddings",
                "drug": "drug_embeddings",
                "gene": "gene_embeddings",
            },
            data_location=adl.obsm["X_pca"],
        )

        gd1 = dm.prepare_data(adata_shuffled1)
        gd2 = dm.prepare_data(adata_shuffled2)

        # Split both with the same random_state
        splitter1 = GroupedDistributionSplitter(
            gd=gd1,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )
        splitter2 = GroupedDistributionSplitter(
            gd=gd2,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        split1 = splitter1.split()
        split2 = splitter2.split()

        # Both should have the same tgt_dist_idx in each split
        for split_name in ["train", "val", "test"]:
            tgt_idxs1 = set(split1[split_name].data.tgt_data.keys())
            tgt_idxs2 = set(split2[split_name].data.tgt_data.keys())
            assert tgt_idxs1 == tgt_idxs2, f"Same random_state should produce same tgt_dist_idx in {split_name}"
