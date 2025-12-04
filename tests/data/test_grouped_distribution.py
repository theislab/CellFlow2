import numpy as np
import pandas as pd
import pytest
import zarr

from scaleflow.data import AnnDataLocation
from scaleflow.data._data import (
    GroupedDistribution,
    GroupedDistributionAnnotation,
    GroupedDistributionData,
)


@pytest.fixture
def dummy_grouped_distribution_data():
    src_to_tgt_dist_map = {0: [0, 1], 1: [2]}
    src_data = {0: np.random.rand(10, 5), 1: np.random.rand(8, 5)}
    tgt_data = {0: np.random.rand(5, 5), 1: np.random.rand(5, 5), 2: np.random.rand(8, 5)}
    conditions = {
        0: {"cond1": np.array([1, 2, 3]), "cond2": np.array([4, 5])},
        1: {"cond1": np.array([6, 7]), "cond2": np.array([8, 9])},
        2: {"cond1": np.array([10]), "cond2": np.array([11])},
    }
    return GroupedDistributionData(
        src_to_tgt_dist_map=src_to_tgt_dist_map,
        src_data=src_data,
        tgt_data=tgt_data,
        conditions=conditions,
    )


@pytest.fixture
def dummy_grouped_distribution_annotation():
    old_obs_index = np.arange(20)
    src_dist_idx_to_labels = {0: ["label1", "label2"], 1: ["label3"]}
    tgt_dist_idx_to_labels = {0: ["tlabel1"], 1: ["tlabel2"], 2: ["tlabel3"]}
    src_tgt_dist_df = pd.DataFrame({"src_dist_idx": [0, 0, 1], "tgt_dist_idx": [0, 1, 2], "other_col": ["a", "b", "c"]})
    default_values = {"param1": 1, "param2": "val"}
    tgt_dist_keys = ["key1", "key2"]
    src_dist_keys = ["skey1"]
    dist_flag_key = "flag"
    data_location = AnnDataLocation().obsm["X_pca"]

    return GroupedDistributionAnnotation(
        old_obs_index=old_obs_index,
        src_dist_idx_to_labels=src_dist_idx_to_labels,
        tgt_dist_idx_to_labels=tgt_dist_idx_to_labels,
        src_tgt_dist_df=src_tgt_dist_df,
        default_values=default_values,
        tgt_dist_keys=tgt_dist_keys,
        src_dist_keys=src_dist_keys,
        dist_flag_key=dist_flag_key,
        data_location=data_location,
    )


def test_grouped_distribution_data_io(tmp_path, dummy_grouped_distribution_data):
    store_path = tmp_path / "test_data.zarr"
    zgroup = zarr.open_group(str(store_path), mode="w")

    dummy_grouped_distribution_data.write_zarr_group(group=zgroup, chunk_size=10, shard_size=100, max_workers=1)

    read_data = GroupedDistributionData.read_zarr(zgroup["data"])

    # Check src_to_tgt_dist_map
    assert read_data.src_to_tgt_dist_map.keys() == dummy_grouped_distribution_data.src_to_tgt_dist_map.keys()
    for k in read_data.src_to_tgt_dist_map:
        np.testing.assert_array_equal(
            read_data.src_to_tgt_dist_map[k], dummy_grouped_distribution_data.src_to_tgt_dist_map[k]
        )

    # Check src_data
    assert read_data.src_data.keys() == dummy_grouped_distribution_data.src_data.keys()
    for k in read_data.src_data:
        np.testing.assert_array_equal(read_data.src_data[k], dummy_grouped_distribution_data.src_data[k])

    # Check tgt_data
    assert read_data.tgt_data.keys() == dummy_grouped_distribution_data.tgt_data.keys()
    for k in read_data.tgt_data:
        np.testing.assert_array_equal(read_data.tgt_data[k], dummy_grouped_distribution_data.tgt_data[k])

    # Check conditions
    assert read_data.conditions.keys() == dummy_grouped_distribution_data.conditions.keys()
    for k in read_data.conditions:
        assert read_data.conditions[k].keys() == dummy_grouped_distribution_data.conditions[k].keys()
        for sub_k in read_data.conditions[k]:
            np.testing.assert_array_equal(
                read_data.conditions[k][sub_k], dummy_grouped_distribution_data.conditions[k][sub_k]
            )


def test_grouped_distribution_annotation_io(tmp_path, dummy_grouped_distribution_annotation):
    store_path = tmp_path / "test_annotation.zarr"
    zgroup = zarr.open_group(str(store_path), mode="w")

    dummy_grouped_distribution_annotation.write_zarr_group(group=zgroup, chunk_size=10, shard_size=100)

    read_annotation = GroupedDistributionAnnotation.read_zarr(zgroup["annotation"])

    np.testing.assert_array_equal(read_annotation.old_obs_index, dummy_grouped_distribution_annotation.old_obs_index)

    assert (
        read_annotation.src_dist_idx_to_labels.keys()
        == dummy_grouped_distribution_annotation.src_dist_idx_to_labels.keys()
    )
    for k in read_annotation.src_dist_idx_to_labels:
        np.testing.assert_array_equal(
            read_annotation.src_dist_idx_to_labels[k], dummy_grouped_distribution_annotation.src_dist_idx_to_labels[k]
        )

    assert (
        read_annotation.tgt_dist_idx_to_labels.keys()
        == dummy_grouped_distribution_annotation.tgt_dist_idx_to_labels.keys()
    )
    for k in read_annotation.tgt_dist_idx_to_labels:
        np.testing.assert_array_equal(
            read_annotation.tgt_dist_idx_to_labels[k], dummy_grouped_distribution_annotation.tgt_dist_idx_to_labels[k]
        )

    pd.testing.assert_frame_equal(
        read_annotation.src_tgt_dist_df, dummy_grouped_distribution_annotation.src_tgt_dist_df
    )
    assert read_annotation.default_values == dummy_grouped_distribution_annotation.default_values
    assert read_annotation.tgt_dist_keys == dummy_grouped_distribution_annotation.tgt_dist_keys
    assert read_annotation.src_dist_keys == dummy_grouped_distribution_annotation.src_dist_keys
    assert read_annotation.dist_flag_key == dummy_grouped_distribution_annotation.dist_flag_key

    # Check data_location is preserved
    assert read_annotation.data_location is not None
    assert (
        read_annotation.data_location._path
        == dummy_grouped_distribution_annotation.data_location._path
    )


def test_grouped_distribution_io(tmp_path, dummy_grouped_distribution_data, dummy_grouped_distribution_annotation):
    gd = GroupedDistribution(data=dummy_grouped_distribution_data, annotation=dummy_grouped_distribution_annotation)

    store_path = tmp_path / "test_grouped_distribution.zarr"

    gd.write_zarr(path=str(store_path), chunk_size=10, shard_size=100, max_workers=1)

    read_gd = GroupedDistribution.read_zarr(str(store_path))

    # Verify data (reuse logic or just spot check)
    assert read_gd.data.src_to_tgt_dist_map.keys() == gd.data.src_to_tgt_dist_map.keys()

    # Verify annotation
    assert read_gd.annotation.dist_flag_key == gd.annotation.dist_flag_key


class TestConditionsWriteRead:
    """Tests for conditions write/read cycle with various array shapes."""

    def test_conditions_2d_arrays(self, tmp_path):
        """Test conditions with 2D arrays like drug embeddings."""
        # Simulate drug embeddings: each distribution has 3 drugs with 128-dim embeddings
        conditions = {
            0: {"drug_emb": np.random.rand(3, 128).astype(np.float32)},
            1: {"drug_emb": np.random.rand(5, 128).astype(np.float32)},
            2: {"drug_emb": np.random.rand(1, 128).astype(np.float32)},
        }

        data = GroupedDistributionData(
            src_to_tgt_dist_map={0: [0, 1], 1: [2]},
            src_data={0: np.random.rand(10, 5), 1: np.random.rand(8, 5)},
            tgt_data={0: np.random.rand(5, 5), 1: np.random.rand(5, 5), 2: np.random.rand(8, 5)},
            conditions=conditions,
        )

        store_path = tmp_path / "test_2d_conditions.zarr"
        zgroup = zarr.open_group(str(store_path), mode="w")

        data.write_zarr_group(group=zgroup, chunk_size=10, shard_size=100, max_workers=1)
        read_data = GroupedDistributionData.read_zarr(zgroup["data"])

        # Verify conditions structure
        assert read_data.conditions.keys() == conditions.keys()
        for dist_id in conditions:
            assert read_data.conditions[dist_id].keys() == conditions[dist_id].keys()
            for col_name in conditions[dist_id]:
                np.testing.assert_array_almost_equal(
                    read_data.conditions[dist_id][col_name],
                    conditions[dist_id][col_name],
                    err_msg=f"Mismatch at dist {dist_id}, col {col_name}",
                )

    def test_conditions_multiple_2d_arrays_same_shape(self, tmp_path):
        """Test multiple 2D arrays with the same shape per distribution."""
        conditions = {
            0: {
                "drug1_emb": np.random.rand(3, 64).astype(np.float32),
                "drug2_emb": np.random.rand(3, 64).astype(np.float32),
            },
            1: {
                "drug1_emb": np.random.rand(5, 64).astype(np.float32),
                "drug2_emb": np.random.rand(5, 64).astype(np.float32),
            },
        }

        data = GroupedDistributionData(
            src_to_tgt_dist_map={0: [0, 1]},
            src_data={0: np.random.rand(10, 5)},
            tgt_data={0: np.random.rand(5, 5), 1: np.random.rand(5, 5)},
            conditions=conditions,
        )

        store_path = tmp_path / "test_multi_2d.zarr"
        zgroup = zarr.open_group(str(store_path), mode="w")

        data.write_zarr_group(group=zgroup, chunk_size=10, shard_size=100, max_workers=1)
        read_data = GroupedDistributionData.read_zarr(zgroup["data"])

        for dist_id in conditions:
            for col_name in conditions[dist_id]:
                np.testing.assert_array_almost_equal(
                    read_data.conditions[dist_id][col_name],
                    conditions[dist_id][col_name],
                )

    def test_conditions_mixed_1d_arrays(self, tmp_path):
        """Test conditions with multiple 1D arrays of different lengths."""
        conditions = {
            0: {
                "dose": np.array([0.1, 0.5, 1.0]),
                "time": np.array([24, 48]),
                "cell_type": np.array([1, 2, 3, 4, 5]),
            },
        }

        data = GroupedDistributionData(
            src_to_tgt_dist_map={0: [0]},
            src_data={0: np.random.rand(10, 5)},
            tgt_data={0: np.random.rand(5, 5)},
            conditions=conditions,
        )

        store_path = tmp_path / "test_mixed_1d.zarr"
        zgroup = zarr.open_group(str(store_path), mode="w")

        data.write_zarr_group(group=zgroup, chunk_size=10, shard_size=100, max_workers=1)
        read_data = GroupedDistributionData.read_zarr(zgroup["data"])

        assert sorted(read_data.conditions[0].keys()) == sorted(conditions[0].keys())
        for col_name in conditions[0]:
            np.testing.assert_array_almost_equal(
                read_data.conditions[0][col_name],
                conditions[0][col_name],
            )

    def test_conditions_empty(self, tmp_path):
        """Test with empty conditions dict."""
        conditions = {}

        data = GroupedDistributionData(
            src_to_tgt_dist_map={0: [0]},
            src_data={0: np.random.rand(10, 5)},
            tgt_data={0: np.random.rand(5, 5)},
            conditions=conditions,
        )

        store_path = tmp_path / "test_empty.zarr"
        zgroup = zarr.open_group(str(store_path), mode="w")

        data.write_zarr_group(group=zgroup, chunk_size=10, shard_size=100, max_workers=1)
        read_data = GroupedDistributionData.read_zarr(zgroup["data"])

        assert read_data.conditions == {}

    def test_conditions_single_element_arrays(self, tmp_path):
        """Test conditions with single-element arrays."""
        conditions = {
            0: {"scalar_cond": np.array([42.0])},
            1: {"scalar_cond": np.array([99.0])},
        }

        data = GroupedDistributionData(
            src_to_tgt_dist_map={0: [0, 1]},
            src_data={0: np.random.rand(10, 5)},
            tgt_data={0: np.random.rand(5, 5), 1: np.random.rand(5, 5)},
            conditions=conditions,
        )

        store_path = tmp_path / "test_scalar.zarr"
        zgroup = zarr.open_group(str(store_path), mode="w")

        data.write_zarr_group(group=zgroup, chunk_size=10, shard_size=100, max_workers=1)
        read_data = GroupedDistributionData.read_zarr(zgroup["data"])

        for dist_id in conditions:
            np.testing.assert_array_almost_equal(
                read_data.conditions[dist_id]["scalar_cond"],
                conditions[dist_id]["scalar_cond"],
            )

    def test_conditions_many_distributions(self, tmp_path):
        """Test with many distributions to verify indexing is correct."""
        n_dists = 50
        conditions = {i: {"val": np.array([float(i), float(i) * 2])} for i in range(n_dists)}

        data = GroupedDistributionData(
            src_to_tgt_dist_map={0: list(range(n_dists))},
            src_data={0: np.random.rand(100, 5)},
            tgt_data={i: np.random.rand(5, 5) for i in range(n_dists)},
            conditions=conditions,
        )

        store_path = tmp_path / "test_many_dists.zarr"
        zgroup = zarr.open_group(str(store_path), mode="w")

        data.write_zarr_group(group=zgroup, chunk_size=10, shard_size=100, max_workers=1)
        read_data = GroupedDistributionData.read_zarr(zgroup["data"])

        assert len(read_data.conditions) == n_dists
        for i in range(n_dists):
            np.testing.assert_array_almost_equal(
                read_data.conditions[i]["val"],
                np.array([float(i), float(i) * 2]),
            )

    def test_conditions_key_ordering_preserved(self, tmp_path):
        """Test that keys are correctly associated with their arrays after sorting."""
        # Create conditions where key ordering matters for correctness
        conditions = {
            0: {
                "zebra": np.array([1.0, 2.0, 3.0]),  # Will be last after sort
                "apple": np.array([10.0, 20.0]),  # Will be first after sort
                "mango": np.array([100.0]),  # Will be middle after sort
            },
        }

        data = GroupedDistributionData(
            src_to_tgt_dist_map={0: [0]},
            src_data={0: np.random.rand(10, 5)},
            tgt_data={0: np.random.rand(5, 5)},
            conditions=conditions,
        )

        store_path = tmp_path / "test_key_order.zarr"
        zgroup = zarr.open_group(str(store_path), mode="w")

        data.write_zarr_group(group=zgroup, chunk_size=10, shard_size=100, max_workers=1)
        read_data = GroupedDistributionData.read_zarr(zgroup["data"])

        # Verify each key maps to the correct array (not just any array)
        np.testing.assert_array_almost_equal(read_data.conditions[0]["zebra"], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(read_data.conditions[0]["apple"], np.array([10.0, 20.0]))
        np.testing.assert_array_almost_equal(read_data.conditions[0]["mango"], np.array([100.0]))

    def test_conditions_values_preserved(self, tmp_path):
        """Test that array values are preserved through write/read cycle for various dtypes.

        Note: dtype may not be perfectly preserved due to zarr/numpy conversion behavior
        (e.g. float32 may become float64). Values are preserved regardless.
        """
        conditions = {
            0: {
                "float32": np.array([1.0, 2.0]).astype(np.float32),
                "float64": np.array([1.0, 2.0]).astype(np.float64),
                "int32": np.array([1, 2]).astype(np.int32),
                "int64": np.array([1, 2]).astype(np.int64),
            },
        }

        data = GroupedDistributionData(
            src_to_tgt_dist_map={0: [0]},
            src_data={0: np.random.rand(10, 5)},
            tgt_data={0: np.random.rand(5, 5)},
            conditions=conditions,
        )

        store_path = tmp_path / "test_dtypes.zarr"
        zgroup = zarr.open_group(str(store_path), mode="w")

        data.write_zarr_group(group=zgroup, chunk_size=10, shard_size=100, max_workers=1)
        read_data = GroupedDistributionData.read_zarr(zgroup["data"])

        for col_name in conditions[0]:
            np.testing.assert_array_almost_equal(
                read_data.conditions[0][col_name],
                conditions[0][col_name],
                err_msg=f"Value mismatch for {col_name}",
            )

    def test_realistic_drug_scenario(self, tmp_path):
        """Test a realistic drug perturbation scenario.

        Simulates:
        - Multiple target distributions (different perturbation conditions)
        - Each distribution has drug embeddings (2D: n_drugs x embedding_dim)
        - Each distribution has doses (1D: n_drugs)
        - Different distributions can have different numbers of drugs (combination treatments)

        On disk: arrays are concatenated into a long array per distribution
        In memory: nested dict structure {dist_id: {col_name: array}}
        """
        np.random.seed(42)
        embedding_dim = 128

        # Simulate different perturbation scenarios:
        # dist 0: single drug treatment (1 drug)
        # dist 1: double drug treatment (2 drugs)
        # dist 2: triple drug treatment (3 drugs)
        conditions = {
            0: {
                "drug_embedding": np.random.rand(1, embedding_dim).astype(np.float32),
                "dose": np.array([1.0]),
            },
            1: {
                "drug_embedding": np.random.rand(2, embedding_dim).astype(np.float32),
                "dose": np.array([0.5, 1.0]),
            },
            2: {
                "drug_embedding": np.random.rand(3, embedding_dim).astype(np.float32),
                "dose": np.array([0.1, 0.5, 1.0]),
            },
        }

        data = GroupedDistributionData(
            src_to_tgt_dist_map={0: [0, 1, 2]},  # One source maps to all three perturbations
            src_data={0: np.random.rand(100, 50)},  # 100 cells, 50 features
            tgt_data={
                0: np.random.rand(50, 50),  # 50 cells in each target
                1: np.random.rand(50, 50),
                2: np.random.rand(50, 50),
            },
            conditions=conditions,
        )

        store_path = tmp_path / "test_drug_scenario.zarr"
        zgroup = zarr.open_group(str(store_path), mode="w")

        # Write
        data.write_zarr_group(group=zgroup, chunk_size=10, shard_size=100, max_workers=1)

        # Read back
        read_data = GroupedDistributionData.read_zarr(zgroup["data"])

        # Verify structure is nested in memory
        assert isinstance(read_data.conditions, dict)
        assert isinstance(read_data.conditions[0], dict)

        # Verify each distribution
        for dist_id in conditions:
            assert dist_id in read_data.conditions
            assert "drug_embedding" in read_data.conditions[dist_id]
            assert "dose" in read_data.conditions[dist_id]

            # Check shapes are preserved
            orig_emb = conditions[dist_id]["drug_embedding"]
            read_emb = read_data.conditions[dist_id]["drug_embedding"]
            assert read_emb.shape == orig_emb.shape, f"Shape mismatch at dist {dist_id}"

            # Check values are preserved
            np.testing.assert_array_almost_equal(
                read_emb, orig_emb, err_msg=f"Drug embedding mismatch at dist {dist_id}"
            )
            np.testing.assert_array_almost_equal(
                read_data.conditions[dist_id]["dose"],
                conditions[dist_id]["dose"],
                err_msg=f"Dose mismatch at dist {dist_id}",
            )

        # Verify on-disk structure: CSR-like format
        data_group = zgroup["data"]
        assert "conditions" in data_group
        cond_group = data_group["conditions"]

        # Check metadata exists
        assert "dist_ids" in cond_group.attrs
        assert list(cond_group.attrs["dist_ids"]) == sorted(conditions.keys())

        # Each column should be stored as one contiguous array
        col_names = list(conditions[0].keys())
        for col_name in col_names:
            assert col_name in cond_group, f"Column {col_name} not in conditions group"
            assert f"indptr_{col_name}" in cond_group.attrs, f"indptr_{col_name} not in attrs"

            # Verify the contiguous array has the right total length
            expected_total = sum(conditions[dist_id][col_name].shape[0] for dist_id in conditions)
            actual_total = cond_group[col_name].shape[0]
            assert actual_total == expected_total, (
                f"Total length mismatch for {col_name}: {actual_total} vs {expected_total}"
            )

            # Verify indptr has correct structure
            indptr = cond_group.attrs[f"indptr_{col_name}"]
            assert indptr[0] == 0, "indptr should start with 0"
            assert indptr[-1] == expected_total, "indptr should end with total length"


class TestInMemoryAndToMemory:
    """Tests for in_memory parameter and to_memory method."""

    def test_read_zarr_lazy_by_default(
        self, tmp_path, dummy_grouped_distribution_data, dummy_grouped_distribution_annotation
    ):
        """Test that read_zarr returns lazy zarr arrays by default."""
        gd = GroupedDistribution(
            data=dummy_grouped_distribution_data,
            annotation=dummy_grouped_distribution_annotation,
        )
        store_path = tmp_path / "test_lazy.zarr"
        gd.write_zarr(path=str(store_path), chunk_size=10, shard_size=100, max_workers=1)

        # Read without in_memory flag
        read_gd = GroupedDistribution.read_zarr(str(store_path))

        # Data should be lazy (zarr arrays)
        assert read_gd.data.is_in_memory is False

    def test_read_zarr_in_memory(
        self, tmp_path, dummy_grouped_distribution_data, dummy_grouped_distribution_annotation
    ):
        """Test that read_zarr with in_memory=True loads data into memory."""
        gd = GroupedDistribution(
            data=dummy_grouped_distribution_data,
            annotation=dummy_grouped_distribution_annotation,
        )
        store_path = tmp_path / "test_in_memory.zarr"
        gd.write_zarr(path=str(store_path), chunk_size=10, shard_size=100, max_workers=1)

        # Read with in_memory=True
        read_gd = GroupedDistribution.read_zarr(str(store_path), in_memory=True)

        # Data should be in memory (numpy arrays)
        assert read_gd.data.is_in_memory is True

        # Verify data is correct
        for k in dummy_grouped_distribution_data.src_data:
            np.testing.assert_array_equal(
                read_gd.data.src_data[k],
                dummy_grouped_distribution_data.src_data[k],
            )

    def test_to_memory_converts_lazy_to_numpy(
        self, tmp_path, dummy_grouped_distribution_data, dummy_grouped_distribution_annotation
    ):
        """Test that to_memory converts lazy zarr arrays to numpy arrays."""
        gd = GroupedDistribution(
            data=dummy_grouped_distribution_data,
            annotation=dummy_grouped_distribution_annotation,
        )
        store_path = tmp_path / "test_to_memory.zarr"
        gd.write_zarr(path=str(store_path), chunk_size=10, shard_size=100, max_workers=1)

        # Read lazy
        read_gd = GroupedDistribution.read_zarr(str(store_path))
        assert read_gd.data.is_in_memory is False

        # Convert to memory
        read_gd.to_memory()
        assert read_gd.data.is_in_memory is True

        # Verify data is correct
        for k in dummy_grouped_distribution_data.src_data:
            np.testing.assert_array_equal(
                read_gd.data.src_data[k],
                dummy_grouped_distribution_data.src_data[k],
            )

    def test_to_memory_idempotent(
        self, tmp_path, dummy_grouped_distribution_data, dummy_grouped_distribution_annotation
    ):
        """Test that calling to_memory multiple times is safe."""
        gd = GroupedDistribution(
            data=dummy_grouped_distribution_data,
            annotation=dummy_grouped_distribution_annotation,
        )
        store_path = tmp_path / "test_idempotent.zarr"
        gd.write_zarr(path=str(store_path), chunk_size=10, shard_size=100, max_workers=1)

        read_gd = GroupedDistribution.read_zarr(str(store_path))
        read_gd.to_memory()
        assert read_gd.data.is_in_memory is True

        # Call again - should not raise
        read_gd.to_memory()
        assert read_gd.data.is_in_memory is True

    def test_is_in_memory_for_datamanager_created_data(self, sample_grouped_distribution):
        """Test that data created by DataManager is already in memory."""
        # Data from DataManager fixture should be in memory (numpy arrays)
        assert sample_grouped_distribution.data.is_in_memory is True

    def test_grouped_distribution_data_is_in_memory_property(self):
        """Test is_in_memory property for manually created data."""
        data = GroupedDistributionData(
            src_to_tgt_dist_map={0: [0, 1]},
            src_data={0: np.random.rand(10, 5)},
            tgt_data={0: np.random.rand(5, 5), 1: np.random.rand(5, 5)},
            conditions={0: {"cond": np.array([1, 2])}, 1: {"cond": np.array([3, 4])}},
        )
        # Numpy arrays should be in memory
        assert data.is_in_memory is True


class TestLazyVsInMemorySplitting:
    """Tests comparing lazy zarr data vs in-memory data when splitting."""

    def test_split_lazy_data(self, tmp_path, dummy_grouped_distribution_data, dummy_grouped_distribution_annotation):
        """Test splitting lazy zarr data."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter

        # Write to zarr
        gd = GroupedDistribution(
            data=dummy_grouped_distribution_data,
            annotation=dummy_grouped_distribution_annotation,
        )
        store_path = tmp_path / "test_lazy_split.zarr"
        gd.write_zarr(path=str(store_path), chunk_size=10, shard_size=100, max_workers=1)

        # Read as lazy
        lazy_gd = GroupedDistribution.read_zarr(str(store_path), in_memory=False)
        assert lazy_gd.data.is_in_memory is False

        # Split the lazy data
        splitter = GroupedDistributionSplitter(
            gd=lazy_gd,
            holdout_combinations=False,
            split_by=["other_col"],
            split_key="split",
            force_training_values={},
            ratios=[0.5, 0.25, 0.25],
            random_state=42,
        )

        splits = splitter.split()

        # Splits should still reference zarr arrays (lazy)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

    def test_split_in_memory_data(
        self, tmp_path, dummy_grouped_distribution_data, dummy_grouped_distribution_annotation
    ):
        """Test splitting in-memory data."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter

        # Write to zarr
        gd = GroupedDistribution(
            data=dummy_grouped_distribution_data,
            annotation=dummy_grouped_distribution_annotation,
        )
        store_path = tmp_path / "test_inmem_split.zarr"
        gd.write_zarr(path=str(store_path), chunk_size=10, shard_size=100, max_workers=1)

        # Read as in-memory
        inmem_gd = GroupedDistribution.read_zarr(str(store_path), in_memory=True)
        assert inmem_gd.data.is_in_memory is True

        # Split the in-memory data
        splitter = GroupedDistributionSplitter(
            gd=inmem_gd,
            holdout_combinations=False,
            split_by=["other_col"],
            split_key="split",
            force_training_values={},
            ratios=[0.5, 0.25, 0.25],
            random_state=42,
        )

        splits = splitter.split()

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

    def test_lazy_and_inmemory_splits_produce_same_structure(
        self, tmp_path, dummy_grouped_distribution_data, dummy_grouped_distribution_annotation
    ):
        """Test that lazy and in-memory splits produce equivalent structure."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter

        # Write to zarr
        gd = GroupedDistribution(
            data=dummy_grouped_distribution_data,
            annotation=dummy_grouped_distribution_annotation,
        )
        store_path = tmp_path / "test_compare_split.zarr"
        gd.write_zarr(path=str(store_path), chunk_size=10, shard_size=100, max_workers=1)

        # Read as lazy
        lazy_gd = GroupedDistribution.read_zarr(str(store_path), in_memory=False)

        # Read as in-memory
        inmem_gd = GroupedDistribution.read_zarr(str(store_path), in_memory=True)

        # Split both with same seed
        lazy_splitter = GroupedDistributionSplitter(
            gd=lazy_gd,
            holdout_combinations=False,
            split_by=["other_col"],
            split_key="split",
            force_training_values={},
            ratios=[0.5, 0.25, 0.25],
            random_state=42,
        )

        inmem_splitter = GroupedDistributionSplitter(
            gd=inmem_gd,
            holdout_combinations=False,
            split_by=["other_col"],
            split_key="split",
            force_training_values={},
            ratios=[0.5, 0.25, 0.25],
            random_state=42,
        )

        lazy_splits = lazy_splitter.split()
        inmem_splits = inmem_splitter.split()

        # Compare split keys
        assert lazy_splits.keys() == inmem_splits.keys()

        # Compare annotation structures (should be identical)
        for split_name in lazy_splits:
            lazy_split = lazy_splits[split_name]
            inmem_split = inmem_splits[split_name]

            # Same number of distributions
            assert len(lazy_split.data.src_data) == len(inmem_split.data.src_data)
            assert len(lazy_split.data.tgt_data) == len(inmem_split.data.tgt_data)
            assert len(lazy_split.data.conditions) == len(inmem_split.data.conditions)

            # Same keys
            assert set(lazy_split.data.src_data.keys()) == set(inmem_split.data.src_data.keys())
            assert set(lazy_split.data.tgt_data.keys()) == set(inmem_split.data.tgt_data.keys())

    def test_sampler_works_with_lazy_split_after_to_memory(
        self, tmp_path, dummy_grouped_distribution_data, dummy_grouped_distribution_annotation
    ):
        """Test that InMemorySampler works with lazy split data after calling to_memory."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter
        from scaleflow.data._dataloader import InMemorySampler

        # Write to zarr
        gd = GroupedDistribution(
            data=dummy_grouped_distribution_data,
            annotation=dummy_grouped_distribution_annotation,
        )
        store_path = tmp_path / "test_sampler_lazy.zarr"
        gd.write_zarr(path=str(store_path), chunk_size=10, shard_size=100, max_workers=1)

        # Read as lazy
        lazy_gd = GroupedDistribution.read_zarr(str(store_path), in_memory=False)
        assert lazy_gd.data.is_in_memory is False

        # Split
        splitter = GroupedDistributionSplitter(
            gd=lazy_gd,
            holdout_combinations=False,
            split_by=["other_col"],
            split_key="split",
            force_training_values={},
            ratios=[0.5, 0.25, 0.25],
            random_state=42,
        )

        splits = splitter.split()
        train_data = splits["train"]

        # Create sampler (will call to_memory internally)
        rng = np.random.default_rng(42)
        sampler = InMemorySampler(data=train_data, rng=rng, batch_size=16)
        sampler.init_sampler()

        # Data should now be in memory
        assert train_data.data.is_in_memory is True

        # Sample should work
        result = sampler.sample()
        assert result["src_cell_data"].shape[0] == 16
        assert "condition" in result

    def test_sampler_works_with_inmemory_split(
        self, tmp_path, dummy_grouped_distribution_data, dummy_grouped_distribution_annotation
    ):
        """Test that InMemorySampler works directly with in-memory split data."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter
        from scaleflow.data._dataloader import InMemorySampler

        # Write to zarr
        gd = GroupedDistribution(
            data=dummy_grouped_distribution_data,
            annotation=dummy_grouped_distribution_annotation,
        )
        store_path = tmp_path / "test_sampler_inmem.zarr"
        gd.write_zarr(path=str(store_path), chunk_size=10, shard_size=100, max_workers=1)

        # Read as in-memory
        inmem_gd = GroupedDistribution.read_zarr(str(store_path), in_memory=True)
        assert inmem_gd.data.is_in_memory is True

        # Split
        splitter = GroupedDistributionSplitter(
            gd=inmem_gd,
            holdout_combinations=False,
            split_by=["other_col"],
            split_key="split",
            force_training_values={},
            ratios=[0.5, 0.25, 0.25],
            random_state=42,
        )

        splits = splitter.split()
        train_data = splits["train"]

        # Create sampler
        rng = np.random.default_rng(42)
        sampler = InMemorySampler(data=train_data, rng=rng, batch_size=16)
        sampler.init_sampler()

        # Sample should work
        result = sampler.sample()
        assert result["src_cell_data"].shape[0] == 16
        assert "condition" in result


class TestAnnDataLocationSerialization:
    """Tests for AnnDataLocation serialization (to_path/from_path)."""

    def test_to_path_empty(self):
        """Test to_path for empty AnnDataLocation."""
        adl = AnnDataLocation()
        assert adl.to_path() == []

    def test_to_path_simple_attr(self):
        """Test to_path for simple attribute access."""
        adl = AnnDataLocation().obsm
        assert adl.to_path() == [["getattr", "obsm"]]

    def test_to_path_attr_and_item(self):
        """Test to_path for attribute and item access."""
        adl = AnnDataLocation().obsm["X_pca"]
        expected = [["getattr", "obsm"], ["getitem", "X_pca"]]
        assert adl.to_path() == expected

    def test_to_path_complex(self):
        """Test to_path for complex nested access."""
        adl = AnnDataLocation().layers["counts"]
        expected = [["getattr", "layers"], ["getitem", "counts"]]
        assert adl.to_path() == expected

    def test_from_path_empty(self):
        """Test from_path for empty path."""
        adl = AnnDataLocation.from_path([])
        assert adl._path == []

    def test_from_path_simple(self):
        """Test from_path for simple path."""
        adl = AnnDataLocation.from_path([["getattr", "X"]])
        assert adl._path == [("getattr", "X")]

    def test_from_path_complex(self):
        """Test from_path for complex path."""
        path = [["getattr", "obsm"], ["getitem", "X_pca"]]
        adl = AnnDataLocation.from_path(path)
        expected_path = [("getattr", "obsm"), ("getitem", "X_pca")]
        assert adl._path == expected_path

    def test_roundtrip(self):
        """Test roundtrip: to_path followed by from_path."""
        original = AnnDataLocation().obsm["X_pca"]
        serialized = original.to_path()
        restored = AnnDataLocation.from_path(serialized)
        assert restored._path == original._path

    def test_roundtrip_various_paths(self):
        """Test roundtrip for various path configurations."""
        test_cases = [
            AnnDataLocation(),
            AnnDataLocation().X,
            AnnDataLocation().obs["cell_type"],
            AnnDataLocation().obsm["X_pca"],
            AnnDataLocation().layers["counts"],
            AnnDataLocation().uns["embeddings"]["drug"],
        ]
        for original in test_cases:
            serialized = original.to_path()
            restored = AnnDataLocation.from_path(serialized)
            assert restored._path == original._path, f"Roundtrip failed for {original}"

    def test_to_json(self):
        """Test to_json serialization."""
        adl = AnnDataLocation().obsm["X_pca"]
        json_str = adl.to_json()
        assert json_str == '[["getattr", "obsm"], ["getitem", "X_pca"]]'

    def test_from_json(self):
        """Test from_json deserialization."""
        json_str = '[["getattr", "obsm"], ["getitem", "X_pca"]]'
        adl = AnnDataLocation.from_json(json_str)
        expected_path = [("getattr", "obsm"), ("getitem", "X_pca")]
        assert adl._path == expected_path

    def test_json_roundtrip(self):
        """Test roundtrip through JSON serialization."""
        test_cases = [
            AnnDataLocation(),
            AnnDataLocation().X,
            AnnDataLocation().obs["cell_type"],
            AnnDataLocation().obsm["X_pca"],
            AnnDataLocation().layers["counts"],
            AnnDataLocation().uns["embeddings"]["drug"],
        ]
        for original in test_cases:
            json_str = original.to_json()
            restored = AnnDataLocation.from_json(json_str)
            assert restored._path == original._path, f"JSON roundtrip failed for {original}"


class TestAnnotationDataLocation:
    """Tests for data_location field in GroupedDistributionAnnotation."""

    @pytest.fixture
    def base_annotation_kwargs(self):
        """Base kwargs for creating annotation without data_location."""
        return {
            "old_obs_index": np.arange(20),
            "src_dist_idx_to_labels": {0: ["label1", "label2"], 1: ["label3"]},
            "tgt_dist_idx_to_labels": {0: ["tlabel1"], 1: ["tlabel2"], 2: ["tlabel3"]},
            "src_tgt_dist_df": pd.DataFrame({
                "src_dist_idx": [0, 0, 1],
                "tgt_dist_idx": [0, 1, 2],
                "other_col": ["a", "b", "c"],
            }),
            "default_values": {"param1": 1, "param2": "val"},
            "tgt_dist_keys": ["key1", "key2"],
            "src_dist_keys": ["skey1"],
            "dist_flag_key": "flag",
        }

    def test_annotation_with_none_data_location(self, tmp_path, base_annotation_kwargs):
        """Test annotation IO with data_location=None."""
        annotation = GroupedDistributionAnnotation(**base_annotation_kwargs, data_location=None)

        store_path = tmp_path / "test_none_location.zarr"
        zgroup = zarr.open_group(str(store_path), mode="w")
        annotation.write_zarr_group(group=zgroup, chunk_size=10, shard_size=100)

        read_annotation = GroupedDistributionAnnotation.read_zarr(zgroup["annotation"])
        assert read_annotation.data_location is None

    def test_annotation_with_obsm_data_location(self, tmp_path, base_annotation_kwargs):
        """Test annotation IO with obsm data_location."""
        data_location = AnnDataLocation().obsm["X_pca"]
        annotation = GroupedDistributionAnnotation(
            **base_annotation_kwargs,
            data_location=data_location,
        )

        store_path = tmp_path / "test_obsm_location.zarr"
        zgroup = zarr.open_group(str(store_path), mode="w")
        annotation.write_zarr_group(group=zgroup, chunk_size=10, shard_size=100)

        read_annotation = GroupedDistributionAnnotation.read_zarr(zgroup["annotation"])
        assert read_annotation.data_location is not None
        assert read_annotation.data_location._path == data_location._path

    def test_annotation_with_X_data_location(self, tmp_path, base_annotation_kwargs):
        """Test annotation IO with X (main matrix) data_location."""
        data_location = AnnDataLocation().X
        annotation = GroupedDistributionAnnotation(
            **base_annotation_kwargs,
            data_location=data_location,
        )

        store_path = tmp_path / "test_X_location.zarr"
        zgroup = zarr.open_group(str(store_path), mode="w")
        annotation.write_zarr_group(group=zgroup, chunk_size=10, shard_size=100)

        read_annotation = GroupedDistributionAnnotation.read_zarr(zgroup["annotation"])
        assert read_annotation.data_location is not None
        assert read_annotation.data_location._path == data_location._path

    def test_annotation_with_layers_data_location(self, tmp_path, base_annotation_kwargs):
        """Test annotation IO with layers data_location."""
        data_location = AnnDataLocation().layers["counts"]
        annotation = GroupedDistributionAnnotation(
            **base_annotation_kwargs,
            data_location=data_location,
        )

        store_path = tmp_path / "test_layers_location.zarr"
        zgroup = zarr.open_group(str(store_path), mode="w")
        annotation.write_zarr_group(group=zgroup, chunk_size=10, shard_size=100)

        read_annotation = GroupedDistributionAnnotation.read_zarr(zgroup["annotation"])
        assert read_annotation.data_location is not None
        assert read_annotation.data_location._path == data_location._path

    def test_grouped_distribution_preserves_data_location(
        self,
        tmp_path,
        dummy_grouped_distribution_data,
        base_annotation_kwargs,
    ):
        """Test that GroupedDistribution preserves data_location through write/read."""
        data_location = AnnDataLocation().obsm["X_scVI"]
        annotation = GroupedDistributionAnnotation(
            **base_annotation_kwargs,
            data_location=data_location,
        )

        gd = GroupedDistribution(
            data=dummy_grouped_distribution_data,
            annotation=annotation,
        )

        store_path = tmp_path / "test_gd_location.zarr"
        gd.write_zarr(path=str(store_path), chunk_size=10, shard_size=100, max_workers=1)

        read_gd = GroupedDistribution.read_zarr(str(store_path))
        assert read_gd.annotation.data_location is not None
        assert read_gd.annotation.data_location._path == data_location._path
