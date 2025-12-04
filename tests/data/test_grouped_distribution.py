import numpy as np
import pandas as pd
import pytest
import zarr

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

    return GroupedDistributionAnnotation(
        old_obs_index=old_obs_index,
        src_dist_idx_to_labels=src_dist_idx_to_labels,
        tgt_dist_idx_to_labels=tgt_dist_idx_to_labels,
        src_tgt_dist_df=src_tgt_dist_df,
        default_values=default_values,
        tgt_dist_keys=tgt_dist_keys,
        src_dist_keys=src_dist_keys,
        dist_flag_key=dist_flag_key,
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
