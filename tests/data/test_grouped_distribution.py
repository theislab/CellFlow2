
import pytest
import numpy as np
import pandas as pd
import zarr
import anndata as ad
from scaleflow.data._data import (
    GroupedDistribution,
    GroupedDistributionData,
    GroupedDistributionAnnotation,
)

@pytest.fixture
def dummy_grouped_distribution_data():
    src_to_tgt_dist_map = {0: [0, 1], 1: [2]}
    src_data = {
        0: np.random.rand(10, 5),
        1: np.random.rand(8, 5)
    }
    tgt_data = {
        0: np.random.rand(5, 5),
        1: np.random.rand(5, 5),
        2: np.random.rand(8, 5)
    }
    conditions = {
        0: {"cond1": np.array([1, 2, 3]), "cond2": np.array([4, 5])},
        1: {"cond1": np.array([6, 7]), "cond2": np.array([8, 9])},
        2: {"cond1": np.array([10]), "cond2": np.array([11])}
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
    src_tgt_dist_df = pd.DataFrame({
        "src_dist_idx": [0, 0, 1],
        "tgt_dist_idx": [0, 1, 2],
        "other_col": ["a", "b", "c"]
    })
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
    
    dummy_grouped_distribution_data.write_zarr_group(
        group=zgroup,
        chunk_size=10,
        shard_size=100,
        max_workers=1
    )
    
    read_data = GroupedDistributionData.read_zarr(zgroup["data"])
    
    # Check src_to_tgt_dist_map
    assert read_data.src_to_tgt_dist_map.keys() == dummy_grouped_distribution_data.src_to_tgt_dist_map.keys()
    for k in read_data.src_to_tgt_dist_map:
        np.testing.assert_array_equal(
            read_data.src_to_tgt_dist_map[k], 
            dummy_grouped_distribution_data.src_to_tgt_dist_map[k]
        )
        
    # Check src_data
    assert read_data.src_data.keys() == dummy_grouped_distribution_data.src_data.keys()
    for k in read_data.src_data:
        np.testing.assert_array_equal(
            read_data.src_data[k],
            dummy_grouped_distribution_data.src_data[k]
        )
        
    # Check tgt_data
    assert read_data.tgt_data.keys() == dummy_grouped_distribution_data.tgt_data.keys()
    for k in read_data.tgt_data:
        np.testing.assert_array_equal(
            read_data.tgt_data[k],
            dummy_grouped_distribution_data.tgt_data[k]
        )
        
    # Check conditions
    assert read_data.conditions.keys() == dummy_grouped_distribution_data.conditions.keys()
    for k in read_data.conditions:
        assert read_data.conditions[k].keys() == dummy_grouped_distribution_data.conditions[k].keys()
        for sub_k in read_data.conditions[k]:
            np.testing.assert_array_equal(
                read_data.conditions[k][sub_k],
                dummy_grouped_distribution_data.conditions[k][sub_k]
            )

def test_grouped_distribution_annotation_io(tmp_path, dummy_grouped_distribution_annotation):
    store_path = tmp_path / "test_annotation.zarr"
    zgroup = zarr.open_group(str(store_path), mode="w")
    
    dummy_grouped_distribution_annotation.write_zarr_group(
        group=zgroup,
        chunk_size=10,
        shard_size=100
    )
    
    read_annotation = GroupedDistributionAnnotation.read_zarr(zgroup["annotation"])
    
    np.testing.assert_array_equal(read_annotation.old_obs_index, dummy_grouped_distribution_annotation.old_obs_index)
    
    assert read_annotation.src_dist_idx_to_labels.keys() == dummy_grouped_distribution_annotation.src_dist_idx_to_labels.keys()
    for k in read_annotation.src_dist_idx_to_labels:
        np.testing.assert_array_equal(
            read_annotation.src_dist_idx_to_labels[k],
            dummy_grouped_distribution_annotation.src_dist_idx_to_labels[k]
        )

    assert read_annotation.tgt_dist_idx_to_labels.keys() == dummy_grouped_distribution_annotation.tgt_dist_idx_to_labels.keys()
    for k in read_annotation.tgt_dist_idx_to_labels:
        np.testing.assert_array_equal(
            read_annotation.tgt_dist_idx_to_labels[k],
            dummy_grouped_distribution_annotation.tgt_dist_idx_to_labels[k]
        )
        
    pd.testing.assert_frame_equal(read_annotation.src_tgt_dist_df, dummy_grouped_distribution_annotation.src_tgt_dist_df)
    assert read_annotation.default_values == dummy_grouped_distribution_annotation.default_values
    assert read_annotation.tgt_dist_keys == dummy_grouped_distribution_annotation.tgt_dist_keys
    assert read_annotation.src_dist_keys == dummy_grouped_distribution_annotation.src_dist_keys
    assert read_annotation.dist_flag_key == dummy_grouped_distribution_annotation.dist_flag_key


def test_grouped_distribution_io(tmp_path, dummy_grouped_distribution_data, dummy_grouped_distribution_annotation):
    gd = GroupedDistribution(
        data=dummy_grouped_distribution_data,
        annotation=dummy_grouped_distribution_annotation
    )
    
    store_path = tmp_path / "test_grouped_distribution.zarr"
    
    gd.write_zarr(
        path=str(store_path),
        chunk_size=10,
        shard_size=100,
        max_workers=1
    )
    
    read_gd = GroupedDistribution.read_zarr(str(store_path))
    
    # Verify data (reuse logic or just spot check)
    assert read_gd.data.src_to_tgt_dist_map.keys() == gd.data.src_to_tgt_dist_map.keys()
    
    # Verify annotation
    assert read_gd.annotation.dist_flag_key == gd.annotation.dist_flag_key
