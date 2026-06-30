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
from scaleflow.data._utils import write_nested_dist_data

# Number of obs rows in the dummy fixtures below. The per-row dist-id arrays and the
# ``old_obs_index`` must all have this length (``to_adata(None)`` builds an AnnData with this
# many rows and uses ``old_obs_index`` as the obs index).
_N_OBS = 20


def _make_grouped_distribution(
    conditions,
    *,
    n_obs=_N_OBS,
    src_to_tgt_dist_map=None,
    src_dist_idx_to_labels=None,
    tgt_dist_idx_to_labels=None,
    src_tgt_dist_df=None,
    default_values=None,
    tgt_dist_keys=None,
    src_dist_keys=None,
    dist_flag_key="flag",
    data_location=None,
):
    """Build a small, internally consistent :class:`GroupedDistribution`.

    The per-row arrays (``row_tgt_dist_idx`` / ``row_src_dist_idx``) and ``old_obs_index`` all have
    length ``n_obs`` so the object can round-trip through ``to_adata(None)`` / ``write_zarr``.
    Target dist ids are taken from ``conditions`` (falling back to ``[0]`` when empty); the first
    half of the rows are controls assigned to source dist 0, the second half are assigned to the
    target dists in round-robin fashion.
    """
    tgt_ids = sorted(int(k) for k in conditions) if conditions else [0]

    row_tgt = np.full(n_obs, -1, dtype=np.int64)
    row_src = np.full(n_obs, -1, dtype=np.int64)
    half = n_obs // 2
    for i in range(half):
        row_src[i] = 0
    for i in range(half, n_obs):
        row_tgt[i] = tgt_ids[(i - half) % len(tgt_ids)]

    data = GroupedDistributionData(
        src_to_tgt_dist_map=src_to_tgt_dist_map or {0: tgt_ids},
        row_tgt_dist_idx=row_tgt,
        row_src_dist_idx=row_src,
        conditions=conditions,
    )
    annotation = GroupedDistributionAnnotation(
        old_obs_index=np.array([f"cell_{i}" for i in range(n_obs)]),
        src_dist_idx_to_labels=src_dist_idx_to_labels or {0: ["s_label0"]},
        tgt_dist_idx_to_labels=tgt_dist_idx_to_labels or {t: [f"t_label{t}"] for t in tgt_ids},
        src_tgt_dist_df=src_tgt_dist_df
        if src_tgt_dist_df is not None
        else pd.DataFrame({"src_dist_idx": [0], "tgt_dist_idx": [tgt_ids[0]]}),
        default_values=default_values or {"param1": 1, "param2": "val"},
        tgt_dist_keys=tgt_dist_keys or ["key1", "key2"],
        src_dist_keys=src_dist_keys or ["skey1"],
        dist_flag_key=dist_flag_key,
        data_location=data_location,
    )
    return GroupedDistribution(data=data, annotation=annotation)


def _assert_conditions_equal(read_conditions, expected_conditions, *, almost=False):
    """Assert two ``{dist_id: {col: array}}`` condition dicts are equal.

    ``from_adata`` returns int dist-id keys, so the comparison is keyed by ``int(dist_id)``.
    """
    assert {int(k) for k in read_conditions} == {int(k) for k in expected_conditions}
    cmp = np.testing.assert_array_almost_equal if almost else np.testing.assert_array_equal
    for dist_id, cols in expected_conditions.items():
        read_cols = read_conditions[int(dist_id)]
        assert set(read_cols.keys()) == set(cols.keys())
        for col_name, arr in cols.items():
            cmp(read_cols[col_name], arr, err_msg=f"Mismatch at dist {dist_id!r}, col {col_name!r}")


@pytest.fixture
def dummy_grouped_distribution_data():
    """A :class:`GroupedDistributionData` built with the new per-row dist-id fields."""
    src_to_tgt_dist_map = {0: [0, 1], 1: [2]}
    # 20 obs rows: rows 0..9 are controls (src 0 then src 1), rows 10..19 are targets.
    row_src = np.full(_N_OBS, -1, dtype=np.int64)
    row_src[0:5] = 0
    row_src[5:10] = 1
    row_tgt = np.full(_N_OBS, -1, dtype=np.int64)
    row_tgt[10:14] = 0
    row_tgt[14:17] = 1
    row_tgt[17:20] = 2
    conditions = {
        0: {"cond1": np.array([1, 2, 3]), "cond2": np.array([4, 5])},
        1: {"cond1": np.array([6, 7]), "cond2": np.array([8, 9])},
        2: {"cond1": np.array([10]), "cond2": np.array([11])},
    }
    return GroupedDistributionData(
        src_to_tgt_dist_map=src_to_tgt_dist_map,
        row_tgt_dist_idx=row_tgt,
        row_src_dist_idx=row_src,
        conditions=conditions,
    )


@pytest.fixture
def dummy_grouped_distribution_annotation():
    """A :class:`GroupedDistributionAnnotation` matching :func:`dummy_grouped_distribution_data`.

    ``old_obs_index`` has length ``_N_OBS`` so the pair round-trips through ``to_adata(None)``.
    """
    old_obs_index = np.array([f"cell_{i}" for i in range(_N_OBS)])
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


def test_grouped_distribution_data_io(
    dummy_grouped_distribution_data, dummy_grouped_distribution_annotation
):
    """The data half of a GroupedDistribution survives an AnnData round-trip.

    Serialization now lives on :class:`GroupedDistribution` (via ``to_adata`` / ``from_adata``),
    so the per-row dist arrays, ``src_to_tgt_dist_map`` and conditions are exercised through it.
    """
    gd = GroupedDistribution(
        data=dummy_grouped_distribution_data, annotation=dummy_grouped_distribution_annotation
    )
    read_gd = GroupedDistribution.from_adata(gd.to_adata(None))
    read_data = read_gd.data

    # src_to_tgt_dist_map
    assert read_data.src_to_tgt_dist_map.keys() == dummy_grouped_distribution_data.src_to_tgt_dist_map.keys()
    for k in read_data.src_to_tgt_dist_map:
        np.testing.assert_array_equal(
            read_data.src_to_tgt_dist_map[k], dummy_grouped_distribution_data.src_to_tgt_dist_map[k]
        )

    # Per-row dist-id arrays (replace the old src_dist_to_rows / tgt_dist_to_rows maps)
    np.testing.assert_array_equal(
        read_data.row_tgt_dist_idx, dummy_grouped_distribution_data.row_tgt_dist_idx
    )
    np.testing.assert_array_equal(
        read_data.row_src_dist_idx, dummy_grouped_distribution_data.row_src_dist_idx
    )

    # rows_for inverts the per-row column back to the explicit row indices
    assert (
        GroupedDistributionData.rows_for(read_data.row_tgt_dist_idx).keys()
        == GroupedDistributionData.rows_for(dummy_grouped_distribution_data.row_tgt_dist_idx).keys()
    )

    # conditions
    _assert_conditions_equal(read_data.conditions, dummy_grouped_distribution_data.conditions)


def test_grouped_distribution_annotation_io(
    dummy_grouped_distribution_data, dummy_grouped_distribution_annotation
):
    """The annotation half of a GroupedDistribution survives an AnnData round-trip."""
    gd = GroupedDistribution(
        data=dummy_grouped_distribution_data, annotation=dummy_grouped_distribution_annotation
    )
    read_annotation = GroupedDistribution.from_adata(gd.to_adata(None)).annotation
    expected = dummy_grouped_distribution_annotation

    np.testing.assert_array_equal(
        read_annotation.old_obs_index.astype(str), expected.old_obs_index.astype(str)
    )

    # Label maps: from_adata reconstructs the label collections as tuples.
    assert read_annotation.src_dist_idx_to_labels.keys() == expected.src_dist_idx_to_labels.keys()
    for k in read_annotation.src_dist_idx_to_labels:
        assert list(read_annotation.src_dist_idx_to_labels[k]) == list(expected.src_dist_idx_to_labels[k])

    assert read_annotation.tgt_dist_idx_to_labels.keys() == expected.tgt_dist_idx_to_labels.keys()
    for k in read_annotation.tgt_dist_idx_to_labels:
        assert list(read_annotation.tgt_dist_idx_to_labels[k]) == list(expected.tgt_dist_idx_to_labels[k])

    pd.testing.assert_frame_equal(read_annotation.src_tgt_dist_df, expected.src_tgt_dist_df)
    assert read_annotation.default_values == expected.default_values
    assert read_annotation.tgt_dist_keys == expected.tgt_dist_keys
    assert read_annotation.src_dist_keys == expected.src_dist_keys
    assert read_annotation.dist_flag_key == expected.dist_flag_key

    # data_location is preserved
    assert read_annotation.data_location is not None
    assert read_annotation.data_location._path == expected.data_location._path


def test_grouped_distribution_io(tmp_path, dummy_grouped_distribution_data, dummy_grouped_distribution_annotation):
    """Full GroupedDistribution survives a write_zarr / read_zarr round-trip."""
    gd = GroupedDistribution(data=dummy_grouped_distribution_data, annotation=dummy_grouped_distribution_annotation)

    store_path = tmp_path / "test_grouped_distribution.zarr"

    # Legacy chunk_size/shard_size/max_workers kwargs are accepted and ignored.
    gd.write_zarr(path=str(store_path), chunk_size=10, shard_size=100, max_workers=1)

    read_gd = GroupedDistribution.read_zarr(str(store_path))

    # Verify data
    assert read_gd.data.src_to_tgt_dist_map.keys() == gd.data.src_to_tgt_dist_map.keys()
    np.testing.assert_array_equal(read_gd.data.row_tgt_dist_idx, gd.data.row_tgt_dist_idx)
    np.testing.assert_array_equal(read_gd.data.row_src_dist_idx, gd.data.row_src_dist_idx)
    _assert_conditions_equal(read_gd.data.conditions, gd.data.conditions)

    # Verify annotation
    assert read_gd.annotation.dist_flag_key == gd.annotation.dist_flag_key
    assert read_gd.annotation.data_location._path == gd.annotation.data_location._path


class TestConditionsWriteRead:
    """Conditions of various shapes survive an AnnData (write_zarr / read_zarr) round-trip.

    Serialization now delegates to anndata (no bespoke CSR/indptr layout), so these tests only
    assert fidelity of the in-memory ``{dist_id: {col: array}}`` structure after the round-trip.
    """

    def _roundtrip(self, conditions, tmp_path, name, *, src_to_tgt_dist_map=None, tgt_labels=None):
        gd = _make_grouped_distribution(
            conditions,
            src_to_tgt_dist_map=src_to_tgt_dist_map,
            tgt_dist_idx_to_labels=tgt_labels,
        )
        store_path = tmp_path / f"{name}.zarr"
        gd.write_zarr(path=str(store_path))
        return GroupedDistribution.read_zarr(str(store_path))

    def test_conditions_2d_arrays(self, tmp_path):
        """Conditions with 2D arrays like drug embeddings."""
        # Simulate drug embeddings: each distribution has a different number of drugs (128-dim).
        conditions = {
            0: {"drug_emb": np.random.rand(3, 128).astype(np.float32)},
            1: {"drug_emb": np.random.rand(5, 128).astype(np.float32)},
            2: {"drug_emb": np.random.rand(1, 128).astype(np.float32)},
        }
        read_gd = self._roundtrip(
            conditions, tmp_path, "test_2d_conditions", src_to_tgt_dist_map={0: [0, 1], 1: [2]}
        )
        _assert_conditions_equal(read_gd.data.conditions, conditions, almost=True)

    def test_conditions_multiple_2d_arrays_same_shape(self, tmp_path):
        """Multiple 2D arrays with the same shape per distribution."""
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
        read_gd = self._roundtrip(
            conditions, tmp_path, "test_multi_2d", src_to_tgt_dist_map={0: [0, 1]}
        )
        _assert_conditions_equal(read_gd.data.conditions, conditions, almost=True)

    def test_conditions_mixed_1d_arrays(self, tmp_path):
        """Conditions with multiple 1D arrays of different lengths."""
        conditions = {
            0: {
                "dose": np.array([0.1, 0.5, 1.0]),
                "time": np.array([24, 48]),
                "cell_type": np.array([1, 2, 3, 4, 5]),
            },
        }
        read_gd = self._roundtrip(conditions, tmp_path, "test_mixed_1d")
        _assert_conditions_equal(read_gd.data.conditions, conditions, almost=True)

    def test_conditions_empty(self, tmp_path):
        """An empty conditions dict round-trips to an empty dict."""
        read_gd = self._roundtrip({}, tmp_path, "test_empty")
        assert read_gd.data.conditions == {}

    def test_conditions_single_element_arrays(self, tmp_path):
        """Conditions with single-element arrays."""
        conditions = {
            0: {"scalar_cond": np.array([42.0])},
            1: {"scalar_cond": np.array([99.0])},
        }
        read_gd = self._roundtrip(
            conditions, tmp_path, "test_scalar", src_to_tgt_dist_map={0: [0, 1]}
        )
        _assert_conditions_equal(read_gd.data.conditions, conditions, almost=True)

    def test_conditions_many_distributions(self, tmp_path):
        """Many distributions, to verify each maps to the correct array."""
        n_dists = 50
        conditions = {i: {"val": np.array([float(i), float(i) * 2])} for i in range(n_dists)}
        read_gd = self._roundtrip(
            conditions,
            tmp_path,
            "test_many_dists",
            src_to_tgt_dist_map={0: list(range(n_dists))},
        )
        assert len(read_gd.data.conditions) == n_dists
        for i in range(n_dists):
            np.testing.assert_array_almost_equal(
                read_gd.data.conditions[i]["val"], np.array([float(i), float(i) * 2])
            )

    def test_conditions_key_ordering_preserved(self, tmp_path):
        """Each column key maps to the correct array (not just any array)."""
        conditions = {
            0: {
                "zebra": np.array([1.0, 2.0, 3.0]),
                "apple": np.array([10.0, 20.0]),
                "mango": np.array([100.0]),
            },
        }
        read_gd = self._roundtrip(conditions, tmp_path, "test_key_order")
        np.testing.assert_array_almost_equal(read_gd.data.conditions[0]["zebra"], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(read_gd.data.conditions[0]["apple"], np.array([10.0, 20.0]))
        np.testing.assert_array_almost_equal(read_gd.data.conditions[0]["mango"], np.array([100.0]))

    def test_conditions_values_preserved(self, tmp_path):
        """Array values are preserved through the round-trip for various dtypes.

        Note: dtype may not be perfectly preserved by anndata zarr encoding; values are
        preserved regardless, so we compare values (not dtypes).
        """
        conditions = {
            0: {
                "float32": np.array([1.0, 2.0]).astype(np.float32),
                "float64": np.array([1.0, 2.0]).astype(np.float64),
                "int32": np.array([1, 2]).astype(np.int32),
                "int64": np.array([1, 2]).astype(np.int64),
            },
        }
        read_gd = self._roundtrip(conditions, tmp_path, "test_dtypes")
        _assert_conditions_equal(read_gd.data.conditions, conditions, almost=True)

    def test_realistic_drug_scenario(self, tmp_path):
        """A realistic drug perturbation scenario.

        Simulates multiple target distributions (perturbation conditions), each with a 2D drug
        embedding (n_drugs x embedding_dim) and a 1D dose vector (n_drugs), where different
        distributions have different numbers of drugs (combination treatments). The in-memory
        structure is a nested dict ``{dist_id: {col_name: array}}``.
        """
        np.random.seed(42)
        embedding_dim = 128
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
        read_gd = self._roundtrip(
            conditions, tmp_path, "test_drug_scenario", src_to_tgt_dist_map={0: [0, 1, 2]}
        )
        read_conditions = read_gd.data.conditions

        # Verify structure is nested in memory
        assert isinstance(read_conditions, dict)
        assert isinstance(read_conditions[0], dict)

        for dist_id in conditions:
            assert dist_id in read_conditions
            assert "drug_embedding" in read_conditions[dist_id]
            assert "dose" in read_conditions[dist_id]

            # Shapes are preserved (anndata keeps the full ndim, no flattening)
            orig_emb = conditions[dist_id]["drug_embedding"]
            read_emb = read_conditions[dist_id]["drug_embedding"]
            assert read_emb.shape == orig_emb.shape, f"Shape mismatch at dist {dist_id}"

        _assert_conditions_equal(read_conditions, conditions, almost=True)


class TestConditionsStringEdgeCases:
    """Condition column-key encoding edge cases survive an AnnData round-trip.

    These test what happens when condition column names (keys) have weird string values, or when
    dist_id keys are strings instead of ints. The serialization now stores conditions as a nested
    ``dict[str, dict[str, array]]`` in ``adata.uns``; the zarr-key constraints of that nested
    layout still reject forward slashes and empty-string keys (the ``xfail`` cases below).
    """

    def _roundtrip(self, conditions, tmp_path, name, *, src_to_tgt_dist_map=None, tgt_labels=None):
        gd = _make_grouped_distribution(
            conditions,
            src_to_tgt_dist_map=src_to_tgt_dist_map,
            tgt_dist_idx_to_labels=tgt_labels,
        )
        store_path = tmp_path / f"{name}.zarr"
        gd.write_zarr(path=str(store_path))
        return GroupedDistribution.read_zarr(str(store_path))

    def test_conditions_with_unicode_column_keys(self, tmp_path):
        """Column keys containing Unicode (Greek) characters."""
        conditions = {
            0: {
                "α-blocker": np.array([1.0, 2.0, 3.0]),
                "β-agonist": np.array([4.0, 5.0, 6.0]),
            },
            1: {
                "α-blocker": np.array([7.0]),
                "β-agonist": np.array([8.0]),
            },
        }
        read_gd = self._roundtrip(
            conditions, tmp_path, "test_unicode_keys", src_to_tgt_dist_map={0: [0, 1]}
        )
        _assert_conditions_equal(read_gd.data.conditions, conditions)

    @pytest.mark.xfail(reason="Slashes in column keys cause zarr group-key issues", strict=True)
    def test_conditions_with_special_char_column_keys(self, tmp_path):
        """Column keys containing special characters (incl. forward slashes)."""
        conditions = {
            0: {
                "drug+combo/v1": np.array([1.0, 2.0]),
                "gene:subtype=1": np.array([3.0, 4.0]),
                "name,with,commas": np.array([5.0, 6.0]),
            },
        }
        read_gd = self._roundtrip(conditions, tmp_path, "test_special_char_keys")
        _assert_conditions_equal(read_gd.data.conditions, conditions)

    def test_conditions_with_string_dist_id_keys(self, tmp_path):
        """dist_id keys that are strings instead of ints (e.g. '0' instead of 0).

        ``from_adata`` canonicalizes dist ids back to ``int``.
        """
        conditions = {
            "0": {"drug_name": np.array([1.0, 2.0])},
            "1": {"drug_name": np.array([3.0, 4.0])},
        }
        read_gd = self._roundtrip(
            conditions, tmp_path, "test_string_dist_ids", src_to_tgt_dist_map={0: [0, 1]}
        )
        # dist ids come back as ints
        assert set(read_gd.data.conditions.keys()) == {0, 1}
        np.testing.assert_array_equal(read_gd.data.conditions[0]["drug_name"], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(read_gd.data.conditions[1]["drug_name"], np.array([3.0, 4.0]))

    def test_conditions_with_cjk_column_keys(self, tmp_path):
        """Column keys containing CJK characters."""
        conditions = {
            0: {
                "日本語": np.array([1.0, 2.0]),
                "中文": np.array([3.0, 4.0]),
                "한국어": np.array([5.0, 6.0]),
            },
        }
        read_gd = self._roundtrip(conditions, tmp_path, "test_cjk_keys")
        _assert_conditions_equal(read_gd.data.conditions, conditions)

    @pytest.mark.xfail(reason="Empty-string column keys cause zarr group-key issues", strict=True)
    def test_conditions_with_empty_string_column_key(self, tmp_path):
        """An empty-string column key."""
        conditions = {
            0: {
                "": np.array([1.0, 2.0]),  # Empty string as key
                "normal": np.array([3.0, 4.0]),
            },
        }
        read_gd = self._roundtrip(conditions, tmp_path, "test_empty_key")
        _assert_conditions_equal(read_gd.data.conditions, conditions)

    def test_conditions_with_numeric_string_column_keys(self, tmp_path):
        """Column keys that look like numbers (e.g. '123')."""
        conditions = {
            0: {
                "123": np.array([1.0, 2.0]),
                "456.78": np.array([3.0, 4.0]),
                "-999": np.array([5.0, 6.0]),
            },
        }
        read_gd = self._roundtrip(conditions, tmp_path, "test_numeric_keys")
        _assert_conditions_equal(read_gd.data.conditions, conditions)

    @pytest.mark.xfail(reason="Empty strings and slashes in column keys cause zarr group-key issues", strict=True)
    def test_conditions_with_mixed_weird_keys(self, tmp_path):
        """A mix of different weird column keys (Unicode + slash + empty + numeric)."""
        conditions = {
            0: {
                "α-blocker+combo/v1": np.array([1.0]),
                "日本語:gene": np.array([2.0]),
                "": np.array([3.0]),
                "123": np.array([4.0]),
            },
            1: {
                "α-blocker+combo/v1": np.array([5.0, 6.0]),
                "日本語:gene": np.array([7.0, 8.0]),
                "": np.array([9.0, 10.0]),
                "123": np.array([11.0, 12.0]),
            },
        }
        read_gd = self._roundtrip(
            conditions, tmp_path, "test_mixed_weird_keys", src_to_tgt_dist_map={0: [0, 1]}
        )
        _assert_conditions_equal(read_gd.data.conditions, conditions)

    def test_full_roundtrip_with_weird_keys(self, tmp_path):
        """Full GroupedDistribution round-trip with Unicode/CJK condition keys and a rich df."""
        src_tgt_dist_df = pd.DataFrame({
            "src_dist_idx": [0, 0, 1],
            "tgt_dist_idx": [0, 1, 2],
            "drug": ["a", "b", "c"],
            "gene": ["x", "y", "z"],
        })

        conditions = {
            0: {"α-blocker": np.array([1.0]), "日本語": np.array([2.0])},
            1: {"α-blocker": np.array([3.0]), "日本語": np.array([4.0])},
            2: {"α-blocker": np.array([5.0]), "日本語": np.array([6.0])},
        }

        gd = _make_grouped_distribution(
            conditions,
            src_to_tgt_dist_map={0: [0, 1], 1: [2]},
            src_dist_idx_to_labels={0: ["cell_line_0"], 1: ["cell_line_1"]},
            tgt_dist_idx_to_labels={0: ["tlabel1"], 1: ["tlabel2"], 2: ["tlabel3"]},
            src_tgt_dist_df=src_tgt_dist_df,
            default_values={"drug": "control", "gene": "control"},
            tgt_dist_keys=["drug", "gene"],
            src_dist_keys=["cell_line"],
            dist_flag_key="control",
        )

        store_path = tmp_path / "test_full_roundtrip_weird_keys.zarr"
        gd.write_zarr(path=str(store_path))

        read_gd = GroupedDistribution.read_zarr(str(store_path))

        # Conditions survived the round-trip
        _assert_conditions_equal(read_gd.data.conditions, conditions)
        # The rich src/tgt pairing dataframe survived too
        pd.testing.assert_frame_equal(read_gd.annotation.src_tgt_dist_df, src_tgt_dist_df)



class TestWriteNestedDistData:
    """Direct tests for the nested writer utility."""

    def _read_back(self, group):
        """Reconstruct nested structure from the on-disk CSR-like layout."""
        dist_ids = list(group.attrs["dist_ids"])
        # Column names correspond to stored arrays (skip attrs)
        col_names = list(group.array_keys())

        reconstructed = {}
        for dist_idx, dist_id in enumerate(dist_ids):
            reconstructed[dist_id] = {}
            for col_name in col_names:
                indptr = group.attrs[f"indptr_{col_name}"]
                start, end = indptr[dist_idx], indptr[dist_idx + 1]
                reconstructed[dist_id][col_name] = group[col_name][start:end]
        return reconstructed

    def test_basic_write_and_metadata(self, tmp_path):
        dist_data = {
            2: {
                "a": np.array([[1, 2], [3, 4]]),  # 2 rows
                "b": np.array([10.0, 11.0]),
            },
            1: {
                "a": np.array([[5, 6]]),  # 1 row
                "b": np.array([12.0]),
            },
        }

        store_path = tmp_path / "nested_basic.zarr"
        zgroup = zarr.open_group(str(store_path), mode="w")

        write_nested_dist_data(
            group=zgroup,
            dist_data=dist_data,
            chunk_size=10,
            shard_size=100,
        )

        # Dist IDs stored sorted
        assert list(zgroup.attrs["dist_ids"]) == [1, 2]

        # Columns stored once, concatenated
        assert set(zgroup.array_keys()) == {"a", "b"}

        # Each column has an indptr
        for col in ("a", "b"):
            assert f"indptr_{col}" in zgroup.attrs

        reconstructed = self._read_back(zgroup)
        # Values should match original per dist_id/column
        for dist_id in dist_data:
            for col in dist_data[dist_id]:
                np.testing.assert_array_equal(reconstructed[dist_id][col], dist_data[dist_id][col])

    def test_empty_input_no_side_effects(self, tmp_path):
        store_path = tmp_path / "nested_empty.zarr"
        zgroup = zarr.open_group(str(store_path), mode="w")

        write_nested_dist_data(
            group=zgroup,
            dist_data={},
            chunk_size=10,
            shard_size=100,
        )

        # No arrays or attrs created
        assert list(zgroup.array_keys()) == []
        assert list(zgroup.attrs.keys()) == []

    def test_many_distributions_sequential_write(self, tmp_path):
        """Test writing many distributions sequentially."""
        n_dists = 500
        n_rows_per_dist = 100

        # Create large dataset with many distributions and multiple columns
        dist_data = {}
        for i in range(n_dists):
            dist_data[i] = {
                "col_a": np.random.rand(n_rows_per_dist),
                "col_b": np.random.rand(n_rows_per_dist),
                "col_c": np.random.rand(n_rows_per_dist),
            }

        store_path = tmp_path / "nested_many_dists.zarr"
        zgroup = zarr.open_group(str(store_path), mode="w")

        # Use settings similar to production (high workers, large shards)
        chunk_size = 1024
        shard_size = chunk_size * 4

        write_nested_dist_data(
            group=zgroup,
            dist_data=dist_data,
            chunk_size=chunk_size,
            shard_size=shard_size,
        )

        # Verify data integrity by reading back
        reconstructed = self._read_back(zgroup)

        for dist_id in dist_data:
            for col in dist_data[dist_id]:
                np.testing.assert_array_almost_equal(
                    reconstructed[dist_id][col],
                    dist_data[dist_id][col],
                )

    def test_large_shards_with_large_data(self, tmp_path):
        """Test with large shards and large data."""
        n_dists = 200
        n_rows_per_dist = 500  # Large data per distribution

        dist_data = {}
        for i in range(n_dists):
            dist_data[i] = {
                "embedding": np.random.rand(n_rows_per_dist, 50),  # 2D array
            }

        store_path = tmp_path / "nested_large_shards.zarr"
        zgroup = zarr.open_group(str(store_path), mode="w")

        # Settings that might trigger the bug
        chunk_size = 131072
        shard_size = chunk_size * 8

        write_nested_dist_data(
            group=zgroup,
            dist_data=dist_data,
            chunk_size=chunk_size,
            shard_size=shard_size,
        )

        # Verify by reading back
        reconstructed = self._read_back(zgroup)

        for dist_id in dist_data:
            np.testing.assert_array_almost_equal(
                reconstructed[dist_id]["embedding"],
                dist_data[dist_id]["embedding"],
            )

    @pytest.mark.parametrize("n_iterations", [10])
    def test_repeated_writes_stress_test(self, tmp_path, n_iterations):
        """Stress test with repeated writes."""
        for iteration in range(n_iterations):
            n_dists = 100
            n_rows_per_dist = 200

            dist_data = {}
            for i in range(n_dists):
                dist_data[i] = {
                    "col_a": np.random.rand(n_rows_per_dist),
                    "col_b": np.random.rand(n_rows_per_dist),
                }

            store_path = tmp_path / f"nested_stress_{iteration}.zarr"
            zgroup = zarr.open_group(str(store_path), mode="w")

            chunk_size = 4096
            shard_size = chunk_size * 4

            write_nested_dist_data(
                group=zgroup,
                dist_data=dist_data,
                chunk_size=chunk_size,
                shard_size=shard_size,
            )

            # Quick verify
            reconstructed = self._read_back(zgroup)
            assert len(reconstructed) == n_dists


class TestInMemoryAndToMemory:
    """Tests for in_memory parameter and to_memory method."""

    def test_read_zarr_roundtrips_row_indices(
        self, tmp_path, dummy_grouped_distribution_data, dummy_grouped_distribution_annotation
    ):
        """read_zarr roundtrips the per-row dist-id metadata (always in memory)."""
        gd = GroupedDistribution(
            data=dummy_grouped_distribution_data,
            annotation=dummy_grouped_distribution_annotation,
        )
        store_path = tmp_path / "test_in_memory.zarr"
        gd.write_zarr(path=str(store_path))

        read_gd = GroupedDistribution.read_zarr(str(store_path))

        # Row-index metadata is always in memory now
        assert read_gd.data.is_in_memory is True

        # Verify per-row dist-id arrays roundtrip correctly
        np.testing.assert_array_equal(
            read_gd.data.row_src_dist_idx, dummy_grouped_distribution_data.row_src_dist_idx
        )
        np.testing.assert_array_equal(
            read_gd.data.row_tgt_dist_idx, dummy_grouped_distribution_data.row_tgt_dist_idx
        )

        # rows_for() reconstructs the same per-dist row groups from the roundtripped column
        def _as_index(rows, n):
            return np.arange(n)[rows] if isinstance(rows, slice) else np.asarray(rows)

        n = read_gd.data.row_src_dist_idx.shape[0]
        original_rows = GroupedDistributionData.rows_for(dummy_grouped_distribution_data.row_src_dist_idx)
        read_rows = GroupedDistributionData.rows_for(read_gd.data.row_src_dist_idx)
        assert read_rows.keys() == original_rows.keys()
        for k in original_rows:
            np.testing.assert_array_equal(_as_index(read_rows[k], n), _as_index(original_rows[k], n))

    def test_is_in_memory_for_datamanager_created_data(self, sample_grouped_distribution):
        """Test that data created by DataManager is already in memory."""
        # Data from DataManager fixture should be in memory (numpy arrays)
        assert sample_grouped_distribution.data.is_in_memory is True

    def test_grouped_distribution_data_is_in_memory_property(self):
        """Test is_in_memory property for manually created data."""
        row_tgt = np.array([-1, -1, 0, 0, 1], dtype=np.int64)
        row_src = np.array([0, 0, -1, -1, -1], dtype=np.int64)
        data = GroupedDistributionData(
            src_to_tgt_dist_map={0: [0, 1]},
            row_tgt_dist_idx=row_tgt,
            row_src_dist_idx=row_src,
            conditions={0: {"cond": np.array([1, 2])}, 1: {"cond": np.array([3, 4])}},
        )
        # Metadata is always in memory
        assert data.is_in_memory is True
        # to_memory() is a no-op that returns None
        assert data.to_memory() is None


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
            "src_tgt_dist_df": pd.DataFrame(
                {
                    "src_dist_idx": [0, 0, 1],
                    "tgt_dist_idx": [0, 1, 2],
                    "other_col": ["a", "b", "c"],
                }
            ),
            "default_values": {"param1": 1, "param2": "val"},
            "tgt_dist_keys": ["key1", "key2"],
            "src_dist_keys": ["skey1"],
            "dist_flag_key": "flag",
        }

    def test_data_location_via_datamanager_and_write(self, tmp_path, sample_grouped_distribution):
        """Test that data_location survives full roundtrip via DataManager and write_zarr.

        This tests the integration path: DataManager.prepare_data -> write_zarr -> read_zarr
        which is how users typically use the library.
        """
        # sample_grouped_distribution comes from DataManager.prepare_data with data_location set
        assert sample_grouped_distribution.annotation.data_location is not None

        store_path = tmp_path / "test_datamanager_data_location.zarr"
        sample_grouped_distribution.write_zarr(
            path=str(store_path),
            chunk_size=10,
            shard_size=100,
            max_workers=1,
        )

        # Read back and verify
        read_gd = GroupedDistribution.read_zarr(str(store_path))
        assert read_gd.annotation.data_location is not None
        assert read_gd.annotation.data_location._path == sample_grouped_distribution.annotation.data_location._path

    def test_data_location_via_prepare_datasets(self, tmp_path, adata_test):
        """Test data_location roundtrip using prepare_datasets convenience function.

        This matches the exact usage pattern:
        gd1 = prepare_datasets({"adata1": adata1}, data_manager=data_manager)
        gd1 = gd1["adata1"]
        gd1.write_zarr("data/gd1.zarr")
        """
        from scaleflow.data import DataManager, prepare_datasets

        adl = AnnDataLocation()
        data_manager = DataManager(
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

        gd_dict = prepare_datasets({"adata1": adata_test}, data_manager=data_manager)
        gd = gd_dict["adata1"]

        # Verify data_location is set
        assert gd.annotation.data_location is not None
        assert gd.annotation.data_location._path == [("getattr", "obsm"), ("getitem", "X_pca")]

        # Write and read back
        store_path = tmp_path / "test_prepare_datasets_data_location.zarr"
        gd.write_zarr(path=str(store_path), chunk_size=10, shard_size=100, max_workers=1)

        read_gd = GroupedDistribution.read_zarr(str(store_path))
        assert read_gd.annotation.data_location is not None
        assert read_gd.annotation.data_location._path == gd.annotation.data_location._path

    def _roundtrip_data_location(self, tmp_path, name, base_annotation_kwargs, data_location):
        """Round-trip a GroupedDistribution carrying ``data_location`` and return the read-back one.

        Serialization is now AnnData-based on :class:`GroupedDistribution`; ``data_location`` is
        stored (as JSON) in ``adata.uns`` and reconstructed by ``from_adata``.
        """
        annotation = GroupedDistributionAnnotation(**base_annotation_kwargs, data_location=data_location)
        # base_annotation_kwargs has old_obs_index of length 20; build a matching data half.
        data = GroupedDistributionData(
            src_to_tgt_dist_map={0: [0, 1], 1: [2]},
            row_tgt_dist_idx=np.full(20, -1, dtype=np.int64),
            row_src_dist_idx=np.full(20, -1, dtype=np.int64),
            conditions={},
        )
        gd = GroupedDistribution(data=data, annotation=annotation)
        store_path = tmp_path / f"{name}.zarr"
        gd.write_zarr(path=str(store_path))
        return GroupedDistribution.read_zarr(str(store_path))

    def test_annotation_with_none_data_location(self, tmp_path, base_annotation_kwargs):
        """data_location=None survives the round-trip as None."""
        read_gd = self._roundtrip_data_location(tmp_path, "test_none_location", base_annotation_kwargs, None)
        assert read_gd.annotation.data_location is None

    def test_annotation_with_obsm_data_location(self, tmp_path, base_annotation_kwargs):
        """obsm data_location survives the round-trip."""
        data_location = AnnDataLocation().obsm["X_pca"]
        read_gd = self._roundtrip_data_location(
            tmp_path, "test_obsm_location", base_annotation_kwargs, data_location
        )
        assert read_gd.annotation.data_location is not None
        assert read_gd.annotation.data_location._path == data_location._path

    def test_annotation_with_X_data_location(self, tmp_path, base_annotation_kwargs):
        """X (main matrix) data_location survives the round-trip."""
        data_location = AnnDataLocation().X
        read_gd = self._roundtrip_data_location(
            tmp_path, "test_X_location", base_annotation_kwargs, data_location
        )
        assert read_gd.annotation.data_location is not None
        assert read_gd.annotation.data_location._path == data_location._path

    def test_annotation_with_layers_data_location(self, tmp_path, base_annotation_kwargs):
        """layers data_location survives the round-trip."""
        data_location = AnnDataLocation().layers["counts"]
        read_gd = self._roundtrip_data_location(
            tmp_path, "test_layers_location", base_annotation_kwargs, data_location
        )
        assert read_gd.annotation.data_location is not None
        assert read_gd.annotation.data_location._path == data_location._path

    def test_grouped_distribution_preserves_data_location(
        self,
        tmp_path,
        dummy_grouped_distribution_data,
        base_annotation_kwargs,
    ):
        """GroupedDistribution preserves data_location through write_zarr / read_zarr."""
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
        gd.write_zarr(path=str(store_path))

        read_gd = GroupedDistribution.read_zarr(str(store_path))
        assert read_gd.annotation.data_location is not None
        assert read_gd.annotation.data_location._path == data_location._path
