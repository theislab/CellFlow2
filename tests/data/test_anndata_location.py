"""Tests for AnnDataLocation serialization and access patterns."""

import json

import pytest

from scaleflow.data._anndata_location import AnnDataLocation


class TestAnnDataLocationSerialization:
    """Tests for JSON serialization of AnnDataLocation."""

    def test_to_json_simple_attribute(self):
        """Test serialization of simple attribute access."""
        loc = AnnDataLocation().obs
        json_str = loc.to_json()
        assert json.loads(json_str) == [["getattr", "obs"]]

    def test_to_json_nested_access(self):
        """Test serialization of nested attribute and item access."""
        loc = AnnDataLocation().obsm["X_pca"]
        json_str = loc.to_json()
        assert json.loads(json_str) == [["getattr", "obsm"], ["getitem", "X_pca"]]

    def test_to_json_integer_key(self):
        """Test serialization with integer key."""
        loc = AnnDataLocation().layers["counts"]
        json_str = loc.to_json()
        parsed = json.loads(json_str)
        assert parsed == [["getattr", "layers"], ["getitem", "counts"]]

    def test_from_json_roundtrip(self):
        """Test that from_json correctly reconstructs the location."""
        original = AnnDataLocation().obsm["X_umap"]
        json_str = original.to_json()
        restored = AnnDataLocation.from_json(json_str)
        assert restored.to_json() == json_str

    def test_slice_serialization(self):
        """Test that slice indexing is serialized to JSON correctly."""
        loc = AnnDataLocation().obs[0:10]
        json_str = loc.to_json()
        parsed = json.loads(json_str)
        assert parsed == [["getattr", "obs"], ["getitem", {"__slice__": [0, 10, None]}]]

    def test_slice_with_none_serialization(self):
        """Test that slice with None values is serialized correctly."""
        loc = AnnDataLocation().X[:]
        json_str = loc.to_json()
        parsed = json.loads(json_str)
        assert parsed == [["getattr", "X"], ["getitem", {"__slice__": [None, None, None]}]]

    def test_slice_with_step_serialization(self):
        """Test that slice with step is serialized correctly."""
        loc = AnnDataLocation()[::2]
        json_str = loc.to_json()
        parsed = json.loads(json_str)
        assert parsed == [["getitem", {"__slice__": [None, None, 2]}]]

    def test_slice_roundtrip(self):
        """Test that slices survive JSON serialization roundtrip."""
        original = AnnDataLocation().obs[5:15:2]
        json_str = original.to_json()
        restored = AnnDataLocation.from_json(json_str)
        assert restored.to_json() == json_str

    def test_multidim_slice_serialization(self):
        """Test that multi-dimensional slicing [:, :50] is serialized correctly."""
        loc = AnnDataLocation().obsm["X_pca"][:, :50]
        json_str = loc.to_json()
        parsed = json.loads(json_str)
        expected = [
            ["getattr", "obsm"],
            ["getitem", "X_pca"],
            ["getitem", {"__tuple__": [{"__slice__": [None, None, None]}, {"__slice__": [None, 50, None]}]}],
        ]
        assert parsed == expected

    def test_multidim_slice_roundtrip(self):
        """Test that multi-dimensional slices survive JSON roundtrip."""
        original = AnnDataLocation().obsm["X_pca"][:, :50]
        json_str = original.to_json()
        restored = AnnDataLocation.from_json(json_str)
        assert restored.to_json() == json_str

    def test_mixed_tuple_serialization(self):
        """Test tuple with mixed int and slice values."""
        loc = AnnDataLocation().X[0, :50]
        json_str = loc.to_json()
        parsed = json.loads(json_str)
        expected = [
            ["getattr", "X"],
            ["getitem", {"__tuple__": [0, {"__slice__": [None, 50, None]}]}],
        ]
        assert parsed == expected


class TestAnnDataLocationExecution:
    """Tests for executing AnnDataLocation on AnnData objects."""

    @pytest.fixture
    def simple_adata(self):
        """Create a simple AnnData object for testing."""
        import numpy as np
        from anndata import AnnData

        adata = AnnData(np.random.randn(10, 5))
        adata.obs["cell_type"] = ["A"] * 5 + ["B"] * 5
        adata.obsm["X_pca"] = np.random.randn(10, 2)
        return adata

    def test_execute_obs_access(self, simple_adata):
        """Test executing obs attribute access."""
        loc = AnnDataLocation().obs
        result = loc(simple_adata)
        assert result is simple_adata.obs

    def test_execute_obsm_item_access(self, simple_adata):
        """Test executing obsm item access."""
        loc = AnnDataLocation().obsm["X_pca"]
        result = loc(simple_adata)
        assert result is simple_adata.obsm["X_pca"]

    def test_execute_missing_attribute_raises(self, simple_adata):
        """Test that accessing missing attribute raises error."""
        loc = AnnDataLocation().nonexistent
        with pytest.raises(AttributeError):
            loc(simple_adata)

    def test_execute_missing_key_raises(self, simple_adata):
        """Test that accessing missing key raises error."""
        loc = AnnDataLocation().obsm["nonexistent"]
        with pytest.raises(KeyError):
            loc(simple_adata)

    def test_execute_slice_on_obsm(self, simple_adata):
        """Test that slices work when executed on obsm arrays."""
        import numpy as np

        loc = AnnDataLocation().obsm["X_pca"][0:5]
        result = loc(simple_adata)
        np.testing.assert_array_equal(result, simple_adata.obsm["X_pca"][0:5])

    def test_execute_slice_after_json_roundtrip(self, simple_adata):
        """Test that slices work after JSON serialization roundtrip."""
        import numpy as np

        original = AnnDataLocation().obsm["X_pca"][2:8]
        json_str = original.to_json()
        restored = AnnDataLocation.from_json(json_str)
        result = restored(simple_adata)
        np.testing.assert_array_equal(result, simple_adata.obsm["X_pca"][2:8])


class TestAnnDataLocationRepr:
    """Tests for string representation of AnnDataLocation."""

    def test_repr_empty(self):
        """Test repr of empty location."""
        loc = AnnDataLocation()
        assert repr(loc) == "<AnnDataLocation: AnnDataAccessor()>"

    def test_repr_with_attribute(self):
        """Test repr with attribute access."""
        loc = AnnDataLocation().obs
        assert repr(loc) == "<AnnDataLocation: AnnDataAccessor().obs>"

    def test_repr_with_item(self):
        """Test repr with item access."""
        loc = AnnDataLocation().obsm["X_pca"]
        assert repr(loc) == "<AnnDataLocation: AnnDataAccessor().obsm['X_pca']>"
