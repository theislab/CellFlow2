"""Test suite for ReservoirSampler."""

import numpy as np
import pytest

from scaleflow.data import AnnDataLocation, DataManager, GroupedDistribution
from scaleflow.data._dataloader import ReservoirSampler


class TestReservoirSamplerInit:
    """Test ReservoirSampler initialization."""

    def test_init_cache_all_default(self, sample_grouped_distribution):
        """Test that default params (no pool_fraction/replacement_prob) enables cache_all mode."""
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
        )
        assert sampler._cache_all is True
        assert sampler._pool_fraction is None
        assert sampler._replacement_prob is None
        assert sampler._pool_size is None
        assert sampler.batch_size == 64
        assert sampler.n_source_dists == len(sample_grouped_distribution.data.src_data)
        assert sampler.n_target_dists == len(sample_grouped_distribution.data.tgt_data)

    def test_init_cache_all_explicit(self, sample_grouped_distribution):
        """Test that pool_fraction=1.0 enables cache_all mode."""
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
            pool_fraction=1.0,
            replacement_prob=0.1,  # This should be ignored
        )
        assert sampler._cache_all is True

    def test_init_pool_mode(self, sample_grouped_distribution):
        """Test initialization with pool_fraction and replacement_prob."""
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
            pool_fraction=0.5,
            replacement_prob=0.1,
        )
        assert sampler._cache_all is False
        assert sampler._pool_fraction == 0.5
        assert sampler._replacement_prob == 0.1
        expected_pool_size = int(np.ceil(0.5 * sampler.n_source_dists))
        assert sampler._pool_size == expected_pool_size

    def test_init_missing_replacement_prob_raises(self, sample_grouped_distribution):
        """Test that pool_fraction without replacement_prob raises ValueError."""
        with pytest.raises(ValueError, match="replacement_prob must be provided"):
            ReservoirSampler(
                data=sample_grouped_distribution,
                batch_size=64,
                pool_fraction=0.5,
            )

    def test_init_missing_pool_fraction_raises(self, sample_grouped_distribution):
        """Test that replacement_prob without pool_fraction raises ValueError."""
        with pytest.raises(ValueError, match="pool_fraction must be provided"):
            ReservoirSampler(
                data=sample_grouped_distribution,
                batch_size=64,
                replacement_prob=0.1,
            )

    def test_init_invalid_pool_fraction_zero(self, sample_grouped_distribution):
        """Test that pool_fraction=0 raises ValueError."""
        with pytest.raises(ValueError, match="pool_fraction must be in"):
            ReservoirSampler(
                data=sample_grouped_distribution,
                batch_size=64,
                pool_fraction=0.0,
                replacement_prob=0.1,
            )

    def test_init_invalid_pool_fraction_negative(self, sample_grouped_distribution):
        """Test that negative pool_fraction raises ValueError."""
        with pytest.raises(ValueError, match="pool_fraction must be in"):
            ReservoirSampler(
                data=sample_grouped_distribution,
                batch_size=64,
                pool_fraction=-0.5,
                replacement_prob=0.1,
            )

    def test_init_invalid_pool_fraction_greater_than_one(self, sample_grouped_distribution):
        """Test that pool_fraction > 1 raises ValueError."""
        with pytest.raises(ValueError, match="pool_fraction must be in"):
            ReservoirSampler(
                data=sample_grouped_distribution,
                batch_size=64,
                pool_fraction=1.5,
                replacement_prob=0.1,
            )


class TestReservoirSamplerInitialization:
    """Test ReservoirSampler init_sampler method."""

    def test_init_sampler_cache_all(self, sample_grouped_distribution):
        """Test init_sampler in cache_all mode."""
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
        )
        rng = np.random.default_rng(42)
        sampler.init_sampler(rng)

        assert sampler._initialized is True
        assert len(sampler._cached_srcs) == sampler.n_source_dists
        assert sampler._src_idx_pool is not None
        assert len(sampler._src_idx_pool) == sampler.n_source_dists

    def test_init_sampler_pool_mode(self, sample_grouped_distribution):
        """Test init_sampler in pool mode."""
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
            pool_fraction=0.5,
            replacement_prob=0.1,
        )
        rng = np.random.default_rng(42)
        sampler.init_sampler(rng)

        assert sampler._initialized is True
        assert len(sampler._cached_srcs) == sampler._pool_size
        assert len(sampler._src_idx_pool) == sampler._pool_size

    def test_init_sampler_double_init_raises(self, sample_grouped_distribution):
        """Test that calling init_sampler twice raises ValueError."""
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
        )
        rng = np.random.default_rng(42)
        sampler.init_sampler(rng)

        with pytest.raises(ValueError, match="already initialized"):
            sampler.init_sampler(rng)


class TestReservoirSamplerSampling:
    """Test ReservoirSampler sampling functionality."""

    def test_sample_not_initialized_raises(self, sample_grouped_distribution):
        """Test that sampling without init raises ValueError."""
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
        )
        rng = np.random.default_rng(42)

        with pytest.raises(ValueError, match="not initialized"):
            sampler.sample(rng)

    def test_sample_returns_correct_structure(self, sample_grouped_distribution):
        """Test that sample returns dict with expected keys."""
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
        )
        rng = np.random.default_rng(42)
        sampler.init_sampler(rng)

        result = sampler.sample(rng)

        assert isinstance(result, dict)
        assert "src_cell_data" in result
        assert "tgt_cell_data" in result
        assert "condition" in result

    def test_sample_batch_size(self, sample_grouped_distribution):
        """Test that sample returns correct batch size."""
        batch_size = 32
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=batch_size,
        )
        rng = np.random.default_rng(42)
        sampler.init_sampler(rng)

        result = sampler.sample(rng)

        assert result["src_cell_data"].shape[0] == batch_size
        assert result["tgt_cell_data"].shape[0] == batch_size

    def test_sample_deterministic_with_same_seed(self, sample_grouped_distribution):
        """Test that sampling is deterministic with same RNG seed."""
        sampler1 = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
        )
        sampler2 = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
        )

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        sampler1.init_sampler(rng1)
        sampler2.init_sampler(rng2)

        result1 = sampler1.sample(rng1)
        result2 = sampler2.sample(rng2)

        np.testing.assert_array_equal(result1["src_cell_data"], result2["src_cell_data"])
        np.testing.assert_array_equal(result1["tgt_cell_data"], result2["tgt_cell_data"])

    def test_sample_different_with_different_seed(self, sample_grouped_distribution):
        """Test that sampling differs with different RNG seeds."""
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
        )

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)

        sampler.init_sampler(rng1)
        result1 = sampler.sample(rng1)
        
        # Re-initialize for clean comparison
        sampler2 = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
        )
        sampler2.init_sampler(rng2)
        result2 = sampler2.sample(rng2)

        # At least one should differ (with very high probability)
        assert not np.array_equal(result1["src_cell_data"], result2["src_cell_data"]) or \
               not np.array_equal(result1["tgt_cell_data"], result2["tgt_cell_data"])

    def test_multiple_samples(self, sample_grouped_distribution):
        """Test that multiple samples can be drawn."""
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
        )
        rng = np.random.default_rng(42)
        sampler.init_sampler(rng)

        for _ in range(10):
            result = sampler.sample(rng)
            assert result["src_cell_data"].shape[0] == 64
            assert result["tgt_cell_data"].shape[0] == 64


class TestReservoirSamplerPoolMode:
    """Test ReservoirSampler pool replacement functionality."""

    def test_pool_mode_initialization(self, sample_grouped_distribution):
        """Test pool mode initializes with correct pool size."""
        pool_fraction = 0.5
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
            pool_fraction=pool_fraction,
            replacement_prob=0.1,
        )
        rng = np.random.default_rng(42)
        sampler.init_sampler(rng)

        expected_pool_size = int(np.ceil(pool_fraction * sampler.n_source_dists))
        assert len(sampler._src_idx_pool) == expected_pool_size
        assert len(sampler._cached_srcs) == expected_pool_size

    def test_pool_mode_sampling(self, sample_grouped_distribution):
        """Test that pool mode sampling works correctly."""
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
            pool_fraction=0.5,
            replacement_prob=0.1,
        )
        rng = np.random.default_rng(42)
        sampler.init_sampler(rng)

        # Should be able to sample multiple times
        for _ in range(20):
            result = sampler.sample(rng)
            assert result["src_cell_data"].shape[0] == 64

    def test_pool_usage_count_increments(self, sample_grouped_distribution):
        """Test that pool usage count is tracked."""
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
            pool_fraction=0.5,
            replacement_prob=0.0,  # No replacement to simplify test
        )
        rng = np.random.default_rng(42)
        sampler.init_sampler(rng)

        initial_usage = sampler._pool_usage_count.copy()

        # Sample a few times
        for _ in range(10):
            sampler.sample(rng)

        # Usage count should have increased
        assert sampler._pool_usage_count.sum() > initial_usage.sum()

    def test_high_replacement_prob(self, sample_grouped_distribution):
        """Test sampling with high replacement probability."""
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
            pool_fraction=0.3,
            replacement_prob=0.9,  # High replacement probability
        )
        rng = np.random.default_rng(42)
        sampler.init_sampler(rng)

        # Should still work with high replacement
        for _ in range(50):
            result = sampler.sample(rng)
            assert result["src_cell_data"].shape[0] == 64


class TestReservoirSamplerWithSplitData:
    """Test ReservoirSampler with split GroupedDistribution data."""

    def test_sampler_with_split_train_data(self, sample_grouped_distribution):
        """Test that sampler works with split training data."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter

        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        splits = splitter.split()
        train_data = splits["train"]

        # Create sampler with training split
        sampler = ReservoirSampler(
            data=train_data,
            batch_size=32,
        )
        rng = np.random.default_rng(42)
        sampler.init_sampler(rng)

        result = sampler.sample(rng)
        assert result["src_cell_data"].shape[0] == 32
        assert result["tgt_cell_data"].shape[0] == 32

    def test_sampler_with_split_val_data(self, sample_grouped_distribution):
        """Test that sampler works with split validation data."""
        from scaleflow.data._data_splitter import GroupedDistributionSplitter

        splitter = GroupedDistributionSplitter(
            gd=sample_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        splits = splitter.split()
        val_data = splits["val"]

        # Create sampler with validation split
        sampler = ReservoirSampler(
            data=val_data,
            batch_size=16,
        )
        rng = np.random.default_rng(42)
        sampler.init_sampler(rng)

        result = sampler.sample(rng)
        assert result["src_cell_data"].shape[0] == 16


class TestReservoirSamplerEdgeCases:
    """Test edge cases for ReservoirSampler."""

    def test_small_pool_fraction(self, sample_grouped_distribution):
        """Test with very small pool fraction."""
        # Use minimum pool size that's at least 1
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=16,
            pool_fraction=0.01,
            replacement_prob=0.5,
        )
        rng = np.random.default_rng(42)
        sampler.init_sampler(rng)

        assert sampler._pool_size >= 1
        result = sampler.sample(rng)
        assert result["src_cell_data"].shape[0] == 16

    def test_batch_size_larger_than_cell_count(self, sample_grouped_distribution):
        """Test that batch size larger than available cells still works (with replacement)."""
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=10000,  # Larger than typical cell count per distribution
        )
        rng = np.random.default_rng(42)
        sampler.init_sampler(rng)

        result = sampler.sample(rng)
        assert result["src_cell_data"].shape[0] == 10000

    def test_condition_is_numpy_array(self, sample_grouped_distribution):
        """Test that returned condition is a numpy array."""
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            batch_size=64,
        )
        rng = np.random.default_rng(42)
        sampler.init_sampler(rng)

        result = sampler.sample(rng)
        assert isinstance(result["condition"], np.ndarray)
