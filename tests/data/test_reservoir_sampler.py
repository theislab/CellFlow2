"""Test suite for ReservoirSampler, InMemorySampler, and CombinedSampler."""

import numpy as np
import pytest

from scaleflow.data._dataloader import CombinedSampler, InMemorySampler, ReservoirSampler


class TestInMemorySamplerInit:
    """Test InMemorySampler initialization."""

    def test_init_basic(self, sample_grouped_distribution):
        """Test basic initialization."""
        rng = np.random.default_rng(42)
        sampler = InMemorySampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
        )
        assert sampler.batch_size == 64
        assert sampler.n_source_dists == len(sample_grouped_distribution.data.src_data)
        assert sampler.n_target_dists == len(sample_grouped_distribution.data.tgt_data)
        assert sampler._initialized is False


class TestInMemorySamplerInitialization:
    """Test InMemorySampler init_sampler method."""

    def test_init_sampler(self, sample_grouped_distribution):
        """Test init_sampler loads all data."""
        rng = np.random.default_rng(42)
        sampler = InMemorySampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
        )
        sampler.init_sampler()

        assert sampler._initialized is True
        assert len(sampler._cached_srcs) == sampler.n_source_dists
        assert len(sampler._cached_tgts) == sampler.n_target_dists

    def test_init_sampler_double_init_raises(self, sample_grouped_distribution):
        """Test that calling init_sampler twice raises ValueError."""
        rng = np.random.default_rng(42)
        sampler = InMemorySampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
        )
        sampler.init_sampler()

        with pytest.raises(ValueError, match="already initialized"):
            sampler.init_sampler()


class TestInMemorySamplerSampling:
    """Test InMemorySampler sampling functionality."""

    def test_sample_not_initialized_raises(self, sample_grouped_distribution):
        """Test that sampling without init raises ValueError."""
        rng = np.random.default_rng(42)
        sampler = InMemorySampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
        )

        with pytest.raises(ValueError, match="not initialized"):
            sampler.sample()

    def test_sample_returns_correct_structure(self, sample_grouped_distribution):
        """Test that sample returns dict with expected keys."""
        rng = np.random.default_rng(42)
        sampler = InMemorySampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
        )
        sampler.init_sampler()

        result = sampler.sample()

        assert isinstance(result, dict)
        assert "src_cell_data" in result
        assert "tgt_cell_data" in result
        assert "condition" in result

    def test_sample_batch_size(self, sample_grouped_distribution):
        """Test that sample returns correct batch size."""
        batch_size = 32
        rng = np.random.default_rng(42)
        sampler = InMemorySampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=batch_size,
        )
        sampler.init_sampler()

        result = sampler.sample()

        assert result["src_cell_data"].shape[0] == batch_size
        assert result["tgt_cell_data"].shape[0] == batch_size

    def test_sample_deterministic_with_same_seed(self, sample_grouped_distribution):
        """Test that sampling is deterministic with same RNG seed."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        sampler1 = InMemorySampler(
            data=sample_grouped_distribution,
            rng=rng1,
            batch_size=64,
        )
        sampler2 = InMemorySampler(
            data=sample_grouped_distribution,
            rng=rng2,
            batch_size=64,
        )

        sampler1.init_sampler()
        sampler2.init_sampler()

        result1 = sampler1.sample()
        result2 = sampler2.sample()

        np.testing.assert_array_equal(result1["src_cell_data"], result2["src_cell_data"])
        np.testing.assert_array_equal(result1["tgt_cell_data"], result2["tgt_cell_data"])

    def test_multiple_samples(self, sample_grouped_distribution):
        """Test that multiple samples can be drawn."""
        rng = np.random.default_rng(42)
        sampler = InMemorySampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
        )
        sampler.init_sampler()

        for _ in range(10):
            result = sampler.sample()
            assert result["src_cell_data"].shape[0] == 64
            assert result["tgt_cell_data"].shape[0] == 64

    def test_condition_is_dict(self, sample_grouped_distribution):
        """Test that returned condition is a dict (nested conditions format)."""
        rng = np.random.default_rng(42)
        sampler = InMemorySampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
        )
        sampler.init_sampler()

        result = sampler.sample()
        assert isinstance(result["condition"], dict)


class TestInMemorySamplerWithSplitData:
    """Test InMemorySampler with split GroupedDistribution data."""

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

        rng = np.random.default_rng(42)
        sampler = InMemorySampler(
            data=train_data,
            rng=rng,
            batch_size=32,
        )
        sampler.init_sampler()

        result = sampler.sample()
        assert result["src_cell_data"].shape[0] == 32
        assert result["tgt_cell_data"].shape[0] == 32


class TestTwoInMemorySamplersIndependent:
    """Test that two InMemorySampler instances work independently."""

    def test_two_samplers_different_seeds_independent(self, sample_grouped_distribution):
        """Test two samplers with different seeds produce different results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)

        sampler1 = InMemorySampler(
            data=sample_grouped_distribution,
            rng=rng1,
            batch_size=64,
        )
        sampler2 = InMemorySampler(
            data=sample_grouped_distribution,
            rng=rng2,
            batch_size=64,
        )

        sampler1.init_sampler()
        sampler2.init_sampler()

        # Sample alternately from both
        results1 = []
        results2 = []
        for _ in range(5):
            results1.append(sampler1.sample())
            results2.append(sampler2.sample())

        # Results should be different
        for r1, r2 in zip(results1, results2, strict=False):
            assert not np.array_equal(r1["src_cell_data"], r2["src_cell_data"]) or not np.array_equal(
                r1["tgt_cell_data"], r2["tgt_cell_data"]
            )

    def test_two_samplers_same_seed_produce_same_results(self, sample_grouped_distribution):
        """Test two samplers with same seed produce identical results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        sampler1 = InMemorySampler(
            data=sample_grouped_distribution,
            rng=rng1,
            batch_size=64,
        )
        sampler2 = InMemorySampler(
            data=sample_grouped_distribution,
            rng=rng2,
            batch_size=64,
        )

        sampler1.init_sampler()
        sampler2.init_sampler()

        for _ in range(5):
            result1 = sampler1.sample()
            result2 = sampler2.sample()

            np.testing.assert_array_equal(result1["src_cell_data"], result2["src_cell_data"])
            np.testing.assert_array_equal(result1["tgt_cell_data"], result2["tgt_cell_data"])


class TestReservoirSamplerInit:
    """Test ReservoirSampler initialization."""

    def test_init_pool_mode(self, sample_grouped_distribution):
        """Test initialization with pool_fraction and replacement_prob."""
        rng = np.random.default_rng(42)
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
            pool_fraction=0.5,
            replacement_prob=0.1,
        )
        assert sampler._pool_fraction == 0.5
        assert sampler._replacement_prob == 0.1
        expected_pool_size = int(np.ceil(0.5 * sampler.n_source_dists))
        assert sampler._pool_size == expected_pool_size

    def test_init_pool_fraction_1_raises(self, sample_grouped_distribution):
        """Test that pool_fraction=1.0 raises ValueError (use InMemorySampler instead)."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="InMemorySampler"):
            ReservoirSampler(
                data=sample_grouped_distribution,
                rng=rng,
                batch_size=64,
                pool_fraction=1.0,
                replacement_prob=0.1,
            )

    def test_init_invalid_pool_fraction_zero(self, sample_grouped_distribution):
        """Test that pool_fraction=0 raises ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="pool_fraction must be in"):
            ReservoirSampler(
                data=sample_grouped_distribution,
                rng=rng,
                batch_size=64,
                pool_fraction=0.0,
                replacement_prob=0.1,
            )

    def test_init_invalid_pool_fraction_negative(self, sample_grouped_distribution):
        """Test that negative pool_fraction raises ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="pool_fraction must be in"):
            ReservoirSampler(
                data=sample_grouped_distribution,
                rng=rng,
                batch_size=64,
                pool_fraction=-0.5,
                replacement_prob=0.1,
            )


class TestReservoirSamplerInitialization:
    """Test ReservoirSampler init_sampler method."""

    def test_init_sampler_pool_mode(self, sample_grouped_distribution):
        """Test init_sampler in pool mode."""
        rng = np.random.default_rng(42)
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
            pool_fraction=0.5,
            replacement_prob=0.1,
        )
        sampler.init_sampler()

        assert sampler._initialized is True
        assert len(sampler._cached_srcs) == sampler._pool_size
        assert len(sampler._src_idx_pool) == sampler._pool_size

    def test_init_sampler_double_init_raises(self, sample_grouped_distribution):
        """Test that calling init_sampler twice raises ValueError."""
        rng = np.random.default_rng(42)
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
            pool_fraction=0.5,
            replacement_prob=0.1,
        )
        sampler.init_sampler()

        with pytest.raises(ValueError, match="already initialized"):
            sampler.init_sampler()


class TestReservoirSamplerSampling:
    """Test ReservoirSampler sampling functionality."""

    def test_sample_not_initialized_raises(self, sample_grouped_distribution):
        """Test that sampling without init raises ValueError."""
        rng = np.random.default_rng(42)
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
            pool_fraction=0.5,
            replacement_prob=0.1,
        )

        with pytest.raises(ValueError, match="not initialized"):
            sampler.sample()

    def test_sample_returns_correct_structure(self, sample_grouped_distribution):
        """Test that sample returns dict with expected keys."""
        rng = np.random.default_rng(42)
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
            pool_fraction=0.5,
            replacement_prob=0.1,
        )
        sampler.init_sampler()

        result = sampler.sample()

        assert isinstance(result, dict)
        assert "src_cell_data" in result
        assert "tgt_cell_data" in result
        assert "condition" in result

    def test_sample_batch_size(self, sample_grouped_distribution):
        """Test that sample returns correct batch size."""
        batch_size = 32
        rng = np.random.default_rng(42)
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=batch_size,
            pool_fraction=0.5,
            replacement_prob=0.1,
        )
        sampler.init_sampler()

        result = sampler.sample()

        assert result["src_cell_data"].shape[0] == batch_size
        assert result["tgt_cell_data"].shape[0] == batch_size

    def test_multiple_samples(self, sample_grouped_distribution):
        """Test that multiple samples can be drawn."""
        rng = np.random.default_rng(42)
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
            pool_fraction=0.5,
            replacement_prob=0.1,
        )
        sampler.init_sampler()

        for _ in range(10):
            result = sampler.sample()
            assert result["src_cell_data"].shape[0] == 64
            assert result["tgt_cell_data"].shape[0] == 64


class TestReservoirSamplerPoolMode:
    """Test ReservoirSampler pool replacement functionality."""

    def test_pool_mode_initialization(self, sample_grouped_distribution):
        """Test pool mode initializes with correct pool size."""
        pool_fraction = 0.5
        rng = np.random.default_rng(42)
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
            pool_fraction=pool_fraction,
            replacement_prob=0.1,
        )
        sampler.init_sampler()

        expected_pool_size = int(np.ceil(pool_fraction * sampler.n_source_dists))
        assert len(sampler._src_idx_pool) == expected_pool_size
        assert len(sampler._cached_srcs) == expected_pool_size

    def test_pool_mode_sampling(self, sample_grouped_distribution):
        """Test that pool mode sampling works correctly."""
        rng = np.random.default_rng(42)
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
            pool_fraction=0.5,
            replacement_prob=0.1,
        )
        sampler.init_sampler()

        # Should be able to sample multiple times
        for _ in range(20):
            result = sampler.sample()
            assert result["src_cell_data"].shape[0] == 64

    def test_pool_usage_count_increments(self, sample_grouped_distribution):
        """Test that pool usage count is tracked."""
        rng = np.random.default_rng(42)
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
            pool_fraction=0.5,
            replacement_prob=0.0,  # No replacement to simplify test
        )
        sampler.init_sampler()

        initial_usage = sampler._pool_usage_count.copy()

        # Sample a few times
        for _ in range(10):
            sampler.sample()

        # Usage count should have increased
        assert sampler._pool_usage_count.sum() > initial_usage.sum()

    def test_high_replacement_prob(self, sample_grouped_distribution):
        """Test sampling with high replacement probability."""
        rng = np.random.default_rng(42)
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
            pool_fraction=0.3,
            replacement_prob=0.9,  # High replacement probability
        )
        sampler.init_sampler()

        # Should still work with high replacement
        for _ in range(50):
            result = sampler.sample()
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
        rng = np.random.default_rng(42)
        sampler = ReservoirSampler(
            data=train_data,
            rng=rng,
            batch_size=32,
            pool_fraction=0.5,
            replacement_prob=0.1,
        )
        sampler.init_sampler()

        result = sampler.sample()
        assert result["src_cell_data"].shape[0] == 32
        assert result["tgt_cell_data"].shape[0] == 32


class TestReservoirSamplerEdgeCases:
    """Test edge cases for ReservoirSampler."""

    def test_small_pool_fraction(self, sample_grouped_distribution):
        """Test with very small pool fraction."""
        rng = np.random.default_rng(42)
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=16,
            pool_fraction=0.01,
            replacement_prob=0.5,
        )
        sampler.init_sampler()

        assert sampler._pool_size >= 1
        result = sampler.sample()
        assert result["src_cell_data"].shape[0] == 16

    def test_condition_is_dict(self, sample_grouped_distribution):
        """Test that returned condition is a dict (nested conditions format)."""
        rng = np.random.default_rng(42)
        sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng,
            batch_size=64,
            pool_fraction=0.5,
            replacement_prob=0.1,
        )
        sampler.init_sampler()

        result = sampler.sample()
        assert isinstance(result["condition"], dict)


class TestTwoReservoirSamplersIndependent:
    """Test that two ReservoirSampler instances work independently."""

    def test_two_samplers_interleaved_sampling(self, sample_grouped_distribution):
        """Test interleaved sampling from two samplers doesn't cause issues."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)

        sampler1 = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng1,
            batch_size=32,
            pool_fraction=0.5,
            replacement_prob=0.1,
        )
        sampler2 = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng2,
            batch_size=64,
            pool_fraction=0.3,
            replacement_prob=0.2,
        )

        sampler1.init_sampler()
        sampler2.init_sampler()

        # Interleave sampling
        for _ in range(20):
            r1 = sampler1.sample()
            r2 = sampler2.sample()
            r1_again = sampler1.sample()

            assert r1["src_cell_data"].shape[0] == 32
            assert r2["src_cell_data"].shape[0] == 64
            assert r1_again["src_cell_data"].shape[0] == 32

    def test_two_samplers_pool_mode_independent(self, sample_grouped_distribution):
        """Test two samplers in pool mode work independently."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)

        sampler1 = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng1,
            batch_size=32,
            pool_fraction=0.5,
            replacement_prob=0.3,
        )
        sampler2 = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng2,
            batch_size=32,
            pool_fraction=0.3,
            replacement_prob=0.5,
        )

        sampler1.init_sampler()
        sampler2.init_sampler()

        # Samplers should have independent pools
        for _ in range(30):
            r1 = sampler1.sample()
            r2 = sampler2.sample()
            assert r1["src_cell_data"].shape[0] == 32
            assert r2["src_cell_data"].shape[0] == 32

        # Each sampler should have tracked its own usage
        assert sampler1._pool_usage_count.sum() == 30
        assert sampler2._pool_usage_count.sum() == 30


# =============================================================================
# CombinedSampler Tests
# =============================================================================


class TestCombinedSamplerInit:
    """Test CombinedSampler initialization."""

    def test_init_basic(self, sample_grouped_distribution):
        """Test basic initialization with two samplers."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        rng_combined = np.random.default_rng(44)

        sampler1 = InMemorySampler(data=sample_grouped_distribution, rng=rng1, batch_size=64)
        sampler2 = InMemorySampler(data=sample_grouped_distribution, rng=rng2, batch_size=64)

        combined = CombinedSampler(
            samplers={"dataset1": sampler1, "dataset2": sampler2},
            rng=rng_combined,
        )

        assert combined._initialized is False
        assert combined.dataset_names == ["dataset1", "dataset2"]

    def test_init_uniform_weights_by_default(self, sample_grouped_distribution):
        """Test that uniform weights are used by default."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        rng3 = np.random.default_rng(44)
        rng_combined = np.random.default_rng(45)

        sampler1 = InMemorySampler(data=sample_grouped_distribution, rng=rng1, batch_size=64)
        sampler2 = InMemorySampler(data=sample_grouped_distribution, rng=rng2, batch_size=64)
        sampler3 = InMemorySampler(data=sample_grouped_distribution, rng=rng3, batch_size=64)

        combined = CombinedSampler(
            samplers={"a": sampler1, "b": sampler2, "c": sampler3},
            rng=rng_combined,
        )

        weights = combined.weights
        assert np.isclose(weights["a"], 1 / 3)
        assert np.isclose(weights["b"], 1 / 3)
        assert np.isclose(weights["c"], 1 / 3)

    def test_init_custom_weights(self, sample_grouped_distribution):
        """Test initialization with custom weights."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        rng_combined = np.random.default_rng(44)

        sampler1 = InMemorySampler(data=sample_grouped_distribution, rng=rng1, batch_size=64)
        sampler2 = InMemorySampler(data=sample_grouped_distribution, rng=rng2, batch_size=64)

        combined = CombinedSampler(
            samplers={"dataset1": sampler1, "dataset2": sampler2},
            rng=rng_combined,
            weights={"dataset1": 0.7, "dataset2": 0.3},
        )

        weights = combined.weights
        assert np.isclose(weights["dataset1"], 0.7)
        assert np.isclose(weights["dataset2"], 0.3)

    def test_init_weights_are_normalized(self, sample_grouped_distribution):
        """Test that weights are normalized to sum to 1."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        rng_combined = np.random.default_rng(44)

        sampler1 = InMemorySampler(data=sample_grouped_distribution, rng=rng1, batch_size=64)
        sampler2 = InMemorySampler(data=sample_grouped_distribution, rng=rng2, batch_size=64)

        # Weights that don't sum to 1
        combined = CombinedSampler(
            samplers={"dataset1": sampler1, "dataset2": sampler2},
            rng=rng_combined,
            weights={"dataset1": 7, "dataset2": 3},  # sum = 10
        )

        weights = combined.weights
        assert np.isclose(weights["dataset1"], 0.7)
        assert np.isclose(weights["dataset2"], 0.3)
        assert np.isclose(sum(weights.values()), 1.0)

    def test_init_empty_samplers_raises(self):
        """Test that empty samplers dict raises ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="must not be empty"):
            CombinedSampler(samplers={}, rng=rng)

    def test_init_weights_keys_mismatch_raises(self, sample_grouped_distribution):
        """Test that mismatched weights keys raises ValueError."""
        rng1 = np.random.default_rng(42)
        rng_combined = np.random.default_rng(43)

        sampler1 = InMemorySampler(data=sample_grouped_distribution, rng=rng1, batch_size=64)

        with pytest.raises(ValueError, match="must match samplers keys"):
            CombinedSampler(
                samplers={"dataset1": sampler1},
                rng=rng_combined,
                weights={"wrong_name": 1.0},
            )

    def test_init_negative_weights_raises(self, sample_grouped_distribution):
        """Test that negative weights raises ValueError."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        rng_combined = np.random.default_rng(44)

        sampler1 = InMemorySampler(data=sample_grouped_distribution, rng=rng1, batch_size=64)
        sampler2 = InMemorySampler(data=sample_grouped_distribution, rng=rng2, batch_size=64)

        with pytest.raises(ValueError, match="non-negative"):
            CombinedSampler(
                samplers={"dataset1": sampler1, "dataset2": sampler2},
                rng=rng_combined,
                weights={"dataset1": -0.5, "dataset2": 0.5},
            )

    def test_init_zero_weights_raises(self, sample_grouped_distribution):
        """Test that all-zero weights raises ValueError."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        rng_combined = np.random.default_rng(44)

        sampler1 = InMemorySampler(data=sample_grouped_distribution, rng=rng1, batch_size=64)
        sampler2 = InMemorySampler(data=sample_grouped_distribution, rng=rng2, batch_size=64)

        with pytest.raises(ValueError, match="positive value"):
            CombinedSampler(
                samplers={"dataset1": sampler1, "dataset2": sampler2},
                rng=rng_combined,
                weights={"dataset1": 0.0, "dataset2": 0.0},
            )


class TestCombinedSamplerInitialization:
    """Test CombinedSampler init_sampler method."""

    def test_init_sampler_initializes_all_underlying(self, sample_grouped_distribution):
        """Test that init_sampler initializes all underlying samplers."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        rng_combined = np.random.default_rng(44)

        sampler1 = InMemorySampler(data=sample_grouped_distribution, rng=rng1, batch_size=64)
        sampler2 = InMemorySampler(data=sample_grouped_distribution, rng=rng2, batch_size=64)

        combined = CombinedSampler(
            samplers={"dataset1": sampler1, "dataset2": sampler2},
            rng=rng_combined,
        )

        assert sampler1._initialized is False
        assert sampler2._initialized is False

        combined.init_sampler()

        assert combined._initialized is True
        assert sampler1._initialized is True
        assert sampler2._initialized is True

    def test_init_sampler_double_init_raises(self, sample_grouped_distribution):
        """Test that calling init_sampler twice raises ValueError."""
        rng1 = np.random.default_rng(42)
        rng_combined = np.random.default_rng(43)

        sampler1 = InMemorySampler(data=sample_grouped_distribution, rng=rng1, batch_size=64)

        combined = CombinedSampler(
            samplers={"dataset1": sampler1},
            rng=rng_combined,
        )

        combined.init_sampler()

        with pytest.raises(ValueError, match="already initialized"):
            combined.init_sampler()


class TestCombinedSamplerSampling:
    """Test CombinedSampler sampling functionality."""

    def test_sample_not_initialized_raises(self, sample_grouped_distribution):
        """Test that sampling without init raises ValueError."""
        rng1 = np.random.default_rng(42)
        rng_combined = np.random.default_rng(43)

        sampler1 = InMemorySampler(data=sample_grouped_distribution, rng=rng1, batch_size=64)

        combined = CombinedSampler(
            samplers={"dataset1": sampler1},
            rng=rng_combined,
        )

        with pytest.raises(ValueError, match="not initialized"):
            combined.sample()

    def test_sample_returns_correct_structure(self, sample_grouped_distribution):
        """Test that sample returns dict with expected keys including dataset_name."""
        rng1 = np.random.default_rng(42)
        rng_combined = np.random.default_rng(43)

        sampler1 = InMemorySampler(data=sample_grouped_distribution, rng=rng1, batch_size=64)

        combined = CombinedSampler(
            samplers={"my_dataset": sampler1},
            rng=rng_combined,
        )
        combined.init_sampler()

        result = combined.sample()

        assert isinstance(result, dict)
        assert "src_cell_data" in result
        assert "tgt_cell_data" in result
        assert "condition" in result
        assert "dataset_name" in result
        assert result["dataset_name"] == "my_dataset"

    def test_sample_returns_correct_dataset_name(self, sample_grouped_distribution):
        """Test that sample returns the correct dataset name."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        rng_combined = np.random.default_rng(44)

        sampler1 = InMemorySampler(data=sample_grouped_distribution, rng=rng1, batch_size=64)
        sampler2 = InMemorySampler(data=sample_grouped_distribution, rng=rng2, batch_size=64)

        combined = CombinedSampler(
            samplers={"alpha": sampler1, "beta": sampler2},
            rng=rng_combined,
        )
        combined.init_sampler()

        for _ in range(20):
            result = combined.sample()
            assert result["dataset_name"] in ["alpha", "beta"]

    def test_sample_respects_weights_distribution(self, sample_grouped_distribution):
        """Test that sampling approximately follows the weight distribution."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        rng_combined = np.random.default_rng(44)

        sampler1 = InMemorySampler(data=sample_grouped_distribution, rng=rng1, batch_size=64)
        sampler2 = InMemorySampler(data=sample_grouped_distribution, rng=rng2, batch_size=64)

        combined = CombinedSampler(
            samplers={"heavy": sampler1, "light": sampler2},
            rng=rng_combined,
            weights={"heavy": 0.9, "light": 0.1},
        )
        combined.init_sampler()

        # Sample many times and count
        counts = {"heavy": 0, "light": 0}
        n_samples = 1000
        for _ in range(n_samples):
            result = combined.sample()
            counts[result["dataset_name"]] += 1

        # Check distribution is approximately correct (with tolerance)
        heavy_ratio = counts["heavy"] / n_samples
        light_ratio = counts["light"] / n_samples

        assert 0.85 < heavy_ratio < 0.95, f"Expected ~0.9, got {heavy_ratio}"
        assert 0.05 < light_ratio < 0.15, f"Expected ~0.1, got {light_ratio}"

    def test_sample_with_single_weight_1(self, sample_grouped_distribution):
        """Test sampling when one weight is 1 and others are 0."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        rng_combined = np.random.default_rng(44)

        sampler1 = InMemorySampler(data=sample_grouped_distribution, rng=rng1, batch_size=64)
        sampler2 = InMemorySampler(data=sample_grouped_distribution, rng=rng2, batch_size=64)

        combined = CombinedSampler(
            samplers={"only_this": sampler1, "never_this": sampler2},
            rng=rng_combined,
            weights={"only_this": 1.0, "never_this": 0.0},
        )
        combined.init_sampler()

        # All samples should come from "only_this"
        for _ in range(100):
            result = combined.sample()
            assert result["dataset_name"] == "only_this"

    def test_sample_batch_sizes_vary_by_underlying_sampler(self, sample_grouped_distribution):
        """Test that batch sizes come from the underlying samplers."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        rng_combined = np.random.default_rng(44)

        sampler1 = InMemorySampler(data=sample_grouped_distribution, rng=rng1, batch_size=32)
        sampler2 = InMemorySampler(data=sample_grouped_distribution, rng=rng2, batch_size=128)

        combined = CombinedSampler(
            samplers={"small": sampler1, "large": sampler2},
            rng=rng_combined,
        )
        combined.init_sampler()

        for _ in range(50):
            result = combined.sample()
            if result["dataset_name"] == "small":
                assert result["src_cell_data"].shape[0] == 32
            else:
                assert result["src_cell_data"].shape[0] == 128


class TestCombinedSamplerWithMixedSamplerTypes:
    """Test CombinedSampler with different sampler types."""

    def test_mixed_inmemory_and_reservoir_samplers(self, sample_grouped_distribution):
        """Test combining InMemorySampler and ReservoirSampler."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        rng_combined = np.random.default_rng(44)

        inmem_sampler = InMemorySampler(data=sample_grouped_distribution, rng=rng1, batch_size=64)
        reservoir_sampler = ReservoirSampler(
            data=sample_grouped_distribution,
            rng=rng2,
            batch_size=64,
            pool_fraction=0.5,
            replacement_prob=0.1,
        )

        combined = CombinedSampler(
            samplers={"inmem": inmem_sampler, "reservoir": reservoir_sampler},
            rng=rng_combined,
        )
        combined.init_sampler()

        for _ in range(30):
            result = combined.sample()
            assert result["dataset_name"] in ["inmem", "reservoir"]
            assert result["src_cell_data"].shape[0] == 64


class TestCombinedSamplerWithSplitData:
    """Test CombinedSampler with split data."""

    def test_combined_sampler_with_train_val_splits(self, sample_grouped_distribution):
        """Test combining samplers from train and val splits."""
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
        val_data = splits["val"]

        rng_train = np.random.default_rng(42)
        rng_val = np.random.default_rng(43)
        rng_combined = np.random.default_rng(44)

        train_sampler = InMemorySampler(data=train_data, rng=rng_train, batch_size=32)
        val_sampler = InMemorySampler(data=val_data, rng=rng_val, batch_size=32)

        combined = CombinedSampler(
            samplers={"train": train_sampler, "val": val_sampler},
            rng=rng_combined,
            weights={"train": 0.8, "val": 0.2},
        )
        combined.init_sampler()

        for _ in range(20):
            result = combined.sample()
            assert result["dataset_name"] in ["train", "val"]
            assert result["src_cell_data"].shape[0] == 32


class TestCombinedSamplerDeterminism:
    """Test CombinedSampler determinism."""

    def test_deterministic_with_same_seeds(self, sample_grouped_distribution):
        """Test that same seeds produce same results."""

        def create_combined(seed_offset):
            rng1 = np.random.default_rng(42 + seed_offset)
            rng2 = np.random.default_rng(43 + seed_offset)
            rng_combined = np.random.default_rng(44 + seed_offset)

            sampler1 = InMemorySampler(data=sample_grouped_distribution, rng=rng1, batch_size=64)
            sampler2 = InMemorySampler(data=sample_grouped_distribution, rng=rng2, batch_size=64)

            combined = CombinedSampler(
                samplers={"a": sampler1, "b": sampler2},
                rng=rng_combined,
            )
            combined.init_sampler()
            return combined

        combined1 = create_combined(0)
        combined2 = create_combined(0)

        for _ in range(10):
            result1 = combined1.sample()
            result2 = combined2.sample()

            assert result1["dataset_name"] == result2["dataset_name"]
            np.testing.assert_array_equal(result1["src_cell_data"], result2["src_cell_data"])
            np.testing.assert_array_equal(result1["tgt_cell_data"], result2["tgt_cell_data"])


class TestCombinedSamplerManyDatasets:
    """Test CombinedSampler with many datasets."""

    def test_many_datasets(self, sample_grouped_distribution):
        """Test combining many samplers."""
        n_datasets = 10
        samplers = {}
        weights = {}

        for i in range(n_datasets):
            rng = np.random.default_rng(42 + i)
            samplers[f"dataset_{i}"] = InMemorySampler(data=sample_grouped_distribution, rng=rng, batch_size=32)
            weights[f"dataset_{i}"] = i + 1  # Increasing weights

        rng_combined = np.random.default_rng(100)
        combined = CombinedSampler(samplers=samplers, rng=rng_combined, weights=weights)
        combined.init_sampler()

        # Sample many times
        counts = {f"dataset_{i}": 0 for i in range(n_datasets)}
        for _ in range(500):
            result = combined.sample()
            counts[result["dataset_name"]] += 1

        # Higher-weighted datasets should have more samples
        for i in range(n_datasets - 1):
            # dataset_i should have fewer samples than dataset_{i+1}
            # (with some tolerance for randomness)
            assert counts[f"dataset_{i}"] < counts[f"dataset_{i + 1}"] * 2
