import math
import os
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np

from scaleflow.data._data import (
    GroupedDistribution,
)

__all__ = [
    "CombinedSampler",
    "InMemorySampler",
    "PredictionSampler",
    "ReservoirSampler",
    "SamplerABC",
    "ValidationSampler",
]


class SamplerABC(ABC):
    @abstractmethod
    def sample(self, *args, **kwargs) -> dict[str, Any]:
        pass

    @abstractmethod
    def init_sampler(self, *args, **kwargs) -> None:
        pass

    @property
    @abstractmethod
    def initialized(self) -> bool:
        pass


class InMemorySampler(SamplerABC):
    """Simple in-memory data sampler that caches all data.

    This sampler loads all source and target distributions into memory
    during initialization. Best for smaller datasets that fit in memory.

    Parameters
    ----------
    data
        The training data.
    rng
        Random number generator to use for all sampling operations.
    batch_size
        The batch size.
    """

    def __init__(
        self,
        data: GroupedDistribution,
        rng: np.random.Generator,
        batch_size: int = 1024,
    ) -> None:
        self.batch_size = batch_size
        self.n_source_dists = len(data.data.src_data)
        self.n_target_dists = len(data.data.tgt_data)
        self._data = data
        self._rng = rng
        self._initialized = False
        self._cached_srcs: dict[int, np.ndarray] = {}
        self._cached_tgts: dict[int, np.ndarray] = {}

    def init_sampler(self) -> None:
        """Initialize the sampler by loading all data into memory."""
        if self._initialized:
            raise ValueError("Sampler already initialized. Call init_sampler() only once.")

        # Load data into memory if it's lazy (zarr arrays)
        if not self._data.data.is_in_memory:
            self._data.data.to_memory()

        # Reference the in-memory arrays directly
        self._cached_srcs = {int(k): v for k, v in self._data.data.src_data.items()}
        self._cached_tgts = {int(k): v for k, v in self._data.data.tgt_data.items()}

        self._initialized = True
        return None

    @property
    def initialized(self) -> bool:
        return self._initialized

    def sample(self) -> dict[str, Any]:
        """Sample a batch for gene expression (flow matching) task.

        Returns
        -------
        Dictionary with source cells, target cells, condition, and task type
        """
        if not self._initialized:
            raise ValueError("Sampler not initialized. Call init_sampler() first.")

        # Sample source distribution
        src_keys = list(self._cached_srcs.keys())
        source_dist_idx = self._rng.choice(src_keys)

        # Sample target distribution from those mapped to this source
        tgt_indices = self._data.data.src_to_tgt_dist_map[source_dist_idx]
        target_dist_idx = self._rng.choice(tgt_indices)

        # Sample cells
        src_arr = self._cached_srcs[source_dist_idx]
        src_idxs = self._rng.choice(src_arr.shape[0], size=self.batch_size, replace=True)
        source_batch = src_arr[src_idxs]

        tgt_arr = self._cached_tgts[target_dist_idx]
        tgt_idxs = self._rng.choice(tgt_arr.shape[0], size=self.batch_size, replace=True)
        target_batch = tgt_arr[tgt_idxs]

        # Get condition
        cond_dict = self._data.data.conditions[target_dist_idx]

        return {"src_cell_data": source_batch, "tgt_cell_data": target_batch, "condition": cond_dict}


class CombinedSampler(SamplerABC):
    """Sampler that combines multiple samplers with configurable sampling weights.

    This allows sampling from multiple datasets with different probabilities,
    useful for multi-dataset training scenarios.

    Parameters
    ----------
    samplers
        Dictionary mapping dataset names to their samplers.
    rng
        Random number generator for selecting which sampler to use.
    weights
        Optional dictionary mapping dataset names to sampling weights.
        If None, uniform weights are used. Weights are normalized to sum to 1.

    Examples
    --------
    >>> sampler1 = InMemorySampler(data1, rng1, batch_size=64)
    >>> sampler2 = InMemorySampler(data2, rng2, batch_size=64)
    >>> combined = CombinedSampler(
    ...     samplers={"dataset1": sampler1, "dataset2": sampler2},
    ...     rng=np.random.default_rng(42),
    ...     weights={"dataset1": 0.7, "dataset2": 0.3},
    ... )
    >>> combined.init_sampler()
    >>> batch = combined.sample()
    >>> batch["dataset_name"]  # Returns "dataset1" or "dataset2"
    """

    def __init__(
        self,
        samplers: dict[str, SamplerABC],
        rng: np.random.Generator,
        weights: dict[str, float] | None = None,
    ) -> None:
        if not samplers:
            raise ValueError("samplers dict must not be empty")

        self._samplers = samplers
        self._rng = rng
        self._dataset_names = list(samplers.keys())
        self._initialized = False

        # Normalize weights
        if weights is None:
            # Uniform weights
            n = len(self._dataset_names)
            self._weights = np.array([1.0 / n] * n)
        else:
            # Validate weights keys match samplers keys
            if set(weights.keys()) != set(self._dataset_names):
                raise ValueError(
                    f"weights keys {set(weights.keys())} must match samplers keys {set(self._dataset_names)}"
                )
            # Normalize to sum to 1
            weight_values = np.array([weights[name] for name in self._dataset_names])
            if np.any(weight_values < 0):
                raise ValueError("weights must be non-negative")
            if weight_values.sum() == 0:
                raise ValueError("weights must sum to a positive value")
            self._weights = weight_values / weight_values.sum()

    def init_sampler(self) -> None:
        """Initialize all underlying samplers."""
        if self._initialized:
            raise ValueError("Sampler already initialized. Call init_sampler() only once.")

        for _name, sampler in self._samplers.items():
            sampler.init_sampler()

        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def sample(self) -> dict[str, Any]:
        """Sample from one of the underlying samplers based on weights.

        Returns
        -------
        Dictionary with source cells, target cells, condition, and dataset_name
        """
        if not self._initialized:
            raise ValueError("Sampler not initialized. Call init_sampler() first.")

        # Select dataset based on weights
        dataset_idx = self._rng.choice(len(self._dataset_names), p=self._weights)
        dataset_name = self._dataset_names[dataset_idx]

        # Sample from selected sampler
        batch = self._samplers[dataset_name].sample()

        # Add dataset name to batch
        batch["dataset_name"] = dataset_name

        return batch

    @property
    def dataset_names(self) -> list[str]:
        """Return list of dataset names."""
        return self._dataset_names.copy()

    @property
    def weights(self) -> dict[str, float]:
        """Return dictionary of dataset weights."""
        return {name: float(w) for name, w in zip(self._dataset_names, self._weights, strict=False)}


class ValidationSampler:
    """Sampler for validation that returns data organized by condition key.

    Unlike training samplers that run infinitely, this returns conditions
    for finite validation. The sample() method returns data structured as:
    {"source": dict, "condition": dict, "target": dict}
    where each dict maps condition_key (tuple) → data.

    This matches the old CellFlow's ValidationSampler behavior and enables
    efficient per-condition prediction using jax.tree.map.

    Parameters
    ----------
    data
        The validation data (GroupedDistribution).
    n_conditions_on_log_iteration
        Max number of conditions to return during log iteration.
        If None, return all.
    n_conditions_on_train_end
        Max number of conditions to return at end of training.
        If None, return all.
    seed
        Random seed for sampling conditions.

    Examples
    --------
    >>> val_sampler = ValidationSampler(val_gd, n_conditions_on_log_iteration=10)
    >>> val_sampler.init_sampler()
    >>> batch = val_sampler.sample(mode="on_log_iteration")
    >>> batch["source"]  # dict mapping condition_key -> source cells
    >>> batch["condition"]  # dict mapping condition_key -> condition embeddings
    >>> batch["target"]  # dict mapping condition_key -> target cells
    """

    def __init__(
        self,
        data: GroupedDistribution,
        n_conditions_on_log_iteration: int | None = None,
        n_conditions_on_train_end: int | None = None,
        seed: int = 0,
    ) -> None:
        self._data = data
        n_total = len(data.data.conditions)
        self._n_conditions_on_log_iteration = (
            n_conditions_on_log_iteration if n_conditions_on_log_iteration is not None else n_total
        )
        self._n_conditions_on_train_end = (
            n_conditions_on_train_end if n_conditions_on_train_end is not None else n_total
        )
        self._rng = np.random.default_rng(seed)
        self._initialized = False

        # Build reverse mapping: tgt_dist_idx -> src_dist_idx
        self._tgt_to_src: dict[int, int] = {}
        for src_idx, tgt_idxs in data.data.src_to_tgt_dist_map.items():
            for tgt_idx in tgt_idxs:
                self._tgt_to_src[tgt_idx] = src_idx

    def init_sampler(self) -> None:
        """Initialize the sampler by loading data into memory if needed."""
        if self._initialized:
            raise ValueError("Sampler already initialized. Call init_sampler() only once.")

        # Load data into memory if it's lazy (zarr arrays)
        if not self._data.data.is_in_memory:
            self._data.data.to_memory()

        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _get_key(self, tgt_idx: int) -> tuple[str, ...]:
        """Get the condition key for a target distribution index.

        Returns a tuple of labels from tgt_dist_idx_to_labels annotation.
        """
        labels = self._data.annotation.tgt_dist_idx_to_labels.get(tgt_idx)
        if labels is not None:
            # Convert to tuple of strings
            if isinstance(labels, (list, np.ndarray)):
                return tuple(str(lbl) for lbl in labels)
            return (str(labels),)
        # Fallback to string index if no labels
        return (str(tgt_idx),)

    def sample(
        self, mode: str = "on_log_iteration"
    ) -> dict[str, dict[tuple[str, ...], Any]]:
        """Sample validation data organized by condition key.

        Parameters
        ----------
        mode
            Sampling mode. Either ``"on_log_iteration"`` or ``"on_train_end"``.

        Returns
        -------
        Dictionary with keys "source", "condition", "target", each mapping
        condition_key (tuple) -> data arrays.
        """
        if not self._initialized:
            raise ValueError("Sampler not initialized. Call init_sampler() first.")

        # Determine number of conditions based on mode
        if mode == "on_log_iteration":
            n_conditions = self._n_conditions_on_log_iteration
        elif mode == "on_train_end":
            n_conditions = self._n_conditions_on_train_end
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'on_log_iteration' or 'on_train_end'.")

        # Get all target distribution indices (conditions)
        all_tgt_indices = list(self._data.data.conditions.keys())
        n_total = len(all_tgt_indices)

        # Sample a subset of conditions if needed
        if n_conditions < n_total:
            selected_indices = self._rng.choice(
                all_tgt_indices, size=n_conditions, replace=False
            ).tolist()
        else:
            selected_indices = all_tgt_indices

        source_dict: dict[tuple[str, ...], Any] = {}
        condition_dict: dict[tuple[str, ...], dict[str, Any]] = {}
        target_dict: dict[tuple[str, ...], Any] = {}

        for tgt_idx in selected_indices:
            # Get condition key as tuple
            cond_key = self._get_key(tgt_idx)

            # Get source distribution index for this target
            src_idx = self._tgt_to_src.get(tgt_idx)
            if src_idx is None:
                continue

            # Get source cells
            source_dict[cond_key] = self._data.data.src_data[src_idx]

            # Get target cells
            target_dict[cond_key] = self._data.data.tgt_data[tgt_idx]

            # Get condition embeddings
            condition_dict[cond_key] = self._data.data.conditions[tgt_idx]

        return {"source": source_dict, "condition": condition_dict, "target": target_dict}

    @property
    def n_conditions(self) -> int:
        """Number of conditions (target distributions)."""
        return len(self._data.data.conditions)

    @property
    def n_conditions_on_log_iteration(self) -> int:
        """Number of conditions sampled during log iteration."""
        return self._n_conditions_on_log_iteration

    @property
    def n_conditions_on_train_end(self) -> int:
        """Number of conditions sampled at end of training."""
        return self._n_conditions_on_train_end

    @property
    def data(self) -> GroupedDistribution:
        """The validation data."""
        return self._data


class PredictionSampler:
    """Sampler for prediction that returns all conditions with source data.

    Unlike ValidationSampler, this returns all conditions without targets,
    suitable for making predictions on new/unseen perturbations.

    The sample() method returns data structured as:
    {"source": dict, "condition": dict}
    where each dict maps condition_key (tuple) → data.

    Parameters
    ----------
    data
        The prediction data (GroupedDistribution).

    Examples
    --------
    >>> pred_sampler = PredictionSampler(pred_gd)
    >>> pred_sampler.init_sampler()
    >>> batch = pred_sampler.sample()
    >>> batch["source"]  # dict mapping condition_key -> source cells
    >>> batch["condition"]  # dict mapping condition_key -> condition embeddings
    """

    def __init__(
        self,
        data: GroupedDistribution,
    ) -> None:
        self._data = data
        self._initialized = False

        # Build reverse mapping: tgt_dist_idx -> src_dist_idx
        self._tgt_to_src: dict[int, int] = {}
        for src_idx, tgt_idxs in data.data.src_to_tgt_dist_map.items():
            for tgt_idx in tgt_idxs:
                self._tgt_to_src[tgt_idx] = src_idx

    def init_sampler(self) -> None:
        """Initialize the sampler by loading data into memory if needed."""
        if self._initialized:
            raise ValueError("Sampler already initialized. Call init_sampler() only once.")

        # Load data into memory if it's lazy (zarr arrays)
        if not self._data.data.is_in_memory:
            self._data.data.to_memory()

        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _get_key(self, tgt_idx: int) -> tuple[str, ...]:
        """Get the condition key for a target distribution index.

        Returns a tuple of labels from tgt_dist_idx_to_labels annotation.
        """
        labels = self._data.annotation.tgt_dist_idx_to_labels.get(tgt_idx)
        if labels is not None:
            # Convert to tuple of strings
            if isinstance(labels, (list, np.ndarray)):
                return tuple(str(lbl) for lbl in labels)
            return (str(labels),)
        # Fallback to string index if no labels
        return (str(tgt_idx),)

    def sample(self) -> dict[str, dict[tuple[str, ...], Any]]:
        """Sample prediction data organized by condition key.

        Returns
        -------
        Dictionary with keys "source" and "condition", each mapping
        condition_key (tuple) -> data arrays.
        """
        if not self._initialized:
            raise ValueError("Sampler not initialized. Call init_sampler() first.")

        # Get all target distribution indices (conditions)
        all_tgt_indices = list(self._data.data.conditions.keys())

        source_dict: dict[tuple[str, ...], Any] = {}
        condition_dict: dict[tuple[str, ...], dict[str, Any]] = {}

        for tgt_idx in all_tgt_indices:
            # Get condition key as tuple
            cond_key = self._get_key(tgt_idx)

            # Get source distribution index for this target
            src_idx = self._tgt_to_src.get(tgt_idx)
            if src_idx is None:
                continue

            # Get source cells
            source_dict[cond_key] = self._data.data.src_data[src_idx]

            # Get condition embeddings
            condition_dict[cond_key] = self._data.data.conditions[tgt_idx]

        return {"source": source_dict, "condition": condition_dict}

    @property
    def n_conditions(self) -> int:
        """Number of conditions (target distributions)."""
        return len(self._data.data.conditions)

    @property
    def data(self) -> GroupedDistribution:
        """The prediction data."""
        return self._data


class ReservoirSampler(SamplerABC):
    """Data sampler with gradual pool replacement using reservoir sampling.

    This approach replaces pool elements one by one rather than refreshing
    the entire pool, providing better cache locality while maintaining
    reasonable randomness. Use `InMemorySampler` for simpler use cases
    where all data fits in memory.

    Parameters
    ----------
    data
        The training data.
    rng
        Random number generator to use for all sampling operations.
    batch_size
        The batch size.
    pool_fraction
        Fraction of source distributions to cache (0 < pool_fraction < 1).
        For caching all sources, use `InMemorySampler` instead.
    replacement_prob
        Probability of replacing a pool element after each sample.
        Lower values = longer cache retention, less randomness.
        Higher values = faster cache turnover, more randomness.
    """

    def __init__(
        self,
        data: GroupedDistribution,
        rng: np.random.Generator,
        batch_size: int = 1024,
        pool_fraction: float = 0.5,
        replacement_prob: float = 0.1,
    ) -> None:
        # Validate pool_fraction
        if pool_fraction is None or pool_fraction >= 1.0:
            raise ValueError("pool_fraction must be in (0, 1). Use InMemorySampler for caching all data.")
        if not (0 < pool_fraction < 1):
            raise ValueError("pool_fraction must be in (0, 1). Use InMemorySampler for caching all data.")

        self.batch_size = batch_size
        self.n_source_dists = len(data.data.src_data)
        self.n_target_dists = len(data.data.tgt_data)
        self._data = data
        self._rng = rng
        self._pool_fraction = pool_fraction
        self._replacement_prob = replacement_prob
        self._pool_size = math.ceil(pool_fraction * self.n_source_dists)

        self._pool_usage_count = np.zeros(self.n_source_dists, dtype=int)
        self._initialized = False
        self._src_idx_pool = None

        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2)  # TODO: avoid magic numbers
        self._pending_replacements: dict[int, dict[str, Any]] = {}

    def init_sampler(self) -> None:
        """Initialize the sampler by loading data into cache.

        Must be called before sampling. Uses the rng provided in the constructor.
        """
        if self._initialized:
            raise ValueError("Sampler already initialized. Call init_sampler() only once.")
        self._init_src_idx_pool()
        self._init_cache_pool_elements()
        self._initialized = True
        return None

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _init_src_idx_pool(self) -> None:
        src_indices = np.array(list(self._data.data.src_data.keys()))
        self._src_idx_pool = self._rng.choice(src_indices, size=self._pool_size, replace=False)
        return None

    def sample(self) -> dict[str, Any]:
        """Sample a batch for gene expression (flow matching) task.

        Returns
        -------
        Dictionary with source cells, target cells, condition, and task type
        """
        source_dist_idx = self._sample_source_dist_idx()
        target_dist_idx = self._sample_target_dist_idx(source_dist_idx)
        source_batch = self._sample_source_cells(source_dist_idx)
        target_batch = self._sample_target_cells(target_dist_idx)

        # Conditions are stored as nested dicts: {col_name: array}
        cond_dict = self._data.data.conditions[target_dist_idx]

        res = {"src_cell_data": source_batch, "tgt_cell_data": target_batch, "condition": cond_dict}
        return res

    def _load_targets_parallel(self, tgt_indices):
        """Load multiple target distributions in parallel."""

        def _load_tgt(j: int):
            return j, self._data.data.tgt_data[j][...]

        max_workers = min(32, (os.cpu_count() or 4))  # TODO: avoid magic numbers
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_load_tgt, tgt_indices))
        return dict(results)

    def _init_cache_pool_elements(self) -> None:
        with self._lock:
            self._cached_srcs = {i: self._data.data.src_data[i][...] for i in self._src_idx_pool}

        tgt_indices = sorted({int(j) for i in self._src_idx_pool for j in self._data.data.src_to_tgt_dist_map[i]})

        with self._lock:
            self._cached_tgts = self._load_targets_parallel(tgt_indices)

        return None

    def _sample_target_dist_idx(self, source_dist_idx: int) -> int:
        """Sample a target distribution index given the source distribution index."""
        return self._rng.choice(self._data.data.src_to_tgt_dist_map[source_dist_idx])

    def _sample_source_dist_idx(self) -> int:
        """Sample a source distribution index with gradual pool replacement."""
        if not self._initialized:
            raise ValueError("Sampler not initialized. Call init_sampler() first.")
        self._apply_ready_replacements()
        # Sample from current pool
        with self._lock:
            source_idx = self._rng.choice(sorted(self._cached_srcs.keys()))

        # Increment usage count for monitoring
        self._pool_usage_count[source_idx] = self._pool_usage_count[source_idx] + 1

        # Gradually replace elements based on replacement probability (schedule only)
        if self._rng.random() < self._replacement_prob:
            self._schedule_replacement()

        return source_idx

    def _schedule_replacement(self):
        # Get usage counts for indices in the pool
        pool_indices = self._src_idx_pool.tolist()
        usage_counts = np.array([self._pool_usage_count[idx] for idx in pool_indices])

        if len(usage_counts) == 0:
            return

        max_usage = usage_counts.max()
        most_used_weight = (usage_counts == max_usage).astype(float)
        if most_used_weight.sum() == 0:
            return
        most_used_weight /= most_used_weight.sum()
        replaced_pool_slot = self._rng.choice(len(pool_indices), p=most_used_weight)
        replaced_pool_idx = pool_indices[replaced_pool_slot]

        with self._lock:
            # If there's already a pending replacement for this pool slot, skip
            if replaced_pool_slot in self._pending_replacements:
                return

            # Find all available source indices (not currently in pool)
            all_src_indices = list(self._data.data.src_data.keys())
            pool_set = set(pool_indices)
            available_indices = [idx for idx in all_src_indices if idx not in pool_set]

            if not available_indices:
                return

            # Get usage counts for available indices
            available_usage = np.array([self._pool_usage_count[idx] for idx in available_indices])
            min_usage = available_usage.min()
            least_used_weight = (available_usage == min_usage).astype(float)
            if least_used_weight.sum() == 0:
                return
            least_used_weight /= least_used_weight.sum()
            new_idx_position = self._rng.choice(len(available_indices), p=least_used_weight)
            new_pool_idx = available_indices[new_idx_position]

            # Kick off background load for new indices
            fut: Future = self._executor.submit(self._load_new_cache, new_pool_idx)
            self._pending_replacements[replaced_pool_slot] = {
                "old": replaced_pool_idx,
                "new": new_pool_idx,
                "future": fut,
            }
            print(f"scheduled replacement of {replaced_pool_idx} with {new_pool_idx} (slot {replaced_pool_slot})")

    def _apply_ready_replacements(self):
        """Apply any finished background loads; non-blocking."""
        to_apply: list[int] = []
        with self._lock:
            for slot, info in self._pending_replacements.items():
                fut: Future = info["future"]
                if fut.done() and not fut.cancelled():
                    to_apply.append(slot)

        for slot in to_apply:
            with self._lock:
                info = self._pending_replacements.pop(slot, None)
                if info is None:
                    continue
                old_idx = int(info["old"])
                new_idx = int(info["new"])
                fut: Future = info["future"]
                try:
                    prepared = fut.result(timeout=0)  # already done
                except Exception as e:
                    print(f"background load failed for {new_idx}: {e}")
                    continue

                # Swap pool index
                self._src_idx_pool[slot] = new_idx

                # Add new entries first
                self._cached_srcs[new_idx] = prepared["src"]
                for k, arr in prepared["tgts"].items():
                    self._cached_tgts[k] = arr

                # Remove old entries
                if old_idx in self._cached_srcs:
                    del self._cached_srcs[old_idx]
                for k in self._data.data.src_to_tgt_dist_map[old_idx]:
                    if k in self._cached_tgts:
                        del self._cached_tgts[k]

                print(f"applied replacement: {old_idx} -> {new_idx} (slot {slot})")

    def _load_new_cache(self, src_idx: int) -> dict[str, Any]:
        """Load new src and corresponding tgt arrays in the background."""
        src_arr = self._data.data.src_data[src_idx][...]
        tgt_dict = {k: self._data.data.tgt_data[k][...] for k in self._data.data.src_to_tgt_dist_map[src_idx]}
        return {"src": src_arr, "tgts": tgt_dict}

    def _sample_source_cells(self, source_dist_idx: int) -> np.ndarray:
        with self._lock:
            arr = self._cached_srcs[source_dist_idx]
        idxs = self._rng.choice(arr.shape[0], size=self.batch_size, replace=True)
        return arr[idxs]

    def _sample_target_cells(self, target_dist_idx: int) -> np.ndarray:
        with self._lock:
            arr = self._cached_tgts[target_dist_idx]
        idxs = self._rng.choice(arr.shape[0], size=self.batch_size, replace=True)
        return arr[idxs]
