import abc
import math
from typing import Any, Literal

import numpy as np
import tqdm
import os
import threading
from concurrent.futures import ThreadPoolExecutor, Future

from scaleflow.data._data import (
    PredictionData,
    TrainingData,
    ValidationData,
    MappedCellData,
    GroupedDistribution,
)

__all__ = [
    "TrainSampler",
    "ValidationSampler",
    "PredictionSampler",
    "ReservoirSampler",
]


class TrainSampler:
    """Data sampler for :class:`~scaleflow.data.TrainingData`.

    Parameters
    ----------
    data
        The training data.
    batch_size
        The batch size.

    """

    def __init__(self, data: TrainingData, batch_size: int = 1024):
        self._data = data
        self._data_idcs = np.arange(data.cell_data.shape[0])
        self.batch_size = batch_size
        self.n_source_dists = data.n_controls
        self.n_target_dists = data.n_perturbations

        self._control_to_perturbation_keys = sorted(data.control_to_perturbation.keys())
        self._has_condition_data = data.condition_data is not None

    def _sample_target_dist_idx(self, rng, source_dist_idx: int) -> int:
        """Sample a target distribution index given the source distribution index."""
        return rng.choice(self._data.control_to_perturbation[source_dist_idx])

    def _sample_source_dist_idx(self, rng) -> int:
        """Sample a source distribution index."""
        return rng.choice(self.n_source_dists)

    def _get_embeddings(self, idx, condition_data) -> dict[str, np.ndarray]:
        """Get embeddings for a given index."""
        result = {}
        for key, arr in condition_data.items():
            result[key] = np.expand_dims(arr[idx], 0)
        return result

    def _sample_from_mask(self, rng, mask) -> np.ndarray:
        """Sample indices according to a mask."""
        # Convert mask to probability distribution
        valid_indices = np.where(mask)[0]
        # Handle case with no valid indices (should not happen in practice)
        if len(valid_indices) == 0:
            raise ValueError("No valid indices found in the mask")

        # Sample from valid indices with equal probability
        batch_idcs = rng.choice(valid_indices, self.batch_size, replace=True)
        return batch_idcs

    def _get_source_cells_mask(self, source_dist_idx: int) -> np.ndarray:
        return self._data.split_covariates_mask == source_dist_idx

    def _get_target_cells_mask(self, source_dist_idx: int, target_dist_idx: int) -> np.ndarray:
        return self._data.perturbation_covariates_mask == target_dist_idx

    def _sample_source_batch_idcs(self, rng, source_dist_idx: int) -> dict[str, Any]:
        source_cells_mask = self._get_source_cells_mask(source_dist_idx)
        source_batch_idcs = self._sample_from_mask(rng, source_cells_mask)
        return source_batch_idcs

    def _sample_target_batch_idcs(self, rng, source_dist_idx: int, target_dist_idx: int) -> dict[str, Any]:
        target_cells_mask = self._get_target_cells_mask(source_dist_idx, target_dist_idx)
        target_batch_idcs = self._sample_from_mask(rng, target_cells_mask)
        return target_batch_idcs

    def _sample_source_cells(self, rng, source_dist_idx: int) -> np.ndarray:
        source_cells_mask = self._get_source_cells_mask(source_dist_idx)
        source_batch_idcs = self._sample_from_mask(rng, source_cells_mask)
        return self._data.cell_data[source_batch_idcs]

    def _sample_target_cells(self, rng, source_dist_idx: int, target_dist_idx: int) -> np.ndarray:
        target_cells_mask = self._get_target_cells_mask(source_dist_idx, target_dist_idx)
        target_batch_idcs = self._sample_from_mask(rng, target_cells_mask)
        return self._data.cell_data[target_batch_idcs]

    def sample(self, rng) -> dict[str, Any]:
        """Sample a batch of data.

        Parameters
        ----------
        seed : int, optional
            Random seed

        Returns
        -------
        Dictionary with source and target data
        """
        # Sample source and target
        source_dist_idx = self._sample_source_dist_idx(rng)
        target_dist_idx = self._sample_target_dist_idx(rng, source_dist_idx)

        # Sample source and target cells
        source_batch = self._sample_source_cells(rng, source_dist_idx)
        target_batch = self._sample_target_cells(rng, source_dist_idx, target_dist_idx)

        res = {"src_cell_data": source_batch, "tgt_cell_data": target_batch}
        if self._has_condition_data:
            condition_batch = self._get_embeddings(target_dist_idx, self._data.condition_data)
            res["condition"] = condition_batch
        return res

    @property
    def data(self) -> TrainingData:
        """The training data."""
        return self._data


class ReservoirSampler(TrainSampler):
    """Data sampler with gradual pool replacement using reservoir sampling.

    This approach replaces pool elements one by one rather than refreshing
    the entire pool, providing better cache locality while maintaining
    reasonable randomness.

    Parameters
    ----------
    data
        The training data.
    batch_size
        The batch size.
    pool_fraction
        Fraction of source distributions to cache (0 < pool_fraction <= 1).
        If 1.0, all sources are cached and no replacement is performed.
    replacement_prob
        Probability of replacing a pool element after each sample.
        Lower values = longer cache retention, less randomness.
        Higher values = faster cache turnover, more randomness.
    """

    def __init__(
        self,
        data: GroupedDistribution,
        batch_size: int = 1024,
        pool_fraction: float = 0.1,
        replacement_prob: float = 0.01,
    ):
        self.batch_size = batch_size
        self.n_source_dists = len(data.data.src_data)
        self.n_target_dists = len(data.data.tgt_data)
        self._data = data

        self._control_to_perturbation_keys = sorted(data.data.src_to_tgt_dist_map.keys())
        self._has_condition_data = data.condition_data is not None

        # Compute pool size from fraction
        if not (0 < pool_fraction <= 1):
            raise ValueError("pool_fraction must be in (0, 1].")
        self._pool_fraction = pool_fraction
        self._pool_size = math.ceil(pool_fraction * self.n_source_dists)
        self._replacement_prob = replacement_prob
        self._pool_usage_count = np.zeros(self.n_source_dists, dtype=int)
        self._initialized = False

        # If caching everything, skip concurrency and replacement logic
        self._cache_all = self._pool_size >= self.n_source_dists

        if not self._cache_all:
            self._lock = threading.RLock()
            self._executor = ThreadPoolExecutor(max_workers=2)
            self._pending_replacements: dict[int, dict[str, Any]] = {}

    def init_pool(self, rng):
        if self._cache_all:
            self._src_idx_pool = np.arange(self.n_source_dists)
            self._initialized = True
            self._init_cache_pool_elements()
        else:
            self._init_pool(rng)
            self._init_cache_pool_elements()

    def _init_cache_pool_elements(self):
        if not self._initialized:
            raise ValueError("Pool not initialized. Call init_pool(rng) first.")
        if self._cache_all:
            # Cache all sources and all targets
            self._cached_srcs = {i: self._data.data.src_data[i][...] for i in range(self.n_source_dists)}
            tgt_indices = sorted({int(j) for i in range(self.n_source_dists) for j in self._data.control_to_perturbation[i]})
            def _load_tgt(j: int):
                return j, self._data.data.tgt_data[j][...]
            max_workers = min(32, (os.cpu_count() or 4))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(_load_tgt, tgt_indices))
            self._cached_tgts = {j: arr for j, arr in results}
        else:
            with self._lock:
                self._cached_srcs = {i: self._data.data.src_data[i][...] for i in self._src_idx_pool}
                tgt_indices = sorted({int(j) for i in self._src_idx_pool for j in self._data.data.src_to_tgt_dist_map[i]})
            def _load_tgt(j: int):
                return j, self._data.data.tgt_data[j][...]
            max_workers = min(32, (os.cpu_count() or 4))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(_load_tgt, tgt_indices))
            with self._lock:
                self._cached_tgts = {j: arr for j, arr in results}

    def _init_pool(self, rng):
        """Initialize the pool with random source distribution indices."""
        self._src_idx_pool = rng.choice(self._data.data.src_to_tgt_dist_map.keys(), size=self._pool_size, replace=False)
        self._initialized = True

    def _sample_source_dist_idx(self, rng) -> int:
        """Sample a source distribution index with gradual pool replacement."""
        if not self._initialized:
            raise ValueError("Pool not initialized. Call init_pool(rng) first.")

        if self._cache_all:
            # Just sample from all sources, no replacement logic
            source_idx = rng.choice(sorted(self._cached_srcs.keys()))
            self._pool_usage_count[source_idx] += 1
            return source_idx

        # Opportunistically apply any ready replacements (non-blocking)
        self._apply_ready_replacements()

        # Sample from current pool
        with self._lock:
            source_idx = rng.choice(sorted(self._cached_srcs.keys()))

        # Increment usage count for monitoring
        self._pool_usage_count[source_idx] += 1

        # Gradually replace elements based on replacement probability (schedule only)
        if rng.random() < self._replacement_prob:
            self._schedule_replacement(rng)

        return source_idx

    def _schedule_replacement(self, rng):
        if self._cache_all:
            return  # No replacement if everything is cached
        # weights same as previous logic
        most_used_weight = (self._pool_usage_count == self._pool_usage_count.max()).astype(float)
        if most_used_weight.sum() == 0:
            return
        most_used_weight /= most_used_weight.sum()
        replaced_pool_idx = rng.choice(self.n_source_dists, p=most_used_weight)

        with self._lock:
            pool_set = set(self._src_idx_pool.tolist())
            if replaced_pool_idx not in pool_set:
                return
            in_pool_idx = int(np.where(self._src_idx_pool == replaced_pool_idx)[0][0])

            # If there's already a pending replacement for this pool slot, skip
            if in_pool_idx in self._pending_replacements:
                return

            least_used_weight = (self._pool_usage_count == self._pool_usage_count.min()).astype(float)
            if least_used_weight.sum() == 0:
                return
            least_used_weight /= least_used_weight.sum()
            new_pool_idx = int(rng.choice(self.n_source_dists, p=least_used_weight))

            # Kick off background load for new indices
            fut: Future = self._executor.submit(self._load_new_cache, new_pool_idx)
            self._pending_replacements[in_pool_idx] = {
                "old": replaced_pool_idx,
                "new": new_pool_idx,
                "future": fut,
            }
            print(f"scheduled replacement of {replaced_pool_idx} with {new_pool_idx} (slot {in_pool_idx})")

    def _apply_ready_replacements(self):
        if self._cache_all:
            return  # No replacement if everything is cached
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
                for k in self._data.control_to_perturbation[old_idx]:
                    if k in self._cached_tgts:
                        del self._cached_tgts[k]

                print(f"applied replacement: {old_idx} -> {new_idx} (slot {slot})")

    def _load_new_cache(self, src_idx: int) -> dict[str, Any]:
        """Load new src and corresponding tgt arrays in the background."""
        src_arr = self._data.src_cell_data[src_idx][...]
        tgt_dict = {k: self._data.tgt_cell_data[k][...] for k in self._data.control_to_perturbation[src_idx]}
        return {"src": src_arr, "tgts": tgt_dict}

    def get_pool_stats(self) -> dict:
        if not hasattr(self, "_src_idx_pool") or self._src_idx_pool is None:
            return {"pool_size": 0, "avg_usage": 0, "unique_sources": 0}
        return {
            "pool_size": self._pool_size,
            "avg_usage": float(np.mean(self._pool_usage_count)),
            "unique_sources": len(set(self._src_idx_pool)),
            "pool_elements": self._src_idx_pool.copy(),
            "usage_counts": self._pool_usage_count.copy(),
            "cache_all": self._cache_all,
        }

    def _sample_source_cells(self, rng, source_dist_idx: int) -> np.ndarray:
        if self._cache_all:
            # No lock needed - cache is immutable after initialization
            arr = self._cached_srcs[source_dist_idx]
        else:
            # Lock needed - cache can be modified by replacements
            with self._lock:
                arr = self._cached_srcs[source_dist_idx]
        return rng.choice(arr, size=self.batch_size, replace=True)

    def _sample_target_cells(self, rng, source_dist_idx: int, target_dist_idx: int) -> np.ndarray:
        del source_dist_idx  # unused
        if self._cache_all:
            # No lock needed - cache is immutable after initialization
            arr = self._cached_tgts[target_dist_idx]
        else:
            # Lock needed - cache can be modified by replacements
            with self._lock:
                arr = self._cached_tgts[target_dist_idx]
        return rng.choice(arr, size=self.batch_size, replace=True)


class BaseValidSampler(abc.ABC):
    @abc.abstractmethod
    def sample(*args, **kwargs):
        pass

    def _get_key(self, cond_idx: int) -> tuple[str, ...]:
        if len(self._data.perturbation_idx_to_id):  # type: ignore[attr-defined]
            return self._data.perturbation_idx_to_id[cond_idx]  # type: ignore[attr-defined]
        cov_combination = self._data.perturbation_idx_to_covariates[cond_idx]  # type: ignore[attr-defined]
        return tuple(cov_combination[i] for i in range(len(cov_combination)))

    def _get_perturbation_to_control(self, data: ValidationData | PredictionData) -> dict[int, np.ndarray]:
        d = {}
        for k, v in data.control_to_perturbation.items():
            for el in v:
                d[el] = k
        return d

    def _get_condition_data(self, cond_idx: int) -> dict[str, np.ndarray]:
        return {k: v[[cond_idx], ...] for k, v in self._data.condition_data.items()}  # type: ignore[attr-defined]


class ValidationSampler(BaseValidSampler):
    """Data sampler for :class:`~scaleflow.data.ValidationData`.

    Parameters
    ----------
    val_data
        The validation data.
    seed
        Random seed.
    validation_batch_size
        Maximum number of cells to sample per condition during validation.
        If None, uses all available cells.
    """

    def __init__(self, val_data: ValidationData, seed: int = 0, validation_batch_size: int | None = None) -> None:
        self._data = val_data
        self.perturbation_to_control = self._get_perturbation_to_control(val_data)
        self.n_conditions_on_log_iteration = (
            val_data.n_conditions_on_log_iteration
            if val_data.n_conditions_on_log_iteration is not None
            else val_data.n_perturbations
        )
        self.n_conditions_on_train_end = (
            val_data.n_conditions_on_train_end
            if val_data.n_conditions_on_train_end is not None
            else val_data.n_perturbations
        )
        self.validation_batch_size = validation_batch_size
        self.rng = np.random.default_rng(seed)
        if self._data.condition_data is None:
            raise NotImplementedError("Validation data must have condition data.")

    def sample(self, mode: Literal["on_log_iteration", "on_train_end"]) -> Any:
        """Sample data for validation.

        Parameters
        ----------
        mode
            Sampling mode. Either ``"on_log_iteration"`` or ``"on_train_end"``.

        Returns
        -------
        Dictionary with source, condition, and target data from the validation data.
        """
        size = self.n_conditions_on_log_iteration if mode == "on_log_iteration" else self.n_conditions_on_train_end
        condition_idcs = self.rng.choice(self._data.n_perturbations, size=(size,), replace=False)

        source_idcs = [self.perturbation_to_control[cond_idx] for cond_idx in condition_idcs]
        source_cells_mask = [self._data.split_covariates_mask == source_idx for source_idx in source_idcs]
        source_cells = [self._data.cell_data[mask] for mask in source_cells_mask]
        target_cells_mask = [cond_idx == self._data.perturbation_covariates_mask for cond_idx in condition_idcs]
        target_cells = [self._data.cell_data[mask] for mask in target_cells_mask]

        # Apply validation batch size if specified
        if self.validation_batch_size is not None:
            source_cells = self._subsample_cells(source_cells)
            target_cells = self._subsample_cells(target_cells)

        conditions = [self._get_condition_data(cond_idx) for cond_idx in condition_idcs]
        cell_rep_dict = {}
        cond_dict = {}
        true_dict = {}
        for i in range(len(condition_idcs)):
            k = self._get_key(condition_idcs[i])
            cell_rep_dict[k] = source_cells[i]
            cond_dict[k] = conditions[i]
            true_dict[k] = target_cells[i]

        return {"source": cell_rep_dict, "condition": cond_dict, "target": true_dict}

    def _subsample_cells(self, cells_list: list[np.ndarray]) -> list[np.ndarray]:
        """Subsample cells from each condition to validation_batch_size."""
        subsampled_cells = []
        for cells in cells_list:
            if len(cells) > self.validation_batch_size:
                indices = self.rng.choice(len(cells), size=self.validation_batch_size, replace=False)
                subsampled_cells.append(cells[indices])
            else:
                subsampled_cells.append(cells)
        return subsampled_cells

    @property
    def data(self) -> ValidationData:
        """The validation data."""
        return self._data


class PredictionSampler(BaseValidSampler):
    """Data sampler for :class:`~scaleflow.data.PredictionData`.

    Parameters
    ----------
    pred_data
        The prediction data.

    """

    def __init__(self, pred_data: PredictionData) -> None:
        self._data = pred_data
        self.perturbation_to_control = self._get_perturbation_to_control(pred_data)
        if self._data.condition_data is None:
            raise NotImplementedError("Validation data must have condition data.")

    def sample(self) -> Any:
        """Sample data for prediction.

        Returns
        -------
        Dictionary with source and condition data from the prediction data.
        """
        condition_idcs = range(self._data.n_perturbations)

        source_idcs = [self.perturbation_to_control[cond_idx] for cond_idx in condition_idcs]
        source_cells_mask = [self._data.split_covariates_mask == source_idx for source_idx in source_idcs]
        source_cells = [self._data.cell_data[mask] for mask in source_cells_mask]
        conditions = [self._get_condition_data(cond_idx) for cond_idx in condition_idcs]
        cell_rep_dict = {}
        cond_dict = {}
        for i in range(len(condition_idcs)):
            k = self._get_key(condition_idcs[i])
            cell_rep_dict[k] = source_cells[i]
            cond_dict[k] = conditions[i]

        return {
            "source": cell_rep_dict,
            "condition": cond_dict,
        }

    @property
    def data(self) -> PredictionData:
        """The training data."""
        return self._data
