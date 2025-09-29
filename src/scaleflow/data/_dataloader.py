import abc
from typing import Any, Literal

import numpy as np
import tqdm

from scaleflow.data._data import (
    PredictionData,
    TrainingData,
    ValidationData,
    ZarrTrainingData,
)

__all__ = [
    "TrainSampler",
    "ValidationSampler",
    "PredictionSampler",
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

    def __init__(self, data: TrainingData | ZarrTrainingData, batch_size: int = 1024):
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
    def data(self) -> TrainingData | ZarrTrainingData:
        """The training data."""
        return self._data


class TrainSamplerWithPool(TrainSampler):
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
    pool_size
        The size of the pool of source distribution indices.
    replacement_prob
        Probability of replacing a pool element after each sample.
        Lower values = longer cache retention, less randomness.
        Higher values = faster cache turnover, more randomness.
    replace_in_pool
        Whether to allow replacement when sampling from the pool.
    """

    def __init__(
        self,
        data: TrainingData | ZarrTrainingData,
        batch_size: int = 1024,
        pool_size: int = 100,
        replacement_prob: float = 0.01,
    ):
        super().__init__(data, batch_size)
        self._pool_size = pool_size
        self._replacement_prob = replacement_prob
        self._src_idx_pool = np.empty(self._pool_size, dtype=int)
        self._pool_usage_count = np.zeros(self.n_source_dists, dtype=int)
        self._initialized = False

    def _compute_idx_mappings(self):
        import cupy as cp

        self._tgt_to_cell_data_idcs = [None] * self.n_target_dists
        gpu_per_cov_mask = cp.asarray(self._data.perturbation_covariates_mask)
        gpu_spl_cov_mask = cp.asarray(self._data.split_covariates_mask)

        for tgt_idx in tqdm.tqdm(range(self.n_target_dists), desc="Computing target to cell data idcs"):
            mask = gpu_per_cov_mask == tgt_idx
            self._tgt_to_cell_data_idcs[tgt_idx] = cp.where(mask)[0].get()
        self._src_to_cell_data_idcs = [None] * self.n_source_dists
        for src_idx in tqdm.tqdm(range(self.n_source_dists), desc="Computing source to cell data idcs"):
            mask = gpu_spl_cov_mask == src_idx
            self._src_to_cell_data_idcs[src_idx] = cp.where(mask)[0].get()

    def init_pool_n_cache(self, rng):
        self._compute_idx_mappings()
        self._init_pool(rng)
        self._init_cache_pool_elements()

    @staticmethod
    def _get_target_idx_pool(src_idx_pool: np.ndarray, control_to_perturbation: dict[int, np.ndarray]) -> set[int]:
        tgt_idx_pool = set()
        for src_idx in src_idx_pool:
            tgt_idx_pool.update(control_to_perturbation[src_idx].tolist())
        return tgt_idx_pool

    def _init_cache_pool_elements(self):
        if not self._initialized:
            raise ValueError("Pool not initialized. Call init_pool_n_cache(rng) first.")

        # Build concatenated row indices and slice maps for sources
        src_concat = []
        src_slices: dict[int, slice] = {}
        offset = 0
        for src_idx in self._src_idx_pool:
            idcs = self._src_to_cell_data_idcs[src_idx]
            n = len(idcs)
            src_slices[src_idx] = slice(offset, offset + n)
            src_concat.append(idcs)
            offset += n
        src_concat = np.concatenate(src_concat) if len(src_concat) else np.empty((0,), dtype=int)

        # Build concatenated row indices and slice maps for targets
        tgt_pool = TrainSamplerWithPool._get_target_idx_pool(self._src_idx_pool, self._data.control_to_perturbation)
        tgt_concat = []
        tgt_slices: dict[int, slice] = {}
        offset = 0
        for tgt_idx in tqdm.tqdm(sorted(tgt_pool), desc="Caching target cells"):
            idcs = self._tgt_to_cell_data_idcs[tgt_idx]
            n = len(idcs)
            tgt_slices[tgt_idx] = slice(offset, offset + n)
            tgt_concat.append(idcs)
            offset += n
        tgt_concat = np.concatenate(tgt_concat) if len(tgt_concat) else np.empty((0,), dtype=int)

        # Single orthogonal-index reads (fast path)
        self._src_block = (
            self._data.cell_data.oindex[src_concat, :]
            if src_concat.size
            else np.empty((0, self._data.cell_data.shape[1]), dtype=self._data.cell_data.dtype)
        )
        self._tgt_block = (
            self._data.cell_data.oindex[tgt_concat, :]
            if tgt_concat.size
            else np.empty((0, self._data.cell_data.shape[1]), dtype=self._data.cell_data.dtype)
        )

        # Views into the blocks (no extra copies)
        self._cached_srcs = {src_idx: self._src_block[sli] for src_idx, sli in src_slices.items()}
        tgt_views = {tgt_idx: self._tgt_block[sli] for tgt_idx, sli in tgt_slices.items()}
        self._cached_tgts = {
            src_idx: {
                tgt_idx: tgt_views[tgt_idx]
                for tgt_idx in self._data.control_to_perturbation[src_idx]
                if tgt_idx in tgt_views
            }
            for src_idx in self._src_idx_pool
        }
        self._initialized = True

    def _init_pool(self, rng):
        """Initialize the pool with random source distribution indices."""
        self._src_idx_pool = rng.choice(self.n_source_dists, size=self._pool_size, replace=False)
        self._initialized = True

    def _sample_source_dist_idx(self, rng) -> int:
        """Sample a source distribution index with gradual pool replacement."""
        if not self._initialized:
            self._init_pool(rng)

        # Sample from current pool
        pool_idx = rng.choice(self._pool_size)
        source_idx = self._src_idx_pool[pool_idx]

        # Increment usage count for monitoring
        self._pool_usage_count[source_idx] += 1

        # Gradually replace elements based on replacement probability
        if rng.random() < self._replacement_prob:
            self._replace_pool_element(rng)

        return source_idx

    def _replace_pool_element(self, rng):
        """Replace a single pool element with a new one."""
        # instead sample weighted by usage count
        # let's only consider the pool_usage_count.min() for least used
        # and the pool_usage_count.max() for most used
        most_used_weight = (self._pool_usage_count == self._pool_usage_count.max()).astype(float)
        most_used_weight /= most_used_weight.sum()

        # weight by most used
        replaced_pool_idx = rng.choice(self.n_source_dists, p=most_used_weight)
        if replaced_pool_idx in set(self._src_idx_pool):
            in_pool_idx = np.where(self._src_idx_pool == replaced_pool_idx)[0][0]
            least_used_weight = (self._pool_usage_count == self._pool_usage_count.min()).astype(float)
            least_used_weight /= least_used_weight.sum()
            new_pool_idx = rng.choice(self.n_source_dists, p=least_used_weight)
            self._src_idx_pool[in_pool_idx] = new_pool_idx
            print(f"replaced {replaced_pool_idx} with {new_pool_idx}")

    def get_pool_stats(self) -> dict:
        """Get statistics about the current pool state."""
        if self._src_idx_pool is None:
            return {"pool_size": 0, "avg_usage": 0, "unique_sources": 0}
        return {
            "pool_size": self._pool_size,
            "avg_usage": float(np.mean(self._pool_usage_count)),
            "unique_sources": len(set(self._src_idx_pool)),
            "pool_elements": self._src_idx_pool.copy(),
            "usage_counts": self._pool_usage_count.copy(),
        }

    def _sample_source_cells(self, rng, source_dist_idx: int) -> np.ndarray:
        return rng.choice(self._cached_srcs[source_dist_idx], size=self.batch_size, replace=True)

    def _sample_target_cells(self, rng, source_dist_idx: int, target_dist_idx: int) -> np.ndarray:
        return rng.choice(self._cached_tgts[source_dist_idx][target_dist_idx], size=self.batch_size, replace=True)


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
