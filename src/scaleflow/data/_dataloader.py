from abc import ABC, abstractmethod
from typing import Any

import numpy as np

__all__ = [
    "CombinedSampler",
    "SamplerABC",
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
