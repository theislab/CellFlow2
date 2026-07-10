from collections.abc import Sequence

import numpy as np
from cellflow.metrics._metrics import maximum_mean_discrepancy
from jax.typing import ArrayLike

__all__ = ["compute_scalar_mmd_sf"]


def compute_scalar_mmd_sf(
    x: ArrayLike, y: ArrayLike, gammas: Sequence[float] | None = None, max_samples: int = 5000
) -> float:
    """Compute the Mean Maximum Discrepancy (MMD) across different length scales

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].
        gammas
            A sequence of values for the paramater gamma of the rbf kernel.
        max_samples
            Maximum number of samples to use for MMD computation. Default is 5000.
            If either x or y has more samples, they will be randomly subsampled.

    Returns
    -------
        A scalar denoting the average MMD over all gammas.
    """
    if x.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(x.shape[0], max_samples, replace=False)
        x = x[idx]
    if y.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(y.shape[0], max_samples, replace=False)
        y = y[idx]
    if gammas is None:
        gammas = [2, 1, 0.5]  # , 0.1, 0.01, 0.005]
    mmds = [maximum_mean_discrepancy(x, y, gamma=gamma) for gamma in gammas]  # type: ignore[union-attr]
    return np.nanmean(np.array(mmds))
