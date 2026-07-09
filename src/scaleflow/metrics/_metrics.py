from collections.abc import Sequence

import jax
import numpy as np

# dedup: re-exported from cellflow (identical implementations)
from cellflow.metrics._metrics import (
    compute_e_distance as compute_e_distance,
)
from cellflow.metrics._metrics import (
    compute_e_distance_fast as compute_e_distance_fast,
)
from cellflow.metrics._metrics import (
    compute_mean_metrics as compute_mean_metrics,
)
from cellflow.metrics._metrics import (
    compute_metrics as compute_metrics,
)
from cellflow.metrics._metrics import (
    compute_metrics_fast as compute_metrics_fast,
)
from cellflow.metrics._metrics import (
    compute_r_squared as compute_r_squared,
)
from cellflow.metrics._metrics import (
    compute_scalar_mmd as compute_scalar_mmd_cf,
)
from cellflow.metrics._metrics import (
    compute_sinkhorn_div as compute_sinkhorn_div,
)
from cellflow.metrics._metrics import (
    maximum_mean_discrepancy as maximum_mean_discrepancy,
)
from cellflow.metrics._metrics import (
    pairwise_squeuclidean as pairwise_squeuclidean,
)
from cellflow.metrics._metrics import (
    rbf_kernel_fast as rbf_kernel_fast,
)
from jax import numpy as jnp
from jax.typing import ArrayLike

__all__ = [
    "compute_metrics",
    "compute_metrics_fast",
    "compute_metrics_fast_gpu",
    "compute_mean_metrics",
    "compute_scalar_mmd_cf",
    "compute_scalar_mmd_sf",
    "compute_scalar_mmd_gpu",
    "compute_r_squared",
    "compute_r_squared_gpu",
    "compute_sinkhorn_div",
    "compute_e_distance",
    "compute_e_distance_fast",
    "compute_e_distance_gpu",
    "maximum_mean_discrepancy",
]




















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




@jax.jit
def compute_r_squared_gpu(x: ArrayLike, y: ArrayLike) -> float:
    """GPU-optimized R² computation without CPU transfers.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].

    Returns
    -------
        A scalar denoting the R squared score.
    """
    x_mean = jnp.mean(x, axis=0)
    y_mean = jnp.mean(y, axis=0)

    ss_res = jnp.sum((x_mean - y_mean) ** 2)
    ss_tot = jnp.sum((x_mean - jnp.mean(x_mean)) ** 2)

    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    return float(r2)


@jax.jit
def subsample_on_gpu(key: jax.Array, x: ArrayLike, max_samples: int) -> ArrayLike:
    """Subsample array on GPU without CPU transfer."""
    n_samples = x.shape[0]
    if n_samples <= max_samples:
        return x
    indices = jax.random.choice(key, n_samples, shape=(max_samples,), replace=False)
    return x[indices]


def compute_scalar_mmd_gpu(
    x: ArrayLike,
    y: ArrayLike,
    gammas: Sequence[float] | None = None,
    max_samples: int = 5000,
    key: jax.Array | None = None,
    precision: str = "float32",
) -> float:
    """GPU-optimized MMD without CPU transfers.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].
        gammas
            A sequence of values for the paramater gamma of the rbf kernel.
        max_samples
            Maximum number of samples to use for MMD computation.
        key
            JAX random key for subsampling on GPU.
        precision
            Precision for computation: 'float32', 'bfloat16', or 'float16'.

    Returns
    -------
        A scalar denoting the average MMD over all gammas.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    key1, key2 = jax.random.split(key)

    x_sub = subsample_on_gpu(key1, x, max_samples)
    y_sub = subsample_on_gpu(key2, y, max_samples)

    if precision == "bfloat16":
        x_sub = x_sub.astype(jnp.bfloat16)
        y_sub = y_sub.astype(jnp.bfloat16)
    elif precision == "float16":
        x_sub = x_sub.astype(jnp.float16)
        y_sub = y_sub.astype(jnp.float16)

    if gammas is None:
        gammas = [2, 1, 0.5]

    @jax.jit
    def compute_mmd_single_gamma(gamma):
        xx = rbf_kernel_fast(x_sub, x_sub, gamma)
        xy = rbf_kernel_fast(x_sub, y_sub, gamma)
        yy = rbf_kernel_fast(y_sub, y_sub, gamma)
        return xx.mean() + yy.mean() - 2 * xy.mean()

    mmds = jnp.array([compute_mmd_single_gamma(g) for g in gammas])
    return float(jnp.mean(mmds))


@jax.jit
def compute_e_distance_gpu(x: ArrayLike, y: ArrayLike, precision: str = "float32") -> float:
    """GPU-optimized energy distance with optional reduced precision.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].
        precision
            Precision for computation: 'float32', 'bfloat16', or 'float16'.

    Returns
    -------
        A scalar denoting the energy distance value.
    """
    if precision == "bfloat16":
        x = x.astype(jnp.bfloat16)
        y = y.astype(jnp.bfloat16)
    elif precision == "float16":
        x = x.astype(jnp.float16)
        y = y.astype(jnp.float16)

    sigma_X = pairwise_squeuclidean(x, x).mean()
    sigma_Y = pairwise_squeuclidean(y, y).mean()
    delta = pairwise_squeuclidean(x, y).mean()
    return 2 * delta - sigma_X - sigma_Y


def compute_metrics_fast_gpu(
    x: ArrayLike, y: ArrayLike, key: jax.Array | None = None, precision: str = "float32", max_samples: int = 5000
) -> dict[str, float]:
    """Compute metrics entirely on GPU with optional reduced precision.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].
        key
            JAX random key for subsampling.
        precision
            Precision for computation: 'float32', 'bfloat16', or 'float16'.
            bfloat16 is recommended for speed with minimal accuracy loss.
        max_samples
            Maximum samples for MMD computation.

    Returns
    -------
        A dictionary containing the following computed metrics:

        - the r squared score.
        - the energy distance value.
        - the mean maximum discrepancy loss

    Note
    ----
        All operations stay on GPU, minimizing data transfers.
        Using bfloat16 can provide ~2x speedup with negligible accuracy loss.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    metrics = {}
    metrics["r_squared"] = compute_r_squared_gpu(x, y)
    metrics["e_distance"] = compute_e_distance_gpu(x, y, precision=precision)
    metrics["mmd"] = compute_scalar_mmd_gpu(x, y, key=key, precision=precision, max_samples=max_samples)
    return metrics
