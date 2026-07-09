import jax
import jax.numpy as jnp

# dedup: re-exported from cellflow (identical implementations)
from cellflow.solvers.utils import (
    ema_update as ema_update,
)


def _multivariate_normal(
    rng: jax.Array,
    shape: tuple[int, ...],
    dim: int,
    mean: float = 0.0,
    cov: float = 1.0,
) -> jnp.ndarray:
    mean = jnp.full(dim, fill_value=mean)
    cov = jnp.diag(jnp.full(dim, fill_value=cov))
    return jax.random.multivariate_normal(rng, mean=mean, cov=cov, shape=shape)
