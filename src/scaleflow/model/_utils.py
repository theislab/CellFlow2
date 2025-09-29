import anndata as ad
import jax
import jax.numpy as jnp

from scaleflow._types import ArrayLike


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


def _write_predictions(
    adata: ad.AnnData,
    predictions: dict[str, ArrayLike],
    key_added_prefix: str,
) -> None:
    for pred_key, pred_value in predictions.items():
        if pred_value.ndim == 2:
            adata.obsm[f"{key_added_prefix}{pred_key}"] = pred_value
        elif pred_value.ndim == 3:
            for i in range(pred_value.shape[2]):
                adata.obsm[f"{key_added_prefix}{pred_key}_{i}"] = pred_value[..., i]
        else:
            raise ValueError(f"Predictions for '{pred_key}' have an invalid shape: {pred_value.shape}")
