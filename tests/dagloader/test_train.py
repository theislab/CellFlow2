"""Case: end-to-end — scheme → DAGClassLoader → a real flow-matching training step.

Proves the data path (scheme → ScheduledClassSampler → {source, target, condition}) drives training:
the loss is finite and decreases. Minimal rectified-flow objective, tiny pure-jax MLP (no model deps).
Unit tests for the loader/sampler contract live in the sibling case files.
"""

import numpy as np
import pytest

pytest.importorskip("annbatch")
jax = pytest.importorskip("jax")
optax = pytest.importorskip("optax")
import jax.numpy as jnp  # noqa: E402

from scaleflow.dagloader import DAGClassLoader, SamplerConfig, perturbation_scheme  # noqa: E402

from _toydata import PCA_D, drug_cond_fn, toy_adata, write_collection  # noqa: E402


def _scheme(source, **kw):
    return perturbation_scheme(source, context=["cell_line"], perturbation=["drug"],
                               control_values={"drug": "control"}, **kw)


def _mlp_init(key, dims):
    ps = []
    for din, dout in zip(dims[:-1], dims[1:], strict=True):
        key, k = jax.random.split(key)
        ps.append((jax.random.normal(k, (din, dout)) / np.sqrt(din), jnp.zeros(dout)))
    return ps


def _mlp(params, h):
    *hidden, (wl, bl) = params
    for w, b in hidden:
        h = jax.nn.relu(h @ w + b)
    return h @ wl + bl


def _train_losses(loader, steps: int = 400) -> list[float]:
    """Minimal rectified-flow / conditional-FM training over the loader's batches. Returns losses."""
    key = jax.random.PRNGKey(0)
    b0 = next(loader)  # peek to size the net to the streamed rep (X dim or obsm dim)
    d, emb = b0["target"].shape[1], b0["condition"].shape[1]
    params = _mlp_init(key, [d + 1 + emb, 64, 64, d])
    opt = optax.adam(1e-3)
    state = opt.init(params)

    def loss_fn(params, s, t, c, k):
        tt = jax.random.uniform(k, (s.shape[0], 1))
        xt = (1 - tt) * s + tt * t
        v = _mlp(params, jnp.concatenate([xt, tt, c], axis=1))
        return jnp.mean((v - (t - s)) ** 2)

    @jax.jit
    def step(params, state, s, t, c, k):
        loss, grads = jax.value_and_grad(loss_fn)(params, s, t, c, k)
        updates, state = opt.update(grads, state)
        return optax.apply_updates(params, updates), state, loss

    losses = []
    for _ in range(steps):
        b = next(loader)
        key, k = jax.random.split(key)
        params, state, loss = step(params, state, jnp.asarray(b["source"]), jnp.asarray(b["target"]),
                                    jnp.asarray(b["condition"]), k)
        losses.append(float(loss))
    return losses


def _assert_learned(losses):
    assert all(np.isfinite(losses))
    assert np.mean(losses[-30:]) < 0.5 * np.mean(losses[:30])  # learned the conditional velocity


def test_training_decreases_loss_in_memory():
    """In-memory AnnData source."""
    scheme = _scheme(toy_adata(), seed=0)
    loader = DAGClassLoader(scheme, SamplerConfig(batch_size=64), condition_fn=drug_cond_fn(scheme.nodes["pert"].cols))
    _assert_learned(_train_losses(loader))


def test_training_datasetcollection(tmp_path):
    """Same end-to-end training, but the source is an ON-DISK annbatch DatasetCollection."""
    coll = write_collection(toy_adata(), tmp_path)
    scheme = _scheme(coll, seed=0)
    loader = DAGClassLoader(scheme, SamplerConfig(batch_size=64), condition_fn=drug_cond_fn(scheme.nodes["pert"].cols))
    _assert_learned(_train_losses(loader))


def test_training_obsm_from_collection(tmp_path):
    """The streamed rep is an OBSM key (not X), read from an on-disk DatasetCollection."""
    coll = write_collection(toy_adata(), tmp_path)
    scheme = _scheme(coll, key="obsm/pca", seed=0)
    loader = DAGClassLoader(scheme, SamplerConfig(batch_size=64), condition_fn=drug_cond_fn(scheme.nodes["pert"].cols))
    assert next(loader)["target"].shape[1] == PCA_D  # streamed obsm rep, not X (=8)
    _assert_learned(_train_losses(loader))


def test_training_chunk_size_gt1_on_sorted():
    """chunk_size>1 (contiguous chunked reads) on a condition-sorted source still trains."""
    scheme = _scheme(toy_adata(), seed=0)  # toy blocks are contiguous
    loader = DAGClassLoader(scheme, SamplerConfig(batch_size=64, chunk_size=8),
                            condition_fn=drug_cond_fn(scheme.nodes["pert"].cols))
    assert loader._samplers["pert"]._chunk_size == 8
    _assert_learned(_train_losses(loader))
