"""End-to-end: declarative Scheme → annbatch TreeSampler → a real flow-matching training step.

Proves the data path (scheme → annbatch ClassSampler → {source, target, condition}) drives training:
the loss is finite and decreases. Minimal rectified-flow objective, tiny pure-jax MLP (no model deps).
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")
jax = pytest.importorskip("jax")
optax = pytest.importorskip("optax")
import anndata as ad  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from scaleflow.treesampler import TreeSampler, perturbation_scheme  # noqa: E402

G, EMB, PCA_D = 8, 4, 5
DRUG_SHIFT = {"control": np.zeros(G), "d1": np.eye(G)[0] * 4.0, "d2": np.eye(G)[1] * 4.0}
NOISE_STD = 0.4
DRUG_EMB = {"control": np.zeros(EMB), "d1": np.eye(EMB)[0], "d2": np.eye(EMB)[1]}


def _toy_adata(seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    rows, blocks = [], []
    for cl in ["A", "B"]:
        cl_mean = np.full(G, 1.0 if cl == "A" else -1.0)
        for drug in ["control", "d1", "d2"]:
            n = 60
            blocks.append(rng.normal(cl_mean + DRUG_SHIFT[drug], NOISE_STD, size=(n, G)).astype(np.float32))
            rows += [{"cell_line": cl, "drug": drug}] * n
    obs = pd.DataFrame(rows)
    for c in ("cell_line", "drug"):
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=np.vstack(blocks), obs=obs)
    adata.obsm["pca"] = adata.X[:, :PCA_D].copy()  # a lower-dim rep carrying the same conditional signal
    adata.uns["drug_emb"] = DRUG_EMB
    return adata


def test_scheme_builds_and_binds():
    adata = _toy_adata()
    scheme = perturbation_scheme(adata, context=["cell_line"], perturbation=["drug"],
                                 control_values={"drug": "control"}, n_rows_per_leaf=16)
    assert scheme.root == "pert"
    # pert weights only non-control combos; ctrl only control combos (absent/0 ⇒ excluded = selection)
    pert_w = scheme.nodes["pert"].weights
    assert all(combo[1] != "control" for combo in pert_w) and len(pert_w) == 4      # (A,d1)(A,d2)(B,d1)(B,d2)
    ctrl_w = scheme.nodes["ctrl"].weights
    assert all(combo[1] == "control" for combo in ctrl_w) and len(ctrl_w) == 2       # (A,ctrl)(B,ctrl)


def test_treesampler_yields_matched_batches():
    adata = _toy_adata()
    scheme = perturbation_scheme(adata, context=["cell_line"], perturbation=["drug"],
                                 control_values={"drug": "control"}, n_rows_per_leaf=16)
    cols = scheme.nodes["pert"].cols
    sampler = TreeSampler(scheme, condition_fn=lambda leaf: DRUG_EMB[leaf[cols.index("drug")]])
    batch = next(sampler)
    assert batch["source"].shape == (16, G)
    assert batch["target"].shape == (16, G)
    assert batch["condition"].shape == (16, EMB)
    # target is class-coherent (one non-control drug); condition matches that drug
    assert not np.allclose(batch["condition"][0], DRUG_EMB["control"])


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


def _train_losses(sampler, steps: int = 400) -> list[float]:
    """Minimal rectified-flow / conditional-FM training over the sampler's batches. Returns losses."""
    key = jax.random.PRNGKey(0)
    b0 = next(sampler)  # peek to size the net to the streamed rep (X dim or obsm dim)
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
        b = next(sampler)
        key, k = jax.random.split(key)
        params, state, loss = step(params, state, jnp.asarray(b["source"]), jnp.asarray(b["target"]),
                                    jnp.asarray(b["condition"]), k)
        losses.append(float(loss))
    return losses


def _write_collection(adata, tmp_path):
    ad.settings.zarr_write_format = 3  # annbatch sharding needs zarr v3
    from annbatch import DatasetCollection

    ap, cp = tmp_path / "a.zarr", tmp_path / "coll.zarr"
    adata.write_zarr(str(ap))
    DatasetCollection(str(cp), mode="a").add_adatas(adata_paths=[str(ap)], shuffle=False)  # preserve row order
    return DatasetCollection(str(cp), mode="r")


def test_flow_matching_training_decreases_loss():
    """In-memory AnnData source."""
    adata = _toy_adata()
    scheme = perturbation_scheme(adata, context=["cell_line"], perturbation=["drug"],
                                 control_values={"drug": "control"}, n_rows_per_leaf=64, seed=0)
    cols = scheme.nodes["pert"].cols
    sampler = TreeSampler(scheme, condition_fn=lambda leaf: DRUG_EMB[leaf[cols.index("drug")]])
    losses = _train_losses(sampler)
    assert all(np.isfinite(losses))
    assert np.mean(losses[-30:]) < 0.5 * np.mean(losses[:30])  # learned the conditional velocity


def test_flow_matching_training_datasetcollection(tmp_path):
    """Same end-to-end training, but the source is an ON-DISK annbatch DatasetCollection."""
    coll = _write_collection(_toy_adata(), tmp_path)
    scheme = perturbation_scheme(coll, context=["cell_line"], perturbation=["drug"],
                                 control_values={"drug": "control"}, n_rows_per_leaf=64, seed=0)
    cols = scheme.nodes["pert"].cols
    sampler = TreeSampler(scheme, condition_fn=lambda leaf: DRUG_EMB[leaf[cols.index("drug")]])
    losses = _train_losses(sampler)
    assert all(np.isfinite(losses))
    assert np.mean(losses[-30:]) < 0.5 * np.mean(losses[:30])  # out-of-core path trains too


def test_obsm_streaming_from_collection(tmp_path):
    """The streamed rep is an OBSM key (not X), read from an on-disk DatasetCollection."""
    coll = _write_collection(_toy_adata(), tmp_path)
    scheme = perturbation_scheme(coll, context=["cell_line"], perturbation=["drug"],
                                 control_values={"drug": "control"}, key="obsm/pca", n_rows_per_leaf=64, seed=0)
    cols = scheme.nodes["pert"].cols
    sampler = TreeSampler(scheme, condition_fn=lambda leaf: DRUG_EMB[leaf[cols.index("drug")]])
    b = next(sampler)
    assert b["target"].shape[1] == PCA_D and b["source"].shape[1] == PCA_D  # streamed obsm rep, not X (=8)
    losses = _train_losses(sampler)
    assert all(np.isfinite(losses)) and np.mean(losses[-30:]) < 0.5 * np.mean(losses[:30])
