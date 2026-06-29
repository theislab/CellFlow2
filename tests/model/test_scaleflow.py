"""Tests for the ScaleFlow model over an annbatch-backed (sorted) collection.

ScaleFlow is the only model: cells are streamed from a DatasetCollection via
GroupedAnnbatchSampler (annbatch ClassSampler), so these tests build a small collection
sorted by condition and drive prepare_data -> prepare_model -> train -> predict.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr

import scaleflow
from scaleflow.data import AnnDataLocation, DataManager, write_sorted_collection
from scaleflow.data._data_splitter import GroupedDistributionSplitter
from scaleflow.model import ScaleFlow

# Small read-slice size: <= every condition in adata_test (~27-81 cells per condition).
CHUNK_SIZE = 8
REP_KEYS = {"cell_line": "cell_line_embeddings", "drug": "drug_embeddings", "gene": "gene_embeddings"}


def _write_rep_store(adata, path):
    g = zarr.open_group(str(path), mode="w")
    for k in ["cell_line_embeddings", "drug_embeddings", "gene_embeddings"]:
        ad.io.write_elem(g, k, adata.uns[k])


def _build_collection(adata, tmp_path):
    a = adata.copy()
    a.X = np.asarray(a.X, dtype=np.float32)
    a.uns = {}
    coll_path = tmp_path / "coll.zarr"
    write_sorted_collection(
        a,
        str(coll_path),
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys=["drug", "gene"],
        sorted_adata_path=str(tmp_path / "adata_sorted.zarr"),
    )
    return coll_path


def _prepared_model(adata_test, tmp_path, solver, **prepare_model_kwargs):
    """Build a prepared (data + model) ScaleFlow over a sorted collection."""
    rep_path = tmp_path / "uns.zarr"
    _write_rep_store(adata_test, rep_path)
    coll_path = _build_collection(adata_test, tmp_path)

    model = ScaleFlow(solver=solver)
    model.prepare_data(
        str(coll_path),
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys=["drug", "gene"],
        rep_keys=REP_KEYS,
        rep_path=str(rep_path),
    )
    defaults = dict(
        condition_embedding_dim=8,
        time_freqs=8,
        time_encoder_dims=(16,),
        hidden_dims=(16,),
        decoder_dims=(16,),
        seed=0,
    )
    defaults.update(prepare_model_kwargs)
    model.prepare_model(**defaults)
    return model, str(coll_path), str(rep_path)


def _predict_kwargs(solver):
    return {"max_steps": 3, "eta": 0.01} if solver == "eqm" else {"max_steps": 3, "throw": False}


@pytest.mark.slow
@pytest.mark.parametrize("solver", ["otfm", "genot", "eqm"])
def test_scaleflow_train_predict(adata_test, tmp_path, solver):
    """prepare -> train -> predict_covariates + get_condition_embedding for each solver."""
    vf_kwargs = {"genot_source_dims": (16, 16), "genot_source_dropout": 0.1} if solver == "genot" else None
    model, coll_path, rep_path = _prepared_model(adata_test, tmp_path, solver, vf_kwargs=vf_kwargs)

    model.train(num_iterations=2, batch_size=32, chunk_size=CHUNK_SIZE, valid_freq=1000)
    assert model.solver is not None and model.solver.is_trained
    assert model.dataloader is not None
    losses = model.trainer.training_logs["loss"]
    assert len(losses) == 2 and all(np.isfinite(losses))

    cov = pd.DataFrame(
        {"cell_line": ["cell_line_0", "cell_line_1"], "drug": ["drug_0", "drug_1"], "gene": ["gene_0", "control"]}
    )
    preds = model.predict_covariates(cov, rep_path=rep_path, **_predict_kwargs(solver))
    assert len(preds) == 2
    for arr in preds.values():
        arr = np.asarray(arr)
        assert arr.ndim == 2 and arr.shape[1] == adata_test.n_vars
        assert np.all(np.isfinite(arr))

    df_mean, df_var = model.get_condition_embedding(cov, rep_path=rep_path)
    assert len(df_mean) == len(df_var) == 2
    assert df_mean.shape[1] == 8
    assert np.all(np.isfinite(df_mean.values)) and np.all(np.isfinite(df_var.values))


@pytest.mark.slow
@pytest.mark.parametrize("conditioning", ["concatenation", "film", "resnet"])
def test_scaleflow_conditioning_variants(adata_test, tmp_path, conditioning):
    """The conditioning mechanisms all train and predict (otfm)."""
    model, coll_path, rep_path = _prepared_model(adata_test, tmp_path, "otfm", conditioning=conditioning)
    model.train(num_iterations=2, batch_size=32, chunk_size=CHUNK_SIZE, valid_freq=1000)
    assert model.solver.is_trained
    cov = pd.DataFrame({"cell_line": ["cell_line_0"], "drug": ["drug_0"], "gene": ["gene_0"]})
    preds = model.predict_covariates(cov, rep_path=rep_path, max_steps=3, throw=False)
    assert len(preds) == 1
    assert np.all(np.isfinite(np.asarray(next(iter(preds.values())))))


@pytest.mark.slow
def test_scaleflow_stochastic_condition_embedding(adata_test, tmp_path):
    """Stochastic condition embeddings train (with regularization>0) and give finite variance."""
    model, _, rep_path = _prepared_model(
        adata_test, tmp_path, "otfm", condition_mode="stochastic", regularization=0.1
    )
    model.train(num_iterations=2, batch_size=32, chunk_size=CHUNK_SIZE, valid_freq=1000)
    cov = pd.DataFrame({"cell_line": ["cell_line_0"], "drug": ["drug_0"], "gene": ["gene_0"]})
    _, df_var = model.get_condition_embedding(cov, rep_path=rep_path)
    assert np.all(np.isfinite(df_var.values))


def test_scaleflow_stochastic_requires_regularization(adata_test, tmp_path):
    """Stochastic condition mode with regularization=0 raises at prepare_model."""
    with pytest.raises(ValueError, match=r"[Ss]tochastic condition embeddings require"):
        _prepared_model(adata_test, tmp_path, "otfm", condition_mode="stochastic", regularization=0.0)


def test_raise_otfm_vf_kwargs_passed(adata_test, tmp_path):
    """otfm with vf_kwargs is rejected at prepare_model."""
    with pytest.raises(ValueError, match=r"vf_kwargs` must be `None`"):
        _prepared_model(
            adata_test, tmp_path, "otfm", vf_kwargs={"genot_source_dims": (2, 2), "genot_source_dropout": 0.1}
        )


@pytest.mark.parametrize("time_max_period", [-3, 0])
def test_time_embedding_invalid_period(adata_test, tmp_path, time_max_period):
    """A non-positive time_max_period is rejected at prepare_model."""
    with pytest.raises(ValueError):
        _prepared_model(adata_test, tmp_path, "otfm", time_freqs=8, time_max_period=time_max_period)


def _combo_adata(n=16):
    """AnnData where each non-control condition is a 2-drug combination (drug_1, drug_2)."""
    cell_lines, drugs = ["cl0", "cl1"], ["control", "dA", "dB", "dC"]
    rows = []
    for cl in cell_lines:
        rows += [{"cell_line": cl, "drug_1": "control", "drug_2": "control", "control": True}] * n
        for d1, d2 in [("dA", "dB"), ("dB", "dC")]:
            rows += [{"cell_line": cl, "drug_1": d1, "drug_2": d2, "control": False}] * n
    obs = pd.DataFrame(rows)
    for c in ["cell_line", "drug_1", "drug_2"]:
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=np.random.randn(len(obs), 12).astype(np.float32), obs=obs)
    adata.uns["cell_line_embeddings"] = {cl: np.eye(2, dtype=np.float32)[i] for i, cl in enumerate(cell_lines)}
    adata.uns["drug_embeddings"] = {d: np.eye(4, dtype=np.float32)[i] for i, d in enumerate(drugs)}
    return adata


@pytest.mark.slow
def test_scaleflow_combination_trains(tmp_path):
    """A multi-column tgt group is pooled at the model level (set length K=2), no padding.

    max_combination_length is derived from the data (= 2) and the single covariate (cell_line)
    is encoded as a not-pooled context covariate.
    """
    adata = _combo_adata()
    rep_path = tmp_path / "uns.zarr"
    g = zarr.open_group(str(rep_path), mode="w")
    for k in ["cell_line_embeddings", "drug_embeddings"]:
        ad.io.write_elem(g, k, adata.uns[k])

    a = adata.copy()
    a.uns = {}
    coll_path = tmp_path / "coll.zarr"
    write_sorted_collection(
        a,
        str(coll_path),
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys={"drug": ["drug_1", "drug_2"]},
        sorted_adata_path=str(tmp_path / "sorted.zarr"),
    )

    model = ScaleFlow(solver="otfm")
    model.prepare_data(
        str(coll_path),
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys={"drug": ["drug_1", "drug_2"]},
        rep_keys={"cell_line": "cell_line_embeddings", "drug": "drug_embeddings"},
        rep_path=str(rep_path),
    )
    # condition carries a 2-element drug set, no padding
    assert np.asarray(next(iter(model.train_data.data.conditions.values()))["drug"]).shape == (1, 2, 4)

    model.prepare_model(
        condition_embedding_dim=8, time_freqs=8, time_encoder_dims=(16,),
        hidden_dims=(16,), decoder_dims=(16,), seed=0,
    )
    # max_combination_length was derived from the data (=2), not pinned to 1
    assert model.vf.max_combination_length == 2

    model.train(num_iterations=2, batch_size=16, chunk_size=4, valid_freq=1000)
    assert model.solver.is_trained
    assert all(np.isfinite(model.trainer.training_logs["loss"]))

    cov = pd.DataFrame({"cell_line": ["cl0"], "drug_1": ["dA"], "drug_2": ["dB"]})
    preds = model.predict_covariates(cov, rep_path=str(rep_path), max_steps=3, throw=False)
    assert len(preds) == 1
    arr = np.asarray(next(iter(preds.values())))
    assert arr.ndim == 2 and arr.shape[1] == adata.n_vars and np.all(np.isfinite(arr))


@pytest.mark.slow
def test_scaleflow_with_validation(adata_test, tmp_path):
    """A registered validation split produces a logged validation metric during training."""
    rep_path = tmp_path / "uns.zarr"
    _write_rep_store(adata_test, rep_path)
    coll_path = _build_collection(adata_test, tmp_path)

    dm = DataManager(
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys=["drug", "gene"],
        rep_keys=REP_KEYS,
        data_location=AnnDataLocation().X,
    )
    gd = dm.prepare_data_from_collection(str(coll_path), rep_path=str(rep_path))
    splits = GroupedDistributionSplitter(
        gd=gd,
        holdout_combinations=False,
        split_by=["drug", "gene"],
        split_key="split",
        force_training_values={},
        ratios=[0.6, 0.2, 0.2],
        random_state=0,
    ).split()

    model = ScaleFlow(solver="otfm")
    model.prepare_data(
        str(coll_path),
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys=["drug", "gene"],
        rep_keys=REP_KEYS,
        rep_path=str(rep_path),
    )
    model.train_data = splits["train"]
    model.prepare_validation_data("val", splits["val"], collection=str(coll_path))
    model.prepare_model(
        condition_embedding_dim=8,
        time_freqs=8,
        time_encoder_dims=(16,),
        hidden_dims=(16,),
        decoder_dims=(16,),
        seed=0,
    )

    metrics_callback = scaleflow.training.Metrics(metrics=["r_squared"])
    model.train(num_iterations=4, batch_size=32, chunk_size=CHUNK_SIZE, valid_freq=1, callbacks=[metrics_callback])
    assert model.solver.is_trained
    assert "val_r_squared_mean" in model.trainer.training_logs


def _combo_collection_with_nulls(tmp_path, n=16):
    """Sorted collection with mixed combos incl. a (dA, control) null slot; control -> 0 emb."""
    cell_lines, drugs = ["cl0", "cl1"], ["control", "dA", "dB"]
    rows = []
    for cl in cell_lines:
        rows += [{"cell_line": cl, "drug_1": "control", "drug_2": "control", "control": True}] * n
        for d1, d2 in [("dA", "dB"), ("dA", "control")]:  # second condition has a null slot
            rows += [{"cell_line": cl, "drug_1": d1, "drug_2": d2, "control": False}] * n
    obs = pd.DataFrame(rows)
    for c in ["cell_line", "drug_1", "drug_2"]:
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=np.random.randn(len(obs), 12).astype(np.float32), obs=obs)
    cl_emb = {cl: np.eye(2, dtype=np.float32)[i] for i, cl in enumerate(cell_lines)}
    drug_emb = dict(zip(drugs, np.concatenate([np.zeros((1, 2), np.float32), np.eye(2, dtype=np.float32)], axis=0)))

    rep_path = tmp_path / "uns.zarr"
    g = zarr.open_group(str(rep_path), mode="w")
    ad.io.write_elem(g, "cell_line_embeddings", cl_emb)
    ad.io.write_elem(g, "drug_embeddings", drug_emb)

    coll_path = tmp_path / "coll.zarr"
    write_sorted_collection(
        adata, str(coll_path), dist_flag_key="control", src_dist_keys=["cell_line"],
        tgt_dist_keys={"drug": ["drug_1", "drug_2"]}, sorted_adata_path=str(tmp_path / "sorted.zarr"),
    )
    return str(coll_path), str(rep_path), adata.n_vars


@pytest.mark.slow
def test_scaleflow_combination_null_slot_trains(tmp_path):
    """A combo with a control (null/zero) slot is masked by the pooler; attention pooling must not NaN."""
    coll_path, rep_path, n_vars = _combo_collection_with_nulls(tmp_path)
    model = ScaleFlow(solver="otfm")
    model.prepare_data(
        coll_path, dist_flag_key="control", src_dist_keys=["cell_line"],
        tgt_dist_keys={"drug": ["drug_1", "drug_2"]},
        rep_keys={"cell_line": "cell_line_embeddings", "drug": "drug_embeddings"}, rep_path=rep_path,
    )
    model.prepare_model(
        condition_embedding_dim=8, time_freqs=8, time_encoder_dims=(16,),
        hidden_dims=(16,), decoder_dims=(16,), pooling="attention_token", seed=0,
    )
    model.train(num_iterations=2, batch_size=16, chunk_size=4, valid_freq=1000)
    assert model.solver.is_trained
    assert all(np.isfinite(model.trainer.training_logs["loss"]))

    # predict both a full combo and the null-slot combo; both must be finite
    cov = pd.DataFrame({"cell_line": ["cl0", "cl0"], "drug_1": ["dA", "dA"], "drug_2": ["dB", "control"]})
    preds = model.predict_covariates(cov, rep_path=rep_path, max_steps=3, throw=False)
    assert len(preds) == 2
    for arr in preds.values():
        arr = np.asarray(arr)
        assert arr.shape[1] == n_vars and np.all(np.isfinite(arr))


@pytest.mark.slow
def test_scaleflow_combination_mean_pooling_null_slot(tmp_path):
    """Same as above but with mean pooling (the other masked-reduction path)."""
    coll_path, rep_path, n_vars = _combo_collection_with_nulls(tmp_path)
    model = ScaleFlow(solver="otfm")
    model.prepare_data(
        coll_path, dist_flag_key="control", src_dist_keys=["cell_line"],
        tgt_dist_keys={"drug": ["drug_1", "drug_2"]},
        rep_keys={"cell_line": "cell_line_embeddings", "drug": "drug_embeddings"}, rep_path=rep_path,
    )
    model.prepare_model(
        condition_embedding_dim=8, time_freqs=8, time_encoder_dims=(16,),
        hidden_dims=(16,), decoder_dims=(16,), pooling="mean", seed=0,
    )
    model.train(num_iterations=2, batch_size=16, chunk_size=4, valid_freq=1000)
    assert model.solver.is_trained and all(np.isfinite(model.trainer.training_logs["loss"]))


@pytest.mark.slow
def test_scaleflow_combination_stochastic(tmp_path):
    """Stochastic condition embeddings + combinations train and give finite variance."""
    coll_path, rep_path, n_vars = _combo_collection_with_nulls(tmp_path)
    model = ScaleFlow(solver="otfm")
    model.prepare_data(
        coll_path, dist_flag_key="control", src_dist_keys=["cell_line"],
        tgt_dist_keys={"drug": ["drug_1", "drug_2"]},
        rep_keys={"cell_line": "cell_line_embeddings", "drug": "drug_embeddings"}, rep_path=rep_path,
    )
    model.prepare_model(
        condition_embedding_dim=8, time_freqs=8, time_encoder_dims=(16,),
        hidden_dims=(16,), decoder_dims=(16,), condition_mode="stochastic", regularization=0.1, seed=0,
    )
    model.train(num_iterations=2, batch_size=16, chunk_size=4, valid_freq=1000)
    assert model.solver.is_trained
    cov = pd.DataFrame({"cell_line": ["cl0"], "drug_1": ["dA"], "drug_2": ["dB"]})
    _, df_var = model.get_condition_embedding(cov, rep_path=rep_path)
    assert np.all(np.isfinite(df_var.values))
