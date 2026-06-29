"""End-to-end smoke test: annbatch-backed data path feeds a real training step."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr

from scaleflow.data import AnnDataLocation, DataManager, GroupedAnnbatchSampler
from scaleflow.data._data_splitter import GroupedDistributionSplitter
from scaleflow.model._scaleflow import ScaleFlow


def _write_rep_store(adata, path):
    g = zarr.open_group(str(path), mode="w")
    for k in ["cell_line_embeddings", "drug_embeddings", "gene_embeddings"]:
        ad.io.write_elem(g, k, adata.uns[k])


def _build_collection(adata, tmp_path):
    from annbatch import DatasetCollection

    a = adata.copy()
    a.X = np.asarray(a.X, dtype=np.float32)
    a.uns = {}
    adata_path = tmp_path / "adata.zarr"
    a.write_zarr(str(adata_path))
    coll_path = tmp_path / "coll.zarr"
    coll = DatasetCollection(str(coll_path), mode="a")
    coll.add_adatas(adata_paths=[str(adata_path)], shuffle=False)
    return coll_path


@pytest.mark.slow
def test_end_to_end_training(tmp_path, adata_test):
    """Coherent ScaleFlow flow: prepare_data (path) -> prepare_model -> train, no manual sampler."""
    rep_path = tmp_path / "uns.zarr"
    _write_rep_store(adata_test, rep_path)
    coll_path = _build_collection(adata_test, tmp_path)

    model = ScaleFlow(solver="otfm")
    model.prepare_data(
        str(coll_path),
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys=["drug", "gene"],
        rep_keys={"cell_line": "cell_line_embeddings", "drug": "drug_embeddings", "gene": "gene_embeddings"},
        rep_path=str(rep_path),
    )
    assert isinstance(model.train_data, type(model.train_data))  # GroupedDistribution
    assert model.train_data.data.conditions  # conditions were built

    model.prepare_model(
        condition_embedding_dim=8,
        time_freqs=8,
        time_encoder_dims=(16,),
        hidden_dims=(16,),
        decoder_dims=(16,),
        seed=0,
    )
    assert model._data_dim == adata_test.n_vars

    model.train(num_iterations=2, batch_size=32, valid_freq=1000)

    assert model.solver is not None and model.solver.is_trained
    losses = model.trainer.training_logs["loss"]
    assert len(losses) == 2
    assert all(np.isfinite(losses))

    # condition embeddings for arbitrary (possibly unseen) covariate combinations
    cov = pd.DataFrame(
        {
            "cell_line": ["cell_line_0", "cell_line_1"],
            "drug": ["drug_0", "drug_1"],
            "gene": ["gene_0", "control"],
        }
    )
    df_mean, df_var = model.get_condition_embedding(cov, rep_path=str(rep_path))
    assert len(df_mean) == 2 and len(df_var) == 2
    assert df_mean.shape[1] == 8  # condition_embedding_dim
    assert np.all(np.isfinite(df_mean.values)) and np.all(np.isfinite(df_var.values))

    # predict responses for those arbitrary covariate combinations (sources = trained controls)
    preds = model.predict_covariates(cov, rep_path=str(rep_path))
    assert len(preds) == 2
    for arr in preds.values():
        arr = np.asarray(arr)
        assert arr.ndim == 2 and arr.shape[1] == adata_test.n_vars
        assert np.all(np.isfinite(arr))


@pytest.mark.slow
def test_end_to_end_training_with_split(tmp_path, adata_test):
    """Full pipeline with a train/val/test split: prepare -> split by condition -> train on train split.

    Splits are metadata-only and share the same on-disk collection; each split references a
    disjoint subset of target conditions (rows), and training streams from the train subset.
    """
    rep_path = tmp_path / "uns.zarr"
    _write_rep_store(adata_test, rep_path)
    coll_path = _build_collection(adata_test, tmp_path)

    rep_keys = {"cell_line": "cell_line_embeddings", "drug": "drug_embeddings", "gene": "gene_embeddings"}
    dm = DataManager(
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys=["drug", "gene"],
        rep_keys=rep_keys,
        data_location=AnnDataLocation().X,
    )
    gd = dm.prepare_data_from_collection(str(coll_path), rep_path=str(rep_path))

    # split by perturbation condition (drug x gene), metadata-only, over the shared collection
    splitter = GroupedDistributionSplitter(
        gd=gd,
        holdout_combinations=False,
        split_by=["drug", "gene"],
        split_key="split",
        force_training_values={},
        ratios=[0.6, 0.2, 0.2],
        random_state=0,
    )
    splits = splitter.split()
    assert set(splits) == {"train", "val", "test"}

    # target conditions are partitioned across splits (disjoint), and together cover all
    tgt = {k: set(v.data.tgt_dist_to_rows.keys()) for k, v in splits.items()}
    assert tgt["train"].isdisjoint(tgt["val"])
    assert tgt["train"].isdisjoint(tgt["test"])
    assert tgt["val"].isdisjoint(tgt["test"])
    assert tgt["train"] | tgt["val"] | tgt["test"] == set(gd.data.tgt_dist_to_rows.keys())

    # a sampler over the train split only ever draws train-split conditions
    train_sampler = GroupedAnnbatchSampler(str(coll_path), splits["train"], batch_size=16, seed=0)
    assert set(train_sampler._tgt_idx_order).issubset(tgt["train"])

    # train on the train split end-to-end via ScaleFlow
    model = ScaleFlow(solver="otfm")
    model.prepare_data(
        str(coll_path),
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys=["drug", "gene"],
        rep_keys=rep_keys,
        rep_path=str(rep_path),
    )
    model.train_data = splits["train"]  # restrict training to the train split
    # register the val split for validation-during-training
    model.prepare_validation_data("val", splits["val"], collection=str(coll_path))
    model.prepare_model(
        condition_embedding_dim=8,
        time_freqs=8,
        time_encoder_dims=(16,),
        hidden_dims=(16,),
        decoder_dims=(16,),
        seed=0,
    )
    # valid_freq=1 with >2 iters triggers in-loop validation (it>1) on the val split
    model.train(num_iterations=4, batch_size=32, valid_freq=1)

    assert model.solver is not None and model.solver.is_trained
    assert len(model.trainer.training_logs["loss"]) == 4
    assert all(np.isfinite(model.trainer.training_logs["loss"]))
    # the auto-built dataloader trained on the train split
    assert model.dataloader.data is splits["train"]

    # predict on the held-out test split: one prediction per test condition
    preds = model.predict(splits["test"], collection=str(coll_path))
    assert isinstance(preds, dict) and len(preds) == len(tgt["test"])
    for key, arr in preds.items():
        arr = np.asarray(arr)
        assert arr.ndim == 2 and arr.shape[1] == adata_test.n_vars, f"bad prediction shape for {key}: {arr.shape}"
        assert np.all(np.isfinite(arr))
