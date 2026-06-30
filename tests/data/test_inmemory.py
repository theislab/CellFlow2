"""In-memory training path: prepare_data_in_memory -> Loader.add_adata, chunk_size=1, slices.

Covers the in-memory two-path flow (annbatch ``Loader.add_adata`` instead of an on-disk
collection), the self-contained AnnData artifact (``to_adata``/``from_adata``, ``output_path``),
and that ``rows_for`` yields contiguous slices once the data is sorted by condition.
"""

import anndata as ad
import numpy as np
import pytest

from scaleflow.data import GroupedDistribution
from scaleflow.data._data import OBS_SRC_DIST_IDX, OBS_TGT_DIST_IDX, GroupedDistributionData
from scaleflow.model._scaleflow import ScaleFlow

REP_KEYS = {"cell_line": "cell_line_embeddings", "drug": "drug_embeddings", "gene": "gene_embeddings"}


def _adata(adata_test):
    a = adata_test.copy()
    a.X = np.asarray(a.X, dtype=np.float32)
    return a


def _prepare(model, a, **kwargs):
    model.prepare_data_in_memory(
        a,
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys=["drug", "gene"],
        rep_keys=REP_KEYS,
        **kwargs,
    )


def test_rows_for_sorted_yields_slices():
    col = np.array([-1, 0, 0, 0, 1, 1, 2, -1, -1])
    rows = GroupedDistributionData.rows_for(col)
    assert rows == {0: slice(1, 4), 1: slice(4, 6), 2: slice(6, 7)}


def test_rows_for_scattered_yields_index_array():
    col = np.array([0, 1, 0, 1])  # non-contiguous -> index arrays, not slices
    rows = GroupedDistributionData.rows_for(col)
    assert isinstance(rows[0], np.ndarray) and rows[0].tolist() == [0, 2]


def test_prepare_in_memory_sorts_and_assigns_per_row(adata_test):
    model = ScaleFlow(solver="otfm")
    _prepare(model, _adata(adata_test))

    gd = model.train_data
    assert isinstance(gd, GroupedDistribution)
    assert gd.data.conditions  # embeddings built from adata.uns
    assert model._in_memory is True

    # every sorted condition is one contiguous run -> rows_for returns slices
    tgt_rows = GroupedDistributionData.rows_for(gd.data.row_tgt_dist_idx)
    assert tgt_rows and all(isinstance(v, slice) for v in tgt_rows.values())
    src_rows = GroupedDistributionData.rows_for(gd.data.row_src_dist_idx)
    assert src_rows and all(isinstance(v, slice) for v in src_rows.values())


def test_to_adata_from_adata_roundtrip(adata_test):
    model = ScaleFlow(solver="otfm")
    a = _adata(adata_test)
    _prepare(model, a)
    gd = model.train_data

    enriched = gd.to_adata(a)
    assert OBS_TGT_DIST_IDX in enriched.obs and OBS_SRC_DIST_IDX in enriched.obs
    assert "scaleflow" in enriched.uns

    gd2 = GroupedDistribution.from_adata(enriched)
    np.testing.assert_array_equal(gd.data.row_tgt_dist_idx, gd2.data.row_tgt_dist_idx)
    np.testing.assert_array_equal(gd.data.row_src_dist_idx, gd2.data.row_src_dist_idx)
    assert gd.data.src_to_tgt_dist_map == gd2.data.src_to_tgt_dist_map
    assert set(gd.data.conditions) == set(gd2.data.conditions)
    assert gd.annotation.dist_flag_key == gd2.annotation.dist_flag_key
    assert gd.annotation.src_dist_keys == gd2.annotation.src_dist_keys


def test_output_path_writes_and_reuses(tmp_path, adata_test):
    out = tmp_path / "prepared.zarr"

    model = ScaleFlow(solver="otfm")
    _prepare(model, _adata(adata_test), output_path=str(out))
    assert out.exists()
    rows_before = np.asarray(model.train_data.data.row_tgt_dist_idx).copy()

    # reused on a second call (file already exists): reload, do not recompute
    model2 = ScaleFlow(solver="otfm")
    _prepare(model2, _adata(adata_test), output_path=str(out))
    np.testing.assert_array_equal(model2.train_data.data.row_tgt_dist_idx, rows_before)

    # the persisted artifact is a self-contained AnnData (X + metadata) round-trippable on its own
    reloaded = ad.read_zarr(str(out))
    assert reloaded.n_obs == adata_test.n_obs
    GroupedDistribution.from_adata(reloaded)


def test_make_dataloader_pins_chunk_size_one(adata_test):
    model = ScaleFlow(solver="otfm")
    _prepare(model, _adata(adata_test))
    with pytest.raises(ValueError, match="chunk_size=1"):
        model.make_dataloader(batch_size=32, chunk_size=8)
    loader = model.make_dataloader(batch_size=32)  # chunk_size defaults to 1 in memory
    assert loader._chunk_size == 1


@pytest.mark.slow
def test_end_to_end_in_memory_training(adata_test):
    model = ScaleFlow(solver="otfm")
    _prepare(model, _adata(adata_test))
    model.prepare_model(
        condition_embedding_dim=8,
        time_freqs=8,
        time_encoder_dims=(16,),
        hidden_dims=(16,),
        decoder_dims=(16,),
        seed=0,
    )
    assert model._data_dim == adata_test.n_vars

    model.train(num_iterations=2, batch_size=32, valid_freq=1000)  # no chunk_size: in memory -> 1
    assert model.solver is not None and model.solver.is_trained
    losses = model.trainer.training_logs["loss"]
    assert len(losses) == 2 and all(np.isfinite(losses))
