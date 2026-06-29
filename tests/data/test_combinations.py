"""Tests for perturbation combinations (grouped tgt_dist_keys) in DataManager.

A combination is a multi-column target group whose columns are stacked into a set of
length K = number of columns (no padding); single covariates stay length 1. The set is
pooled at the model level (tested in tests/model/test_scaleflow.py).
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scaleflow.data import (
    AnnDataLocation,
    DataManager,
    GroupedAnnbatchSampler,
    GroupedDistribution,
    write_sorted_collection,
)

CELL_LINES = ["cl0", "cl1"]
DRUGS = ["control", "dA", "dB", "dC"]


def _combo_adata(n=6):
    """AnnData where each non-control condition is a 2-drug combination (drug_1, drug_2)."""
    rows = []
    for cl in CELL_LINES:
        rows += [{"cell_line": cl, "drug_1": "control", "drug_2": "control", "control": True}] * n
        for d1, d2 in [("dA", "dB"), ("dB", "dC")]:
            rows += [{"cell_line": cl, "drug_1": d1, "drug_2": d2, "control": False}] * n
    obs = pd.DataFrame(rows)
    for c in ["cell_line", "drug_1", "drug_2"]:
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=np.random.randn(len(obs), 5).astype(np.float32), obs=obs)
    adata.uns["cell_line_embeddings"] = {cl: np.eye(len(CELL_LINES), dtype=np.float32)[i] for i, cl in enumerate(CELL_LINES)}
    adata.uns["drug_embeddings"] = {d: np.eye(len(DRUGS), dtype=np.float32)[i] for i, d in enumerate(DRUGS)}
    return adata


def _dm(tgt_dist_keys):
    rep_keys = {"cell_line": "cell_line_embeddings"}
    # rep_keys is keyed by covariate name: a group name (grouped) or a column (flat)
    if isinstance(tgt_dist_keys, dict):
        rep_keys.update(dict.fromkeys(tgt_dist_keys, "drug_embeddings"))
    else:
        rep_keys.update(dict.fromkeys(tgt_dist_keys, "drug_embeddings"))
    return DataManager(
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys=tgt_dist_keys,
        rep_keys=rep_keys,
        data_location=AnnDataLocation().X,
    )


def test_col_to_repr_accepts_numpy_scalars():
    """Numeric covariates: _col_to_repr accepts numpy scalars (predict path uses iterrows).

    Regression: predict_covariates/get_condition_embedding iterate with DataFrame.iterrows(),
    yielding np.float32/np.int64 for numeric columns; isinstance(label, float) used to reject them.
    """
    f = DataManager._col_to_repr
    np.testing.assert_array_equal(f({}, "dose", np.float32(10.0)), np.array([10.0]))
    np.testing.assert_array_equal(f({}, "dose", np.int64(5)), np.array([5.0]))
    np.testing.assert_array_equal(f({}, "dose", 2.5), np.array([2.5]))
    with pytest.raises(ValueError, match="not found in col_to_repr"):
        f({}, "drug", "some_string_label")  # non-numeric, unknown store still raises


def test_grouped_condition_is_a_stacked_set():
    """A 2-column group → (1, 2, emb) with the columns stacked in order; src stays (1, 1, emb)."""
    adata = _combo_adata()
    dm = _dm({"drug": ["drug_1", "drug_2"]})
    gd = dm.prepare_data(adata)

    drug_emb = adata.uns["drug_embeddings"]
    for t, cond in gd.data.conditions.items():
        assert np.asarray(cond["cell_line"]).shape == (1, 1, len(CELL_LINES))
        assert np.asarray(cond["drug"]).shape == (1, 2, len(DRUGS))  # K=2, no padding
        # labels are (cell_line, drug_1, drug_2); the drug set stacks drug_1 then drug_2
        _, d1, d2 = gd.annotation.tgt_dist_idx_to_labels[t]
        np.testing.assert_array_equal(np.asarray(cond["drug"])[0, 0], drug_emb[d1])
        np.testing.assert_array_equal(np.asarray(cond["drug"])[0, 1], drug_emb[d2])


def test_flat_keys_stay_singletons():
    """Flat tgt_dist_keys keep set length 1 (backward compatible)."""
    adata = _combo_adata()
    gd = _dm(["drug_1", "drug_2"]).prepare_data(adata)
    cond = next(iter(gd.data.conditions.values()))
    assert np.asarray(cond["drug_1"]).shape == (1, 1, len(DRUGS))
    assert np.asarray(cond["drug_2"]).shape == (1, 1, len(DRUGS))
    assert np.asarray(cond["cell_line"]).shape == (1, 1, len(CELL_LINES))


def test_unequal_combination_widths_raise():
    """Two multi-column groups with different widths is rejected at construction."""
    with pytest.raises(ValueError, match="combination width"):
        _dm({"drug": ["drug_1", "drug_2"], "gene": ["g1", "g2", "g3"]})


def test_get_condition_data_grouped():
    """get_condition_data (arbitrary covariates) also stacks grouped columns into a set."""
    adata = _combo_adata()
    dm = _dm({"drug": ["drug_1", "drug_2"]})
    cov = pd.DataFrame({"cell_line": ["cl0"], "drug_1": ["dA"], "drug_2": ["dC"]})
    conds = dm.get_condition_data(cov, rep_dict=adata.uns)
    cond = next(iter(conds.values()))
    assert np.asarray(cond["drug"]).shape == (1, 2, len(DRUGS))
    np.testing.assert_array_equal(np.asarray(cond["drug"])[0, 0], adata.uns["drug_embeddings"]["dA"])
    np.testing.assert_array_equal(np.asarray(cond["drug"])[0, 1], adata.uns["drug_embeddings"]["dC"])


def _adata_with_control_embedding(n=6):
    """Combo adata where 'control' maps to a ZERO drug embedding (like the real fixtures).

    Includes a (dA, control) condition so one set slot is the null/zero perturbation.
    """
    cell_lines, drugs = ["cl0", "cl1"], ["control", "dA", "dB"]
    rows = []
    for cl in cell_lines:
        rows += [{"cell_line": cl, "drug_1": "control", "drug_2": "control", "control": True}] * n
        for d1, d2 in [("dA", "dB"), ("dA", "control")]:
            rows += [{"cell_line": cl, "drug_1": d1, "drug_2": d2, "control": False}] * n
    obs = pd.DataFrame(rows)
    for c in ["cell_line", "drug_1", "drug_2"]:
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=np.random.randn(len(obs), 5).astype(np.float32), obs=obs)
    adata.uns["cell_line_embeddings"] = {cl: np.eye(2, dtype=np.float32)[i] for i, cl in enumerate(cell_lines)}
    # control -> zero row (null embedding), real drugs -> identity rows
    drug_emb = np.concatenate([np.zeros((1, 2), dtype=np.float32), np.eye(2, dtype=np.float32)], axis=0)
    adata.uns["drug_embeddings"] = dict(zip(drugs, drug_emb, strict=False))
    return adata


def test_control_slot_is_null_embedding():
    """A 'control' value in one combo slot yields a zero (maskable) embedding at that position."""
    adata = _adata_with_control_embedding()
    dm = DataManager(
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys={"drug": ["drug_1", "drug_2"]},
        rep_keys={"cell_line": "cell_line_embeddings", "drug": "drug_embeddings"},
        data_location=AnnDataLocation().X,
    )
    gd = dm.prepare_data(adata)
    found = False
    for t, cond in gd.data.conditions.items():
        _, d1, d2 = gd.annotation.tgt_dist_idx_to_labels[t]
        drug = np.asarray(cond["drug"])
        assert drug.shape == (1, 2, 2)
        if (d1, d2) == ("dA", "control"):
            found = True
            np.testing.assert_array_equal(drug[0, 1], np.zeros(2))  # control slot -> null/zero
            assert not np.all(drug[0, 0] == 0)  # real slot is non-null
    assert found, "expected a (dA, control) condition"


def test_numeric_dosage_combination():
    """A combination of numeric columns (not in rep_keys) stacks scalars into (1, K, 1)."""
    cell_lines = ["cl0", "cl1"]
    rows = []
    for cl in cell_lines:
        rows += [{"cell_line": cl, "dose_1": 0.0, "dose_2": 0.0, "control": True}] * 6
        for v1, v2 in [(10.0, 100.0), (100.0, 1000.0)]:
            rows += [{"cell_line": cl, "dose_1": v1, "dose_2": v2, "control": False}] * 6
    obs = pd.DataFrame(rows)
    obs["cell_line"] = obs["cell_line"].astype("category")
    adata = ad.AnnData(X=np.random.randn(len(obs), 4).astype(np.float32), obs=obs)
    adata.uns["cell_line_embeddings"] = {cl: np.eye(2, dtype=np.float32)[i] for i, cl in enumerate(cell_lines)}
    dm = DataManager(
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys={"dose": ["dose_1", "dose_2"]},  # numeric group, no rep_keys entry
        rep_keys={"cell_line": "cell_line_embeddings"},
        data_location=AnnDataLocation().X,
    )
    gd = dm.prepare_data(adata)
    for t, cond in gd.data.conditions.items():
        dose = np.asarray(cond["dose"])
        assert dose.shape == (1, 2, 1)  # K=2 scalars stacked
        _, v1, v2 = gd.annotation.tgt_dist_idx_to_labels[t]
        np.testing.assert_allclose(dose[0, :, 0], [float(v1), float(v2)])


def test_multiple_aligned_pooled_groups():
    """Two equal-width pooled groups (drug + dosage) both become (1, K, emb), cellflow-style."""
    cell_lines, drugs = ["cl0", "cl1"], ["control", "dA", "dB"]
    rows = []
    for cl in cell_lines:
        rows += [{"cell_line": cl, "drug_1": "control", "drug_2": "control", "dose_1": 0.0, "dose_2": 0.0, "control": True}] * 6
        for (a, b) in [("dA", "dB"), ("dB", "dA")]:
            rows += [{"cell_line": cl, "drug_1": a, "drug_2": b, "dose_1": 10.0, "dose_2": 100.0, "control": False}] * 6
    obs = pd.DataFrame(rows)
    for c in ["cell_line", "drug_1", "drug_2"]:
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=np.random.randn(len(obs), 4).astype(np.float32), obs=obs)
    adata.uns["cell_line_embeddings"] = {cl: np.eye(2, dtype=np.float32)[i] for i, cl in enumerate(cell_lines)}
    adata.uns["drug_embeddings"] = {d: np.eye(3, dtype=np.float32)[i] for i, d in enumerate(drugs)}
    dm = DataManager(
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys={"drug": ["drug_1", "drug_2"], "dose": ["dose_1", "dose_2"]},
        rep_keys={"cell_line": "cell_line_embeddings", "drug": "drug_embeddings"},
        data_location=AnnDataLocation().X,
    )
    gd = dm.prepare_data(adata)
    cond = next(iter(gd.data.conditions.values()))
    assert np.asarray(cond["drug"]).shape == (1, 2, 3)
    assert np.asarray(cond["dose"]).shape == (1, 2, 1)
    assert np.asarray(cond["cell_line"]).shape == (1, 1, 2)


def test_sampler_emits_combination_condition(tmp_path):
    """GroupedAnnbatchSampler streams a (1, K, emb) combination condition with the batch."""
    n = 8
    rows = []
    for cl in CELL_LINES:
        rows += [{"cell_line": cl, "drug_1": "control", "drug_2": "control", "control": True}] * n
        for d1, d2 in [("dA", "dB"), ("dB", "dC")]:
            rows += [{"cell_line": cl, "drug_1": d1, "drug_2": d2, "control": False}] * n
    obs = pd.DataFrame(rows)
    for c in ["cell_line", "drug_1", "drug_2"]:
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=np.random.randn(len(obs), 5).astype(np.float32), obs=obs)
    cl_emb = {cl: np.eye(len(CELL_LINES), dtype=np.float32)[i] for i, cl in enumerate(CELL_LINES)}
    drug_emb = {d: np.eye(len(DRUGS), dtype=np.float32)[i] for i, d in enumerate(DRUGS)}

    coll = str(tmp_path / "coll.zarr")
    write_sorted_collection(
        adata, coll, dist_flag_key="control", src_dist_keys=["cell_line"],
        tgt_dist_keys={"drug": ["drug_1", "drug_2"]}, sorted_adata_path=str(tmp_path / "sorted.zarr"),
    )
    dm = DataManager(
        dist_flag_key="control", src_dist_keys=["cell_line"], tgt_dist_keys={"drug": ["drug_1", "drug_2"]},
        rep_keys={"cell_line": "cell_line_embeddings", "drug": "drug_embeddings"},
        data_location=AnnDataLocation().X,
    )
    gd = dm.prepare_data_from_collection(coll, rep_dict={"cell_line_embeddings": cl_emb, "drug_embeddings": drug_emb})

    sampler = GroupedAnnbatchSampler(coll, gd, batch_size=8, chunk_size=4, seed=0)
    for _ in range(5):
        batch = sampler.sample()
        assert np.asarray(batch["condition"]["drug"]).shape == (1, 2, len(DRUGS))   # pooled set
        assert np.asarray(batch["condition"]["cell_line"]).shape == (1, 1, len(CELL_LINES))  # context
        assert np.asarray(batch["tgt_cell_data"]).shape == (8, 5)


def test_grouped_conditions_zarr_roundtrip(tmp_path):
    """(1, K, emb) condition sets round-trip through GroupedDistribution zarr IO."""
    adata = _combo_adata()
    gd = _dm({"drug": ["drug_1", "drug_2"]}).prepare_data(adata)
    path = str(tmp_path / "gd.zarr")
    gd.write_zarr(path, chunk_size=8, shard_size=64)
    gd2 = GroupedDistribution.read_zarr(path)

    assert set(gd2.data.conditions) == set(gd.data.conditions)
    for t, cond in gd.data.conditions.items():
        got = np.asarray(gd2.data.conditions[t]["drug"])
        assert got.shape == (1, 2, len(DRUGS))
        np.testing.assert_allclose(got, np.asarray(cond["drug"]))
