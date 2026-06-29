"""Unit tests for the ClassSampler-backed GroupedAnnbatchSampler.

Each cell's first feature ``X[:, 0]`` encodes a unique ``(cell_line, drug)`` code, so a
returned target batch can be traced back to a single condition and the matched source/
condition pairing can be checked directly.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scaleflow.data import (
    AnnDataLocation,
    DataManager,
    GroupedAnnbatchSampler,
    write_sorted_collection,
)

CELL_LINES = ["cl0", "cl1"]
TGT_DRUGS = ["drug0", "drug1"]
ALL_DRUGS = ["control", *TGT_DRUGS]


def _make_collection(tmp_path, *, n_target=20, n_control=20):
    """Build a sorted DatasetCollection + GroupedDistribution with traceable X codes.

    Returns ``(coll_path, gd, pair_to_code, rep)`` where ``pair_to_code`` maps
    ``"{cell_line}|{drug}"`` to the float stored in ``X[:, 0]`` for those cells.
    """
    rows = []
    for cl in CELL_LINES:
        rows += [{"cell_line": cl, "drug": "control", "control": True}] * n_control
        for drug in TGT_DRUGS:
            rows += [{"cell_line": cl, "drug": drug, "control": False}] * n_target
    obs = pd.DataFrame(rows)

    pair = obs["cell_line"].astype(str) + "|" + obs["drug"].astype(str)
    pair_to_code = {p: float(i) for i, p in enumerate(sorted(pair.unique()))}
    codes = pair.map(pair_to_code).to_numpy(dtype=np.float32)

    n_obs = len(obs)
    X = np.zeros((n_obs, 4), dtype=np.float32)
    X[:, 0] = codes
    for c in ["cell_line", "drug"]:
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=X, obs=obs)

    coll_path = tmp_path / "coll.zarr"
    write_sorted_collection(
        adata,
        str(coll_path),
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys=["drug"],
        sorted_adata_path=str(tmp_path / "sorted.zarr"),
    )

    drug_emb = {d: np.eye(len(ALL_DRUGS), dtype=np.float32)[i] for i, d in enumerate(ALL_DRUGS)}
    cl_emb = {cl: np.eye(len(CELL_LINES), dtype=np.float32)[i] for i, cl in enumerate(CELL_LINES)}
    rep = {"drug_emb": drug_emb, "cell_line_emb": cl_emb}

    dm = DataManager(
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys=["drug"],
        rep_keys={"cell_line": "cell_line_emb", "drug": "drug_emb"},
        data_location=AnnDataLocation().X,
    )
    gd = dm.prepare_data_from_collection(str(coll_path), rep_dict=rep)
    return str(coll_path), gd, pair_to_code, rep


def _code_to_pair(pair_to_code):
    return {v: k for k, v in pair_to_code.items()}


def test_batches_are_class_coherent(tmp_path):
    """Every cell in a returned target batch comes from a single condition."""
    coll_path, gd, pair_to_code, _ = _make_collection(tmp_path)
    target_codes = {c for p, c in pair_to_code.items() if not p.endswith("|control")}

    sampler = GroupedAnnbatchSampler(coll_path, gd, batch_size=16, chunk_size=8, seed=0)
    for _ in range(20):
        batch = sampler.sample()
        tgt = np.asarray(batch["tgt_cell_data"])
        assert tgt.shape == (16, 4)  # drop_last => exactly batch_size
        uniq = np.unique(tgt[:, 0])
        assert uniq.shape == (1,), f"batch mixed conditions: {uniq}"
        assert float(uniq[0]) in target_codes


def test_source_and_condition_pairing(tmp_path):
    """The returned source/condition match the sampled target's (cell_line, drug)."""
    coll_path, gd, pair_to_code, rep = _make_collection(tmp_path)
    code_to_pair = _code_to_pair(pair_to_code)

    sampler = GroupedAnnbatchSampler(coll_path, gd, batch_size=16, chunk_size=8, seed=1)
    for _ in range(20):
        batch = sampler.sample()
        tgt = np.asarray(batch["tgt_cell_data"])
        src = np.asarray(batch["src_cell_data"])
        cl, drug = code_to_pair[float(tgt[0, 0])].split("|")

        # source cells are the matched cell_line's controls (single control code)
        assert src.shape == (16, 4)
        assert np.unique(src[:, 0]).tolist() == [pair_to_code[f"{cl}|control"]]

        # condition embeddings correspond to the sampled (cell_line, drug)
        cond = batch["condition"]
        np.testing.assert_array_equal(np.asarray(cond["drug"])[0, 0], rep["drug_emb"][drug])
        np.testing.assert_array_equal(np.asarray(cond["cell_line"])[0, 0], rep["cell_line_emb"][cl])


def test_weights_restrict_sampled_conditions(tmp_path):
    """A one-hot weight vector makes the sampler draw only that one condition."""
    coll_path, gd, pair_to_code, _ = _make_collection(tmp_path)

    order = GroupedAnnbatchSampler(coll_path, gd, batch_size=16, chunk_size=8)._tgt_idx_order
    chosen = order[0]
    weights = np.zeros(len(order))
    weights[0] = 1.0

    cl, drug = gd.annotation.tgt_dist_idx_to_labels[chosen]
    expected_code = pair_to_code[f"{cl}|{drug}"]

    sampler = GroupedAnnbatchSampler(coll_path, gd, batch_size=16, chunk_size=8, seed=2, weights=weights)
    for _ in range(20):
        tgt = np.asarray(sampler.sample()["tgt_cell_data"])
        assert np.unique(tgt[:, 0]).tolist() == [expected_code]


def test_too_small_condition_raises(tmp_path):
    """chunk_size larger than a sampled condition raises with a clear message."""
    coll_path, gd, _, _ = _make_collection(tmp_path, n_target=20)
    sampler = GroupedAnnbatchSampler(coll_path, gd, batch_size=8, chunk_size=25, seed=0)
    with pytest.raises(ValueError, match="chunk_size"):
        sampler.init_sampler()


def test_zero_weight_excludes_condition(tmp_path):
    """A zero-weighted condition is never drawn (ClassSampler excludes weight-0 classes)."""
    coll_path, gd, pair_to_code, _ = _make_collection(tmp_path, n_target=20)

    order = GroupedAnnbatchSampler(coll_path, gd, batch_size=16, chunk_size=8)._tgt_idx_order
    weights = np.ones(len(order))
    weights[0] = 0.0
    excluded = order[0]
    cl, drug = gd.annotation.tgt_dist_idx_to_labels[excluded]
    excluded_code = pair_to_code[f"{cl}|{drug}"]

    sampler = GroupedAnnbatchSampler(coll_path, gd, batch_size=16, chunk_size=8, seed=3, weights=weights)
    seen = set()
    for _ in range(40):
        tgt = np.asarray(sampler.sample()["tgt_cell_data"])
        seen.add(float(np.unique(tgt[:, 0])[0]))
    assert excluded_code not in seen
