"""Smoke tests for the self-contained scaleflow.classmap subpackage."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import scaleflow.classmap as cm


def _toy_adata(seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    rows = []
    for cl in ["cl0", "cl1"]:
        for drug in ["control", "drugA", "drugB"]:
            for _ in range(6):
                rows.append({"cell_line": cl, "drug": drug})
    obs = pd.DataFrame(rows)
    adata = ad.AnnData(X=rng.standard_normal((len(obs), 5)).astype(np.float32), obs=obs)
    adata.uns["drug_emb"] = {"drugA": np.array([1.0, 0.0]), "drugB": np.array([0.0, 1.0]), "control": np.array([0.0, 0.0])}
    return adata


def test_public_api_imports():
    for name in ["HierarchicalIndexer", "Domain", "ClassMapping", "ClassMappingAnnbatchLoader", "GroupingSpec",
                 "prepare", "as_source", "AnnDataLocation"]:
        assert hasattr(cm, name)


def test_indexer_sorted_gives_contiguous_slices():
    df = pd.DataFrame({"cell_line": ["cl0", "cl0", "cl1", "cl1"], "drug": ["a", "b", "a", "b"]})
    for c in df.columns:
        df[c] = df[c].astype("category")
    idx = cm.HierarchicalIndexer(groups_cols=["cell_line"], conditions_cols=["drug"])
    assert idx.sort_columns == ("cell_line", "drug")
    mi = idx.create_index(df)
    assert isinstance(mi, pd.MultiIndex)
    assert mi.names[0] == cm.BASE_LEVEL_NAME


def test_source_seam_in_memory_slice_read():
    adata = _toy_adata()
    src = cm.as_source(adata)
    assert src.forces_chunk_size_one is True
    out = src.read_rows({0: slice(0, 6), 1: slice(6, 12)})
    assert out[0].shape == (6, 5) and out[1].shape == (6, 5)
    np.testing.assert_array_equal(out[0], adata.X[0:6])


def test_prepare_builds_classmapping():
    adata = _toy_adata()
    spec = cm.GroupingSpec(
        group_cols=["cell_line"], condition_cols=["drug"],
        control_values={"drug": "control"}, reps={"drug": "drug_emb"},
    )
    mapping = cm.prepare(adata, spec)
    assert isinstance(mapping, cm.ClassMapping)
    assert mapping.names == [cm.DEFAULT_DOMAIN]
    d = mapping[cm.DEFAULT_DOMAIN]
    assert d.is_control.sum() == 12  # 2 cell lines * 6 control cells
    # non-control conditions present, keyed by the condition tuple
    assert ("drugA",) in d.conditions and ("drugB",) in d.conditions


def test_loader_in_memory_sample():
    adata = _toy_adata()
    spec = cm.GroupingSpec(
        group_cols=["cell_line"], condition_cols=["drug"],
        control_values={"drug": "control"}, reps={"drug": "drug_emb"},
    )
    mapping = cm.prepare(adata, spec)
    loader = cm.ClassMappingAnnbatchLoader(mapping, batch_size=4, n_batches_per_pass=4, seed=0)
    batch = loader.sample()
    assert set(batch) == {"src_cell_data", "tgt_cell_data", "condition"}
    assert batch["src_cell_data"].shape == (4, 5)
    assert batch["tgt_cell_data"].shape == (4, 5)
