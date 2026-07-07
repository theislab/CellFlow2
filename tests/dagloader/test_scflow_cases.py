"""Case: sc-flow-tools — schemes assembled directly from a grouping/matching config.

Demonstrates the pieces a ``DataManager`` config maps onto: a multi-covariate condition (a
combination of a cell-line embedding and a drug embedding), explicit fixed source→target pairing via
the bind's ``common`` columns (``matched_keys`` / ``control_values_dict``), an ``obsm`` representation
as the streamed state (``sample_rep``), all working out-of-core over a ``DatasetCollection``.
"""

from __future__ import annotations

import numpy as np
import pytest

from scaleflow.dagloader import Bind, DAGClassLoader, Node, SamplerConfig, Scheme, uniform

from _toydata import (
    CELL_D,
    CELL_EMB,
    CELL_LINES,
    DRUG_D,
    G,
    PCA_D,
    cell_drug_cond_fn,
    toy_adata,
    write_collection,
)

pytest.importorskip("annbatch")

COLS = ("cell_line", "drug")
PERTURBED = [(cl, d) for cl in CELL_LINES for d in ("d1", "d2")]
CONTROL = [(cl, "control") for cl in CELL_LINES]


def _scheme(source, *, key="X", seed=0):
    """Two-node scheme (target ← control) assembled directly, sc-flow-tools style."""
    return Scheme(
        sources={"data": source},
        nodes={
            "target": Node("data", COLS, key, uniform(PERTURBED)),
            "control": Node("data", COLS, key, uniform(CONTROL)),  # control_values_dict → weighted combos
        },
        root="target",
        binds=(Bind("target", "control", common=("cell_line",)),),
        seed=seed,
    )


def test_multi_covariate_condition():
    """Condition is a combination of two covariates' embeddings: [cell_line ‖ drug]."""
    loader = DAGClassLoader(_scheme(toy_adata()), SamplerConfig(batch_size=16),
                            condition_fn=cell_drug_cond_fn(COLS))
    b = next(loader)
    assert b["condition"].shape == (16, CELL_D + DRUG_D)  # concatenated combination width
    # the first CELL_D entries are a one-hot cell-line embedding (one of the two known rows)
    assert any(np.allclose(b["condition"][0, :CELL_D], CELL_EMB[cl]) for cl in CELL_LINES)


def test_explicit_matched_pairing():
    """With one control combo per cell_line, the bind forces a deterministic source→target pairing."""
    loader = DAGClassLoader(_scheme(toy_adata()), SamplerConfig(batch_size=16))
    m = loader._bindmap["control"]
    assert all(len(codes) == 1 for codes in m["common_to_child"].values())  # exactly one control per context
    loader._start_pass()
    tl, ctl = loader._st["target"]["leaves"], loader._st["control"]["leaves"]
    for rc, cc in zip(loader._schedules["target"], loader._schedules["control"]):
        assert tl[int(rc)][0] == ctl[int(cc)][0]  # derived control shares the target's cell_line


def test_obsm_rep_streamed():
    """State is an obsm representation (sc-flow-tools' PCA-space ``sample_rep``), both target and source."""
    loader = DAGClassLoader(_scheme(toy_adata(), key="obsm/pca"), SamplerConfig(batch_size=32),
                            condition_fn=cell_drug_cond_fn(COLS))
    b = next(loader)
    assert b["target"].shape == (32, PCA_D) and b["source"].shape == (32, PCA_D)  # obsm rep, not X (=8)
    assert b["condition"].shape == (32, CELL_D + DRUG_D)


def test_out_of_core(tmp_path):
    """Same scheme, streamed from an on-disk DatasetCollection."""
    coll = write_collection(toy_adata(), tmp_path)
    loader = DAGClassLoader(_scheme(coll), SamplerConfig(batch_size=32), condition_fn=cell_drug_cond_fn(COLS))
    b = next(loader)
    assert b["target"].shape == (32, G) and b["source"].shape == (32, G)
    assert np.sign(b["target"].mean()) == np.sign(b["source"].mean())  # control matched to target context
