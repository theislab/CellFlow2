"""prepare() — the AnnData → ClassMapping transform, as a process function (not a class).

PRE-SYNC DRAFT. This is the lightweight counterpart to sc-flow-tools' ``DataManager``: stateless
given a :class:`GroupingSpec`. Condition-embedding building here is a simple per-column lookup;
the richer encoding/pooling (combinations, transforms, external encoders) lives in sc-flow-tools'
``DataManager`` and is reconciled after the sync.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import anndata as ad
import numpy as np

from scaleflow.classmap._container import UNS_KEY, ClassMapping
from scaleflow.classmap._indexer import HierarchicalIndexer

__all__ = ["GroupingSpec", "prepare"]


@dataclass
class GroupingSpec:
    """Configuration for :func:`prepare` (behavior-free)."""

    group_cols: list[str]  # context / source populations (HierarchicalIndexer GROUP level)
    condition_cols: list[str]  # perturbation (CONDITION level)
    control_values: dict[str, Any]  # per condition col: the value (or np.nan) meaning "control"
    reps: dict[str, str] = field(default_factory=dict)  # condition col -> uns embedding-store key
    sample_rep: str | None = None  # obsm key for the cell representation, else X


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def prepare(
    adata: ad.AnnData,
    spec: GroupingSpec,
    *,
    rep_store: Mapping[str, Mapping[Any, np.ndarray]] | None = None,
    sort: bool = True,
) -> ClassMapping:
    """Transform an in-memory AnnData into a self-contained :class:`ClassMapping` (in-memory).

    Casts the group/condition columns to Categorical, sorts by condition (so each group is a
    contiguous run), builds per-condition embeddings, and writes the per-domain mappings into the
    adata's obs/uns. Returns ``ClassMapping.in_memory(adata)`` — a single artifact (X + obs + uns).

    ``rep_store`` maps a rep key to ``{covariate_value: embedding}``; defaults to ``adata.uns``.
    """
    if rep_store is None:
        rep_store = adata.uns

    adata = adata.copy()
    for col in (*spec.group_cols, *spec.condition_cols):
        if not hasattr(adata.obs[col], "cat"):
            adata.obs[col] = adata.obs[col].astype("category")

    indexer = HierarchicalIndexer(groups_cols=spec.group_cols, conditions_cols=spec.condition_cols)
    if sort and indexer.sort_columns:
        order = adata.obs.sort_values(list(indexer.sort_columns), kind="stable").index
        adata = adata[order].copy()

    # control mask: condition cols all equal their control value
    obs = adata.obs
    is_ctrl = np.ones(adata.n_obs, dtype=bool)
    for col, ctrl in spec.control_values.items():
        vals = obs[col]
        is_ctrl &= vals.isna().to_numpy() if (isinstance(ctrl, float) and np.isnan(ctrl)) else (vals == ctrl).to_numpy()

    # per-condition embeddings, keyed by the condition-label tuple (non-control conditions only)
    cond_view = obs.loc[~is_ctrl, spec.condition_cols].drop_duplicates()
    conditions: dict[str, dict[str, list]] = {}
    for row in cond_view.itertuples(index=False, name=None):
        emb = {}
        for col, val in zip(spec.condition_cols, row, strict=True):
            if col in spec.reps:
                emb[col] = np.asarray(rep_store[spec.reps[col]][val])
            else:
                emb[col] = np.asarray([val])  # numeric covariate (e.g. dosage) used as-is
        conditions[json.dumps(list(row), default=_json_default)] = {c: a.tolist() for c, a in emb.items()}

    meta = {
        "group_cols": list(spec.group_cols),
        "condition_cols": list(spec.condition_cols),
        "control_values": spec.control_values,
        "reps": dict(spec.reps),
        "sample_rep": spec.sample_rep,
    }
    adata.uns[UNS_KEY] = {"meta_json": json.dumps(meta, default=_json_default), "conditions": conditions}
    return ClassMapping.in_memory(adata)
