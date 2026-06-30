"""ClassMapping (cells + per-domain mappings) and Domain (mappings only).

PRE-SYNC DRAFT. The container model overlaps sc-flow-tools' ``data/containers`` + ``DataManager``
(``DistributionData``/``CouplingData``/``control_values_dict``); finalize against upstream after the
sync. What is stable here: the *boundary* (container = the only cells; Domain = mappings only,
backed by an AnnData's obs/uns; grouping via :class:`HierarchicalIndexer`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from scaleflow.classmap._indexer import CONDITION_LEVEL_NAME, GROUP_LEVEL_NAME, HierarchicalIndexer

__all__ = ["Domain", "ClassMapping", "DEFAULT_DOMAIN", "UNS_KEY"]

DEFAULT_DOMAIN = "domain_0"
UNS_KEY = "classmap"  # project-neutral (was scaleflow_*) so both projects share the contract


@dataclass(eq=False)
class Domain:
    """Mappings only (never cells) — a typed view over a metadata AnnData's obs/uns.

    The obs holds the (Categorical) group + condition columns; ``uns["classmap"]`` holds the
    grouping spec, control values, and per-condition embeddings. A :class:`HierarchicalIndexer`
    turns the obs columns into a ``MultiIndex`` whose lexsorted lookups return contiguous slices.
    """

    adata: ad.AnnData

    @cached_property
    def _store(self) -> Mapping[str, Any]:
        if UNS_KEY not in self.adata.uns:
            raise KeyError(f"adata.uns has no {UNS_KEY!r}; build it with classmap.prepare().")
        return self.adata.uns[UNS_KEY]

    @cached_property
    def _meta(self) -> dict:
        return json.loads(self._store["meta_json"])

    @property
    def group_cols(self) -> list[str]:
        return list(self._meta["group_cols"])

    @property
    def condition_cols(self) -> list[str]:
        return list(self._meta["condition_cols"])

    @property
    def control_values(self) -> dict[str, Any]:
        return self._meta["control_values"]

    @cached_property
    def indexer(self) -> HierarchicalIndexer:
        return HierarchicalIndexer(groups_cols=self.group_cols, conditions_cols=self.condition_cols)

    @cached_property
    def index(self) -> pd.MultiIndex:
        """The hierarchical MultiIndex over the (sorted) obs rows."""
        return self.indexer.create_index(self.adata.obs)

    @cached_property
    def conditions(self) -> dict[tuple, dict[str, np.ndarray]]:
        """Per-condition embeddings, keyed by the condition-label tuple (stable across new cats)."""
        raw = self._store["conditions"]
        return {tuple(json.loads(k)): {c: np.asarray(a) for c, a in v.items()} for k, v in raw.items()}

    @cached_property
    def is_control(self) -> np.ndarray:
        """Boolean per-row mask: condition columns all equal their control value."""
        obs = self.adata.obs
        mask = np.ones(self.adata.n_obs, dtype=bool)
        for col, ctrl in self.control_values.items():
            vals = obs[col]
            mask &= vals.isna().to_numpy() if pd.isna(ctrl) else (vals == ctrl).to_numpy()
        return mask

    def slice_for(self, level: str, key: tuple) -> slice:
        """Contiguous ``slice`` of rows whose ``level`` sub-key equals ``key`` (sorted data).

        ``level`` is GROUP_LEVEL_NAME or CONDITION_LEVEL_NAME. Raises if the data is not sorted
        (the run is not contiguous) — enforcing the sorted-by-condition invariant.
        """
        loc = self.index.get_locs([slice(None)] + self._level_query(level, key))
        if isinstance(loc, slice):
            return loc
        arr = np.asarray(loc)
        if arr.size and (arr[-1] - arr[0] + 1) == arr.size:
            return slice(int(arr[0]), int(arr[-1]) + 1)
        raise ValueError(f"rows for {level}={key} are not contiguous — container must be sorted by condition.")

    def _level_query(self, level: str, key: tuple) -> list:
        cols = self.group_cols if level == GROUP_LEVEL_NAME else self.condition_cols
        if len(key) != len(cols):
            raise ValueError(f"{level} key {key} does not match columns {cols}.")
        # NOTE(sync): get_locs query construction is a placeholder; reconcile with sc-flow-tools'
        # IndexSelector/QueryFactory which already does robust per-level selection.
        return list(key)


@dataclass(eq=False)
class ClassMapping:
    """Cells (``container``) + per-domain mappings (``domains``).

    ``container`` is the only cell store: in-memory :class:`~anndata.AnnData` or on-disk
    :class:`annbatch.DatasetCollection`. ``domains`` maps domain_name → :class:`Domain`; ≥1 allowed,
    default key ``"domain_0"``. Only :class:`ClassMappingAnnbatchLoader` restricts to one domain.
    """

    container: Any  # ad.AnnData | annbatch.DatasetCollection
    domains: Mapping[str, Domain] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Domain:
        return self.domains[name]

    def __iter__(self):
        return iter(self.domains)

    def __len__(self) -> int:
        return len(self.domains)

    @property
    def names(self) -> list[str]:
        return list(self.domains)

    @classmethod
    def in_memory(cls, adata: ad.AnnData) -> "ClassMapping":
        # standard #5: AnnData container ⇒ single domain whose mappings live in that same adata
        return cls(container=adata, domains={DEFAULT_DOMAIN: Domain(adata)})

    @classmethod
    def on_disk(cls, collection: Any, domains: "ad.AnnData | Mapping[str, ad.AnnData]") -> "ClassMapping":
        if isinstance(domains, ad.AnnData):
            domains = {DEFAULT_DOMAIN: domains}
        return cls(container=collection, domains={name: Domain(a) for name, a in domains.items()})
