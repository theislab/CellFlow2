"""classmap — shared, self-contained class-mapping data layer for flow-matching on single cells.

This subpackage is the candidate **shared module** between scaleflow (CellFlow2) and sc-flow-tools.
It depends only on ``anndata`` + ``annbatch`` + ``numpy``/``pandas`` (no other ``scaleflow`` imports),
so extraction to a standalone package later is just moving this folder + adding a ``pyproject``.

Boundary / standards (see the ``shared-data-layer-scflow`` decision):
- ``container`` is the only cell store (in-memory AnnData or on-disk annbatch DatasetCollection).
- :class:`Domain` exposes *mappings only* (no cells), backed by an AnnData's obs/uns.
- grouping is hierarchical via :class:`HierarchicalIndexer` (MultiIndex from Categorical codes →
  contiguous slices; robust to new categories) — VENDORED from sc-flow-tools (Lorenzo Consoli).
- scaleflow contributes :class:`ClassMappingAnnbatchLoader` (annbatch ClassSampler streaming +
  in-memory mask-range source cache), which sc-flow-tools lacks.

STATUS: pre-sync. The grouping/indexer and source seam are stable; the container/loader/prepare
glue is a first cut to be reconciled with sc-flow-tools' ``DataManager``/containers after the sync.
Requires Python ≥3.12 (annbatch uses PEP 695).
"""

from scaleflow.classmap._anndata_location import AnnDataLocation
from scaleflow.classmap._container import DEFAULT_DOMAIN, UNS_KEY, ClassMapping, Domain
from scaleflow.classmap._indexer import (
    BASE_LEVEL_NAME,
    CONDITION_LEVEL_NAME,
    GROUP_LEVEL_NAME,
    HierarchicalIndexer,
)
from scaleflow.classmap._loader import ClassMappingAnnbatchLoader
from scaleflow.classmap._prepare import GroupingSpec, prepare
from scaleflow.classmap._source import as_source, make_loader, read_distributions

__all__ = [
    "AnnDataLocation",
    "HierarchicalIndexer",
    "BASE_LEVEL_NAME",
    "GROUP_LEVEL_NAME",
    "CONDITION_LEVEL_NAME",
    "Domain",
    "ClassMapping",
    "DEFAULT_DOMAIN",
    "UNS_KEY",
    "ClassMappingAnnbatchLoader",
    "GroupingSpec",
    "prepare",
    "as_source",
    "make_loader",
    "read_distributions",
]
