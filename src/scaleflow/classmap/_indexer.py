"""Hierarchical MultiIndex grouping.

PORTED from sc-flow-tools (``theislab/sc-flow-tools``, author Lorenzo Consoli), branch
``feat/rebased/preproc``: ``src/sc_flow/data/grouping/_indexer.py``. Vendored here so the
shared data layer is self-contained and extractable; this is the agreed common foundation
(see the ``shared-data-layer-scflow`` decision). **Pending the upstream sync on ownership**,
treat sc-flow-tools as the source of truth and re-sync before diverging.

Why this over per-row integer ids (``groupby.ngroup``): the MultiIndex key is the *category
tuple* (built from ``Categorical.cat.codes``), so adding a new category never renumbers existing
groups, and a lexsorted MultiIndex maps each group to a contiguous ``slice`` via ``get_loc``.
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd

__all__ = ["HierarchicalIndexer", "BASE_LEVEL_NAME", "GROUP_LEVEL_NAME", "CONDITION_LEVEL_NAME"]

# Level names (inlined from sc_flow._constants).
BASE_LEVEL_NAME = "base"
GROUP_LEVEL_NAME = "groups"
CONDITION_LEVEL_NAME = "conditions"


def _check_columns_present(query: Collection[str], reference: Collection[str]) -> None:
    """Raise if any ``query`` column is absent from ``reference`` (the df columns)."""
    missing = set(query) - set(reference)
    if missing:
        raise ValueError(f"The following columns are missing from the dataframe: {sorted(missing)}")


class HierarchicalIndexer:
    """Hierarchical indexing of a dataframe into subpopulations via a ``pd.MultiIndex``.

    Two levels are always present — GROUP (context/source) and CONDITION (perturbation); more can
    be stacked. Each level is a set of columns; every column becomes a sub-level named
    ``(level_name, col)``. Columns must be Categorical (codes are used directly — no factorization),
    which keeps group ids stable across new categories.
    """

    def __init__(
        self,
        groups_cols: Collection[str] | None = None,
        conditions_cols: Collection[str] | None = None,
    ) -> None:
        self._groups_cols = [] if groups_cols is None else list(groups_cols)
        self._conditions_cols = [] if conditions_cols is None else list(conditions_cols)
        self._init_registry()

    def _init_registry(self) -> None:
        self._registry: dict[str, list[str]] = {
            GROUP_LEVEL_NAME: sorted(self._groups_cols),
            CONDITION_LEVEL_NAME: sorted(self._conditions_cols),
        }
        self._hierarchy_levels: list[str] = [GROUP_LEVEL_NAME, CONDITION_LEVEL_NAME]

    def create_index(self, df: pd.DataFrame) -> pd.MultiIndex:
        """Build the hierarchical ``MultiIndex`` from ``df``'s Categorical columns + its row index."""
        all_levels: list = [df.index]
        all_codes: list = [np.arange(len(df))]
        names: list = [BASE_LEVEL_NAME]

        for level_name, level_cols in self._registry.items():
            _check_columns_present(level_cols, df.columns)
            for col in level_cols:
                series = df[col]
                if not hasattr(series, "cat"):
                    raise TypeError(
                        f"Column '{col}' must be Categorical. Cast with: "
                        f"df['{col}'] = df['{col}'].astype('category')"
                    )
                all_levels.append(series.cat.categories)
                all_codes.append(series.cat.codes.values)
                names.append((level_name, col))

        return pd.MultiIndex(levels=all_levels, codes=all_codes, names=names)

    def update_registry(
        self,
        level_name: str,
        level_cols: Collection[str] | None = None,
        level_hierarchy: int = -1,
        allow_override: bool = False,
    ) -> None:
        """Add (or override) a hierarchy level; ``level_hierarchy`` is its insert position."""
        if level_name in self._registry and not allow_override:
            raise ValueError(f"Level {level_name} already present, cannot override.")
        self._registry[level_name] = [] if level_cols is None else sorted(level_cols)
        self._hierarchy_levels.insert(level_hierarchy, level_name)

    @property
    def groups_cols(self) -> list[str]:
        return self._groups_cols

    @property
    def conditions_cols(self) -> list[str]:
        return self._conditions_cols

    @property
    def hierarchy_levels(self) -> list[str]:
        return self._hierarchy_levels

    @property
    def registry(self) -> dict[str, list[str]]:
        return self._registry

    @property
    def sort_columns(self) -> tuple[str, ...]:
        """Columns to sort the data by (groups then conditions) so each group is contiguous."""
        cols: list[str] = []
        for level_name in self._hierarchy_levels:
            cols.extend(self._registry.get(level_name, []))
        return tuple(cols)
