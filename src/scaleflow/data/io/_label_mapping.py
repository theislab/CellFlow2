from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

# TODO: Generalize to IndexedMapping[V] with pluggable ValueSerializer strategy
# to unify with conditions data (CSRArraySerializer for dict[str, ndarray])

__all__ = ["CSRLabelMapping"]


@dataclass
class CSRLabelMapping:
    """Simple storage for dist_idx -> tuple[label_values] mappings.

    Stores labels as JSON-encoded strings for flexibility with ragged data.
    Provides efficient zarr I/O by writing just 2 arrays (indices + json_labels).

    Parameters
    ----------
    indices
        Array of distribution indices (n_dists,).
    json_labels
        Array of JSON-encoded label tuples (n_dists,).
    """

    indices: np.ndarray  # (n_dists,) distribution IDs
    json_labels: np.ndarray  # (n_dists,) JSON-encoded label strings
    _lookup: dict[int, int] | None = None  # lazy-built: dist_idx -> position

    @classmethod
    def from_dict(cls, mapping: dict[int, Any]) -> "CSRLabelMapping":
        """Create CSRLabelMapping from a dict.

        Parameters
        ----------
        mapping
            Dict mapping distribution indices to label tuples/lists.

        Returns
        -------
        CSRLabelMapping
            New instance.
        """
        if not mapping:
            return cls(
                indices=np.array([], dtype=np.int64),
                json_labels=np.array([], dtype=object),
            )

        sorted_keys = sorted(mapping.keys())
        indices = np.array(sorted_keys, dtype=np.int64)

        # Encode each label tuple as JSON string
        json_labels = np.array([json.dumps(list(mapping[k])) for k in sorted_keys], dtype=object)

        return cls(indices=indices, json_labels=json_labels)

    def to_dict(self) -> dict[int, tuple]:
        """Convert back to dict format.

        Returns
        -------
        dict[int, tuple]
            Dict mapping distribution indices to tuples of label values.
        """
        result = {}
        for i, dist_idx in enumerate(self.indices):
            result[int(dist_idx)] = tuple(json.loads(self.json_labels[i]))
        return result

    def _build_lookup(self) -> None:
        """Build the index lookup table if not already built."""
        if self._lookup is None:
            self._lookup = {int(idx): i for i, idx in enumerate(self.indices)}

    def __getitem__(self, dist_idx: int) -> tuple:
        """Get labels for a distribution index."""
        self._build_lookup()
        if dist_idx not in self._lookup:
            raise KeyError(dist_idx)
        pos = self._lookup[dist_idx]
        return tuple(json.loads(self.json_labels[pos]))

    def get(self, dist_idx: int, default=None) -> tuple | None:
        """Get labels with a default."""
        try:
            return self[dist_idx]
        except KeyError:
            return default

    def __contains__(self, dist_idx: int) -> bool:
        self._build_lookup()
        return dist_idx in self._lookup

    def __len__(self) -> int:
        return len(self.indices)

    def keys(self):
        return (int(idx) for idx in self.indices)

    def values(self):
        for label in self.json_labels:
            yield tuple(json.loads(label))

    def items(self):
        for i, dist_idx in enumerate(self.indices):
            yield int(dist_idx), tuple(json.loads(self.json_labels[i]))

    def write_zarr(self, group: zarr.Group, name: str) -> None:
        """Write to zarr as 2 arrays (indices + json_labels)."""
        subgroup = group.create_group(name)
        subgroup.create_array("indices", data=self.indices)
        subgroup.create_array("json_labels", data=self.json_labels.astype(str))

    @classmethod
    def read_zarr(cls, group: zarr.Group) -> "CSRLabelMapping":
        """Read from zarr group."""
        indices = np.asarray(group["indices"])
        json_labels = np.asarray(group["json_labels"]).astype(object)
        return cls(indices=indices, json_labels=json_labels)

    def __repr__(self) -> str:
        return f"CSRLabelMapping(n_dists={len(self)})"
