"""ClassMappingAnnbatchLoader — the annbatch streaming loader (scaleflow's contribution).

Plain class (no scaleflow ``SamplerABC``, no ``init_sampler`` pattern): everything is set up in
``__init__`` and ``sample()`` pulls the next class-coherent batch from the internal annbatch
``Loader``. Source (control) cells are loaded into memory once via mask-range slices; the internal
``ClassSampler`` is driven by the per-row TARGET categories.

PRE-SYNC DRAFT: the class-building + source pairing (group=context/source, condition=perturbation,
pair = same group) follows the natural reading of sc-flow-tools' hierarchy, but the precise
container/coupling model is to be reconciled upstream (see ``shared-data-layer-scflow``). Marked
NOTE(sync) below.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from annbatch.samplers import ClassSampler

from scaleflow.classmap._container import ClassMapping
from scaleflow.classmap._source import as_source, make_loader

__all__ = ["ClassMappingAnnbatchLoader"]

_EXCLUDED = "__excluded__"  # ClassSampler sentinel (weight 0) for control / non-sampled rows


class ClassMappingAnnbatchLoader:
    """Yield matched ``{src_cell_data, tgt_cell_data, condition}`` batches from a single-domain ClassMapping."""

    def __init__(
        self,
        class_mapping: ClassMapping,
        *,
        batch_size: int = 1024,
        chunk_size: int | None = None,
        preload_nchunks: int | None = None,
        seed: int = 0,
        condition_transform=None,
        n_batches_per_pass: int = 4096,
    ) -> None:
        if len(class_mapping) != 1:
            raise NotImplementedError("ClassMappingAnnbatchLoader supports a single domain for now.")
        self._cm = class_mapping
        self._domain = d = class_mapping[class_mapping.names[0]]
        self._source = as_source(class_mapping.container)
        self.batch_size = batch_size
        self._condition_transform = condition_transform
        self._n_batches_per_pass = int(n_batches_per_pass)

        # in-memory ⇒ per-row sampling (chunk_size=1); on-disk ⇒ required
        if self._source.forces_chunk_size_one:
            if chunk_size not in (None, 1):
                raise ValueError("In-memory training uses chunk_size=1; do not pass another value.")
            chunk_size = 1
        elif chunk_size is None:
            raise ValueError("chunk_size is required for on-disk training.")
        self._chunk_size = int(chunk_size)

        cls_ss, src_ss = np.random.SeedSequence(seed).spawn(2)
        self._src_rng = np.random.default_rng(src_ss)

        obs = d.adata.obs
        is_ctrl = np.asarray(d.is_control)

        # NOTE(sync): combined per-row keys; group = context/source, condition = perturbation.
        group_key = self._combined_key(obs, d.group_cols)
        cond_key = self._combined_key(obs, d.condition_cols)

        # SOURCE cache: control rows grouped by GROUP key, read once via contiguous slices.
        self._source_cells: dict[Any, Any] = {}
        ctrl_groups = pd.Series(group_key)[is_ctrl]
        for gkey, idx in ctrl_groups.groupby(ctrl_groups, observed=True).groups.items():
            rows = np.asarray(idx)
            sl = slice(int(rows[0]), int(rows[-1]) + 1) if (rows[-1] - rows[0] + 1) == rows.size else rows
            self._source_cells[gkey] = self._source.read_rows({0: sl})[0]

        # TARGET classes: each non-control row → its (group, condition) class; controls → excluded.
        row_class = np.where(is_ctrl, _EXCLUDED, [repr((g, c)) for g, c in zip(group_key, cond_key)])
        target_cats = sorted({c for c in row_class.tolist() if c != _EXCLUDED})
        categories = [*target_cats, _EXCLUDED]
        codes = pd.Categorical(row_class, categories=categories).codes.astype(np.int64)
        classes = pd.Categorical.from_codes(codes, categories=categories)

        # recover (group_key, cond_key, condition_tuple) per class code for sampling
        self._class_to_group: dict[int, Any] = {}
        self._class_to_cond: dict[int, tuple] = {}
        for code, label in enumerate(target_cats):
            i = int(np.flatnonzero(row_class == label)[0])
            self._class_to_group[code] = group_key[i]
            self._class_to_cond[code] = tuple(cond_key[i]) if isinstance(cond_key[i], tuple) else (cond_key[i],)
        self._row_to_code = codes

        class_weights = np.concatenate([np.ones(len(target_cats)), [0.0]])
        group_chunks = self.batch_size // np.gcd(self._chunk_size, self.batch_size)
        preload = preload_nchunks if preload_nchunks is not None else int(group_chunks) * 4
        sampler = ClassSampler(
            chunk_size=self._chunk_size,
            preload_nchunks=preload,
            batch_size=self.batch_size,
            classes=classes,
            num_samples=self._n_batches_per_pass * self.batch_size,
            class_weights=class_weights,
            drop_last=True,
            rng=np.random.default_rng(cls_ss),
        )
        self._loader = make_loader(self._source, batch_sampler=sampler, return_index=True)
        self._iter = iter(self._loader)

    @staticmethod
    def _combined_key(obs: pd.DataFrame, cols: list[str]) -> list:
        if not cols:
            return [()] * len(obs)
        return list(map(tuple, obs[cols].to_numpy()))

    def sample(self) -> dict[str, Any]:
        try:
            batch = next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            batch = next(self._iter)
        code = int(self._row_to_code[int(np.asarray(batch["index"])[0])])
        x = self._source_cells[self._class_to_group[code]]
        src = x[self._src_rng.integers(0, x.shape[0], size=self.batch_size)]
        cond = self._domain.conditions[self._class_to_cond[code]]
        if self._condition_transform is not None:
            cond = self._condition_transform(cond)
        return {"src_cell_data": src, "tgt_cell_data": batch["X"], "condition": cond}

    @property
    def data(self) -> ClassMapping:
        return self._cm
