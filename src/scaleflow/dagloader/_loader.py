"""``DAGClassLoader`` — streams matched ``{source, target, condition}`` batches from a :class:`Scheme`.

Every node streams through its own :class:`~scaleflow.dagloader.ScheduledClassSampler` + annbatch
``Loader``, configured by a :class:`SamplerConfig` (one shared, or one per node). Each pass, the loader
draws the root's per-batch category schedule from the root's weights and *derives* each bound child's
schedule from the parent's (via the bind's shared columns), pushes the schedules onto the samplers, and
zips the loaders batch-for-batch. See ``README.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

import numpy as np
import pandas as pd
from annbatch import Loader

from scaleflow.dagloader._io import key_backings, leaf_codes, obs_columns
from scaleflow.dagloader._schema import Bind, Container, SamplerConfig, Scheme, _weight_vector
from scaleflow.dagloader._scheduled_sampler import ScheduledClassSampler

__all__ = ["DAGClassLoader"]


class DAGClassLoader:
    """Yields ``{"source", "target", "condition"}`` batches; every node streams through its own loader."""

    def __init__(
        self,
        scheme: Scheme,
        sampler_config: SamplerConfig | Mapping[str, SamplerConfig],
        condition_fn: Callable[[tuple], np.ndarray] | None = None,
    ) -> None:
        self.s = scheme
        self._cond_fn = condition_fn
        self._n_batches = scheme.steps_per_pass
        self._cfg = self._resolve_configs(sampler_config)
        self._B = next(iter(self._cfg.values())).batch_size  # uniform across nodes (validated)

        self._children: dict[str, list[Bind]] = {}
        for b in scheme.binds:
            self._children.setdefault(b.parent, []).append(b)

        # per-node independent RNG streams spawned from one seed (assigned by sorted node name)
        seqs = np.random.SeedSequence(scheme.seed).spawn(len(scheme.nodes))
        self._rngs = {name: np.random.default_rng(s) for name, s in zip(sorted(scheme.nodes), seqs)}

        # per-node leaf partition + weights (obs only — no cell matrices)
        self._st: dict[str, dict] = {}
        for name, node in scheme.nodes.items():
            obs = obs_columns(scheme.sources[node.source], node.cols)
            codes, leaves = leaf_codes(obs, node.cols)
            self._st[name] = {"node": node, "codes": codes, "leaves": leaves, "w": _weight_vector(node.weights, leaves)}

        self._build_loaders()
        self._build_bind_maps()
        self._iters: dict[str, Iterator[dict]] | None = None
        self._schedules: dict[str, np.ndarray] = {}
        self._pos = 0

    def _resolve_configs(self, cfg: SamplerConfig | Mapping[str, SamplerConfig]) -> dict[str, SamplerConfig]:
        """Normalize to one config per node and enforce a uniform batch_size (batches must align)."""
        if isinstance(cfg, SamplerConfig):
            resolved = {name: cfg for name in self.s.nodes}
        else:
            missing = set(self.s.nodes) - set(cfg)
            if missing:
                raise ValueError(f"sampler_config mapping is missing node(s): {sorted(missing)}.")
            resolved = {name: cfg[name] for name in self.s.nodes}
        batch_sizes = {c.batch_size for c in resolved.values()}
        if len(batch_sizes) > 1:
            raise ValueError(f"all nodes must share batch_size (target rows == source rows); got {batch_sizes}.")
        return resolved

    # ── build ────────────────────────────────────────────────────────────
    def _build_loaders(self) -> None:
        """One ScheduledClassSampler + Loader per node, configured by its SamplerConfig."""
        self._samplers: dict[str, ScheduledClassSampler] = {}
        self._loaders: dict[str, Loader] = {}
        self._srcs: dict[str, Container] = {}
        num_samples = self._n_batches * self._B  # steps_per_pass full batches per pass
        for name, st in self._st.items():
            node = st["node"]
            src = self.s.sources[node.source]
            cfg = self._cfg[name]
            K = len(st["leaves"])
            classes = pd.Categorical([str(c) for c in st["codes"]], categories=[str(i) for i in range(K)])
            try:  # annbatch enforces its own run-length rule for chunk>1; forward with node context
                sampler = ScheduledClassSampler(
                    chunk_size=cfg.chunk_size, preload_nchunks=cfg.resolved_preload, batch_size=cfg.batch_size,
                    classes=classes, num_samples=num_samples, class_weights=st["w"],
                    drop_last=True, rng=self._rngs[name],
                )
            except ValueError as e:
                raise ValueError(f"node {name!r}: {e}") from e
            loader = Loader(batch_sampler=sampler, return_index=(name == self.s.root),
                            to_torch=False, preload_to_gpu=False).add_datasets(key_backings(src, node.key))
            self._samplers[name] = sampler
            self._loaders[name] = loader
            self._srcs[name] = src

    def _build_bind_maps(self) -> None:
        """Precompute, per bound child, the maps to turn the parent's schedule into the child's."""
        self._bindmap: dict[str, dict] = {}
        for b in self._children.get(self.s.root, []):
            rst, cst = self._st[b.parent], self._st[b.child]
            rcols, ccols = rst["node"].cols, cst["node"].cols
            # parent leaf code → shared-column value
            root_code_to_cv = {i: tuple(lf[rcols.index(c)] for c in b.common) for i, lf in enumerate(rst["leaves"])}
            # shared-column value → positive-weight child leaf codes carrying it
            common_to_child: dict[tuple, list[int]] = {}
            for code, lf in enumerate(cst["leaves"]):
                if cst["w"][code] > 0:
                    common_to_child.setdefault(tuple(lf[ccols.index(c)] for c in b.common), []).append(code)
            self._bindmap[b.child] = {
                "root_code_to_cv": root_code_to_cv,
                "common_to_child": common_to_child,
                "positive_child_codes": np.flatnonzero(cst["w"] > 0).astype(np.int64),
            }

    # ── per-pass scheduling ────────────────────────────────────────────────
    def _draw_root_schedule(self) -> np.ndarray:
        """Draw ``steps_per_pass`` root leaf codes ∝ the root's weights (the root's own RNG stream)."""
        w = self._st[self.s.root]["w"]
        pos = np.flatnonzero(w > 0)
        return self._rngs[self.s.root].choice(pos, size=self._n_batches, p=w[pos] / w[pos].sum()).astype(np.int64)

    def _derive_child_schedule(self, child: str, root_sched: np.ndarray) -> np.ndarray:
        """Map each root batch's category to a matching child leaf (child's RNG for ties/fallback)."""
        m = self._bindmap[child]
        rng = self._rngs[child]
        out = np.empty(len(root_sched), dtype=np.int64)
        for j, rc in enumerate(root_sched):
            cands = m["common_to_child"].get(m["root_code_to_cv"][int(rc)])
            if not cands:  # no child leaf shares the parent's value → unconditional fallback
                cands = m["positive_child_codes"]
            out[j] = int(cands[0]) if len(cands) == 1 else int(rng.choice(cands))
        return out

    def _start_pass(self) -> None:
        """Draw schedules (root + derived children), push them onto the samplers, rebuild iterators.

        Ordering matters: ``Loader.__iter__`` re-reads ``sampler.sample()`` and the scheduled draw is
        read up front, so ``set_schedule`` must land before ``iter(loader)``.
        """
        root_sched = self._draw_root_schedule()
        self._schedules = {self.s.root: root_sched}
        self._samplers[self.s.root].set_schedule(root_sched)
        for b in self._children.get(self.s.root, []):
            child_sched = self._derive_child_schedule(b.child, root_sched)
            self._schedules[b.child] = child_sched
            self._samplers[b.child].set_schedule(child_sched)
        self._iters = {name: iter(ld) for name, ld in self._loaders.items()}
        self._pos = 0

    # ── iteration ──────────────────────────────────────────────────────────
    def __iter__(self) -> DAGClassLoader:
        return self

    def __next__(self) -> dict[str, np.ndarray]:
        if self._iters is None or self._pos >= self._n_batches:
            self._start_pass()
        j = self._pos
        self._pos += 1

        st = self._st[self.s.root]
        root_batch = next(self._iters[self.s.root])
        leaf = st["leaves"][int(self._schedules[self.s.root][j])]  # per-batch category — from the schedule
        target = np.asarray(root_batch["X"], dtype=np.float32)     # the node's streamed rep (Node.key)
        B = target.shape[0]

        out: dict = {"target": target}
        if self._cond_fn is not None:
            cond = np.asarray(self._cond_fn(leaf), dtype=np.float32)
            out["condition"] = np.broadcast_to(cond, (B, cond.shape[-1])).copy()

        for b in self._children.get(self.s.root, []):  # bound child source, conditioned via its schedule
            child_batch = next(self._iters[b.child])
            out["source"] = np.asarray(child_batch["X"], dtype=np.float32)
        return out
