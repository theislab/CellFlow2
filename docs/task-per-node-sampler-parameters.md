# Task: a per-node sampling-parameter framework (kill the hidden ClassSampler settings)

Audience: an agent picking this up to **design + build it in `src/scaleflow/treesampler.py`, discussing
the API with the user before finalizing.** This is context + goal — design the shape yourself; don't
inherit a prescribed API.

## Background

`TreeSampler._build_root_loader` streams the root node through annbatch `ClassSampler` with a block of
**hardcoded, hidden settings**:

```python
ClassSampler(
    chunk_size=1,        # per-row
    preload_nchunks=B,   # B = scheme.n_rows_per_leaf
    batch_size=B,
    num_samples=B * 512, # magic
    class_weights=...,
    drop_last=True,
    rng=np.random.default_rng(scheme.seed),
)
```

These are wrong for a generalized framework: not exposed, not per-node, and partly magic (`B*512`,
`chunk_size=1`). Sampling behavior should be **controlled and explicit**, not buried.

## Goal

A framework to decide the annbatch/sampler parameters **per node** — explicit and overridable, with
sensible **data-derived defaults** and validation. Each node/source can need different values, so the
choice belongs at the node level, not a global constant.

## Settings to bring under the framework (currently hardcoded)

- **`chunk_size`** — annbatch read-slice size. `1` ⇒ per-row (any layout, no run-length constraint) but
  scattered/slow on disk. `>1` ⇒ contiguous chunked reads (fast) **but** requires each sampled class's
  rows to be contiguous runs `≥ chunk_size` ⇒ the source **sorted by condition**. So `chunk_size` is
  coupled to sortedness and to each leaf's cell count.
- **`preload_nchunks`** — chunks per read window. Constraint: `chunk_size * preload_nchunks` divisible by
  `batch_size`.
- **`batch_size`** — currently `= scheme.n_rows_per_leaf`. Decide whether these stay unified.
- **`num_samples`** — length of one with-replacement pass (`ClassSampler` has no epoch; the loader
  restarts). Currently magic `B*512`. Governs precompute size, restart cadence, and the exact draw
  sequence for a seed.
- **`drop_last`** — batch completeness.
- **`rng` / seed** — reproducibility. Needs a scheme for **per-node independent RNG streams** (spawned
  from one root seed) so nodes don't correlate.
- **(related) sortedness** — whether the framework should *require* and/or *produce* a condition-sorted
  source (cf. the old `write_sorted_collection`) to unlock `chunk_size>1`.

## annbatch facts the framework must respect

- `ClassSampler` is **with-replacement** (no epoch); `num_samples` bounds a pass.
- **Run-length rule**: every positive-weight class's contiguous run must be `≥ chunk_size`, else
  `ClassSampler` raises → `chunk_size>1` demands sorted-by-condition layout AND each sampled leaf having
  `≥ chunk_size` cells.
- `chunk_size * preload_nchunks % batch_size == 0`.
- Single worker only.
- `return_index=True` is how the batch's leaf/condition is recovered.

## Per-node rationale

Nodes differ in size/layout/role, so one global setting can't be right:
- **root** (streamed target, possibly huge, on-disk, sortable) → `chunk_size>1` for throughput.
- **control/child** (small, reused, cached — doesn't even use `ClassSampler`) → different or N/A.

So the framework should let each node **declare or derive** its sampler params, **validated against its
data** (is it sorted? does every sampled leaf have `≥ chunk_size` cells? divisibility?).

## Open questions to settle with the user

1. Explicit per-node config object vs. derived-with-overrides? (Defaults derivable: `chunk_size` from
   sortedness + min leaf size; `preload_nchunks` from `batch_size`; `num_samples` from a target #steps.)
2. Where does `batch_size` / `n_rows_per_leaf` live — scheme-level or per-node?
3. `num_samples`/epoch semantics — expose as "steps per pass," or infinite-with-fixed-restart?
4. Reproducibility — the per-node RNG-spawning scheme.
5. Should the framework own the **sortedness requirement** (validate and/or produce a sorted source)?

## Where

`src/scaleflow/treesampler.py` (`_build_root_loader` holds the hardcoded block). Keep the project's
principles from the design discussion: no hidden settings; one source of truth; conveniences in
constructors/factories, not in the schema. Design the per-node parameter API + validation, then confirm
the shape with the user before building it out.

## Resolution — what was built (2026-07-06)

Design settled with the user and implemented in `src/scaleflow/dag_class_loader.py` — the loader class
is **`DAGClassLoader`** (21 tests green: unit in `tests/test_dag_class_loader.py`, end-to-end training in
`tests/test_dag_class_loader_train.py`; shared toy data in `tests/_toydata.py`). The hardcoded
`ClassSampler` block is gone; every node now streams through its own **`ScheduledClassSampler`** — an
annbatch `ClassSampler` subclass whose per-batch category sequence is *supplied* (`set_schedule`)
instead of drawn internally. This keeps annbatch's **chunk math** (contiguous `chunk_size` reads, RLE,
slice sampling — the throughput win) while letting `DAGClassLoader` own *which* category each batch
draws. Only `_iter_requests` is copied from annbatch, with
one line changed to source `group_classes` from a `_group_positions(n_groups)` hook → **candidate to
upstream** (if annbatch adds that hook, the subclass collapses to a one-method override).

Decisions on the open questions:
1. **Per-node vs derived** → the only genuine per-node knob is **`Node.chunk_size`** (`None`⇒1); every
   other param is a principled *derivation*, not magic. `chunk_size>1` is validated (run-length rule,
   wrapped with node context) so opting into throughput is explicit, not hidden.
2. **`batch_size` location** → **scheme-level** (`Scheme.n_rows_per_leaf`): a yielded batch has one row
   count (target rows == source rows == B), so it can't vary per node.
3. **`num_samples`/epoch** → **`Scheme.steps_per_pass`** (default 512); `num_samples = steps_per_pass *
   batch_size`, `drop_last=True`, loader restarts each pass → effectively infinite with a fixed,
   reproducible restart cadence. `preload_nchunks = batch_size // chunk_size` (one batch per window).
4. **Reproducibility** → per-node RNG spawned from one `SeedSequence(seed)` (one stream per node, by
   sorted name); fixes the old shared-seed correlation bug.
5. **Sortedness** → **validate only** (run-length rule at build). Producing a sorted source stays a
   separate concern (not in the schema).

Key constraint (validated): **`chunk_size` divides `batch_size`** ⇒ exactly one category per batch ⇒ a
schedule is a length-`steps_per_pass` array of leaf codes, and root/child align batch-for-batch
regardless of each node's own `chunk_size`. **The bind falls out for free**: the root schedule is drawn
from the root's weights; each bound child's schedule is *derived* from the parent's (parent leaf →
shared-column value → matching child leaf, child RNG for ties/fallback), so loaders zip with no per-step
reconfiguration. `DAGClassLoader._start_pass()` draws + pushes all schedules, then rebuilds the iterators
(order matters: `Loader.__iter__` re-reads `sampler.sample()`, so `set_schedule` must land before
`iter(loader)`). Bound children are now streamed (their own loader) instead of the old in-memory cache.
