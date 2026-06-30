# Shared data-layer with sc-flow-tools — status & plan (handoff)

Portable record of the design work on branch `feat/annbatch-data-layer`. (Auto-memory lives on a
single machine; this doc travels with the branch.) Last updated 2026-06-30.

## TL;DR

CellFlow2 (scaleflow) and **sc-flow-tools** (theislab, Lorenzo Consoli) should share one data layer.
sc-flow-tools already has nearly all of it (grouping, matching, representations, containers,
preprocessing, an in-memory sampler, jax+torch nn backends). **The one thing it lacks is annbatch
out-of-core streaming.** So the plan: **build the annbatch streaming loader in sc-flow-tools, and
CellFlow2 depends on sc-flow-tools and retires its own data layer** (keeping only model/solvers/training).

## Decisions reached (converged)

- **Grouping**: converge on sc-flow-tools' `HierarchicalIndexer` — a `pd.MultiIndex` built from
  Categorical `.cat.codes`, computed **on the fly from `obs`**; lexsorted lookups → contiguous slices;
  robust to new categories (no fragile `groupby.ngroup` integer ids).
- **Representations**: the only thing persisted separately is the embeddings — `uns` lookup tables
  `{covariate_value: embedding}`, referenced by reps keys (`conditions_reps`/`groups_reps`). Grouping/
  conditions are NOT persisted; deduced from `obs` at load. (= scaleflow's existing on-disk `rep_path`.)
- **Coupling/matching**: adopt sc-flow-tools' `control_values_dict` + `matched_keys`. Default
  (one-to-many, source = control subpop in same group/context) **is** scaleflow's "control = same group".
  `matched_keys: dict[tuple, tuple]` adds explicit fixed source→target pairings. `control_values`/
  `matched_keys` can be passed at inference for arbitrary control selection.
- **Sampler**: shared sampler = scaleflow's annbatch `ClassSampler` **one-class-per-batch** loader
  (both tools use it). sc-flow-tools' multi-node weighted `TrainSampler` added later as an alternative.
- **Python ≥3.12** (annbatch uses PEP 695 `type` aliases).
- **Ownership direction (pending final confirm with Lorenzo)**: sc-flow-tools hosts the shared data
  layer; CellFlow2 depends on it. Needs the data layer landed on `main`/released for a stable dep.

## sc-flow-tools inventory (branches `feat/rebased/preproc` / `feat/setup-repository`, not `main`)

HAS: `data/grouping/_indexer.py` (HierarchicalIndexer), `_selector.py`/`_query.py`;
`data/containers/` (`DistributionData`, `StateData`, `CouplingData`, `CategoricalData`, `NestedData`
= the matched-subpopulation *tree*); `data/schemas/` (Condition/Coupling/Groups/Response/State);
`data/_manager.py` `DataManager` (conditions, `*_reps`, `control_values_dict`, `matched_keys`,
coupling `source_rep`/`n_shared_dims`); `preprocessing/` (transforms: PCA, zscore); `data/samplers/`
(`TrainSampler` — in-memory, tree-based, multi-node-per-batch); jax + torch nn backends.

LACKS: **annbatch / `DatasetCollection` out-of-core streaming** (TrainSampler assumes cells are
already materialized in memory).

## The seam: loading vs indexing in sc-flow-tools (verified locally)

- **Indexing/matching is `obs`-only**: `_get_matched_distributions` keys off `data.ann_df`; the
  indexer reads obs Categorical cols; conditions read `obs` + `uns`. The matched-subpopulation tree is
  an **index** structure — no `X` needed.
- **Loading is one materialization point**: `schemas/_base_schema._extract_array` →
  `adata.X if repr is None else adata.obsm[repr]` (eager) → `StateData(X)`; `StateData.__getitem__(idxs)`
  = `self.X[idxs]`; `TrainSampler._dispatch_node` draws `batch_size` rows per node by slicing.
- So the **only** thing fusing loading to indexing is that tree nodes carry an *eagerly materialized*
  `StateData(X)`. The access pattern is already "fetch these row indices" — that IS the load op.

## Integration plan (in sc-flow-tools), phased

**Phase 1 — separate loading/indexing + DatasetCollection backing** (the prerequisite). Make tree nodes
carry **row indices** (not arrays); add a cell-source abstraction (`AnnData` → numpy slice;
`DatasetCollection` → annbatch read); lazy `StateData`. Touch points (additive — keep eager path
alongside): `containers/_state.py`(+`_distribution.py`), `schemas/_base_schema._extract_array` +
`_state`/`_coupling` schemas, `_manager.get_distribution_data` (accept a `DatasetCollection`, index off
`collection.obs(...)`, defer cells), `samplers/_base._dispatch_node` (stream the sampled idxs).

**Phase 2 — annbatch loader**. Port the prototype (see below) onto the Phase-1 source abstraction:
annbatch `ClassSampler` (chunked, class-coherent, one-condition-per-batch) reading from the collection.
NOTE: scflow's `TrainSampler` draws *arbitrary* rows per node → scattered disk reads (slow); the
ClassSampler loader is what makes DatasetCollection backing *fast*. Phase 1 = correct out-of-core;
Phase 2 = efficient.

**Coordination**: Phase 1 edits scflow core (Lorenzo's) and is the #1 sync item ("can the
matched-distribution computation run on obs alone, cells streamed?"). Do it as a **branch + draft PR**
(additive, minimal) — the PR doubles as the sync artifact. The code already points this way, so it's a
contained refactor, not a redesign.

## Prototype already built (this branch): `src/scaleflow/classmap/`

Self-contained subpackage (deps: anndata+annbatch+numpy+pandas only). Commit `2b66433`.
- `_indexer.py` — `HierarchicalIndexer` **vendored** from sc-flow-tools (the shared grouping base).
- `_source.py` — annbatch source seam (in-memory `add_adata` slices vs on-disk `DatasetCollection`).
  **This + `_loader.py` logic are the genuinely-new, upstreamable pieces.**
- `_container.py`/`_loader.py`/`_prepare.py` — PRE-SYNC drafts (duplicate scflow's mature versions;
  not for upstream — drop in favor of scflow's).
- `_anndata_location.py` — copied (decoupled). `tests/classmap/test_smoke.py` — green (incl. e2e sample).

## Repo state (branch `feat/annbatch-data-layer`)

- `e840023` — baseline (CLAUDE.md; collection-based data layer).
- `41315ef` — WIP in-memory + GroupedDistribution→AnnData wrapper refactor. **NON-IMPORTING, abandoned
  approach**, kept for reference only (superseded by the converge-on-scflow plan).
- `2b66433` — added `scaleflow.classmap` prototype; **data layer restored to `e840023` baseline**
  (imports + tests green). Current functional baseline.
- This doc — handoff.

`scaleflow.data` is the original collection-based layer (green). `scaleflow.classmap` is the prototype.

## Next steps (resume here)

1. Draft **Phase 1** on a branch in `sc-flow-tools` (cell-source abstraction + index-backed `StateData`
   + obs-only `get_distribution_data` accepting a `DatasetCollection`) → draft PR for Lorenzo.
2. **Phase 2**: port `classmap/_source.py` + `_loader.py` logic onto Phase 1 (annbatch ClassSampler).
3. Point `ScaleFlow` (CellFlow2 model) at the sc-flow-tools data layer; retire `scaleflow.data` +
   `scaleflow.classmap`.

Open sync items with Lorenzo: hosting/ownership + landing the data layer on `main`; confirm Phase-1
obs-only matched-index split fits his model; final `obs`/`uns` contract key names; batch contract
(`{src_cell_data, tgt_cell_data, condition}`) vs his node-dispatch output.

Local refs: sc-flow-tools checked out at `/Users/selman/projects/sc-flow-tools` (machine-local — on
another computer, clone `theislab/sc-flow-tools`; data layer is on the feature branches, not `main`).
