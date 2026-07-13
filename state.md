# CellFlow2 / scaleflow ↔ cellflow dedup — handoff state

**Goal of the effort:** deduplicate the `scaleflow` package (this repo, `src/scaleflow`) by
reusing/inheriting from the sibling `cellflow` package instead of maintaining forks.

## Repos & branches

- **scaleflow (this repo):** `/Users/selman.ozleyen/projects/CellFlow2`, branch
  `dedup/reexport-cellflow`, remote `theislab/CellFlow2`.
- **cellflow (sibling):** `/Users/selman.ozleyen/projects/cellflow`, branch `feat/annbatch-loader`.
  scaleflow's test venv installs this cellflow **editable**, so edits there are picked up immediately.
  cellflow packages `src/dagloader` as a top-level `dagloader` module (vendored, to be extracted later).

## Test environment (there is no working repo `.venv` for this)

Dedicated uv venv at `~/.cache/cellflow2-dedup-testvenv` (Python 3.13, editable installs of BOTH
`/Users/selman.ozleyen/projects/cellflow` and this repo `[test]`). Run with `JAX_PLATFORMS=cpu`:

```
JAX_PLATFORMS=cpu ~/.cache/cellflow2-dedup-testvenv/bin/python -m pytest tests/model tests/networks tests/solver tests/trainer -q
```

Recreate if missing:
```
uv venv ~/.cache/cellflow2-dedup-testvenv --python 3.13
uv pip install --python ~/.cache/cellflow2-dedup-testvenv/bin/python -e /Users/selman.ozleyen/projects/cellflow -e ".[test]"
```

## Current status

- **Package (`src/scaleflow`) is GREEN:** `tests/model`, `tests/networks`, `tests/solver`,
  `tests/trainer` all pass (0 failures). cellflow's own suite also green (240 solver/VF).
- **Experiment scripts are BROKEN** (see "OPEN" below) — they import the now-deleted data layer.
  They are NOT in the pytest suite, so they don't affect the green status.
- Pre-existing, unrelated failures (do not touch): `tests/networks/test_condencoder.py`
  (a jax concat-shape encoder inconsistency), `tests/preprocessing/*`.

## What's DONE (dedup arc, newest first)

Solvers, velocity fields, CFG, and now the model/data layer are deduped by **inheritance**
(subclass the cellflow class, override only what genuinely diverges) — NOT copy/paste.

1. **Model + data layer (commit `6e5bb28`):** `ScaleFlow` now subclasses
   `cellflow.model.CellFlow`. It overrides ONLY `__init__` (resolves the `sf_*` solver registry
   then `super().__init__(adata=None, solver=solver)`) and `prepare_model` (scaleflow's VF/solver
   construction). Everything else is inherited: `prepare_data`, **`prepare_annbatch_data` (the
   dagloader streaming path)**, `prepare_validation_data`, `train`, `predict`, `get_condition_embedding`,
   `save`/`load`, all properties. `src/scaleflow/data/` was **deleted entirely** (~2540 lines);
   `tests/data/` deleted; the split-perf benchmark deleted. Trainer (`src/scaleflow/training/_trainer.py`)
   rewired off `scaleflow.data.SamplerABC` (→ `Any`) and to cellflow's sampler contract
   (`sampler.sample(rng_np)` in the train loop, `vdl.sample(mode=mode)` in validation). The
   scaleflow CFG **w-sweep trainer is preserved**: `prepare_model` still builds
   `scaleflow.training.CellFlowTrainer`, and the inherited `train` uses `self.trainer`.
   - **Dropped functionality (noted intentionally):** scaleflow's old `train()` extra params
     (`train_dataloader`/`val_dataloader`/`num_workers`/`prefetch_factor`/`log_every`). cellflow's
     inherited `train` builds the loader from the prepared data instead.
   - `tests/model/test_scaleflow.py` rewritten to cellflow's path: `prepare_data(adata=...)` →
     build a `sample_batch` from `cellflow.data.TrainSampler` → `prepare_model(sample_batch=...)` → `train`.
2. **Velocity fields** all inherit cellflow's (`74ce5ed` Conditional, `8205c47` GENOT, `03c1167` EqM
   via a `_setup_time` hook). Shared adaln helpers (`5f005d4`, `5baebdc`) in `networks/_utils.py`
   (`build_adaln_blocks`/`apply_adaln`). scaleflow keeps only `cell_transformer_*` fields + `adaln_zero`
   + no-time EqM overrides. All param-tree byte-identical (checkpoint-safe, verified).
3. **GENOT classifier-free guidance** added in cellflow (`49389f4`): signature-agnostic
   `ClassifierFreeGuidance.wrap(vf)` (base velocity takes a `force_uncond` flag) shared by OTFM+GENOT;
   GENOT solver got `guidance`/`cfg_enabled`/per-call `guidance_scale` threading `x_0`. scaleflow's
   `sf_genot` inherits it.
4. Earlier: `_normalize_vf_kwargs`/`_match_kwargs` hooks, solver registry (`sf_`-prefixed), CFG
   `condition_null` (zero_embedding / mask_value), per-call guidance upstreamed to OTFM.

See the persistent memory files for full detail: `cellflow2-dedup-effort.md`, `scaleflow-data-layer.md`,
`shared-classmap-package.md` (classmap is now considered OUTDATED — the shared loader is the dagloader).

## OPEN — needs discussion with the user before doing

**Porting the experiment scripts to the dagloader.** `experiments/{train_zarr,train_comparison,
model_comparison,temp_edit}.py` all import the deleted `scaleflow.data` and are built on a
**multi-dataset** workflow that has NO drop-in cellflow equivalent:
- `CombinedSampler` over N `GroupedDistribution`s (train jointly on tahoe + sciplex + …),
- `ReservoirSampler` (pool_fraction / replacement_prob),
- per-dataset `split_datasets` holdout,
- ablation `ConditionTransform`,
- `sf.train(train_dataloader=combined_sampler)`.

cellflow's inherited model is **single-adata** (`prepare_data` → `TrainSampler`) or
**annbatch-collection** (`prepare_annbatch_data` → `dagloader.DAGLoader`). The faithful target is the
**dagloader** (`prepare_annbatch_data` + a multi-source `SamplerConfig`/`Scheme`; see
`dagloader.perturbation_scheme` / `split_scheme` / `resolve_split_configs`), but that:
- needs the data as an **annbatch collection**, not scaleflow's zarr `GroupedDistribution`;
- requires mapping each script's multi-dataset / holdout / reservoir / ablation intent onto the
  scheme API;
- **cannot be verified here** (no test data; experiments not in the pytest suite).

**DISCUSS with the user:** how should the multi-dataset combined/reservoir workflow be expressed on
the dagloader? (multi-source `Scheme` with per-source `Bind`s? one concatenated collection? is the
reservoir/ablation behavior still needed, or acceptable to drop?) The user chose "best-effort
dagloader port" — but agreed this needs their data + a workflow decision, so align on the scheme
design first, then port + have the user run against real data and iterate from the errors.
`experiments/temp_edit.py` is a not-yet-integrated staging/diagnostics module (EffectSizeMonitor +
full_diagnostics) — port with the rest or drop.

## Other remaining / smaller items

- **Retire the `sf_` prefix (Path A):** the VFs now inherit cellflow's, so the last step is
  upstreaming `cell_transformer` + `adaln_zero` into cellflow's VF, then scaleflow re-exports and
  drops its VF subclasses + the `sf_` registry keys entirely.
- Dead `src/scaleflow/solvers/utils.py` (`_multivariate_normal`, unused since GENOT is cellflow's).
- cellflow's own GENOT VF is still a copy-paste fork of its base (a cellflow-internal dedup).
- cellflow `feat/annbatch-loader` GENOT-CFG (`49389f4`) may want a PR to cellflow `main`.

## How to resume

1. Read this file + the memory files above.
2. Confirm green: run the pytest command in "Test environment".
3. Discuss the multi-dataset dagloader design with the user (the OPEN section) BEFORE editing experiments.
4. Then port `experiments/*.py` to `prepare_annbatch_data` + dagloader schemes; verify against the
   user's real data.
