# CLAUDE.md — scaleflow (CellFlow2)

Project context and conventions for working in this repo. Keep this current when the data/model
architecture changes.

## Architecture (data + model)

The data layer streams cells from an `annbatch.DatasetCollection`; `GroupedDistribution` is
**metadata-only** (per-condition row indices + condition embeddings, no cell matrices).

- **`ScaleFlow` is the only model** (`src/scaleflow/model/_scaleflow.py`). The old in-memory
  `CellFlow` class was removed. Use:
  - `prepare_data(collection_path, *, dist_flag_key, src_dist_keys, tgt_dist_keys, rep_keys, rep_path|rep_dict, ...)`
  - `prepare_model(...)` then `train(num_iterations, batch_size, chunk_size, ...)`.
- **Training sampler** = `GroupedAnnbatchSampler` (annbatch `ClassSampler`, class-coherent batches,
  one condition per batch). `chunk_size` is **required** and must be ≤ the smallest trained
  condition's cell count. The on-disk collection MUST be sorted by condition — build it with
  `scaleflow.data.write_sorted_collection` (single write only; it raises on a non-empty collection
  because appending scatters rows and breaks per-condition contiguity).
- **Validation/Prediction** = `ValidationSampler` / `PredictionSampler` (take `collection` as the
  first positional arg). `predict(data: GroupedDistribution, collection=...)`; arbitrary-covariate
  prediction is `predict_covariates(covariate_data, ...)`.

## Perturbation combinations (model-level, no padding)

`tgt_dist_keys` accepts a flat `list[str]` (each column is a single-value condition key,
length-1 set — unchanged behavior) **or** a grouped `dict[str, list[str]]`, e.g.
`{"drug": ["drug_1", "drug_2"]}`. A multi-column group is stacked into a `(1, K, emb)` set
(K = #columns, **no padding**) and pooled by the condition encoder; `max_combination_length` is
derived from the data. Single covariates (src keys, 1-column groups) are encoded as
`covariates_not_pooled`. See `docs/perturbation_combinations.md` for the full contract, caveats
(control/null slots map to a zero embedding and are masked via `mask_value==0`; don't give real
values an all-zero embedding; column order = set-stacking order, canonicalize upstream if order
shouldn't matter), and cellflow porting notes.

All conditions must expose the **same** covariate key set (the encoder is built from one
condition's keys); `prepare_model` fails fast on heterogeneous keys (e.g. partial `extra_rep_keys`
coverage — make each extra embedding available for all conditions or none).

## Test environment

- `requires-python>=3.12`. Use the project `.venv`
  (`uv pip install -e ".[dev]" pytest pytest-benchmark`).
- Run tests with `-o addopts=""` to skip the `--benchmark-skip` plugin option, e.g.
  `.venv/bin/python -m pytest tests/data -q -o addopts="" -p no:cacheprovider`.
- Benchmarks (`tests/benchmarks/`) use the `benchmark` fixture; run with `--benchmark-only`.
- **Known pre-existing failures (unrelated; do not attribute to new work):**
  `tests/networks/test_condencoder.py` (~72; the encoder's `separate_inputs`/dict-`layers_before_pool`
  branch concatenates pooled keys on the SET axis, requiring uniform feature dims, vs the default
  FEATURE-axis branch) and `tests/preprocessing/*`. `tests/external` and `tests/preprocessing/test_gene_emb.py`
  need optional deps (`scvi-tools`, `torch`).

## Conventions

- Commit only when asked (feature branches OK to checkpoint); end commit messages with the
  `Co-Authored-By: Claude ...` trailer.
- Prefer additive API changes; this branch's annbatch migration is breaking vs the old AnnData API —
  when porting to `cellflow`, add the collection path as *new* methods alongside the AnnData API.
