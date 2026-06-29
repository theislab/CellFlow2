# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

-   Basic tool, preprocessing and plotting functions
-   annbatch-backed streaming training: cells are streamed from a sorted
    `annbatch.DatasetCollection` via `GroupedAnnbatchSampler` (class-coherent batches);
    `write_sorted_collection` builds the required sorted-by-condition collection.
-   Model-level perturbation **combinations**: `tgt_dist_keys` accepts a grouped form
    (`{"drug": ["drug_1", "drug_2"]}`) whose columns are stacked into a `(1, K, emb)` set and
    pooled by the condition encoder, with no padding (see `docs/perturbation_combinations.md`).
    `max_combination_length` is derived from the data. A flat `tgt_dist_keys` list is unchanged.

### Changed

-   `ScaleFlow` is the single model (the in-memory `CellFlow` class was removed). `prepare_data`
    takes a collection path + distribution keys; `train` requires `chunk_size` (the ClassSampler
    read-slice size) when it builds the default sampler. `predict`/`prepare_validation_data` take a
    `GroupedDistribution` + collection; arbitrary-covariate prediction moved to `predict_covariates`.
    `GroupedDistribution` is metadata-only (cells live in the collection).
