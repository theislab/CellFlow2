# Perturbation combinations (model-level, no padding)

This note documents how scaleflow represents **perturbation combinations** (multiple
same-type perturbations applied together, e.g. two drugs) and the data→model contract, so the
design can be ported to the original `cellflow` with minimal user-facing change.

## User-facing API (additive)

`tgt_dist_keys` (on `DataManager` and `ScaleFlow.prepare_data`) accepts either the existing flat
list **or** a grouped dict:

```python
# flat (unchanged): each column is its own single-value condition key
tgt_dist_keys=["drug", "gene"]

# grouped: a covariate name -> the obs columns that form a combination of that type
tgt_dist_keys={"drug": ["drug_1", "drug_2"], "gene": ["gene_1"]}
```

A flat list behaves **byte-identically** to before (every group has width `K = 1`).
`rep_keys` is keyed by the covariate *name* (the group name for grouped keys, the column for flat
keys). `write_sorted_collection` accepts the grouped form too (it flattens to columns for sorting).
`prepare_model` no longer needs `max_combination_length` — it is derived from the data.

## Data → model contract

- **Data prep** (`DataManager._build_condition`): a multi-column group's columns are stacked into a
  set of length `K = len(columns)` → `conditions[t][group] = (1, K, emb_dim)`. Single covariates
  (every `src_dist_key`, and any 1-column group) stay `(1, 1, emb_dim)`. **No padding / null rows /
  mask tensor** are introduced — `K` is the true number of columns. All multi-column groups must
  share the same width `K` (position-aligned); construction raises otherwise.
- **Grouping** is unchanged: a condition is still a unique combination of *all* the individual
  column values (the grouped columns are flattened for `sort_values`/`groupby`/labels).
- **Model** (`ScaleFlow.prepare_model` + `ConditionEncoder`): `max_combination_length` is derived as
  the max set width across all prepared conditions (per-key, robust to conditionally-present keys).
  Keys with set width 1 are auto-assigned to `covariates_not_pooled` (context covariates,
  concatenated after pooling) **only when** a pooled (width > 1) key exists; otherwise the flat
  behavior (all pooled, width 1) is preserved exactly. The set encoder pools the width-`K` set
  (mean / attention) and `_get_masks` computes the set mask over the **pooled keys only**, so pooled
  `(1,K,emb)` and not-pooled `(1,1,emb)` keys coexist without padding/tiling.

### Why no padding is possible

scaleflow feeds **one condition per training batch** (`GroupedAnnbatchSampler.sample` →
`conditions[c_t]`, broadcast over the cell batch) and processes validation/prediction one condition
at a time (`jax.tree.map`). Conditions are never stacked across each other, so each condition's set
is its true length and the permutation-invariant pooling handles any `K`. The only cost is a JIT
recompile per *distinct* `K`; for a fixed combination width (`K` = number of columns, constant per
dataset) that is a single compile.

## Semantics & caveats

- **Absent slot / control.** A `"control"` value mapping to a zero embedding becomes a zero set
  element; the encoder masks elements equal to `mask_value` (0.0), so the slot is ignored by
  pooling — `(drugX, control)` pools like a single `drugX`. Because `K` is fixed per dataset, mean
  pooling divides by the constant `K` (a consistent scale across conditions). **Caveat:** a *real*
  covariate value whose embedding is all-zero would also be masked — do not use zero embeddings for
  real values (`mask_value` collision; inherited from the padding design).
- **Column order = stacking order.** Columns are position-aligned across groups (column `i` of every
  group → set element `i`). Pooling is permutation-invariant, so element order does not change the
  pooled embedding, but distinct on-disk column orders are still distinct conditions — canonicalize
  the column values upstream (e.g. sort) if order should not matter.
- **Numeric covariates** (e.g. dosage columns with no `rep_keys` entry) stack their scalar values
  into `(1, K, 1)`.

## Porting to cellflow — watch-outs

- The original cellflow **pads** combinations to a fixed `max_combination_length` and treats sample
  covariates as broadcast across the full set axis. This design instead emits true-`K` sets and
  marks width-1 covariates as not-pooled. If ported, carry the **pooled / not-pooled split
  explicitly** rather than re-deriving it from set width — otherwise cellflow's padded/broadcast
  covariates (set width = max, not 1) would be misclassified as pooled and would no longer be
  excluded from `_get_masks`.
- The `mask_value`-based masking of zero embeddings is shared with cellflow; combinations just make
  control-filled slots routine, so audit that real covariate embeddings are never all-zero.
- The combination API (`tgt_dist_keys` grouped dict) maps to cellflow's `perturbation_covariates`
  dict; the key difference is no padding and a data-derived (not user-fixed) combination width.
