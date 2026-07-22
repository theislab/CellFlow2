# Metacell construction + responder-metacell prediction

A small pipeline to (1) aggregate single-cell perturbation data into **SEACells
metacells** and (2) label each metacell as a **responder** (statistically
out-of-control vs. its cell line's control population).

```
 single cells ──► metacells ──► responder labels
   (h5ad/zarr)     (zarr)         (CSV [+ responders-only zarr])
```

The metacell target distributions feed CellFlow2 training; the responder labels
are an "option-3" non-parametric filter (no Gaussian assumption on the control).

---

## Scripts

| script | stage | input → output | env |
|---|---|---|---|
| `build_perturbation_metacells.py` | construct (route A) | single-cell **h5ad** → metacell **h5ad** | SEACells (`bio-agent`) |
| `process_metacell_dose_to_zarr.py` | construct (route A) | metacell **h5ad** → metacell **zarr** | scaleflow (`cfp`) |
| `construct_metacells_from_zarr.py` | construct (route B) | single-cell **zarr** → metacell **zarr** | SEACells **and** scaleflow |
| `define_responder_metacells.py` | responder | metacell **zarr** → labels (+ optional zarr) | scaleflow (`cfp`) |

There are **two construction routes**. Use whichever matches what you have:

- **Route A (h5ad, recommended / proven):** build metacells from a single-cell
  h5ad with SEACells, then convert the metacell h5ad to a GroupedDistribution
  zarr. This cleanly separates the SEACells step (`bio-agent`, zarr 2.x) from the
  zarr I/O (`cfp`, zarr 3.x).
- **Route B (zarr-native):** aggregate an existing single-cell GroupedDistribution
  zarr directly. Conceptually simplest, but needs one env with **both** SEACells
  and scaleflow (see *Environments*).

Both routes produce an identical-format metacell zarr that `define_responder_metacells.py` consumes.

---

## Inputs expected

**Single-cell h5ad** (route A) — `adata.obs`: `cell_line`, `drug`, `control` (bool);
optionally `dose_value` for dose-resolved grouping. `adata.obsm[<embedding>]`
(e.g. `X_scconcept`, `X_state`). `adata.uns`: `cell_line_ccle_embeddings`,
`drug_0_embeddings`, `prophet_emb` (used by the zarr conversion).

**Single-cell zarr** (route B) — a `GroupedDistribution` where source dists are
raw control cells (keyed by `cell_line`) and target dists are perturbed single
cells per condition.

---

## Usage

### Route A — h5ad → metacell zarr
```bash
# 1) SEACells metacells (controls kept raw). --extra-group-keys adds dose etc.
conda run -n bio-agent python build_perturbation_metacells.py \
    --input  /path/sciplex3_prophet_filtered.h5ad \
    --output outputs/sciplex_metacell_dose_concept.h5ad \
    --obsm   X_scconcept \
    --cells-per-metacell 25 \
    --extra-group-keys dose_value

# 2) metacell h5ad → GroupedDistribution zarr (target = cell_line × drug × dose)
conda run -n cfp python process_metacell_dose_to_zarr.py \
    --input  outputs/sciplex_metacell_dose_concept.h5ad \
    --output outputs/sciplex_metacell_dose_concept.zarr \
    --obsm   X_scconcept
```

### Route B — single-cell zarr → metacell zarr
```bash
conda run -n <env-with-seacells+scaleflow> python construct_metacells_from_zarr.py \
    --input  outputs/sciplex_raw_concept.zarr \
    --output outputs/sciplex_metacell_from_zarr.zarr \
    --cells-per-metacell 25 \
    --checkpoint-dir outputs/mc_from_zarr_ckpts
```

### Responder labels (either route's zarr)
```bash
conda run -n cfp python define_responder_metacells.py \
    --input  outputs/sciplex_metacell_dose_concept.zarr \
    --out-prefix outputs/responder_labels_sciplex_dose_concept \
    --n-eff 25 \
    --write-zarr        # optional: also emit a responders-only metacell zarr
```
Outputs: `<prefix>.csv` (per condition: counts + responder fraction + threshold),
`<prefix>_permetacell.csv` (per metacell: k-NN distance, threshold, responder
bool), and with `--write-zarr` a `<prefix>_responders.zarr`.

---

## How responder labels are defined (option-3)

Per cell line, using its **raw control cells**:
1. split controls into a k-NN **reference** set and a held-out **pool**;
2. build a **null** of "no-effect" k-NN distances from **control pseudo-metacells**
   = bootstrap means of `--n-eff` pool cells (same aggregation level as real metacells);
3. **threshold** = the `--percentile`-th percentile of that null.

A metacell is a **responder** iff its distance to its `--k`-th nearest control
reference neighbour exceeds the threshold — i.e. it sits farther from the control
population than a typical no-effect metacell of the same size.

### ⚠ `--n-eff` must match the metacell aggregation size
`--n-eff` must equal `--cells-per-metacell` used at construction (≈ cells averaged
per metacell). Averaging shrinks variance ~`1/n_eff`, so a mismatched null
miscalibrates the threshold. We use **25** for these sciplex metacells (the
tahoe pipeline used 76). Other knobs (`--k` 15, `--reference` 5000, `--null`
2000, `--percentile` 95) follow the original option-3 defaults.

---

## Environments

- **`bio-agent`** — SEACells + zarr 2.x (metacell construction, route A step 1).
- **`cfp`** — scaleflow + zarr 3.x (zarr I/O, responder labeling, route A step 2).

SEACells and scaleflow's zarr-3 I/O live in different envs, which is why route A
splits the work across two steps. Route B requires a single env providing both;
if you don't have one, use route A.

`construct_metacells_from_zarr.py` and `build_perturbation_metacells.py`
deferred-import `SEACells`, so everything except the actual aggregation runs
without it.
