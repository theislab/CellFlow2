#!/usr/bin/env python
"""
Unified cell embedding script supporting multiple pretrained foundation models.

Modes:
  - In-memory (small/medium files): omit --chunk-id / --n-chunks
      The model is loaded once; cells are processed in sequential cell-batches
      of --cell-batch-size to keep peak RAM bounded.
  - Chunked (large files, SLURM array): pass --chunk-id and --n-chunks
      Each job handles one row-slice of the h5ad; cell-batching still applies
      within each slice.

Supported models (set via --config YAML):
  se          State Embedding (SE-600M)
  scgpt       scGPT
  scimilarity scimilarity
  scconcept   scConcept

Gene name format:
  var_names may be a mixture of HGNC symbols and Ensembl IDs (e.g. 'TSPAN6',
  'ENSG00000291308') with no extra var columns.  Each model resolves the
  overlap against its own vocabulary internally:
    - scGPT / scimilarity  -> match on HGNC symbols, Ensembl IDs are ignored
    - scConcept            -> HGNC symbols are mapped to Ensembl IDs via the
                             Ensembl REST API before embedding; existing ENSG
                             IDs are passed through unchanged
    - SE                   -> protein_embeds-based auto-detection, unaffected
"""
import argparse
import re
import time
import requests
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import anndata as ad
import yaml
import sys

SCRIPTS_DIR = "/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/scripts"
REPO_SRC    = "/lustre/groups/ml01/code/xiaotong.fu/state/src"
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, REPO_SRC)


# ---------------------------------------------------------------------------
# Ensembl gene-name mapping (used by scConcept)
# ---------------------------------------------------------------------------

_ENSG_RE = re.compile(r"^ENSG\d+")

def _map_hgnc_to_ensembl(symbols: list[str]) -> dict[str, str]:
    """
    Batch-map HGNC gene symbols -> Ensembl IDs via the Ensembl REST API.
    Already-ENSG IDs are returned as-is.  Unmapped symbols are absent from
    the returned dict.

    Uses POST /lookup/symbol for bulk queries (1000 at a time) with a
    per-symbol /xrefs/symbol fallback for anything still missing.
    Adapted from reconstruction/notebooks/model/scconcept.ipynb.
    """
    server  = "https://rest.ensembl.org"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    session = requests.Session()

    to_query = [s for s in symbols if not _ENSG_RE.match(s)]
    sym2ensg: dict[str, str] = {}

    def _post_lookup(batch: list[str]) -> dict:
        for _ in range(6):
            r = session.post(
                server + "/lookup/symbol/homo_sapiens",
                headers=headers,
                json={"symbols": batch},
                timeout=60,
            )
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 1)))
                continue
            if r.ok:
                return r.json()
            if r.status_code in (500, 502, 503, 504):
                time.sleep(1.0)
                continue
            r.raise_for_status()
        return {}

    def _xref_fallback(symbol: str) -> str | None:
        for _ in range(4):
            r = session.get(
                f"{server}/xrefs/symbol/homo_sapiens/{symbol}",
                headers=headers,
                timeout=30,
            )
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 1)))
                continue
            if r.ok:
                hits = r.json()
                return next(
                    (h["id"] for h in hits
                     if h.get("type") == "gene"
                     and isinstance(h.get("id"), str)
                     and h["id"].startswith("ENSG")),
                    None,
                )
            if r.status_code in (400, 404):
                return None
            time.sleep(1.0)
        return None

    # Bulk lookup
    for i in range(0, len(to_query), 1000):
        chunk = to_query[i : i + 1000]
        data  = _post_lookup(chunk)
        if isinstance(data, dict):
            for sym, obj in data.items():
                if isinstance(obj, dict):
                    _id = obj.get("id", "")
                    if isinstance(_id, str) and _id.startswith("ENSG"):
                        sym2ensg[sym] = _id
        time.sleep(0.05)

    # Per-symbol fallback for anything still missing
    missing = [s for s in to_query if s not in sym2ensg]
    print(f"  [Ensembl] {len(missing)} symbols not found in bulk lookup, trying fallback ...")
    for j, sym in enumerate(missing):
        ensg = _xref_fallback(sym)
        if ensg:
            sym2ensg[sym] = ensg
        if (j + 1) % 50 == 0:
            time.sleep(0.2)

    still_missing = [s for s in to_query if s not in sym2ensg]
    print(f"  [Ensembl] mapped {len(sym2ensg)}/{len(to_query)} symbols; "
          f"{len(still_missing)} unmapped (will be ignored by scConcept)")
    return sym2ensg


def build_gene_id_col(var_names: list[str]) -> list[str | None]:
    """
    Return a list aligned to var_names where:
      - existing ENSG IDs are kept as-is
      - HGNC symbols are mapped to Ensembl IDs via REST
      - unmappable symbols → None (scConcept skips None entries)
    """
    hgnc_syms = [v for v in var_names if not _ENSG_RE.match(v)]
    print(f"  [Ensembl] {len(hgnc_syms)} HGNC symbols to map, "
          f"{len(var_names) - len(hgnc_syms)} already Ensembl IDs")

    sym2ensg = _map_hgnc_to_ensembl(hgnc_syms) if hgnc_syms else {}

    return [
        v if _ENSG_RE.match(v) else sym2ensg.get(v)
        for v in var_names
    ]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg: dict):
    """Instantiate a model from a config dict."""
    model_type = cfg["model"].lower()

    if model_type == "se":
        from reconse import ReconPretrainedStateModel
        model = ReconPretrainedStateModel(
            checkpoint_path=cfg["checkpoint"],
            protein_embeds_path=cfg["protein_embeds"],
            emb_key=cfg.get("obsm_key", "X_se"),
            encode_batch_size=cfg.get("encode_batch_size", 256),
        )

    elif model_type == "scgpt":
        from sc_reconstruction.models.reconscgpt import ReconPretrainedscGPT
        # gene_col='index' tells scGPT to read gene names from adata.var.index
        # (var_names), avoiding the assert that a named var column exists.
        model = ReconPretrainedscGPT(
            checkpoint_path=cfg["checkpoint"],
            gene_col="index",
        )

    elif model_type == "scimilarity":
        from sc_reconstruction.models.reconscimilarity import ReconPretrainedscimilarity
        model = ReconPretrainedscimilarity(
            checkpoint_path=cfg["checkpoint"],
        )

    elif model_type == "scconcept":
        from sc_reconstruction.models.reconscconcept import ReconPretrainedscConcept
        model = ReconPretrainedscConcept(
            checkpoint_path=cfg["checkpoint"],
        )

    else:
        raise ValueError(
            f"Unknown model '{model_type}'. Choose from: se, scgpt, scimilarity, scconcept"
        )

    return model


# ---------------------------------------------------------------------------
# Cell-level batching
# ---------------------------------------------------------------------------

def _row_has_nonzero(X) -> np.ndarray:
    """Boolean mask: True for rows with at least one non-zero value."""
    if sp.issparse(X):
        return np.diff(sp.csr_matrix(X).indptr) > 0
    return np.asarray(X.sum(axis=1)).ravel() != 0


def _embed_batch(model, batch: ad.AnnData) -> np.ndarray:
    """Embed a single batch, skipping all-zero cells (zero embeddings for them)."""
    mask = _row_has_nonzero(batch.X)
    if mask.all():
        return np.asarray(model.get_latent_representation(X=batch), dtype=np.float32)

    n_skip = (~mask).sum()
    print(f"    skipping {n_skip}/{batch.n_obs} all-zero cells")

    valid = batch[mask].copy()
    rep_valid = np.asarray(model.get_latent_representation(X=valid), dtype=np.float32)
    rep = np.zeros((batch.n_obs, rep_valid.shape[1]), dtype=np.float32)
    rep[mask] = rep_valid
    return rep


def embed_in_cell_batches(model, adata: ad.AnnData, cell_batch_size: int, prefix: str = "") -> np.ndarray:
    """
    Call model.get_latent_representation in cell-row slices and concatenate.
    Avoids OOM when a single get_latent_representation call would materialise
    too many intermediate tensors for the full adata at once.
    """
    n = adata.n_obs
    if cell_batch_size <= 0 or cell_batch_size >= n:
        return _embed_batch(model, adata)

    parts = []
    n_batches = (n + cell_batch_size - 1) // cell_batch_size
    for i, start in enumerate(range(0, n, cell_batch_size)):
        end = min(start + cell_batch_size, n)
        print(f"{prefix}  cell-batch {i + 1}/{n_batches} [{start}:{end}]")
        batch = adata[start:end].copy()
        rep = _embed_batch(model, batch)
        parts.append(rep)
        del batch
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in",  dest="in_path",  required=True, help="Input .h5ad file")
    ap.add_argument("--out", dest="out_path", required=True, help="Output .h5ad file (full adata copy with embedding in obsm)")
    ap.add_argument("--config", required=True,
                    help="Path to model config YAML (see configs/model_*.yaml)")
    ap.add_argument("--obsm-key", default=None,
                    help="Override obsm key defined in config")
    ap.add_argument("--encode-batch-size", type=int, default=None,
                    help="Override encode_batch_size defined in config")
    ap.add_argument("--cell-batch-size", type=int, default=50_000,
                    help="Max cells per get_latent_representation call (default 50000). "
                         "Reduce to lower peak RAM. 0 = no batching.")
    # Chunked mode (SLURM array)
    ap.add_argument("--chunk-id",  type=int, default=None,
                    help="0-based chunk index (enables chunked mode)")
    ap.add_argument("--n-chunks",  type=int, default=None,
                    help="Total number of chunks (enables chunked mode)")
    args = ap.parse_args()

    # Load and patch config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.obsm_key is not None:
        cfg["obsm_key"] = args.obsm_key
    if args.encode_batch_size is not None:
        cfg["encode_batch_size"] = args.encode_batch_size
    obsm_key = cfg.get("obsm_key", f"X_{cfg['model'].lower()}")

    chunked = (args.chunk_id is not None) and (args.n_chunks is not None)
    prefix  = f"[chunk {args.chunk_id + 1}/{args.n_chunks}] " if chunked else ""

    in_path  = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"{prefix}Loading data: {in_path}")
    if chunked:
        adata_full = ad.read_h5ad(in_path, backed="r")
        n          = adata_full.n_obs
        chunk_sz   = (n + args.n_chunks - 1) // args.n_chunks
        start      = args.chunk_id * chunk_sz
        end        = min(start + chunk_sz, n)
        adata      = adata_full[start:end].to_memory()
        del adata_full
    else:
        adata = ad.read_h5ad(in_path)

    print(f"{prefix}Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"{prefix}Loading model: {cfg['model']}")
    model = load_model(cfg)

    # ------------------------------------------------------------------
    # Gene-name resolution: scGPT and scimilarity expect HGNC symbols.
    # If var_names are Ensembl IDs, fall back to var['symbol'].
    # ------------------------------------------------------------------
    model_type = cfg["model"].lower()
    _original_var_names = None

    if model_type in ("scgpt", "scimilarity"):
        names = adata.var_names.tolist()
        n_ensg = sum(1 for g in names if _ENSG_RE.match(g))
        if n_ensg > len(names) * 0.5 and "symbol" in adata.var.columns:
            print(f"{prefix}var_names are Ensembl IDs ({n_ensg}/{len(names)}); "
                  f"using var['symbol'] for {cfg['model']}")
            _original_var_names = adata.var_names.copy()
            adata.var_names = adata.var["symbol"].values
            adata.var_names_make_unique()

    # Models other than SE need gene names set before embedding
    if model_type != "se":
        model.set_genes(adata.var_names.tolist())

    # ------------------------------------------------------------------
    # Per-model adata preparation (add var columns required by wrappers)
    # ------------------------------------------------------------------
    if model_type == "scconcept":
        # extract_embeddings() expects adata.var['gene_id'] with Ensembl IDs.
        # Map any HGNC symbols in var_names -> Ensembl via REST; pass-through
        # existing ENSG IDs.  Unmappable genes get None and are skipped by
        # scConcept's token matching.
        print(f"{prefix}Mapping gene names to Ensembl IDs for scConcept ...")
        gene_ids = build_gene_id_col(adata.var_names.tolist())
        adata.var["gene_id"] = gene_ids

    # ------------------------------------------------------------------
    # Embed (with cell-level batching to bound peak RAM)
    # ------------------------------------------------------------------
    print(f"{prefix}Embedding (cell_batch_size={args.cell_batch_size}) ...")
    rep = embed_in_cell_batches(model, adata, args.cell_batch_size, prefix=prefix)
    print(f"{prefix}Embedding shape: {rep.shape}")

    # ------------------------------------------------------------------
    # Write output: full adata copy with embedding in obsm
    # ------------------------------------------------------------------
    if _original_var_names is not None:
        adata.var_names = _original_var_names

    print(f"{prefix}Writing: {out_path}")
    adata.obsm[obsm_key] = rep
    adata.write_h5ad(out_path, compression="gzip")
    print(f"{prefix}Done.")


if __name__ == "__main__":
    main()
