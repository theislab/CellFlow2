import os
from collections.abc import Callable, Iterable

import anndata as ad
import pandas as pd
from cellflow.preprocessing._gene_emb import (
    BatchedDataset,
    GeneInfo,
    _get_esm_collate_fn,
    prot_sequence_from_ensembl,
)

from scaleflow._logging import logger

try:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, EsmModel
except ImportError as e:
    torch = None
    DataLoader = None
    AutoTokenizer = None
    EsmModel = None
    raise ImportError(
        "To use gene embedding, please install `transformers` and `torch` \
            e.g. via `pip install scaleflow['embedding']`."
    ) from e

__all__ = [
    "get_esm_embedding",
    "protein_features_from_genes",
    "GeneInfo",
    "prot_sequence_from_ensembl",
]














def create_dataloader(
    prot_names: list[str],
    sequences: list[str],
    toks_per_batch: int,
    collate_fn: Callable,  # type: ignore[type-arg]
) -> DataLoader:
    dataset = BatchedDataset(prot_names, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_sampler=batches,
    )
    return data_loader




def get_model_and_tokenizer(model_name: str, use_cuda: bool, cache_dir: None | str) -> tuple[EsmModel, AutoTokenizer]:
    model_path = os.path.join("facebook", model_name)
    model = EsmModel.from_pretrained(model_path, cache_dir=cache_dir, add_pooling_layer=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    if use_cuda:
        model = model.cuda()
    model.requires_grad_(False)
    return model, tokenizer


def protein_features_from_genes(
    genes: list[str],
    esm_model_name: str = "esm2_t36_3B_UR50D",
    toks_per_batch: int = 4096,
    trunc_len: int | None = 1022,
    truncation: bool = True,
    use_cuda: bool = True,
    cache_dir: None | str = None,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    """
    Compute gene embeddings using ESM2 :cite:`lin:2023`.

    Parameters
    ----------
    genes
        List of gene names.
    esm_model_name
        Name of the ESM model to use.
    toks_per_batch
        Number of tokens per batch.
    trunc_len
        Maximum length of the sequence.
    truncation
        Whether to truncate the sequence.
    use_cuda
        Use GPU if available.
    cache_dir
        Directory to cache the model.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with gene names as keys and embeddings as values.
    """
    if os.getenv("HF_HOME") is None and cache_dir is None:
        logger.warning(
            "HF_HOME environment variable is not set and `cache_dir` is None. \
                Cache will be stored in the current directory."
        )
    metadata = prot_sequence_from_ensembl(genes)
    to_emb = metadata[metadata.protein_sequence.notnull()]
    use_cuda = use_cuda and torch.cuda.is_available()
    esm, tokenizer = get_model_and_tokenizer(esm_model_name, use_cuda, cache_dir)
    data_loader = create_dataloader(
        prot_names=to_emb["gene_id"].to_list(),
        sequences=to_emb["protein_sequence"].to_list(),
        toks_per_batch=toks_per_batch,
        collate_fn=_get_esm_collate_fn(tokenizer, max_length=trunc_len, truncation=truncation, return_tensors="pt"),
    )
    results = {}
    for batch_metadata, batch in data_loader:
        if use_cuda:
            batch = {k: v.cuda() for k, v in batch.items()}
        batch_results = esm(**batch).last_hidden_state
        for i, name in enumerate(batch_metadata["gene_id"]):
            trunc_len = min(trunc_len, len(batch_metadata["protein_sequence"][i]))  # type: ignore[type-var]
            emb = batch_results[i, 1 : trunc_len + 1].mean(0).clone()  # type: ignore[operator]
            results[name] = emb
    return results, metadata


def get_esm_embedding(
    adata: ad.AnnData,
    gene_key: str | Iterable[str] = "gene_target_",
    null_value: str | None = None,
    gene_emb_key: str = "gene_embedding",
    copy: bool = False,
    esm_model_name: str = "esm2_t36_3B_UR50D",
    toks_per_batch: int = 4096,
    trunc_len: int | None = 1022,
    truncation: bool = True,
    use_cuda: bool = True,
    cache_dir: str | None = None,
) -> ad.AnnData | None:
    """
    Create gene embeddings from adata object using ESM2 model :cite:`lin:2023`.

    Parameters
    ----------
    adata
        Annotated data matrix.
    gene_key
        prefix in :attr:~anndata.AnnData.obs' containing gene names or list of keys.
    null_value
        Value to ignore (useful when using combinations of KO).
    gene_emb_key
        Key to store gene embeddings in :attr:`~anndata.AnnData.uns`.
    copy
        Return a copy of ``adata`` instead of updating it in place.
    esm_model_name
        Name of the ESM model to use.
    toks_per_batch
        Number of tokens per batch.
    trunc_len
        Maximum length of the sequence.
    truncation
        Whether to truncate the sequence.
    use_cuda
        Use GPU if available.
    cache_dir
        Directory to cache the model.

    Returns
    -------
    AnnData
        If `copy` is :obj:`True`, returns a new :class:`~anndata.AnnData`
        Sets the following fields:

        - ``adata.uns[gene_emb_key]``: Gene embeddings.
        - ``adata.uns[gene_emb_key + "_metadata"]``: Metadata for gene embeddings.
    """
    if copy:
        adata = adata.copy()
    if isinstance(gene_key, str):
        # We use it as a prefix
        mask_col = adata.obs.columns.str.startswith(gene_key)
        columns = adata.obs.columns[mask_col]
    else:
        assert all(col in adata.obs.columns for col in gene_key)
        columns = gene_key
    genes_todo = []
    for col in columns:
        genes_todo.extend(adata.obs[col].unique().tolist())
    unique_genes = list(set(genes_todo) - {null_value, None})
    results, metadata = protein_features_from_genes(
        genes=unique_genes,
        esm_model_name=esm_model_name,
        toks_per_batch=toks_per_batch,
        trunc_len=trunc_len,
        truncation=truncation,
        use_cuda=use_cuda,
        cache_dir=cache_dir,
    )
    adata.uns[gene_emb_key] = results
    adata.uns[gene_emb_key + "_metadata"] = metadata
    if copy:
        return adata
