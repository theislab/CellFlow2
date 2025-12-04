import os
from typing import Any

import anndata as ad

import numpy as np
import pandas as pd

from scanpy.readwrite import _check_datafile_present_and_download

from scaleflow._types import PathLike

__all__ = [
    "ineurons",
    "pbmc_cytokines",
    "sample_adata",
]


def ineurons(
    path: PathLike = "~/.cache/scaleflow/ineurons.h5ad",
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    """Preprocessed and extracted data as provided in :cite:`lin2023human`.

    The :attr:`anndata.AnnData.X` is based on reprocessing of the counts data using
    :func:`scanpy.pp.normalize_total` and :func:`scanpy.pp.log1p`.

    Parameters
    ----------
    path
        Path where to save the file.
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(
        path,
        backup_url="https://figshare.com/ndownloader/files/52852961",
        expected_shape=(54134, 2000),  # TODO: adapt this, and enable check
        force_download=force_download,
        **kwargs,
    )


def pbmc_cytokines(
    path: PathLike = "~/.cache/scaleflow/pbmc_parse.h5ad",
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    """PBMC samples from 12 donors treated with 90 cytokines.

    Processed data from https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-in-a-single-experiment/,
    subset to 2000 highly varibale genes, containing embeddings for
    donors and cytokines.

    Parameters
    ----------
    path
        Path where to save the file.
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(
        path,
        backup_url="https://figshare.com/ndownloader/files/53372768",
        expected_shape=(54134, 2000),  # TODO: adapt this, and enable check
        force_download=force_download,
        **kwargs,
    )


def zesta(
    path: PathLike = "~/.cache/scaleflow/zesta.h5ad",
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    """Developing zebrafish with genetic perturbations.

    Dataset published in :cite:`saunders2023embryo` containing single-cell
    RNA-seq readouts of the embryonic zebrafish at 5 time points with up
    to 23 different genetic perturbations.

    Parameters
    ----------
    path
        Path where to save the file.
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(
        path,
        backup_url="https://figshare.com/ndownloader/files/52966469",
        expected_shape=(54134, 2000),  # TODO: adapt this, and enable check
        force_download=force_download,
        **kwargs,
    )


def _load_dataset_from_url(
    fpath: PathLike,
    *,
    backup_url: str,
    expected_shape: tuple[int, int],
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    fpath = os.path.expanduser(fpath)
    if not fpath.endswith(".h5ad"):
        fpath += ".h5ad"
    if force_download and os.path.exists(fpath):
        os.remove(fpath)
    if not _check_datafile_present_and_download(backup_url=backup_url, path=fpath):
        raise FileNotFoundError(f"File `{fpath}` not found or download failed.")
    data = ad.read_h5ad(filename=fpath, **kwargs)

    # TODO: enable the dataset shape check
    # if data.shape != expected_shape:
    #    raise ValueError(
    #        f"Expected AnnData object to have shape `{expected_shape}`, found `{data.shape}`."
    #    )

    return data


def sample_adata():
    drugs = ["control", "drug_A", "drug_B"]
    genes = ["control", "gene_A", "gene_B"]
    cell_lines = ["cell_line_A", "cell_line_B"]
    batches = ["batch_1", "batch_2", "batch_3"]
    plates = ["plate_1", "plate_2", "plate_3"]
    days = ["day_1", "day_2", "day_3"]
    doses = [1.0, 10.0, 100.0]

    rows = []
    for drug in drugs:
        for gene in genes:
            for cell_line in cell_lines:
                for batch in batches:
                    for plate in plates:
                        for day in days:
                            if drug != "control":
                                for dose in doses:
                                    rows.append(
                                        {
                                            "drug": drug,
                                            "gene": gene,
                                            "cell_line": cell_line,
                                            "batch": batch,
                                            "plate": plate,
                                            "day": day,
                                            "dose": dose,
                                            "control": False,
                                        }
                                    )
                            else:
                                rows.append(
                                    {
                                        "drug": drug,
                                        "gene": gene,
                                        "cell_line": cell_line,
                                        "batch": batch,
                                        "plate": plate,
                                        "day": day,
                                        "dose": 0.0,
                                        "control": gene == "control" and drug == "control",
                                    }
                                )

    n_obs = len(rows)
    n_vars = 20
    n_pca = 10

    obs = pd.DataFrame(rows)

    # Convert to categorical
    for col in ["cell_line", "drug", "gene", "batch", "plate", "day"]:
        obs[col] = obs[col].astype("category")

    # Simple X matrix (not really used in tests, just needs to exist)
    X = np.random.randn(n_obs, n_vars).astype(np.float32)

    # X_pca: Put cell index at position [idx, 0] for easy tracing
    X_pca = np.zeros((n_obs, n_pca), dtype=np.float32)
    for i in range(n_obs):
        X_pca[i, 0] = float(i)  # Cell 0 has value 0, cell 1 has value 1, etc.

    # Create AnnData
    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["X_pca"] = X_pca

    # Simple embeddings
    adata.uns["cell_line_embeddings"] = {
        "cell_line_A": np.array([1.0, 0.0], dtype=np.float32),
        "cell_line_B": np.array([0.0, 1.0], dtype=np.float32),
    }

    adata.uns["drug_embeddings"] = {
        "drug_A": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "drug_B": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "control": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    }

    adata.uns["gene_embeddings"] = {
        "gene_A": np.array([1.0, 0.0], dtype=np.float32),
        "gene_B": np.array([0.0, 1.0], dtype=np.float32),
        "control": np.array([0.0, 0.0], dtype=np.float32),
    }

    return adata
