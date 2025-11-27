"""Batch utilities for processing multiple datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scaleflow.data._data import GroupedDistribution
from scaleflow.data._data_splitter import GroupedDistributionSplitter
from scaleflow.data._datamanager import DataManager

if TYPE_CHECKING:
    import anndata

__all__ = [
    "prepare_multiple_datasets",
    "split_multiple_datasets",
    "prepare_and_split_multiple_datasets",
]


def prepare_multiple_datasets(
    datasets: dict[str, "anndata.AnnData"],
    data_manager: DataManager,
    verbose: bool = False,
) -> dict[str, GroupedDistribution]:
    """
    Prepare multiple AnnData objects using the same DataManager.

    Parameters
    ----------
    datasets
        Dictionary mapping dataset names to AnnData objects.
    data_manager
        DataManager instance to use for preparing all datasets.
    verbose
        Whether to print timing information.

    Returns
    -------
    dict[str, GroupedDistribution]
        Dictionary mapping dataset names to their prepared GroupedDistribution objects.

    Examples
    --------
    >>> adl = AnnDataLocation()
    >>> dm = DataManager(
    ...     dist_flag_key="control",
    ...     src_dist_keys=["cell_line"],
    ...     tgt_dist_keys=["drug", "gene"],
    ...     rep_keys={
    ...         "cell_line": "cell_line_emb",
    ...         "drug": "drug_emb",
    ...         "gene": "gene_emb",
    ...     },
    ...     data_location=adl.obsm["X_pca"],
    ... )
    >>> datasets = {"pbmc": adata_pbmc, "zebrafish": adata_zebrafish}
    >>> gd_dict = prepare_multiple_datasets(datasets, dm)
    >>> gd_dict["pbmc"]  # GroupedDistribution for pbmc
    """
    return {
        name: data_manager.prepare_data(adata, verbose=verbose)
        for name, adata in datasets.items()
    }


def split_multiple_datasets(
    grouped_distributions: dict[str, GroupedDistribution],
    *,
    holdout_combinations: bool,
    split_by: list[str],
    split_key: str = "split",
    force_training_values: dict[str, Any] | None = None,
    ratios: list[float] | None = None,
    random_state: int = 42,
) -> dict[str, dict[str, GroupedDistribution]]:
    """
    Split multiple GroupedDistribution objects using the same splitter configuration.

    Parameters
    ----------
    grouped_distributions
        Dictionary mapping dataset names to GroupedDistribution objects.
    holdout_combinations
        Whether to hold out specific combinations.
    split_by
        List of keys to split by.
    split_key
        Name of the column to store split assignment.
    force_training_values
        Dictionary of values that must appear in training set.
    ratios
        Train/val/test ratios. Defaults to [0.6, 0.2, 0.2].
    random_state
        Random seed for reproducibility.

    Returns
    -------
    dict[str, dict[str, GroupedDistribution]]
        Nested dictionary: {dataset_name: {"train": gd, "val": gd, "test": gd}}.

    Examples
    --------
    >>> gd_dict = {"pbmc": gd_pbmc, "zebrafish": gd_zebrafish}
    >>> splits = split_multiple_datasets(
    ...     gd_dict,
    ...     holdout_combinations=False,
    ...     split_by=["drug", "gene"],
    ...     ratios=[0.6, 0.2, 0.2],
    ...     random_state=42,
    ... )
    >>> splits["pbmc"]["train"]  # Training GroupedDistribution for pbmc
    >>> splits["zebrafish"]["val"]  # Validation GroupedDistribution for zebrafish
    """
    if force_training_values is None:
        force_training_values = {}
    if ratios is None:
        ratios = [0.6, 0.2, 0.2]

    result = {}
    for name, gd in grouped_distributions.items():
        splitter = GroupedDistributionSplitter(
            gd=gd,
            holdout_combinations=holdout_combinations,
            split_by=split_by,
            split_key=split_key,
            force_training_values=force_training_values,
            ratios=ratios,
            random_state=random_state,
        )
        result[name] = splitter.split()

    return result


def prepare_and_split_multiple_datasets(
    datasets: dict[str, "anndata.AnnData"],
    data_manager: DataManager,
    *,
    holdout_combinations: bool,
    split_by: list[str],
    split_key: str = "split",
    force_training_values: dict[str, Any] | None = None,
    ratios: list[float] | None = None,
    random_state: int = 42,
    verbose: bool = False,
) -> dict[str, dict[str, GroupedDistribution]]:
    """
    Prepare and split multiple AnnData objects in one step.

    This is a convenience function that combines prepare_multiple_datasets
    and split_multiple_datasets.

    Parameters
    ----------
    datasets
        Dictionary mapping dataset names to AnnData objects.
    data_manager
        DataManager instance to use for preparing all datasets.
    holdout_combinations
        Whether to hold out specific combinations.
    split_by
        List of keys to split by.
    split_key
        Name of the column to store split assignment.
    force_training_values
        Dictionary of values that must appear in training set.
    ratios
        Train/val/test ratios. Defaults to [0.6, 0.2, 0.2].
    random_state
        Random seed for reproducibility.
    verbose
        Whether to print timing information.

    Returns
    -------
    dict[str, dict[str, GroupedDistribution]]
        Nested dictionary: {dataset_name: {"train": gd, "val": gd, "test": gd}}.

    Examples
    --------
    >>> adl = AnnDataLocation()
    >>> dm = DataManager(
    ...     dist_flag_key="control",
    ...     src_dist_keys=["cell_line"],
    ...     tgt_dist_keys=["drug", "gene"],
    ...     rep_keys={
    ...         "cell_line": "cell_line_emb",
    ...         "drug": "drug_emb",
    ...         "gene": "gene_emb",
    ...     },
    ...     data_location=adl.obsm["X_pca"],
    ... )
    >>> datasets = {"pbmc": adata_pbmc, "zebrafish": adata_zebrafish}
    >>> splits = prepare_and_split_multiple_datasets(
    ...     datasets,
    ...     dm,
    ...     holdout_combinations=False,
    ...     split_by=["drug", "gene"],
    ...     ratios=[0.6, 0.2, 0.2],
    ... )
    >>> splits["pbmc"]["train"]  # Training data for pbmc
    """
    grouped_distributions = prepare_multiple_datasets(
        datasets=datasets,
        data_manager=data_manager,
        verbose=verbose,
    )

    return split_multiple_datasets(
        grouped_distributions=grouped_distributions,
        holdout_combinations=holdout_combinations,
        split_by=split_by,
        split_key=split_key,
        force_training_values=force_training_values,
        ratios=ratios,
        random_state=random_state,
    )
