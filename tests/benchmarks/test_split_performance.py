
"""Performance benchmarks for GroupedDistributionSplitter."""

import numpy as np
import pytest

from scaleflow.data import AnnDataLocation, DataManager
from scaleflow.data._data_splitter import GroupedDistributionSplitter


def create_large_adata(n_obs: int = 10000, n_drugs: int = 20, n_genes: int = 10):
    """Create a large AnnData for benchmarking."""
    import anndata as ad
    import pandas as pd

    n_vars = 50
    n_pca = 20
    n_cell_lines = 5

    drugs = ["control"] + [f"drug_{i}" for i in range(n_drugs)]
    genes = ["control"] + [f"gene_{i}" for i in range(n_genes)]
    cell_lines = [f"cell_line_{i}" for i in range(n_cell_lines)]

    # Generate random obs data
    np.random.seed(42)
    obs = pd.DataFrame({
        "drug": np.random.choice(drugs, n_obs),
        "gene": np.random.choice(genes, n_obs),
        "cell_line": np.random.choice(cell_lines, n_obs),
    })

    # Mark controls
    obs["control"] = (obs["drug"] == "control") & (obs["gene"] == "control")

    for col in ["drug", "gene", "cell_line"]:
        obs[col] = obs[col].astype("category")

    X = np.random.randn(n_obs, n_vars).astype(np.float32)
    X_pca = np.random.randn(n_obs, n_pca).astype(np.float32)

    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["X_pca"] = X_pca

    # Create embeddings
    adata.uns["cell_line_emb"] = {cl: np.random.randn(10) for cl in cell_lines}
    adata.uns["drug_emb"] = {d: np.random.randn(10) for d in drugs}
    adata.uns["gene_emb"] = {g: np.random.randn(10) for g in genes}

    return adata


@pytest.fixture(scope="module")
def large_grouped_distribution():
    """Create a large GroupedDistribution for benchmarking."""
    adata = create_large_adata(n_obs=10000)
    adl = AnnDataLocation()
    dm = DataManager(
        dist_flag_key="control",
        src_dist_keys=["cell_line"],
        tgt_dist_keys=["drug", "gene"],
        rep_keys={
            "cell_line": "cell_line_emb",
            "drug": "drug_emb",
            "gene": "gene_emb",
        },
        data_location=adl.obsm["X_pca"],
    )
    return dm.prepare_data(adata)


class TestSplitterBenchmarks:
    """Benchmark tests for splitter operations."""

    def test_benchmark_split_annotation(self, benchmark, large_grouped_distribution):
        """Benchmark split_annotation performance."""
        splitter = GroupedDistributionSplitter(
            gd=large_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        # benchmark automatically runs multiple times and reports stats
        result = benchmark(splitter.split_annotation)

        assert len(result) == 3

    def test_benchmark_split_full(self, benchmark, large_grouped_distribution):
        """Benchmark full split performance (annotation + data)."""
        splitter = GroupedDistributionSplitter(
            gd=large_grouped_distribution,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        result = benchmark(splitter.split)

        assert len(result) == 3

    def test_benchmark_prepare_data(self, benchmark):
        """Benchmark DataManager.prepare_data performance."""
        adata = create_large_adata(n_obs=1_000_000)
        adl = AnnDataLocation()
        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_emb",
                "drug": "drug_emb",
                "gene": "gene_emb",
            },
            data_location=adl.obsm["X_pca"],
        )

        result = benchmark(dm.prepare_data, adata)

        assert result is not None


class TestScalingBenchmarks:
    """Benchmark tests for different data sizes."""

    @pytest.mark.parametrize("n_obs", [1000, 5000, 10000, 10_000_000])
    def test_benchmark_split_scaling(self, benchmark, n_obs):
        """Benchmark split performance at different scales."""
        adata = create_large_adata(n_obs=n_obs)
        adl = AnnDataLocation()
        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_emb",
                "drug": "drug_emb",
                "gene": "gene_emb",
            },
            data_location=adl.obsm["X_pca"],
        )
        gd = dm.prepare_data(adata)

        splitter = GroupedDistributionSplitter(
            gd=gd,
            holdout_combinations=False,
            split_by=["drug", "gene"],
            split_key="split",
            force_training_values={},
            ratios=[0.6, 0.2, 0.2],
            random_state=42,
        )

        # Add extra info to benchmark
        benchmark.extra_info["n_obs"] = n_obs
        result = benchmark(splitter.split)

        assert len(result) == 3