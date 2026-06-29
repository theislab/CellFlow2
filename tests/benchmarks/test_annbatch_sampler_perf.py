"""Throughput benchmarks for the annbatch-backed GroupedAnnbatchSampler.

The training sampler streams class-coherent batches from a sorted DatasetCollection via
annbatch's ClassSampler. Throughput scales with ``chunk_size`` (larger chunks = fewer, larger
contiguous on-disk reads), so these benchmarks sweep ``chunk_size`` and report per-batch latency.

Run with: ``pytest tests/benchmarks/test_annbatch_sampler_perf.py --benchmark-only`` (needs
``pytest-benchmark``; skipped in normal runs via ``--benchmark-skip``).
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr

from scaleflow.data import AnnDataLocation, DataManager, GroupedAnnbatchSampler, write_sorted_collection

BATCH_SIZE = 256


def _build_collection(tmp, *, n_per_cond=2000, n_genes=128, n_cell_lines=3, n_drugs=10):
    cell_lines = [f"cl{i}" for i in range(n_cell_lines)]
    drugs = ["control"] + [f"d{i}" for i in range(n_drugs)]
    rows = []
    for cl in cell_lines:
        rows += [{"cell_line": cl, "drug": "control", "control": True}] * n_per_cond
        for d in drugs[1:]:
            rows += [{"cell_line": cl, "drug": d, "control": False}] * n_per_cond
    obs = pd.DataFrame(rows)
    for c in ["cell_line", "drug"]:
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=np.random.randn(len(obs), n_genes).astype(np.float32), obs=obs)
    cl_emb = {cl: np.eye(n_cell_lines, dtype=np.float32)[i] for i, cl in enumerate(cell_lines)}
    drug_emb = {d: np.eye(len(drugs), dtype=np.float32)[i] for i, d in enumerate(drugs)}
    rep_path = f"{tmp}/uns.zarr"
    g = zarr.open_group(rep_path, mode="w")
    ad.io.write_elem(g, "cell_line_embeddings", cl_emb)
    ad.io.write_elem(g, "drug_embeddings", drug_emb)
    coll = f"{tmp}/coll.zarr"
    write_sorted_collection(
        adata, coll, dist_flag_key="control", src_dist_keys=["cell_line"], tgt_dist_keys=["drug"],
        sorted_adata_path=f"{tmp}/sorted.zarr",
    )
    dm = DataManager(
        dist_flag_key="control", src_dist_keys=["cell_line"], tgt_dist_keys=["drug"],
        rep_keys={"cell_line": "cell_line_embeddings", "drug": "drug_embeddings"},
        data_location=AnnDataLocation().X,
    )
    return coll, dm.prepare_data_from_collection(coll, rep_path=rep_path)


@pytest.mark.benchmark
class TestSamplerThroughput:
    @pytest.fixture(scope="class")
    def collection(self, tmp_path_factory):
        return _build_collection(str(tmp_path_factory.mktemp("annbatch_bench")))

    @pytest.mark.parametrize("chunk_size", [64, 256, 1024])
    def test_benchmark_sample(self, benchmark, collection, chunk_size):
        coll, gd = collection
        sampler = GroupedAnnbatchSampler(coll, gd, batch_size=BATCH_SIZE, chunk_size=chunk_size, seed=0)
        sampler.init_sampler()
        for _ in range(10):  # warm up the loader / first-read costs
            sampler.sample()
        benchmark.extra_info["chunk_size"] = chunk_size
        benchmark.extra_info["batch_size"] = BATCH_SIZE
        batch = benchmark(sampler.sample)
        assert np.asarray(batch["tgt_cell_data"]).shape[0] == BATCH_SIZE

    def test_benchmark_init_sampler(self, benchmark, collection):
        coll, gd = collection

        def _init():
            s = GroupedAnnbatchSampler(coll, gd, batch_size=BATCH_SIZE, chunk_size=256, seed=0)
            s.init_sampler()
            return s

        sampler = benchmark(_init)
        assert sampler.initialized
