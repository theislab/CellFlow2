import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scaleflow.data import DataManager, GroupedDistribution
from scaleflow.data._anndata_location import AnnDataLocation


@pytest.fixture
def adata_test():
    drugs = ['control', 'drug_A', 'drug_B']
    genes = ['control', 'gene_A', 'gene_B']
    cell_lines = ['cell_line_A', 'cell_line_B']
    batches = ['batch_1', 'batch_2', 'batch_3']
    plates = ['plate_1', 'plate_2', 'plate_3']
    days = ['day_1', 'day_2', 'day_3']
    doses = [1.0, 10.0, 100.0]

    rows = []
    for drug in drugs:
        for gene in genes:
            for cell_line in cell_lines:
                for batch in batches:
                    for plate in plates:
                        for day in days:
                            if drug != 'control':
                                for dose in doses:
                                    rows.append({
                                        'drug': drug,
                                        'gene': gene,
                                        'cell_line': cell_line,
                                        'batch': batch,
                                        'plate': plate,
                                        'day': day,
                                        'dose': dose,
                                        'control': False
                                    })
                            else:
                                rows.append({
                                    'drug': drug,
                                    'gene': gene,
                                    'cell_line': cell_line,
                                    'batch': batch,
                                    'plate': plate,
                                    'day': day,
                                    'dose': 0.0,
                                    'control': gene == 'control' and drug == 'control'
                                })

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


class TestDataManagerBasic:
    """Test basic DataManager functionality."""

    def test_prepare_data_basic(self, adata_test):
        """Test that prepare_data works and returns correct structure."""
        adl = AnnDataLocation()

        dm = DataManager(
            dist_flag_key="control",
            src_dist_keys=["cell_line"],
            tgt_dist_keys=["drug", "gene"],
            rep_keys={
                "cell_line": "cell_line_embeddings",
                "drug": "drug_embeddings",
                "gene": "gene_embeddings",
            },
            data_location=adl.obsm["X_pca"],
        )

        gd = dm.prepare_data(adata_test)

        assert isinstance(gd, GroupedDistribution)

        # src_dist_idx	tgt_dist_idx	cell_line	drug	gene
        #   0	0	cell_line_A	control	gene_A
        #   0	1	cell_line_A	control	gene_B
        #   0	2	cell_line_A	drug_A	control
        #   0	3	cell_line_A	drug_A	gene_A
        #   0	4	cell_line_A	drug_A	gene_B
        #   0	5	cell_line_A	drug_B	control
        #   0	6	cell_line_A	drug_B	gene_A
        #   0	7	cell_line_A	drug_B	gene_B
        #   1	8	cell_line_B	control	gene_A
        #   1	9	cell_line_B	control	gene_B
        #   1	10	cell_line_B	drug_A	control
        #   1	11	cell_line_B	drug_A	gene_A
        #   1	12	cell_line_B	drug_A	gene_B
        #   1	13	cell_line_B	drug_B	control
        #   1	14	cell_line_B	drug_B	gene_A
        #   1	15	cell_line_B	drug_B	gene_B

        expected_src_data = {
            0: ('cell_line_A',),
            1: ('cell_line_B',)
        }
        expected_tgt_data = {
            0: ('drug_A', 'control'),
            1: ('drug_B', 'control'),
            2: ('gene_A', 'control'),
            3: ('gene_B', 'control'),
            4: ('drug_A', 'gene_A'),
            5: ('drug_A', 'gene_B'),
            6: ('drug_B', 'gene_A'),
            7: ('drug_B', 'gene_B'),
            8: ('drug_A', 'control'),
            9: ('drug_B', 'control'),
            10: ('gene_A', 'control'),
            11: ('gene_B', 'control'),
            12: ('drug_A', 'gene_A'),
            13: ('drug_A', 'gene_B'),
            14: ('drug_B', 'gene_A'),
            15: ('drug_B', 'gene_B'),
        }
        expected_mapping = {
            0: {0, 1, 2, 3, 4, 5, 6, 7},
            1: {8, 9, 10, 11, 12, 13, 14, 15},
        }

        assert len(gd.data.src_data) == len(expected_src_data)
        assert len(gd.data.tgt_data) == len(expected_tgt_data)

        # Test target mapping correctness
        # Verify that src_to_tgt_dist_map exists for each source
        assert len(gd.data.src_to_tgt_dist_map) == len(expected_src_data)

        # sum of the values in src_to_tgt_dist_map should be equal to the number of target distributions
        assert sum(len(v) for v in gd.data.src_to_tgt_dist_map.values()) == len(expected_tgt_data)

        # Each source should have at least one target
        for src_idx, tgt_indices in gd.data.src_to_tgt_dist_map.items():
            assert len(tgt_indices) > 0, f"Source {src_idx} has no targets"
            # All target indices should exist in tgt_data
            for tgt_idx in tgt_indices:
                assert tgt_idx in gd.data.tgt_data, f"Target {tgt_idx} not in tgt_data"

        # Verify that targets are correctly mapped to their source cell_lines
        # using the src_tgt_dist_df
        src_tgt_df = gd.annotation.src_tgt_dist_df

        # For each target, verify it belongs to the correct source
        for _, row in src_tgt_df.iterrows():
            src_idx = row["src_dist_idx"]
            tgt_idx = row["tgt_dist_idx"]
            # Target should be in the source's target list
            assert tgt_idx in gd.data.src_to_tgt_dist_map[src_idx], (
                f"Target {tgt_idx} not found in source {src_idx}'s mapping"
            )
            assert tgt_idx in expected_mapping[src_idx], (
                f"Target {tgt_idx} not found in source {src_idx}'s mapping"
            )
