import pytest
import numpy as np
import pandas as pd
import anndata as ad
from scaleflow.data import DataManager, GroupedDistribution
from scaleflow.data._anndata_location import AnnDataLocation


def create_test_adata() -> ad.AnnData:
    """
    Create test AnnData with simple, traceable values for testing data splitting.
    
    Key design:
    - X_pca: cell index embedded at position [idx, 0] for easy tracing (cell 0 has value 0, cell 1 has value 1, etc.)
    - Simple names: cellline_A, drug_A, gene_A, etc.
    - Multiple metadata columns (batch, plate, day) for testing different split strategies
    - Known perturbation combinations
    """
    
    # Define explicit test cases
    data = [
        # Controls - cellline_A
        {'control': True, 'cellline': 'cellline_A', 'drug': 'control', 'gene': 'control', 'dose': 0.0, 'batch': 'batch_1', 'plate': 'plate_1', 'day': 'day_1'},
        {'control': True, 'cellline': 'cellline_A', 'drug': 'control', 'gene': 'control', 'dose': 0.0, 'batch': 'batch_1', 'plate': 'plate_1', 'day': 'day_1'},
        {'control': True, 'cellline': 'cellline_A', 'drug': 'control', 'gene': 'control', 'dose': 0.0, 'batch': 'batch_2', 'plate': 'plate_2', 'day': 'day_2'},
        
        # Controls - cellline_B
        {'control': True, 'cellline': 'cellline_B', 'drug': 'control', 'gene': 'control', 'dose': 0.0, 'batch': 'batch_1', 'plate': 'plate_2', 'day': 'day_1'},
        {'control': True, 'cellline': 'cellline_B', 'drug': 'control', 'gene': 'control', 'dose': 0.0, 'batch': 'batch_2', 'plate': 'plate_2', 'day': 'day_3'},
        
        # cellline_A + drug_A, low dose
        {'control': False, 'cellline': 'cellline_A', 'drug': 'drug_A', 'gene': 'control', 'dose': 1.0, 'batch': 'batch_1', 'plate': 'plate_1', 'day': 'day_1'},
        {'control': False, 'cellline': 'cellline_A', 'drug': 'drug_A', 'gene': 'control', 'dose': 1.0, 'batch': 'batch_1', 'plate': 'plate_1', 'day': 'day_2'},
        {'control': False, 'cellline': 'cellline_A', 'drug': 'drug_A', 'gene': 'control', 'dose': 1.0, 'batch': 'batch_2', 'plate': 'plate_2', 'day': 'day_1'},
        
        # cellline_A + drug_A, high dose
        {'control': False, 'cellline': 'cellline_A', 'drug': 'drug_A', 'gene': 'control', 'dose': 100.0, 'batch': 'batch_1', 'plate': 'plate_1', 'day': 'day_1'},
        {'control': False, 'cellline': 'cellline_A', 'drug': 'drug_A', 'gene': 'control', 'dose': 100.0, 'batch': 'batch_2', 'plate': 'plate_3', 'day': 'day_2'},
        
        # cellline_A + gene_A knockout
        {'control': False, 'cellline': 'cellline_A', 'drug': 'control', 'gene': 'gene_A', 'dose': 0.0, 'batch': 'batch_1', 'plate': 'plate_1', 'day': 'day_1'},
        {'control': False, 'cellline': 'cellline_A', 'drug': 'control', 'gene': 'gene_A', 'dose': 0.0, 'batch': 'batch_2', 'plate': 'plate_2', 'day': 'day_3'},
        
        # cellline_B + drug_B, mid dose
        {'control': False, 'cellline': 'cellline_B', 'drug': 'drug_B', 'gene': 'control', 'dose': 10.0, 'batch': 'batch_1', 'plate': 'plate_2', 'day': 'day_1'},
        {'control': False, 'cellline': 'cellline_B', 'drug': 'drug_B', 'gene': 'control', 'dose': 10.0, 'batch': 'batch_1', 'plate': 'plate_2', 'day': 'day_2'},
        {'control': False, 'cellline': 'cellline_B', 'drug': 'drug_B', 'gene': 'control', 'dose': 10.0, 'batch': 'batch_3', 'plate': 'plate_3', 'day': 'day_1'},
        
        # cellline_B + gene_B knockout
        {'control': False, 'cellline': 'cellline_B', 'drug': 'control', 'gene': 'gene_B', 'dose': 0.0, 'batch': 'batch_2', 'plate': 'plate_2', 'day': 'day_1'},
        
        # Combination: cellline_A + drug_A + gene_A
        {'control': False, 'cellline': 'cellline_A', 'drug': 'drug_A', 'gene': 'gene_A', 'dose': 10.0, 'batch': 'batch_1', 'plate': 'plate_1', 'day': 'day_1'},
        {'control': False, 'cellline': 'cellline_A', 'drug': 'drug_A', 'gene': 'gene_A', 'dose': 10.0, 'batch': 'batch_2', 'plate': 'plate_2', 'day': 'day_2'},
        
        # Combination: cellline_B + drug_B + gene_B
        {'control': False, 'cellline': 'cellline_B', 'drug': 'drug_B', 'gene': 'gene_B', 'dose': 10.0, 'batch': 'batch_3', 'plate': 'plate_3', 'day': 'day_1'},
    ]
    
    n_obs = len(data)
    n_vars = 20
    n_pca = 10
    
    obs = pd.DataFrame(data)
    
    # Convert to categorical
    for col in ['cellline', 'drug', 'gene', 'batch', 'plate', 'day']:
        obs[col] = obs[col].astype('category')
    
    # Simple X matrix (not really used in tests, just needs to exist)
    X = np.random.randn(n_obs, n_vars).astype(np.float32)
    
    # X_pca: Put cell index at position [idx, 0] for easy tracing
    X_pca = np.zeros((n_obs, n_pca), dtype=np.float32)
    for i in range(n_obs):
        X_pca[i, 0] = float(i)  # Cell 0 has value 0, cell 1 has value 1, etc.
    
    # Create AnnData
    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm['X_pca'] = X_pca
    
    # Simple embeddings
    adata.uns['cellline_embeddings'] = {
        'cellline_A': np.array([1.0, 0.0], dtype=np.float32),
        'cellline_B': np.array([0.0, 1.0], dtype=np.float32),
    }
    
    adata.uns['drug_embeddings'] = {
        'drug_A': np.array([1.0, 0.0, 0.0], dtype=np.float32),
        'drug_B': np.array([0.0, 1.0, 0.0], dtype=np.float32),
        'control': np.array([0.0, 0.0, 0.0], dtype=np.float32),
    }
    
    adata.uns['gene_embeddings'] = {
        'gene_A': np.array([1.0, 0.0], dtype=np.float32),
        'gene_B': np.array([0.0, 1.0], dtype=np.float32),
        'control': np.array([0.0, 0.0], dtype=np.float32),
    }
    
    return adata


@pytest.fixture
def test_adata():
    """Fixture to provide test AnnData."""
    return create_test_adata()


class TestDataManagerBasic:
    """Test basic DataManager functionality."""
    
    def test_prepare_data_basic(self, test_adata):
        """Test that prepare_data works and returns correct structure."""
        adl = AnnDataLocation()
        
        dm = DataManager(
            dist_flag_key='control',
            src_dist_keys=['cellline'],
            tgt_dist_keys=['drug', 'gene'],
            rep_keys={
                'cellline': 'cellline_embeddings',
                'drug': 'drug_embeddings',
                'gene': 'gene_embeddings',
            },
            data_location=adl.obsm['X_pca'],
        )
        
        gd = dm.prepare_data(test_adata)
        
        assert isinstance(gd, GroupedDistribution)
        # 2 source distributions (cellline_A, cellline_B)
        assert len(gd.data.src_data) == 2
        # Multiple target distributions
        assert len(gd.data.tgt_data) > 0
        
        # Test target mapping correctness
        # Verify that src_to_tgt_dist_map exists for each source
        assert len(gd.data.src_to_tgt_dist_map) == 2
        
        # Each source should have at least one target
        for src_idx, tgt_indices in gd.data.src_to_tgt_dist_map.items():
            assert len(tgt_indices) > 0, f"Source {src_idx} has no targets"
            # All target indices should exist in tgt_data
            for tgt_idx in tgt_indices:
                assert tgt_idx in gd.data.tgt_data, f"Target {tgt_idx} not in tgt_data"
        
        # Verify that targets are correctly mapped to their source celllines
        # using the src_tgt_dist_df
        src_tgt_df = gd.annotation.src_tgt_dist_df
        
        # For each target, verify it belongs to the correct source
        for _, row in src_tgt_df.iterrows():
            src_idx = row['src_dist_idx']
            tgt_idx = row['tgt_dist_idx']
            
            # Target should be in the source's target list
            assert tgt_idx in gd.data.src_to_tgt_dist_map[src_idx], \
                f"Target {tgt_idx} not found in source {src_idx}'s mapping"
            
            # Verify that the cellline in target matches the source cellline
            src_label = gd.annotation.src_dist_idx_to_labels[src_idx]
            tgt_label = gd.annotation.tgt_dist_idx_to_labels[tgt_idx]
            
            # The cellline should match between source and target
            assert src_label[0] == tgt_label[0], \
                f"Cellline mismatch: source has {src_label[0]}, target has {tgt_label[0]}"
    