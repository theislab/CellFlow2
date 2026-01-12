import lightning as L
import torch
import torch.nn as nn
import numpy as np
from anndata import AnnData
import os
import sys  
import logging  
from tqdm import tqdm  
from typing import Optional  
import anndata as ad
import warnings

log = logging.getLogger(__name__)  

from sc_reconstruction.models._base_model import BaseReconstructionModel
repo = "/lustre/groups/ml01/code/xiaotong.fu/state/src"
sys.path.insert(0, repo)  

from state.emb.nn.model import StateEmbeddingModel
from omegaconf import OmegaConf
from state.emb.train.trainer import get_embeddings
from state.emb.data import create_dataloader
from state.emb.utils import get_embedding_cfg, get_precision_config

from state.emb import Inference

class ReconPretrainedStateModel(BaseReconstructionModel):
    def __init__(
        self,
        checkpoint_path: str,
        protein_embeds_path: str,
        emb_key: str = 'X_state',
        read_depth: float = 4.0,
        encode_batch_size: int = 64,
        decode_batch_size: int = 64,
        **kwargs
    ):
        """
        Pre-trained State Embedding Model for reconstruction
        
        Args:
            checkpoint_path: Path to the pre-trained model checkpoint
            protein_embeds_path: Path to protein embeddings
            emb_key: Key for cell embeddings in adata.obsm
            read_depth: Read depth for decoding
            batch_size: Batch size for inference
            library_size_mode: For interface compatibility (not used in this model)
        """
        self.model_params = {
            'checkpoint_path': checkpoint_path,
            'protein_embeds_path': protein_embeds_path,
            'emb_key': emb_key,
            'read_depth': read_depth,
            'encode_batch_size': encode_batch_size,
            'decode_batch_size': decode_batch_size
        }
        
        # Initialize the pre-trained model
        self.inferer = self._load_pretrained_model(checkpoint_path, protein_embeds_path)
        self.emb_key = emb_key
        self.read_depth = read_depth
        self.encode_batch_size = encode_batch_size
        self.decode_batch_size = decode_batch_size
        
        # Store genes for later use
        self.genes = None
        self.overlap_genes = None
        
    def _load_pretrained_model(self, checkpoint_path, protein_embeds_path):
        """Load the pre-trained State model"""
        from state.emb.utils import get_precision_config
        from state.emb.nn.model import StateEmbeddingModel
        import torch
        
        print(f"Loading protein embeddings from {protein_embeds_path}")
        print(f"Loading model checkpoint from {checkpoint_path}")
        
        protein_embeds = torch.load(protein_embeds_path, weights_only=False, map_location="cpu")
        
        # Create inference instance
        inferer = ReconInference(cfg=None, protein_embeds=protein_embeds)
        inferer.load_model(checkpoint_path)
        
        return inferer

    def prepare(self, adata: AnnData | None = None, **kwargs):
        """Prepare the model with data"""
        if adata is not None:
            self.adata = adata
            self.genes = adata.var_names.tolist()
            
            # Compute cell embeddings if not already present
            if self.emb_key not in adata.obsm:
                print(f"Computing cell embeddings with key: {self.emb_key}")
                # This would need to be implemented based on your specific setup
                # For now, we assume embeddings are already computed
                pass

    def get_latent_representation(
        self, 
        X: np.ndarray|ad.AnnData
        ) -> np.ndarray:

        device = next(self.inferer.model.parameters()).device
        self.inferer.model.eval()
        
        if isinstance(X, ad.AnnData):
            temp_adata = X
        else:
            temp_adata = AnnData(X=X)
            temp_adata.var_names = self.genes
        cell_embs = self.inferer.encode_adata(
            adata=temp_adata,
            batch_size=self.encode_batch_size
        )
        return cell_embs


    def train(self, datamodule: L.LightningDataModule = None, **train_kwargs):
        """Training method - for pre-trained model, this might be fine-tuning or no-op"""
        print("This is a pre-trained model. Training might not be necessary or would require fine-tuning setup.")
        # If you want to enable fine-tuning, you would need to implement it here
        # For now, we'll just warn the user
        if datamodule is not None:
            print("Fine-tuning not implemented yet. Using pre-trained weights as-is.")

    def set_genes(self, genes: list[str]):
        """Set the genes for reconstruction"""
        self.genes = genes

    def get_overlap_genes(self, genes) -> list[str]:
        """Get overlapping genes between adata and protein embeddings."""
        adata = AnnData(X= np.zeros((1, len(genes))))
        adata.var_names = genes
        _, overlap_genes = ReconInference._auto_detect_gene_column(self.inferer, adata)
        return list(overlap_genes)

    def predict(
        self, 
        X: np.ndarray | ad.AnnData,
        target_genes: Optional[list[str]] = None,
        read_depth: Optional[float] = None,
        ) -> np.ndarray:
        """Predict reconstruction using pre-trained model"""
        device = next(self.inferer.model.parameters()).device
        self.inferer.model.eval()
        if isinstance(X, ad.AnnData):
            temp_adata = X
        else:
            temp_adata = AnnData(X=X)
            temp_adata.obsm[self.emb_key] = X
        
            
        reconstruction = self.inferer.decode_adata(
            adata=temp_adata,
            genes=target_genes if target_genes is not None else self.genes,
            emb_key=self.emb_key,
            read_depth=read_depth if read_depth is not None else self.read_depth,
            batch_size=self.decode_batch_size
        )
        return reconstruction
  
    def predict_relu(self, X: np.ndarray) -> np.ndarray:
        """Predict reconstruction with ReLU activation"""
        reconstruction = self.predict(X)
        return np.maximum(reconstruction, 0)

    def forward(self, x):
        """Forward pass for compatibility"""
        # This would need to be adapted for tensor input
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x
            
        reconstruction = self.predict(x_np)
        return torch.from_numpy(reconstruction).to(x.device if isinstance(x, torch.Tensor) else 'cpu')


    
    def save(self, path: str):
        """Save model configuration (not the actual pre-trained weights)"""
        import json
        
        if not path.endswith('.json'):
            path = path + '.json'
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the configuration
        config = {
            'model_params': self.model_params,
            'emb_key': self.emb_key,
            'read_depth': self.read_depth,
            'encode_batch_size': self.encode_batch_size,
            'decode_batch_size': self.decode_batch_size,
            'genes': self.genes
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"Model configuration saved to {path}")

    def load(self, path: str, map_location=None) -> None:
        """Load model configuration"""
        import json
        
        with open(path, 'r') as f:
            config = json.load(f)
            
        self.model_params = config['model_params']
        self.emb_key = config['emb_key']
        self.read_depth = config['read_depth']
        self.encode_batch_size = config['encode_batch_size']
        self.decode_batch_size = config['decode_batch_size']
        self.genes = config['genes']
        
        # Reload the pre-trained model
        self.inferer = self._load_pretrained_model(
            self.model_params['checkpoint_path'],
            self.model_params['protein_embeds_path']
        )
        
        print(f"Model configuration loaded from {path}")



class ReconInference(Inference):
    def _auto_detect_gene_column(self, adata):
        """Auto-detect the gene column with highest overlap with protein embeddings."""
        if self.protein_embeds is None:
            log.warning("No protein embeddings available for auto-detection, using index")
            return None

        protein_genes = set(self.protein_embeds.keys())
        best_column = None
        best_overlap = 0
        best_overlap_pct = 0

        # Check index first
        if hasattr(adata.var, "index"):
            index_genes = set(adata.var.index)
            overlap = len(protein_genes.intersection(index_genes))
            overlap_genes = protein_genes.intersection(index_genes)
            overlap_pct = overlap / len(index_genes) if len(index_genes) > 0 else 0
            if overlap > best_overlap:
                best_overlap = overlap
                best_overlap_pct = overlap_pct
                best_column = None  # None means use index
        # Check all columns in var
        for col in adata.var.columns:
            col_genes = set(adata.var[col].dropna().astype(str))
            overlap = len(protein_genes.intersection(col_genes))
            overlap_pct = overlap / len(col_genes) if len(col_genes) > 0 else 0
            if overlap > best_overlap:
                best_overlap = overlap
                best_overlap_pct = overlap_pct
                best_column = col
                overlap_genes = protein_genes.intersection(col_genes)

        if best_column is None:
            log.info(
                f"Auto-detected gene column: var.index (overlap: {best_overlap}/{len(protein_genes)} protein embeddings, {best_overlap_pct:.1%} of genes)"
            )
        else:
            log.info(
                f"Auto-detected gene column: var.{best_column} (overlap: {best_overlap}/{len(protein_genes)} protein embeddings, {best_overlap_pct:.1%} of genes)"
            )


        return best_column, overlap_genes

    def get_overlap_genes(self, adata):
        """Get overlapping genes between adata and protein embeddings."""
        gene_column, overlap_genes = self._auto_detect_gene_column(adata)
        return overlap_genes

    def __load_dataset_meta(self, adata):
        num_cells, num_genes = adata.shape
        return {"inference": (num_cells, num_genes)}

    def encode_adata(
        self,
        adata,
        dataset_name: str | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        shape_dict = self.__load_dataset_meta(adata)
        if dataset_name is None:
            dataset_name = "inference"

        # Convert to CSR format if needed
        adata = self._convert_to_csr(adata)

        # Auto-detect the best gene column
        gene_column, _ = self._auto_detect_gene_column(adata)

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        precision = get_precision_config(device_type=device_type)

        # Allow overriding batch size for faster inference if more VRAM is available
        dataloader_cfg = self._vci_conf
        if batch_size is not None:
            try:
                dataloader_cfg = OmegaConf.create(OmegaConf.to_container(self._vci_conf, resolve=True))
                # Ensure nested structure exists
                if not hasattr(dataloader_cfg, "model"):
                    dataloader_cfg["model"] = {}
                dataloader_cfg.model.batch_size = int(batch_size)
                log.info(f"Using override batch size: {batch_size}")
            except Exception:
                # Fallback: attempt direct set; if it fails, proceed with original config
                try:
                    dataloader_cfg.model.batch_size = int(batch_size)
                    log.info(f"Using override batch size: {batch_size}")
                except Exception:
                    log.warning("Failed to override batch size; using config default")

        dataloader = create_dataloader(
            dataloader_cfg,
            adata=adata,
            adata_name= dataset_name or "inference",
            shape_dict=shape_dict,
            # data_dir=os.path.dirname(input_adata_path),
            shuffle=False,
            protein_embeds=self.protein_embeds,
            precision=precision,
            gene_column=gene_column,
        )

        all_embeddings = []
        all_ds_embeddings = []
        for embeddings, ds_embeddings in tqdm(self.encode(dataloader), total=len(dataloader), desc="Encoding"):
            all_embeddings.append(embeddings)
            if ds_embeddings is not None:
                all_ds_embeddings.append(ds_embeddings)

        # attach this as a numpy array to the adata and write it out
        all_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        if len(all_ds_embeddings) > 0:
            all_ds_embeddings = np.concatenate(all_ds_embeddings, axis=0).astype(np.float32)

            # concatenate along axis -1 with all embeddings
            all_embeddings = np.concatenate([all_embeddings, all_ds_embeddings], axis=-1)

        return all_embeddings

        

    @torch.no_grad()
    def decode_generator(self, adata, genes, emb_key: str, read_depth=None, batch_size=64):
        try:
            cell_embs = adata.obsm[emb_key]
        except:
            cell_embs = adata.X

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        precision = get_precision_config(device_type=device_type)

        # changed
        cell_embs = torch.Tensor(cell_embs).to('cpu', dtype=precision)

        use_rda = getattr(self.model.cfg.model, "rda", False)
        if use_rda and read_depth is None:
            read_depth = 4.0

        print('Decoding with read depth:', read_depth)
        gene_embeds = self.get_gene_embedding(genes)
        print('Gene embeddings shape:', gene_embeds.shape)
        # with torch.autocast(device_type=device_type, dtype=precision):
        for i in tqdm(range(0, cell_embs.size(0), batch_size), total=int(cell_embs.size(0) // batch_size)):
            batch_cpu = cell_embs[i : i + batch_size]
            cell_embeds_batch = batch_cpu.to(self.model.device, dtype=precision)
            task_counts = torch.full(
                (cell_embeds_batch.shape[0],), read_depth, device=self.model.device, dtype=precision
            )

            ds_emb = cell_embeds_batch[:, -self.model.z_dim_ds :]  # last ten columns are the dataset embeddings
            merged_embs = StateEmbeddingModel.resize_batch(
                cell_embeds_batch[:, :-self.model.z_dim_ds], gene_embeds, task_counts=task_counts, ds_emb=ds_emb
            )
            logprobs_batch = self.model.binary_decoder(merged_embs)
            logprobs_batch = logprobs_batch.detach().cpu().float().numpy()

            del merged_embs, ds_emb, cell_embeds_batch, task_counts
            torch.cuda.empty_cache()

            yield logprobs_batch.squeeze()

    def decode_adata(self, adata, genes, emb_key: str, read_depth=None, batch_size=64):
        decoded_list = []
        for batch_decoded in self.decode_generator(
            adata, genes, emb_key=emb_key, read_depth=read_depth, batch_size=batch_size
        ):
            decoded_list.append(batch_decoded)
        decoded_array = np.vstack(decoded_list)
        return decoded_array