# %% [markdown]
# ## Embedding for Lamin datasets

# %% [markdown]
# ### Lamin datasets notes:
# - No data for  LINCS
# - GDSC2, PRISM, CTRP, SCOREE -> phenotype

import os
import sys
import zarr
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd

# %%
import lamindb as ln
import scanpy as sc
import wetlab as wl
import bionty as bt
import numpy as np
import os
import sys
sys.path.append('/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/scripts')

checkpoint="/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/sup/SE-600M/se600m_epoch16.ckpt"
protein_embeds="/lustre/groups/ml01/workspace/xiaotong.fu/data/reconstruction/sup/SE-600M/protein_embeddings.pt"
obsm_key="X_state"
encode_batch_size=256
from reconse import ReconPretrainedStateModel
emb_model = ReconPretrainedStateModel(
        checkpoint_path=checkpoint,
        protein_embeds_path=protein_embeds,
        emb_key=obsm_key,
        encode_batch_size=encode_batch_size,
    )


ln.connect('https://lamin.ai/theislab/pertmodeling')

obs_schema = ln.Schema.get("ft0k8kia0erFFrYn")
obs_schema.describe()


projects = ln.Project.lookup()
artifacts = ln.Artifact.filter(suffix=".h5ad", projects=projects.scperturb).order_by("size").all()
artifacts.df()

dataset_num = len(artifacts)
for artifact in artifacts[dataset_num-2:dataset_num-1]:
    out_path = f"/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/{artifact.key}"
    adata=artifact.load()
    rep = emb_model.get_latent_representation(adata)
    rep = np.asarray(rep, dtype=np.float32)
    adata.obsm[obsm_key] = rep
    adata.write_h5ad(out_path, compression="gzip")
    print(f"Saved embedding for {artifact.key} to {out_path}")
