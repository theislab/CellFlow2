import os
import sys
import zarr
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import h5py
import anndata as ad

# Drug embeddings
drug_embeddings = pd.read_csv('/lustre/groups/ml01/projects/2025_llm_exp_design/embeddings/drug_embeddings/rdkit_200.csv')


combosciplex = sc.read_h5ad('/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/combosciplex_se.h5ad')

combosciplex.obs['drug_1'] = combosciplex.obs['pert_compound'].apply(lambda x: x.split('_')[0]).astype("category")
combosciplex.obs['drug_1'] = combosciplex.obs['drug_1'].apply(lambda x: 'control' if x == 'vehicle' else x).astype("category")
combosciplex.obs['drug_0'] = combosciplex.obs['pert_compound'].apply(lambda x: x.split('_')[1]).astype("category")
combosciplex.obs['drug_0'] = combosciplex.obs['drug_0'].apply(lambda x: 'control' if x == 'vehicle' else x).astype("category")

tahoe = sc.read_h5ad('/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe_a549/chunks_emb/chunk_00000_with_state.h5ad')
all_drugs = set(combosciplex.obs['drug_0'].cat.categories).union(set(combosciplex.obs['drug_1'].cat.categories)).union(set(tahoe.obs['drug'].cat.categories))
all_drugs.remove('control')


import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

uncharger = rdMolStandardize.Uncharger()

def to_parent_mol(smiles: str):
    if smiles is None or pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = rdMolStandardize.FragmentParent(mol)
    mol = uncharger.uncharge(mol)
    return mol

bad = []

def to_inchikey(smiles):
    if smiles is None or pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            bad.append((smiles, "MolFromSmiles returned None"))
            return None

        mol = rdMolStandardize.FragmentParent(mol)
        mol = uncharger.uncharge(mol)

        try:
            Chem.Kekulize(mol, clearAromaticFlags=True) 
        except Exception as e:
            pass

        return inchi.MolToInchiKey(mol)

    except Exception as e:
        bad.append((smiles, repr(e)))
        return None

def to_canon_smiles(smiles: str, keep_stereo=False):
    mol = to_parent_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=keep_stereo)

import requests
import pandas as pd

drugs = all_drugs

def name_to_canonical_smiles(name: str, target_field='Isomeric',timeout=30):
    name = name.strip()
    if name.lower() == "vehicle":
        return None

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/property/{target_field}SMILES/JSON"
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        return None

    js = r.json()
    props = js["PropertyTable"]["Properties"]
    if target_field == 'Isomeric':
        return props[0].get("SMILES")
    elif target_field == 'Canonical':
        return props[0].get("ConnectivitySMILES")

target_fields = ['Isomeric','Canonical'] 
merged_dfs = []
for field in target_fields:
    name2smiles = {d: name_to_canonical_smiles(d,field) for d in drugs}
    map_df = pd.DataFrame({"drug_name": list(name2smiles.keys()),
                        "smiles": list(name2smiles.values())})
    drug_embeddings = drug_embeddings.copy()
    map_df = map_df.copy()
    map_df["inchikey"] = map_df["smiles"].map(to_inchikey)
    drug_embeddings["inchikey"] = drug_embeddings["smiles"].map(to_inchikey)
    merged = map_df.merge(drug_embeddings.drop(columns=["smiles"]), on="inchikey", how="left")
    print("Unmatched drugs:", merged.loc[merged["inchikey"].isna() | merged.filter(regex="^emb").isna().all(axis=1), "drug_name"].tolist())
    merged_dfs.append(merged)

final = pd.concat(merged_dfs)
cleaned = final.drop(columns=["smiles", "inchikey"]).dropna().drop_duplicates(subset=["drug_name"]).reset_index(drop=True)
add_embeddings = ['control']+ np.repeat(0.0, 200).tolist()
cleaned.loc[len(cleaned)] = add_embeddings

emb_cols = [c for c in cleaned.columns if c != "drug_name"]
emb_matrix = cleaned[emb_cols].to_numpy(dtype=np.float32)
name_to_vec = dict(zip(cleaned["drug_name"].tolist(), emb_matrix))

valid_0 = combosciplex.obs["drug_0"].isin(name_to_vec.keys())
valid_1 = combosciplex.obs["drug_1"].isin(name_to_vec.keys())
mask = valid_0 & valid_1
print(f"Before: {combosciplex.n_obs} cells")
combosciplex = combosciplex[mask].copy()
print(f"After:  {combosciplex.n_obs} cells (dropped {(~mask).sum()})")
drugs0 = pd.unique(combosciplex.obs["drug_0"])
drugs1 = pd.unique(combosciplex.obs["drug_1"])

combosciplex.uns["drug_0_embeddings"] = {d: name_to_vec[d] for d in drugs0}
combosciplex.uns["drug_1_embeddings"] = {d: name_to_vec[d] for d in drugs1}
cell_lines = np.sort(combosciplex.obs["cell_line"].dropna().astype(str).unique())
eye = np.eye(len(cell_lines), dtype=np.float32)
combosciplex.uns["cell_line_embeddings"] = {
    cl: eye[i] for i, cl in enumerate(cell_lines)
}
combosciplex.uns["cell_line_embeddings"] 
# combosciplex.write_h5ad('/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/combosciplex_with_embeddings.h5ad')

tahoe.obs['drug_0'] = tahoe.obs['drug']
tahoe.obs['drug_0'] = tahoe.obs['drug_0'].apply(lambda x: 'control' if x == 'DMSO_TF' else x)
tahoe.obs['drug_1'] = 'control'
valid_0 = tahoe.obs["drug_0"].isin(name_to_vec.keys())
valid_1 = tahoe.obs["drug_1"].isin(name_to_vec.keys())
mask = valid_0 & valid_1
print(f"Before: {tahoe.n_obs} cells")
tahoe = tahoe[mask].copy()
print(f"After:  {tahoe.n_obs} cells (dropped {(~mask).sum()})")
drugs0 = pd.unique(tahoe.obs["drug_0"])
drugs1 = pd.unique(tahoe.obs["drug_1"])

tahoe.uns["drug_0_embeddings"] = {d: name_to_vec[d] for d in drugs0}
tahoe.uns["drug_1_embeddings"] = {d: name_to_vec[d] for d in drugs1}
cell_lines = np.sort(tahoe.obs["cell_line"].dropna().astype(str).unique())
eye = np.eye(len(cell_lines), dtype=np.float32)
tahoe.uns["cell_line_embeddings"] = {
    cl: eye[i] for i, cl in enumerate(cell_lines)
}
tahoe.uns["cell_line_embeddings"] 
# tahoe.write_h5ad('/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/tahoe_with_embeddings.h5ad')