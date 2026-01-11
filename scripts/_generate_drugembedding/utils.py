import requests
import pandas as pd


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


def to_inchikey(smiles):
    if smiles is None or pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Bad SMILES: {smiles}")
        return None

    mol = rdMolStandardize.FragmentParent(mol)
    mol = uncharger.uncharge(mol)

    try:
        Chem.Kekulize(mol, clearAromaticFlags=True) 
    except Exception as e:
        pass

    return inchi.MolToInchiKey(mol)


def to_canon_smiles(smiles: str, keep_stereo=False):
    mol = to_parent_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=keep_stereo)