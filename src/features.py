import pandas as pd
import numpy as np
from rdkit import Chem
# pip install torch torchvision torchaudio
import torch
# pip install torch-geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random
# ============================ get_atom_features =======================
def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(x == s) for s in permitted_list]
    return binary_encoding



def get_atom_features(atom, use_chirality=True, hydrogens_implicit=True):
    permitted_atoms = ['C','N','O','S','F','Si','P','Cl','Br','I','B','Na','K','Ca','Fe','Zn','Cu','Unknown']
    if not hydrogens_implicit:
        permitted_atoms = ['H'] + permitted_atoms
    
    atom_type_enc = one_hot_encoding(atom.GetSymbol(), permitted_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(min(atom.GetDegree(), 4), [0,1,2,3,4])
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-1,0,1])
    hybridisation_enc = one_hot_encoding(str(atom.GetHybridization()), ["S","SP","SP2","SP3"])
    is_in_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]

    # optional numeric features
    atomic_mass_scaled = [atom.GetMass() / 100]
    
    atom_feature_vector = (
        atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc +
        hybridisation_enc + is_in_ring_enc + is_aromatic_enc + atomic_mass_scaled
    )

    if use_chirality:
        chirality_enc = one_hot_encoding(str(atom.GetChiralTag()),
                                         ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"])
        atom_feature_vector += chirality_enc

    if hydrogens_implicit:
        n_hydrogens_enc = one_hot_encoding(min(atom.GetTotalNumHs(), 4), [0,1,2,3,4])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


# test 
mol = Chem.MolFromSmiles("CCO")

for atom in mol.GetAtoms():
    print("Atom:", atom.GetSymbol())
    print(get_atom_features(atom))
    print("Length:", len(get_atom_features(atom)))
    print("-"*40)

# ====================================================================== 

def get_bond_features(bond, use_stereochemistry=True):
    bond_type_enc = one_hot_encoding(
        bond.GetBondType(),
        [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
         Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    )
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry:
        stereo_enc = one_hot_encoding(
            str(bond.GetStereo()),
            ["STEREOZ","STEREOE","STEREONONE"]
        )
        bond_feature_vector += stereo_enc

    return np.array(bond_feature_vector)



