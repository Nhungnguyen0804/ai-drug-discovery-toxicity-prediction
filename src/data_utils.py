import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import SaltRemover

def clean_smiles(smiles_raw: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles_raw)
        if mol is None: return None
        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        return None

def get_atom_features(atom):
    return [
        atom.GetAtomicNum(),       # Số hiệu nguyên tử
        atom.GetDegree(),          # Bậc nguyên tử
        atom.GetTotalNumHs(),      # Số H liên kết
        int(atom.GetHybridization()), # Trạng thái lai hóa
        int(atom.GetIsAromatic())  # Có phải vòng thơm không
    ]


def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None

    # 1. Node Features
    node_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float)

    # 2. Edge Index
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append((i, j))
        edge_indices.append((j, i)) 

    if not edge_indices: return Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long))

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


