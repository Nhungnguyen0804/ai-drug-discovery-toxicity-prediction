import pandas as pd
import numpy as np
from rdkit import Chem
# pip install torch torchvision torchaudio
import torch
# pip install torch-geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random

TOX21 = "dataset/tox21.csv"
df = pd.read_csv(TOX21)
print('print df ============================================')
print(df.head())
# Tách ra smiles, labels, mol_id từ df ======================== 
label_cols = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]


smiles_df = df["smiles"].tolist()
mol_ids_df = df["mol_id"].tolist()

labels_df = df[label_cols].values  # numpy array (n_samples, 12)

# print(smiles_df)
# print(mol_ids_df)
print("labels_df--------------------")
print(labels_df)

# Tạo labels_clean + mask từ labels_raw =========================
'''
Mỗi molecule có 12 nhãn, nhưng nhiều nhãn bị NaN
Thay NaN = 0 (label_clean) để giữ shape 12
Tạo mask (1 nếu có nhãn, 0 nếu NaN) để mô hình bỏ qua chỗ thiếu nhãn lúc tính loss.

'''
# Tạo labels_clean (thay NaN bằng 0)
labels_clean = np.nan_to_num(labels_df, nan=0.0)
print("labels_clean------------------")
print(labels_clean)

# Tạo mask (vị trí nào có NaN → mask = 0)
mask = ~np.isnan(labels_df)  # True nếu có giá trị, False nếu NaN
mask = mask.astype(np.float32)
print("mask-------------------------")
print(mask)


# Convert SMILES → Mol (molecule) → Graph ======================================
# Convert SMILES → Mol ----- 
def smiles_to_molecule(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles, sanitize=False) #Lọc molecule bị valence lỗi
        Chem.SanitizeMol(molecule)
        return molecule
    except:
        return None
 
    
# Convert Mol → Graph (PyG Data) ----- 
def molecule_to_graph(molecule, y, mask, mol_id):
    # Lấy node feature (ví dụ: atomic number)
    atom_features = []
    for atom in molecule.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        atom_features.append([atomic_num])   # feature 1 chiều cho dễ hiểu

    x = torch.tensor(atom_features, dtype=torch.float)

    # Lấy edge_index (bond pairs)
    edge_index = []
    for bond in molecule.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        edge_index.append([a, b])
        edge_index.append([b, a])  # undirected

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Gán label + mask + mol_id
    y = torch.tensor(y, dtype=torch.float)
    mask = torch.tensor(mask, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        mask=mask,
        mol_id=mol_id
    )

    return data

def build_graph_dataset(smiles_list, labels_clean, mask, mol_ids):
    dataset = []

    for i, smi in enumerate(smiles_list):
        molecule = smiles_to_molecule(smi)
        if molecule is None: # Lọc SMILES không parse được (molecule = None)
            continue
        
        graph = molecule_to_graph(
            molecule=molecule,
            y=labels_clean[i],
            mask=mask[i],
            mol_id=mol_ids[i]
        )

        dataset.append(graph)

    return dataset



molecule = smiles_to_molecule("CCO")
print('test molecule-------------------------')
print(molecule)


dataset = build_graph_dataset(smiles_df, labels_clean, mask, mol_ids_df)

print("Dataset size:", len(dataset))
print("First graph:", dataset[0])


# Chia dữ liệu theo random split 80-10-10
def split_dataset(dataset, ratio=(0.8, 0.1, 0.1), seed=42):
    """
    Chia dataset thành 3 phần: train, val, test theo tỉ lệ.
    
    Tham số:
        - dataset: list chứa các đối tượng Data (PyG Data)
        - ratio: bộ 3 số (train_ratio, val_ratio, test_ratio)
        - seed: số random seed để kết quả tách dữ liệu ổn định mỗi lần chạy
        
    Ý tưởng:
        1. Trộn dữ liệu (shuffle) để không bị lệch thứ tự
        2. Tính số phần tử cho train, val, test dựa trên tỉ lệ
        3. Cắt dataset thành 3 phần bằng slicing
    """

    train_ratio, val_ratio, test_ratio = ratio

    random.seed(seed)   # để lần sau chạy lại vẫn ra cùng kết quả
    random.shuffle(dataset)
    total = len(dataset)
    # số mẫu cho train
    len_train = int(train_ratio * total)

    # số mẫu cho validation
    len_val = int(val_ratio * total)

    # test sẽ lấy phần còn lại
    len_test = total - len_train - len_val

    print("Tổng số mẫu:", total)
    print("Train:", len_train)
    print("Val  :", len_val)
    print("Test :", len_test)

    train_set = dataset[:len_train]
    val_set   = dataset[len_train : len_train + len_val]
    test_set  = dataset[len_train + len_val :]

    return train_set, val_set, test_set


train_dataset, val_dataset, test_dataset = split_dataset(dataset, ratio=(0.8, 0.1, 0.1))

torch.save(train_dataset, "processed/train.pt")
torch.save(val_dataset, "processed/val.pt")
torch.save(test_dataset, "processed/test.pt")

