import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from torch_geometric.data import Data as GraphData
from rdkit import Chem



class SmartDrugDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128, label_cols=None):
        self.data = []
        print("Processing Dataset...")
        
        # Nếu không truyền label_cols, lấy tất cả cột trừ SMILES
        if label_cols is None:
            label_cols = [c for c in df.columns if c != 'cleaned_smiles']

        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            # --- Graph ---
            graph = smiles_to_graph(row['cleaned_smiles'])
            if graph is None:
                continue

            # --- Text ---
            text_enc = tokenizer(
                row['cleaned_smiles'], 
                max_length=max_len, 
                padding='max_length', 
                truncation=True, 
                return_tensors="pt"
            )

            # --- Labels ---
            labels = row[label_cols].to_numpy(dtype=float)  # giữ NaN
            label_tensor = torch.tensor(labels, dtype=torch.float)

            self.data.append({
                'graph': graph,
                'input_ids': text_enc['input_ids'].squeeze(0),
                'attention_mask': text_enc['attention_mask'].squeeze(0),
                'label': label_tensor
            })
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

