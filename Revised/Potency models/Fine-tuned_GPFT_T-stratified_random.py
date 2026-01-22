# -*- coding: utf-8 -*-
"""GPFT_Final_Fixed_Vocab_IMPROVED.py"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import AttentiveFP
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
from typing import Optional
from collections import Counter, OrderedDict
from torch.cuda.amp import autocast, GradScaler

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    PRETRAINED_MODEL = "potency_enhanced_features_best_model.pt"
    VOCAB_FILE = "vocab_enhanced_features.pkl"
    TOKENIZER_FILE = "tokenizer_enhanced_features.pkl"
    
    # Data files
    TRAIN_DATA = "polaris_train.csv"
    TEST_DATA = "polaris_unblinded_test.csv"
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 64
    MAX_EPOCHS = 200
    PATIENCE = 15
    SEED = 42
    
    # Model Params (Must match pretrained)
    HIDDEN_DIM = 256
    OUTPUT_DIM = 32
    DROPOUT = 0.2
    
    # Gradual unfreezing
    PHASE1_EPOCHS = 20
    PHASE1_LR = 1e-4
    PHASE2_LR = 1e-5
    
    # NEW: Performance improvements
    NUM_WORKERS = 4  # Parallel data loading
    USE_MIXED_PRECISION = True  # FP16 training
    USE_SCHEDULER = True  # Learning rate scheduling

    VAL_SIZE = 0.10  # 10% validation from each protein
    
    OUTPUT_PREFIX = "target_specific_random_p1-20"

print(f"Using device: {Config.DEVICE}")
print(f"Mixed Precision: {Config.USE_MIXED_PRECISION}")

# ==============================================================================
# 2. ROBUST VOCAB CLASS (FIXED THE ATTRIBUTE ERROR)
# ==============================================================================
class Vocab:
    """Simple vocabulary class for token-to-index mapping"""
    def __init__(self, counter=None, max_size=None, min_freq=1, specials=('<pad>', '<unk>'), 
                 special_first=True):
        self.freqs = counter if counter is not None else Counter()
        self.stoi = OrderedDict()
        self.itos = []
        
        # Add special tokens
        if special_first:
            for tok in specials:
                self.itos.append(tok)
                self.stoi[tok] = len(self.itos) - 1
        
        # Add regular tokens
        if counter is not None:
            for tok, freq in counter.most_common(max_size):
                if freq >= min_freq and tok not in self.stoi:
                    self.itos.append(tok)
                    self.stoi[tok] = len(self.itos) - 1
        
        # Add special tokens at end if not first
        if not special_first:
            for tok in specials:
                if tok not in self.stoi:
                    self.itos.append(tok)
                    self.stoi[tok] = len(self.itos) - 1
        
        self.unk_index = self.stoi.get('<unk>', 0)
    
    def __len__(self):
        return len(self.itos)
    
    def __getitem__(self, token):
        # --- CRITICAL FIX FOR ATTRIBUTE ERROR ---
        if not hasattr(self, 'unk_index'):
            self.unk_index = self.stoi.get('<unk>', 0)
        return self.stoi.get(token, self.unk_index)
    
    def __call__(self, tokens):
        if not hasattr(self, 'unk_index'):
            self.unk_index = self.stoi.get('<unk>', 0)
        return [self[token] for token in tokens]

# ==============================================================================
# 3. TOKENIZER & UTILS
# ==============================================================================
def protein_tokenizer(seq): return list(seq.upper())

class ProteinTokenizer:
    def __init__(self): self.tokenizer = protein_tokenizer
    def tokenize(self, seq): return self.tokenizer(seq)

def plot_actual_vs_predicted(actual, predicted, path, title):
    if len(actual) == 0: return
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.6, edgecolors='k')
    mn, mx = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.savefig(path, dpi=300)
    plt.close()

def plot_metrics(train_mae, val_mae, val_r2, val_corr, path):
    if not train_mae: return
    epochs = range(1, len(train_mae) + 1)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1); plt.plot(epochs, train_mae, '.-', label='Train'); plt.plot(epochs, val_mae, '.-', label='Val'); plt.title("MAE"); plt.legend(); plt.grid(True)
    plt.subplot(1, 3, 2); plt.plot(epochs, val_r2, '.-', color='green'); plt.title("R² (CoD)"); plt.grid(True)
    plt.subplot(1, 3, 3); plt.plot(epochs, val_corr, '.-', color='purple'); plt.title("r² (Corr)"); plt.grid(True)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

# ==============================================================================
# 4. ENHANCED FEATURIZATION
# ==============================================================================
def get_atom_features(atom):
    atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Other']
    atom_type = atom.GetSymbol()
    atom_type_enc = [int(atom_type == t) for t in atom_types[:-1]] + [int(atom_type not in atom_types[:-1])]
    degree_enc = [int(atom.GetDegree() == i) for i in range(6)]
    h_count_enc = [int(atom.GetTotalNumHs() == i) for i in range(5)]
    hybrid_enc = [int(atom.GetHybridization() == h) for h in [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ]]
    chirality = [
        int(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
        int(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW),
        int(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW),
        int(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_OTHER)
    ]
    electronegativity = {'C': 2.55, 'N': 3.04, 'O': 3.44, 'S': 2.58, 'F': 3.98, 'P': 2.19, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66}.get(atom_type, 2.5) / 4.0
    return np.array(atom_type_enc + degree_enc + h_count_enc + hybrid_enc + [int(atom.GetIsAromatic()), int(atom.IsInRing())] + chirality + [atom.GetMass() / 100.0, electronegativity], dtype=np.float32)

def get_bond_features(bond):
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    stereo = [int(bond.GetStereo() == s) for s in [
        Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOANY
    ]]
    return np.array([int(bond.GetBondType() == bt) for bt in bond_types] + [int(bond.GetIsConjugated()), int(bond.IsInRing())] + stereo, dtype=np.float32)

def smiles_to_graph_enhanced(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        try: rdPartialCharges.ComputeGasteigerCharges(mol)
        except: pass
        atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(np.array(atom_features), dtype=torch.float)
        edge_indices, edge_features = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = get_bond_features(bond)
            edge_indices.extend([[i, j], [j, i]])
            edge_features.extend([bf, bf])
        if not edge_indices:
            return Data(x=x, edge_index=torch.zeros((2,0), dtype=torch.long), edge_attr=torch.zeros((0,10), dtype=torch.float))
        return Data(x=x, edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(), edge_attr=torch.tensor(np.array(edge_features), dtype=torch.float))
    except: return None

# ==============================================================================
# 5. MODEL ARCHITECTURE
# ==============================================================================
class EnhancedProteinLigandModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim*2, dropout, batch_first=True), 
            num_layers=2
        )
        self.ligand_gnn = AttentiveFP(35, hidden_dim, output_dim, 10, 4, 2, dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, p, l):
        p_emb = self.transformer_encoder(self.embedding(p)).mean(dim=1)
        l_emb = self.ligand_gnn(l.x, l.edge_index, l.edge_attr, l.batch)
        return self.fc(torch.cat([p_emb, l_emb], dim=1)).squeeze(1)

# ==============================================================================
# 6. DATASET & LOADING
# ==============================================================================
class PolarisDataset(Dataset):
    def __init__(self, df, vocab, tokenizer):
        self.data = []
        
        # Verbose detection of target type
        target_col = None
        target_mode = 'unknown'
        
        if 'pIC50' in df.columns: target_col='pIC50'; target_mode='pIC50'
        elif 'logIC50' in df.columns: target_col='logIC50'; target_mode='logIC50'
        elif 'IC50' in df.columns: target_col='IC50'; target_mode='IC50'
        
        print(f"Dataset Target Mode: {target_mode} (Col: {target_col})")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                # UPDATED: Look for ligand_smiles as well
                smi = row.get('ligand_smiles', row.get('SMILES', row.get('canonical_smiles')))
                # UPDATED: Look for protein_sequence as well
                seq = row.get('protein_sequence', row.get('PROTEIN_SEQ', row.get('protein_sequence')))
                
                if pd.isna(smi) or pd.isna(seq): continue

                raw_val = float(row[target_col])
                if np.isnan(raw_val) or np.isinf(raw_val): continue
                
                if target_mode == 'pIC50':
                    val = 9.0 - raw_val
                elif target_mode == 'IC50':
                    val = math.log10(raw_val)
                else:
                    val = raw_val

                graph = smiles_to_graph_enhanced(smi)
                if graph:
                    graph.y = torch.tensor([val], dtype=torch.float)
                    tokenized = tokenizer.tokenize(seq)
                    seq_tensor = torch.tensor(vocab(tokenized), dtype=torch.long)
                    self.data.append((seq_tensor, graph, graph.y))
            except Exception as e:
                if idx == 0: print(f"Error on row 0: {e}")
                continue
        print(f"Valid samples: {len(self.data)} / {len(df)}")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch):
    if not batch: return None, None, None
    p, l, t = zip(*batch)
    return nn.utils.rnn.pad_sequence(p, batch_first=True, padding_value=0), Batch.from_data_list(l), torch.cat(t).view(-1)

# ==============================================================================
# 7. TRAINING LOOP (IMPROVED WITH MIXED PRECISION)
# ==============================================================================
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, count = 0, 0
    for p, l, t in tqdm(loader, desc="Training", leave=False):
        if p is None: continue
        p, l, t = p.to(device), l.to(device), t.to(device)
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if scaler:
            with autocast():
                out = model(p, l)
                loss = criterion(out, t)
            if torch.isnan(loss): continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(p, l)
            loss = criterion(out, t)
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item() * p.size(0)
        count += p.size(0)
    return total_loss / count if count > 0 else float('inf')

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, count = 0, 0
    preds, actuals = [], []
    with torch.no_grad():
        for p, l, t in tqdm(loader, desc="Eval", leave=False):
            if p is None: continue
            p, l, t = p.to(device), l.to(device), t.to(device)
            out = model(p, l)
            total_loss += criterion(out, t).item() * p.size(0)
            count += p.size(0)
            preds.extend(out.cpu().numpy())
            actuals.extend(t.cpu().numpy())
            
    if count == 0: return float('nan'), 0.0, 0.0, torch.tensor([]), torch.tensor([])
    mae = total_loss / count
    y_true, y_pred = np.array(actuals), np.array(preds)
    
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
    
    corr = 0.0
    if len(y_true) > 1 and np.std(y_pred) > 1e-9:
        try:
            corr = pearsonr(y_true, y_pred)[0]**2
        except: pass
        
    return mae, r2, corr, torch.tensor(y_pred), torch.tensor(y_true)

# ==============================================================================
# 8. TARGET-SPECIFIC RANDOM SPLIT FUNCTION (UPDATED)
# ==============================================================================
def target_specific_random_split(dataframe, val_size=0.10, seed=42):
    """
    Performs Random splitting SEPARATELY for each protein target.
    Ensures both SARS-CoV-2 and MERS-CoV proteins appear in both train and val sets.
    
    Args:
        dataframe: DataFrame with protein sequences and target values
        val_size: Fraction of data to use for validation (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        train_df, val_df: Training and validation DataFrames
    """
    print(f"\n{'='*60}")
    print(f"CREATING TARGET-SPECIFIC RANDOM SPLIT")
    print(f"{'='*60}")
    
    # Check for required column
    if 'protein_sequence' not in dataframe.columns:
        raise KeyError("DataFrame missing 'protein_sequence' column for split")
    
    print(f"Original dataset size: {len(dataframe)}")
    
    # Define target protein sequences
    SARS_COV2_SEQ = "RKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQC"
    MERS_COV_SEQ = "VKMSHPSGDVEACMVQVTCGSMTLNGLWLDNTVWCPRHVMCPADQLSDPNYDALLISMTNHSFSVQKHIGAPANLRVVGHAMQGTLLKLTVDVANPSTPAYTFTTVKPGAAFSVLACYNGRPTGTFTVVMRPNYTIKGSFLCGSCGSVGYTKEGSVINFCYMHQMELANGTHTGSAFDGTMYGAFMDKQVHQVQLTDKYCSVNVVAWLYAAILNGCAWFVKPNRTSVVSFNEWALANQFTEFVGTQSVDMLAVKTGVAIEQLLYAIQQLYTGFQGKQILGSTMLEDEFTPEDVNMQI"
    
    # Separate data by protein target
    sars_mask = dataframe['protein_sequence'] == SARS_COV2_SEQ
    mers_mask = dataframe['protein_sequence'] == MERS_COV_SEQ
    other_mask = ~(sars_mask | mers_mask)
    
    sars_df = dataframe[sars_mask].reset_index(drop=True)
    mers_df = dataframe[mers_mask].reset_index(drop=True)
    other_df = dataframe[other_mask].reset_index(drop=True)
    
    print(f"\nProtein distribution:")
    print(f"  SARS-CoV-2 (6Y2E): {len(sars_df)} samples")
    print(f"  MERS-CoV (5C3N): {len(mers_df)} samples")
    print(f"  Other proteins: {len(other_df)} samples")
    
    # Split each protein target separately
    train_dfs = []
    val_dfs = []
    
    # Split SARS-CoV-2 data
    if len(sars_df) > 0:
        n_val_sars = max(1, int(len(sars_df) * val_size))
        sars_indices = np.arange(len(sars_df))
        np.random.seed(seed)
        np.random.shuffle(sars_indices)
        
        val_idx_sars = sars_indices[:n_val_sars]
        train_idx_sars = sars_indices[n_val_sars:]
        
        train_dfs.append(sars_df.iloc[train_idx_sars])
        val_dfs.append(sars_df.iloc[val_idx_sars])
        
        print(f"  SARS-CoV-2 split: {len(train_idx_sars)} train, {len(val_idx_sars)} val")
    
    # Split MERS-CoV data
    if len(mers_df) > 0:
        n_val_mers = max(1, int(len(mers_df) * val_size))
        mers_indices = np.arange(len(mers_df))
        np.random.seed(seed + 1)  # Different seed for MERS
        np.random.shuffle(mers_indices)
        
        val_idx_mers = mers_indices[:n_val_mers]
        train_idx_mers = mers_indices[n_val_mers:]
        
        train_dfs.append(mers_df.iloc[train_idx_mers])
        val_dfs.append(mers_df.iloc[val_idx_mers])
        
        print(f"  MERS-CoV split: {len(train_idx_mers)} train, {len(val_idx_mers)} val")
    
    # Split other proteins (if any)
    if len(other_df) > 0:
        n_val_other = max(1, int(len(other_df) * val_size))
        other_indices = np.arange(len(other_df))
        np.random.seed(seed + 2)  # Different seed for others
        np.random.shuffle(other_indices)
        
        val_idx_other = other_indices[:n_val_other]
        train_idx_other = other_indices[n_val_other:]
        
        train_dfs.append(other_df.iloc[train_idx_other])
        val_dfs.append(other_df.iloc[val_idx_other])
        
        print(f"  Other proteins split: {len(train_idx_other)} train, {len(val_idx_other)} val")
    
    # Combine all splits
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    
    # Shuffle the combined datasets
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"\nFinal split sizes:")
    print(f"  Training set: {len(train_df)} samples ({len(train_df)/len(dataframe)*100:.1f}%)")
    print(f"  Validation set: {len(val_df)} samples ({len(val_df)/len(dataframe)*100:.1f}%)")
    
    # Verify protein representation in both sets
    print(f"\nProtein representation in splits:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df)]:
        n_sars = (split_df['protein_sequence'] == SARS_COV2_SEQ).sum()
        n_mers = (split_df['protein_sequence'] == MERS_COV_SEQ).sum()
        n_other = len(split_df) - n_sars - n_mers
        print(f"  {split_name}: SARS-CoV-2={n_sars}, MERS-CoV={n_mers}, Other={n_other}")
    
    print(f"{'='*60}\n")
    
    return train_df, val_df

# ==============================================================================
# 9. MAIN
# ==============================================================================
def main():
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    print("\n" + "="*60 + "\nPOLARIS FINE-TUNING (IMPROVED - STRATIFIED)\n" + "="*60)

    # 1. Load Vocab
    print("1. Loading Vocab...")
    with open(Config.VOCAB_FILE, "rb") as f: vocab = pickle.load(f)
    with open(Config.TOKENIZER_FILE, "rb") as f: tokenizer = pickle.load(f)

    # 2. Load Data
    print("2. Loading Data...")
    if not os.path.exists(Config.TRAIN_DATA) or not os.path.exists(Config.TEST_DATA):
        raise FileNotFoundError("Missing Data Files")

    train_full = pd.read_csv(Config.TRAIN_DATA)
    test_df = pd.read_csv(Config.TEST_DATA)
    
    # --- CRITICAL UPDATE: Standardize columns to match Code 2 ---
    print("   -> Standardizing columns...")
    for df in [train_full, test_df]:
        df.columns = df.columns.str.strip()
        # Rename columns to standard names used by the split strategy
        df.rename(columns={
            'SMILES': 'ligand_smiles', 
            'canonical_smiles': 'ligand_smiles',
            'PROTEIN_SEQ': 'protein_sequence', 
            'PROTEIN': 'protein_sequence'
        }, inplace=True)
        
        # Ensure pIC50 exists or can be calculated (similar to Code 2 logic)
        if 'pIC50' not in df.columns: 
            if 'IC50' in df.columns: 
                # Note: This logic assumes IC50 is in nM if we do log conversion
                pass # Dataset class handles log10 logic, but let's leave as is for Dataset class
            elif 'logIC50' in df.columns: 
                pass # Dataset class handles this

    # --- EXECUTE THE MATCHING STRATEGY ---
    train_df, val_df = target_specific_random_split(train_full, val_size=0.1, seed=Config.SEED)
    
    train_ds = PolarisDataset(train_df, vocab, tokenizer)
    if len(train_ds) == 0: raise ValueError("Train Dataset is Empty")
    
    val_ds = PolarisDataset(val_df, vocab, tokenizer)
    test_ds = PolarisDataset(test_df, vocab, tokenizer)

    # IMPROVED: DataLoaders with parallel loading
    train_loader = DataLoader(
        train_ds, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == 'cuda' else False,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=Config.BATCH_SIZE, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if Config.DEVICE == 'cuda' else False
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=Config.BATCH_SIZE, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if Config.DEVICE == 'cuda' else False
    )

    # 3. Model
    print("3. Model Setup...")
    model = EnhancedProteinLigandModel(len(vocab), Config.HIDDEN_DIM, Config.OUTPUT_DIM, Config.DROPOUT).to(Config.DEVICE)
    ckpt = torch.load(Config.PRETRAINED_MODEL, map_location=Config.DEVICE)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    criterion = nn.L1Loss()
    
    # IMPROVED: Mixed precision scaler
    scaler = GradScaler() if Config.USE_MIXED_PRECISION and Config.DEVICE == 'cuda' else None

    # 4. Zero Shot
    print("4. Zero-Shot Eval...")
    base_mae, base_r2, base_corr, base_preds, base_targets = evaluate(model, test_loader, criterion, Config.DEVICE)
    print(f"   Base MAE: {base_mae:.4f} | R²: {base_r2:.4f} | r²: {base_corr:.4f}")

    # 5. Training
    train_maes, val_maes, val_r2s, val_corrs = [], [], [], []
    best_loss = float('inf')
    best_preds, best_targets = None, None
    save_path = f"{Config.OUTPUT_PREFIX}_best.pt"

    # Phase 1
    print(f"\n--- Phase 1: Head Only ({Config.PHASE1_EPOCHS} epochs) ---")
    for p in model.parameters(): p.requires_grad = False
    for p in model.fc.parameters(): p.requires_grad = True
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.PHASE1_LR)
    
    for epoch in range(Config.PHASE1_EPOCHS):
        epoch_start = time.time()
        t_loss = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE, scaler)
        v_loss, v_r2, v_corr, _, _ = evaluate(model, val_loader, criterion, Config.DEVICE)
        epoch_time = time.time() - epoch_start
        
        print(f"Ep {epoch+1}/{Config.PHASE1_EPOCHS} | T: {t_loss:.4f} | V: {v_loss:.4f} | "
              f"R²: {v_r2:.4f} | Time: {epoch_time:.1f}s")
        train_maes.append(t_loss); val_maes.append(v_loss); val_r2s.append(v_r2); val_corrs.append(v_corr)

    # Phase 2
    print(f"\n--- Phase 2: Unfreezing (up to {Config.MAX_EPOCHS - Config.PHASE1_EPOCHS} epochs) ---")
    for p in model.transformer_encoder.layers[-1].parameters(): p.requires_grad = True
    for p in model.ligand_gnn.parameters(): p.requires_grad = True
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.PHASE2_LR)
    
    # IMPROVED: Learning rate scheduler
    scheduler = None
    if Config.USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
    # IMPROVED: Early stopping counter
    patience_counter = 0
    
    for epoch in range(Config.PHASE1_EPOCHS, Config.MAX_EPOCHS):
        epoch_start = time.time()
        t_loss = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE, scaler)
        v_loss, v_r2, v_corr, preds, targets = evaluate(model, val_loader, criterion, Config.DEVICE)
        epoch_time = time.time() - epoch_start
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch+1}/{Config.MAX_EPOCHS} | T: {t_loss:.4f} | V: {v_loss:.4f} | "
              f"R²: {v_r2:.4f} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        
        train_maes.append(t_loss); val_maes.append(v_loss); val_r2s.append(v_r2); val_corrs.append(v_corr)
        
        # IMPROVED: Comprehensive checkpointing
        if v_loss < best_loss:
            best_loss = v_loss
            best_preds, best_targets = preds, targets
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_loss': best_loss,
                'train_maes': train_maes,
                'val_maes': val_maes,
                'val_r2s': val_r2s,
                'val_corrs': val_corrs
            }, save_path)
            print(f"   ✓ Checkpoint saved (Val MAE: {best_loss:.4f})")
        else:
            patience_counter += 1
        
        # IMPROVED: Learning rate scheduling
        if scheduler:
            scheduler.step(v_loss)
        
        # IMPROVED: Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1} (no improvement for {Config.PATIENCE} epochs)")
            break

    # 6. Final Eval & Report
    print("\n--- Final Evaluation ---")
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    tuned_mae, tuned_r2, tuned_corr, tuned_preds, tuned_targets = evaluate(model, test_loader, criterion, Config.DEVICE)

    print("\n" + "="*80)
    print("FINAL PERFORMANCE COMPARISON ON TEST SET")
    print("="*80)
    print(f"  Metric                   |  Base Model (Zero-Shot) |  Finetuned Model  |  Improvement")
    print(f"  -------------------------|-------------------------|-------------------|---------------")
    print(f"  MAE                      |  {base_mae:<23.4f}|  {tuned_mae:<17.4f}|  {(base_mae - tuned_mae):.4f}")
    print(f"  R² (Coeff of Det.)       |  {base_r2:<23.4f}|  {tuned_r2:<17.4f}|  {(tuned_r2 - base_r2):.4f}")
    print(f"  r² (Sq. Correlation)     |  {base_corr:<23.4f}|  {tuned_corr:<17.4f}|  {(tuned_corr - base_corr):.4f}")
    print("="*80)
    print(f"\nTraining stopped at epoch: {len(train_maes)}")

    # 7. SAVE PREDICTION FILES
    print("\n--- Saving Prediction Files ---")
    
    # Save base model predictions
    base_results_df = pd.DataFrame({
        'Actual': base_targets.numpy(),
        'Predicted': base_preds.numpy(),
        'Residual': base_targets.numpy() - base_preds.numpy()
    })
    base_results_df.to_csv(f"{Config.OUTPUT_PREFIX}_base_predictions.csv", index=False)
    print(f"✓ Saved: {Config.OUTPUT_PREFIX}_base_predictions.csv")
    
    # Save finetuned model predictions
    tuned_results_df = pd.DataFrame({
        'Actual': tuned_targets.numpy(),
        'Predicted': tuned_preds.numpy(),
        'Residual': tuned_targets.numpy() - tuned_preds.numpy()
    })
    tuned_results_df.to_csv(f"{Config.OUTPUT_PREFIX}_tuned_predictions.csv", index=False)
    print(f"✓ Saved: {Config.OUTPUT_PREFIX}_tuned_predictions.csv")
    
    # Save validation set predictions (best epoch)
    if best_preds is not None and best_targets is not None:
        val_results_df = pd.DataFrame({
            'Actual': best_targets.numpy(),
            'Predicted': best_preds.numpy(),
            'Residual': best_targets.numpy() - best_preds.numpy()
        })
        val_results_df.to_csv(f"{Config.OUTPUT_PREFIX}_val_predictions.csv", index=False)
        print(f"✓ Saved: {Config.OUTPUT_PREFIX}_val_predictions.csv")
    
    # Save summary metrics
    summary_df = pd.DataFrame({
        'Model': ['Base (Zero-Shot)', 'Finetuned'],
        'MAE': [base_mae, tuned_mae],
        'R2_CoD': [base_r2, tuned_r2],
        'r2_Corr': [base_corr, tuned_corr],
        'MAE_Improvement': [0.0, base_mae - tuned_mae],
        'R2_Improvement': [0.0, tuned_r2 - base_r2]
    })
    summary_df.to_csv(f"{Config.OUTPUT_PREFIX}_summary_metrics.csv", index=False)
    print(f"✓ Saved: {Config.OUTPUT_PREFIX}_summary_metrics.csv")
    
    # Save training history
    history_df = pd.DataFrame({
        'Epoch': range(1, len(train_maes) + 1),
        'Train_MAE': train_maes,
        'Val_MAE': val_maes,
        'Val_R2': val_r2s,
        'Val_r2_Corr': val_corrs
    })
    history_df.to_csv(f"{Config.OUTPUT_PREFIX}_training_history.csv", index=False)
    print(f"✓ Saved: {Config.OUTPUT_PREFIX}_training_history.csv")
    
    print("\n--- Saving Plots ---")
    plot_actual_vs_predicted(base_targets.numpy(), base_preds.numpy(), f"{Config.OUTPUT_PREFIX}_base.png", "Base")
    print(f"✓ Saved: {Config.OUTPUT_PREFIX}_base.png")
    
    plot_actual_vs_predicted(tuned_targets.numpy(), tuned_preds.numpy(), f"{Config.OUTPUT_PREFIX}_tuned.png", "Tuned")
    print(f"✓ Saved: {Config.OUTPUT_PREFIX}_tuned.png")
    
    plot_metrics(train_maes, val_maes, val_r2s, val_corrs, f"{Config.OUTPUT_PREFIX}_learning_curves.png")
    print(f"✓ Saved: {Config.OUTPUT_PREFIX}_learning_curves.png")
    
    print("\n" + "="*80)
    print("ALL FILES SAVED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
