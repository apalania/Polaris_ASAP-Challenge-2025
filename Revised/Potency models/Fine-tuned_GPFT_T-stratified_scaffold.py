# -*- coding: utf-8 -*-
"""GPFT_Final_Fixed_Vocab_IMPROVED_SCAFFOLD_SPLIT.py"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import time
import json
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

# ===== NEW IMPORTS FOR SCAFFOLD SPLIT =====
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import random

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
    PHASE1_EPOCHS = 5
    PHASE1_LR = 1e-4
    PHASE2_LR = 1e-5
    
    # Performance improvements
    NUM_WORKERS = 4
    USE_MIXED_PRECISION = True
    USE_SCHEDULER = True
    
    OUTPUT_PREFIX = "polaris_finetune_result"

print(f"Using device: {Config.DEVICE}")
print(f"Mixed Precision: {Config.USE_MIXED_PRECISION}")

# ==============================================================================
# 2. SCAFFOLD SPLITTING FUNCTIONS (FIXED VERSION)
# ==============================================================================

def generate_scaffold(smiles, include_chirality=False):
    """Generate Murcko scaffold for a SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
        return scaffold
    except:
        return None


def scaffold_split(df, smiles_col='SMILES', 
                   train_size=0.8, val_size=0.1, test_size=0.1,
                   seed=42):
    """Split dataset based on molecular scaffolds"""
    random.seed(seed)
    np.random.seed(seed)
    
    # IMPORTANT: Reset index to ensure we use positional indices
    df = df.reset_index(drop=True)
    
    # Generate scaffolds for all molecules
    print("  Generating molecular scaffolds...")
    scaffolds = defaultdict(list)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Scaffold generation", leave=False):
        smiles = row[smiles_col]
        scaffold = generate_scaffold(smiles)
        
        if scaffold is not None:
            scaffolds[scaffold].append(idx)
        else:
            scaffolds['_invalid_'].append(idx)
    
    # Sort scaffolds by size
    scaffold_sets = list(scaffolds.values())
    random.shuffle(scaffold_sets)
    scaffold_sets = sorted(scaffold_sets, key=len, reverse=True)
    
    print(f"    → {len(scaffolds)} unique scaffolds")
    print(f"    → Largest: {len(scaffold_sets[0])}, Smallest: {len(scaffold_sets[-1])}")
    
    # Distribute scaffolds to splits
    train_idx, val_idx, test_idx = [], [], []
    train_count, val_count = 0, 0
    
    total_size = len(df)
    train_target = int(train_size * total_size)
    val_target = int(val_size * total_size)
    
    for scaffold_set in scaffold_sets:
        if train_count < train_target:
            train_idx.extend(scaffold_set)
            train_count += len(scaffold_set)
        elif val_count < val_target:
            val_idx.extend(scaffold_set)
            val_count += len(scaffold_set)
        else:
            test_idx.extend(scaffold_set)
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    return train_df, val_df, test_df


def target_specific_scaffold_split(df, target_col='PROTEIN_SEQ', 
                                   smiles_col='SMILES',
                                   train_size=0.9, val_size=0.1, 
                                   test_size=0.0, seed=42,
                                   save_indices_path=None):
    """Perform scaffold split separately for each target AND save indices"""
    print(f"\n{'='*80}")
    print("TARGET-SPECIFIC SCAFFOLD SPLITTING")
    print(f"{'='*80}")
    
    if target_col not in df.columns:
        print(f"⚠️  Warning: '{target_col}' not found!")
        print(f"Available columns: {df.columns.tolist()}")
        print("Performing global scaffold split instead...")
        return scaffold_split(df, smiles_col, train_size, val_size, test_size, seed)
    
    targets = df[target_col].unique()
    print(f"Found {len(targets)} unique protein targets")
    
    all_train, all_val, all_test = [], [], []
    
    # Track ORIGINAL indices from the input dataframe
    train_indices = []
    val_indices = []
    test_indices = []
    
    for i, target in enumerate(targets, 1):
        # Get the subset for this target - KEEP ORIGINAL INDICES
        target_mask = df[target_col] == target
        target_df = df[target_mask].copy()
        original_indices = df[target_mask].index.tolist()  # Store original indices
        
        target_name = target[:50] + "..." if len(str(target)) > 50 else target
        print(f"\n[{i}/{len(targets)}] Target: {target_name}")
        print(f"  Samples: {len(target_df)}")
        
        if len(target_df) < 10:
            print(f"  ⚠️  Too few samples, adding all to training")
            all_train.append(target_df)
            train_indices.extend(original_indices)  # Use original indices
            continue
        
        # Add a temporary ID column to track rows through the scaffold split
        target_df_with_id = target_df.copy()
        target_df_with_id['_temp_original_idx'] = original_indices
        
        # Do the scaffold split with the ID column
        train_t_with_id, val_t_with_id, test_t_with_id = scaffold_split(
            target_df_with_id, smiles_col, train_size, val_size, test_size, seed
        )
        
        print(f"  ✓ Split: Train={len(train_t_with_id)}, Val={len(val_t_with_id)}, Test={len(test_t_with_id)}")
        
        # Extract the original indices
        train_orig_idx = train_t_with_id['_temp_original_idx'].tolist()
        val_orig_idx = val_t_with_id['_temp_original_idx'].tolist()
        test_orig_idx = test_t_with_id['_temp_original_idx'].tolist()
        
        # Remove the temporary column
        train_t_with_id.drop('_temp_original_idx', axis=1, inplace=True)
        val_t_with_id.drop('_temp_original_idx', axis=1, inplace=True)
        test_t_with_id.drop('_temp_original_idx', axis=1, inplace=True)
        
        all_train.append(train_t_with_id)
        all_val.append(val_t_with_id)
        all_test.append(test_t_with_id)
        
        train_indices.extend(train_orig_idx)
        val_indices.extend(val_orig_idx)
        test_indices.extend(test_orig_idx)
    
    # Concatenate all splits
    train_df = pd.concat(all_train, ignore_index=True)
    val_df = pd.concat(all_val, ignore_index=True) if all_val else pd.DataFrame()
    test_df = pd.concat(all_test, ignore_index=True) if all_test else pd.DataFrame()
    
    print(f"\n{'='*80}")
    print("FINAL SPLIT SUMMARY:")
    print(f"{'='*80}")
    print(f"  Train: {len(train_df):>6} samples ({len(train_df)/len(df)*100:>5.1f}%)")
    print(f"  Val:   {len(val_df):>6} samples ({len(val_df)/len(df)*100:>5.1f}%)")
    print(f"  Test:  {len(test_df):>6} samples ({len(test_df)/len(df)*100:>5.1f}%)")
    print(f"  Total: {len(train_df) + len(val_df) + len(test_df):>6} samples")
    print(f"{'='*80}\n")
    
    # Save indices to file
    if save_indices_path:
        split_info = {
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'seed': seed,
            'target_col': target_col,
            'smiles_col': smiles_col,
            'train_size': train_size,
            'val_size': val_size
        }
        with open(save_indices_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"✅ Split indices saved to: {save_indices_path}\n")
    
    return train_df, val_df, test_df

# ==============================================================================
# 3. ROBUST VOCAB CLASS
# ==============================================================================
class Vocab:
    """Simple vocabulary class for token-to-index mapping"""
    def __init__(self, counter=None, max_size=None, min_freq=1, specials=('<pad>', '<unk>'), 
                 special_first=True):
        self.freqs = counter if counter is not None else Counter()
        self.stoi = OrderedDict()
        self.itos = []
        
        if special_first:
            for tok in specials:
                self.itos.append(tok)
                self.stoi[tok] = len(self.itos) - 1
        
        if counter is not None:
            for tok, freq in counter.most_common(max_size):
                if freq >= min_freq and tok not in self.stoi:
                    self.itos.append(tok)
                    self.stoi[tok] = len(self.itos) - 1
        
        if not special_first:
            for tok in specials:
                if tok not in self.stoi:
                    self.itos.append(tok)
                    self.stoi[tok] = len(self.itos) - 1
        
        self.unk_index = self.stoi.get('<unk>', 0)
    
    def __len__(self):
        return len(self.itos)
    
    def __getitem__(self, token):
        if not hasattr(self, 'unk_index'):
            self.unk_index = self.stoi.get('<unk>', 0)
        return self.stoi.get(token, self.unk_index)
    
    def __call__(self, tokens):
        if not hasattr(self, 'unk_index'):
            self.unk_index = self.stoi.get('<unk>', 0)
        return [self[token] for token in tokens]

# ==============================================================================
# 4. TOKENIZER & UTILS
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
# 5. ENHANCED FEATURIZATION
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
# 6. MODEL ARCHITECTURE
# ==============================================================================
class EnhancedProteinLigandModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, output_dim=32, dropout=0.2):
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
# 7. DATASET & LOADING
# ==============================================================================
class PolarisDataset(Dataset):
    def __init__(self, df, vocab, tokenizer):
        self.data = []
        
        target_col = None
        target_mode = 'unknown'
        
        if 'pIC50' in df.columns: target_col='pIC50'; target_mode='pIC50'
        elif 'logIC50' in df.columns: target_col='logIC50'; target_mode='logIC50'
        elif 'IC50' in df.columns: target_col='IC50'; target_mode='IC50'
        
        print(f"Dataset Target Mode: {target_mode} (Col: {target_col})")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing", leave=False):
            try:
                smi = row.get('SMILES', row.get('canonical_smiles'))
                seq = row.get('PROTEIN_SEQ', row.get('protein_sequence'))
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
# 8. TRAINING LOOP
# ==============================================================================
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, count = 0, 0
    for p, l, t in tqdm(loader, desc="Training", leave=False):
        if p is None: continue
        p, l, t = p.to(device), l.to(device), t.to(device)
        optimizer.zero_grad()
        
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
# 9. MAIN FUNCTION
# ==============================================================================
def main():
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    print("\n" + "="*80)
    print("POLARIS FINE-TUNING WITH TARGET-SPECIFIC SCAFFOLD SPLIT")
    print("="*80)

    # 1. Load Vocab
    print("\n1. Loading Vocab...")
    with open(Config.VOCAB_FILE, "rb") as f: vocab = pickle.load(f)
    with open(Config.TOKENIZER_FILE, "rb") as f: tokenizer = pickle.load(f)
    print(f"   ✓ Vocab size: {len(vocab)}")

    # 2. Load Data
    print("\n2. Loading Training Data...")
    if not os.path.exists(Config.TRAIN_DATA):
        raise FileNotFoundError(f"Missing {Config.TRAIN_DATA}")

    train_full = pd.read_csv(Config.TRAIN_DATA)
    train_full.columns = train_full.columns.str.strip()
    print(f"   ✓ Loaded {len(train_full)} training samples")
    print(f"   Columns: {train_full.columns.tolist()}")
    
    # 3. TARGET-SPECIFIC SCAFFOLD SPLIT
    print("\n3. Performing Target-Specific Scaffold Split...")
    
    smiles_col = None
    for col in ['SMILES', 'canonical_smiles', 'smiles']:
        if col in train_full.columns:
            smiles_col = col
            break
    
    if smiles_col is None:
        raise ValueError("No SMILES column found in data")
    
    print(f"   Using SMILES column: {smiles_col}")
    print(f"   Using target column: PROTEIN_SEQ")
    
    train_df, val_df, _ = target_specific_scaffold_split(
        train_full, 
        target_col='PROTEIN_SEQ',
        smiles_col=smiles_col,
        train_size=0.9,
        val_size=0.1,
        test_size=0.0,
        seed=Config.SEED,
        save_indices_path='scaffold_split_indices.json'
    )
    
    # Load test data
    if os.path.exists(Config.TEST_DATA):
        test_df = pd.read_csv(Config.TEST_DATA)
        test_df.columns = test_df.columns.str.strip()
        print(f"\n4. Loaded Test Data: {len(test_df)} samples")
    else:
        print("\n4. ⚠️  No separate test file found")
        test_df = pd.DataFrame()
    
    # 5. Create Datasets
    print("\n5. Creating PyTorch Datasets...")
    train_ds = PolarisDataset(train_df, vocab, tokenizer)
    if len(train_ds) == 0: 
        raise ValueError("Train Dataset is Empty")
    
    val_ds = PolarisDataset(val_df, vocab, tokenizer)
    test_ds = PolarisDataset(test_df, vocab, tokenizer) if len(test_df) > 0 else None

    # 6. DataLoaders
    print("\n6. Creating DataLoaders...")
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS, pin_memory=True if Config.DEVICE == 'cuda' else False,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn,
        num_workers=2, pin_memory=True if Config.DEVICE == 'cuda' else False
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn,
        num_workers=2, pin_memory=True if Config.DEVICE == 'cuda' else False
    ) if test_ds else None

    # 7. Model Setup
    print("\n7. Loading Pretrained Model...")
    model = EnhancedProteinLigandModel(
        len(vocab), Config.HIDDEN_DIM, Config.OUTPUT_DIM, Config.DROPOUT
    ).to(Config.DEVICE)
    
    ckpt = torch.load(Config.PRETRAINED_MODEL, map_location=Config.DEVICE)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    print("   ✓ Loaded pretrained weights")
    
    criterion = nn.L1Loss()
    scaler = GradScaler() if Config.USE_MIXED_PRECISION and Config.DEVICE == 'cuda' else None
    
    # 8. Zero-Shot Evaluation
    if test_loader:
        print("\n8. Zero-Shot Evaluation on Test Set...")
        base_mae, base_r2, base_corr, base_preds, base_targets = evaluate(
            model, test_loader, criterion, Config.DEVICE
        )
        print(f"   Base MAE:  {base_mae:.4f}")
        print(f"   Base R²:   {base_r2:.4f}")
        print(f"   Base r²:   {base_corr:.4f}")
    else:
        base_mae = base_r2 = base_corr = 0.0
        base_preds = base_targets = torch.tensor([])

    # 9. Training Phase 1: Head Only
    print(f"\n{'='*80}")
    print(f"PHASE 1: Fine-tuning Head Only ({Config.PHASE1_EPOCHS} epochs)")
    print(f"{'='*80}")

    train_maes, val_maes, val_r2s, val_corrs = [], [], [], []
    best_loss = float('inf')
    best_preds, best_targets = None, None
    save_path = f"{Config.OUTPUT_PREFIX}_best.pt"

    # Freeze all except head
    for p in model.parameters(): 
        p.requires_grad = False
    for p in model.fc.parameters(): 
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=Config.PHASE1_LR
    )

    for epoch in range(Config.PHASE1_EPOCHS):
        epoch_start = time.time()
        t_loss = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE, scaler)
        v_loss, v_r2, v_corr, _, _ = evaluate(model, val_loader, criterion, Config.DEVICE)
        epoch_time = time.time() - epoch_start
    
        print(f"Epoch {epoch+1:>3}/{Config.PHASE1_EPOCHS} | "
              f"Train MAE: {t_loss:.4f} | Val MAE: {v_loss:.4f} | "
              f"R²: {v_r2:.4f} | Time: {epoch_time:.1f}s")
    
        train_maes.append(t_loss)
        val_maes.append(v_loss)
        val_r2s.append(v_r2)
        val_corrs.append(v_corr)

    # 10. Training Phase 2: Gradual Unfreezing
    print(f"\n{'='*80}")
    print(f"PHASE 2: Unfreezing Encoder (up to {Config.MAX_EPOCHS} total epochs)")
    print(f"{'='*80}")

    # Unfreeze last transformer layer and GNN
    for p in model.transformer_encoder.layers[-1].parameters(): 
        p.requires_grad = True
    for p in model.ligand_gnn.parameters(): 
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=Config.PHASE2_LR
    )

    # Learning rate scheduler
    scheduler = None
    if Config.USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    patience_counter = 0

    for epoch in range(Config.PHASE1_EPOCHS, Config.MAX_EPOCHS):
        epoch_start = time.time()
        t_loss = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE, scaler)
        v_loss, v_r2, v_corr, preds, targets = evaluate(
            model, val_loader, criterion, Config.DEVICE
        )
        epoch_time = time.time() - epoch_start
    
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:>3}/{Config.MAX_EPOCHS} | "
              f"Train MAE: {t_loss:.4f} | Val MAE: {v_loss:.4f} | "
              f"R²: {v_r2:.4f} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
    
        train_maes.append(t_loss)
        val_maes.append(v_loss)
        val_r2s.append(v_r2)
        val_corrs.append(v_corr)
    
        # Checkpointing (FIXED INDENTATION)
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
        
        # Learning rate scheduling (FIXED INDENTATION)
        if scheduler:
            scheduler.step(v_loss)
        
        # Early stopping (FIXED INDENTATION)
        if patience_counter >= Config.PATIENCE:
            print(f"\n⚠️  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {Config.PATIENCE} epochs)")
            break

    # 11. Final Evaluation on Test Set
    print(f"\n{'='*80}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*80}")

    if test_loader:
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
        tuned_mae, tuned_r2, tuned_corr, tuned_preds, tuned_targets = evaluate(
            model, test_loader, criterion, Config.DEVICE
        )
    
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)
        print(f"{'Metric':<25} | {'Base (Zero-Shot)':<20} | {'Finetuned':<15} | {'Improvement'}")
        print("-" * 80)
        print(f"{'MAE':<25} | {base_mae:<20.4f} | {tuned_mae:<15.4f} | {(base_mae - tuned_mae):+.4f}")
        print(f"{'R² (Coefficient)':<25} | {base_r2:<20.4f} | {tuned_r2:<15.4f} | {(tuned_r2 - base_r2):+.4f}")
        print(f"{'r² (Correlation)':<25} | {base_corr:<20.4f} | {tuned_corr:<15.4f} | {(tuned_corr - base_corr):+.4f}")
        print("="*80)
    else:
        tuned_mae = tuned_r2 = tuned_corr = 0.0
        tuned_preds = tuned_targets = torch.tensor([])
        print("⚠️  No test set available for final evaluation")

    # 12. Save Results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    # Save predictions
    if test_loader and len(base_preds) > 0:
        base_results = pd.DataFrame({
            'Actual': base_targets.numpy(),
            'Predicted': base_preds.numpy(),
            'Residual': base_targets.numpy() - base_preds.numpy()
        })
        base_results.to_csv(f"{Config.OUTPUT_PREFIX}_base_predictions.csv", index=False)
        print(f"✓ {Config.OUTPUT_PREFIX}_base_predictions.csv")
    
        tuned_results = pd.DataFrame({
            'Actual': tuned_targets.numpy(),
            'Predicted': tuned_preds.numpy(),
            'Residual': tuned_targets.numpy() - tuned_preds.numpy()
        })
        tuned_results.to_csv(f"{Config.OUTPUT_PREFIX}_tuned_predictions.csv", index=False)
        print(f"✓ {Config.OUTPUT_PREFIX}_tuned_predictions.csv")

    # Save validation predictions (FIXED INDENTATION)
    if best_preds is not None:
        val_results = pd.DataFrame({
            'Actual': best_targets.numpy(),
            'Predicted': best_preds.numpy(),
            'Residual': best_targets.numpy() - best_preds.numpy()
        })
        val_results.to_csv(f"{Config.OUTPUT_PREFIX}_val_predictions.csv", index=False)
        print(f"✓ {Config.OUTPUT_PREFIX}_val_predictions.csv")

    # Save metrics summary (FIXED INDENTATION)
    if test_loader:
        summary = pd.DataFrame({
            'Model': ['Base (Zero-Shot)', 'Finetuned'],
            'MAE': [base_mae, tuned_mae],
            'R2_CoD': [base_r2, tuned_r2],
            'r2_Corr': [base_corr, tuned_corr],
            'MAE_Improvement': [0.0, base_mae - tuned_mae],
            'R2_Improvement': [0.0, tuned_r2 - base_r2]
        })
        summary.to_csv(f"{Config.OUTPUT_PREFIX}_summary_metrics.csv", index=False)
        print(f"✓ {Config.OUTPUT_PREFIX}_summary_metrics.csv")

    # Save training history (FIXED INDENTATION)
    history = pd.DataFrame({
        'Epoch': range(1, len(train_maes) + 1),
        'Train_MAE': train_maes,
        'Val_MAE': val_maes,
        'Val_R2': val_r2s,
        'Val_r2_Corr': val_corrs
    })
    history.to_csv(f"{Config.OUTPUT_PREFIX}_training_history.csv", index=False)
    print(f"✓ {Config.OUTPUT_PREFIX}_training_history.csv")

    # 13. Generate Plots
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}")

    if test_loader and len(base_preds) > 0:
        plot_actual_vs_predicted(
            base_targets.numpy(), base_preds.numpy(), 
            f"{Config.OUTPUT_PREFIX}_base.png", 
            "Base Model (Zero-Shot)"
        )
        print(f"✓ {Config.OUTPUT_PREFIX}_base.png")
        
        plot_actual_vs_predicted(
            tuned_targets.numpy(), tuned_preds.numpy(), 
            f"{Config.OUTPUT_PREFIX}_tuned.png", 
            "Finetuned Model"
        )
        print(f"✓ {Config.OUTPUT_PREFIX}_tuned.png")

    plot_metrics(
        train_maes, val_maes, val_r2s, val_corrs, 
        f"{Config.OUTPUT_PREFIX}_learning_curves.png"
    )
    print(f"✓ {Config.OUTPUT_PREFIX}_learning_curves.png")

    print(f"\n{'='*80}")
    print("✅ ALL DONE! Training completed successfully.")
    print(f"{'='*80}")
    print(f"\nBest model saved to: {save_path}")
    print(f"Split indices saved to: scaffold_split_indices.json")
    print(f"Training stopped at epoch: {len(train_maes)}")
    if test_loader:
        print(f"Final Test MAE: {tuned_mae:.4f}")
        print(f"Improvement: {(base_mae - tuned_mae):.4f}")

if __name__ == "__main__":
    main()
