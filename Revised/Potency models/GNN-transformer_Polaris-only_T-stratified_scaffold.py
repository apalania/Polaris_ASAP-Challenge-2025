import os
# --- 1. STRICT REPRODUCIBILITY SETUP (MUST BE FIRST) ---
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
# -------------------------------------------------------

import math
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import AttentiveFP
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen, rdPartialCharges
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
import pickle
from collections import Counter
from typing import Optional

# Set matplotlib DPI for high resolution
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
sns.set_style("whitegrid")

# --- 2. ENABLE SPEED OPTIMIZATIONS FOR RTX A4000 ---
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
# ---------------------------------------------------

# ------------------- Custom Vocabulary Builder -------------------

class Vocab:
    def __init__(self, tokens, specials=None):
        self.specials = specials if specials else []
        self.itos = []
        self.stoi = {}
        
        for special in self.specials:
            self.itos.append(special)
            self.stoi[special] = len(self.itos) - 1
        
        if isinstance(tokens, Counter):
            tokens = tokens.most_common()
        elif isinstance(tokens, dict):
            tokens = sorted(tokens.items(), key=lambda x: (-x[1], x[0]))
        
        for token, _ in tokens:
            if token not in self.stoi:
                self.itos.append(token)
                self.stoi[token] = len(self.itos) - 1
        
        self.default_index = self.stoi.get("<pad>", 0)
    
    def __len__(self):
        return len(self.itos)
    
    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)
    
    def __call__(self, tokens):
        return [self[token] for token in tokens]
    
    def set_default_index(self, index):
        self.default_index = index

def build_vocab_from_iterator(iterator, specials=None):
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    return Vocab(counter, specials=specials)

# ------------------- Config -------------------

class Config:
    # Data files
    TRAIN_DATA_FILE = "polaris_train.csv"
    TEST_DATA_FILE = "polaris_unblinded_test.csv"
    SCAFFOLD_SPLIT_FILE = "enhanced_scaffold_split_indices.json"
    
    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 200
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    HIDDEN_DIM = 256
    OUTPUT_DIM = 32
    DROPOUT = 0.2
    PATIENCE = 15
    SEED = 42
    USE_MIXED_PRECISION = True
    
    # Output paths
    OUTPUT_DIR = "enhanced_output"
    MODEL_DIR = "enhanced_models"
    MODEL_SAVE = os.path.join(MODEL_DIR, "enhanced_scaffold_best_model.pt")
    METRICS_FILE = os.path.join(OUTPUT_DIR, "enhanced_scaffold_training_metrics.csv")
    VOCAB_FILE = os.path.join(MODEL_DIR, "vocab_enhanced_scaffold.pkl")
    TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer_enhanced_scaffold.pkl")

# Create output directories
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.MODEL_DIR, exist_ok=True)

print(f"Running on: {Config.DEVICE} ({torch.cuda.get_device_name(0) if Config.DEVICE.type == 'cuda' else 'CPU'})")

# ------------------- Tokenizer -------------------

def protein_tokenizer(seq):
    return list(seq.upper())

class ProteinTokenizer:
    def __init__(self):
        self.tokenizer = protein_tokenizer
    
    def tokenize(self, seq):
        return self.tokenizer(seq)

# ------------------- Enhanced Feature Extraction -------------------

def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Other']
    atom_type = atom.GetSymbol()
    atom_type_enc = [int(atom_type == t) for t in atom_types[:-1]]
    atom_type_enc.append(int(atom_type not in atom_types[:-1]))
    
    degree = atom.GetDegree()
    degree_enc = [int(degree == i) for i in range(6)]
    
    h_count = atom.GetTotalNumHs()
    h_count_enc = [int(h_count == i) for i in range(5)]
    
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ]
    hybrid_enc = [int(atom.GetHybridization() == h) for h in hybridizations]
    
    is_aromatic = int(atom.GetIsAromatic())
    is_in_ring = int(atom.IsInRing())
    
    chirality = [
        int(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
        int(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW),
        int(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW),
        int(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_OTHER)
    ]
    
    atomic_mass = atom.GetMass() / 100.0
    electronegativity_map = {'C': 2.55, 'N': 3.04, 'O': 3.44, 'S': 2.58, 'F': 3.98, 'P': 2.19, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66}
    electronegativity = electronegativity_map.get(atom_type, 2.5) / 4.0
    
    features = (atom_type_enc + degree_enc + h_count_enc + hybrid_enc + [is_aromatic] + [is_in_ring] + chirality + [atomic_mass] + [electronegativity])
    return np.array(features, dtype=np.float32)

def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = [int(bond.GetBondType() == bt) for bt in bond_types]
    
    is_conjugated = int(bond.GetIsConjugated())
    is_in_ring = int(bond.IsInRing())
    
    stereo = [
        int(bond.GetStereo() == Chem.rdchem.BondStereo.STEREONONE),
        int(bond.GetStereo() == Chem.rdchem.BondStereo.STEREOZ),
        int(bond.GetStereo() == Chem.rdchem.BondStereo.STEREOE),
        int(bond.GetStereo() == Chem.rdchem.BondStereo.STEREOANY)
    ]
    
    return np.array(bond_type_enc + [is_conjugated, is_in_ring] + stereo, dtype=np.float32)

def smiles_to_graph_enhanced(smiles: str) -> Optional[Data]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
        try: rdPartialCharges.ComputeGasteigerCharges(mol)
        except: pass
        
        atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(np.array(atom_features), dtype=torch.float)
        
        edge_indices, edge_features = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feat = get_bond_features(bond)
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            edge_features.append(bond_feat)
            edge_features.append(bond_feat)
        
        if len(edge_indices) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None

# ------------------- Scaffold Split Functions -------------------

def load_scaffold_split_indices(json_path):
    """Load pre-computed scaffold split indices from JSON file."""
    with open(json_path, 'r') as f:
        split_indices = json.load(f)
    
    print(f"Loaded scaffold split indices from {json_path}")
    print(f"Available keys in JSON: {list(split_indices.keys())}")
    
    train_indices = split_indices.get('train_indices', [])
    val_indices = split_indices.get('val_indices', [])
    
    print(f"Train indices: {len(train_indices)} samples")
    print(f"Val indices: {len(val_indices)} samples")
    
    return {'train': train_indices, 'val': val_indices}

def apply_scaffold_split(df, split_indices, target_col='pIC50'):
    """Apply pre-computed scaffold split indices to the dataframe."""
    train_indices = split_indices.get('train', [])
    val_indices = split_indices.get('val', [])
    
    print(f"\nOriginal dataset size: {len(df)}")
    print(f"Scaffold split indices:")
    print(f"  Training indices: {len(train_indices)}")
    print(f"  Validation indices: {len(val_indices)}")
    
    # Apply split
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    
    print(f"\nBefore filtering NaN:")
    print(f"  Training set: {len(train_df)} samples")
    print(f"  Validation set: {len(val_df)} samples")
    
    # Filter NaN values
    essential_cols = ['canonical_smiles', 'protein_sequence', target_col]
    
    train_mask = train_df[essential_cols].notna().all(axis=1)
    train_df_clean = train_df[train_mask].reset_index(drop=True)
    train_removed = len(train_df) - len(train_df_clean)
    
    val_mask = val_df[essential_cols].notna().all(axis=1)
    val_df_clean = val_df[val_mask].reset_index(drop=True)
    val_removed = len(val_df) - len(val_df_clean)
    
    print(f"\nAfter filtering NaN:")
    print(f"  Training set: {len(train_df_clean)} samples (removed {train_removed})")
    print(f"  Validation set: {len(val_df_clean)} samples (removed {val_removed})")
    
    if len(train_df_clean) > 0 and len(val_df_clean) > 0:
        print(f"\nTarget ({target_col}) distribution:")
        print(f"  Train range: [{train_df_clean[target_col].min():.2f}, {train_df_clean[target_col].max():.2f}]")
        print(f"  Train mean: {train_df_clean[target_col].mean():.2f} ± {train_df_clean[target_col].std():.2f}")
        print(f"  Val range: [{val_df_clean[target_col].min():.2f}, {val_df_clean[target_col].max():.2f}]")
        print(f"  Val mean: {val_df_clean[target_col].mean():.2f} ± {val_df_clean[target_col].std():.2f}")
    
    return train_df_clean, val_df_clean

# ------------------- Dataset -------------------

class EnhancedProteinLigandDataset(Dataset):
    def __init__(self, df, vocab, tokenizer, has_target=True, target_col='pIC50'):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.has_target = has_target
        self.target_col = target_col
        self.processed_data = []

        desc = "Preprocessing Test Set" if not has_target else "Preprocessing Training Data"
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc=desc):
            # Check for valid data
            if pd.isna(row['canonical_smiles']) or pd.isna(row['protein_sequence']):
                continue
            
            if has_target and pd.isna(row[target_col]):
                continue
                
            ligand_graph = smiles_to_graph_enhanced(row['canonical_smiles'])
            if ligand_graph is not None:
                if has_target:
                    pIC50_val = row[target_col]
                    ligand_graph.y = torch.tensor([pIC50_val], dtype=torch.float)
                    self.processed_data.append({
                        'protein_sequence': row['protein_sequence'],
                        'ligand_graph': ligand_graph,
                        'pIC50': pIC50_val,
                        'original_index': idx
                    })
                else:
                    self.processed_data.append({
                        'protein_sequence': row['protein_sequence'],
                        'ligand_graph': ligand_graph,
                        'original_index': idx
                    })
        
        print(f"Final dataset size: {len(self.processed_data)}")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        entry = self.processed_data[idx]
        protein_tokens = self.tokenizer.tokenize(entry['protein_sequence'])
        protein_seq_tensor = torch.tensor(self.vocab(protein_tokens), dtype=torch.long)
        
        if self.has_target:
            return protein_seq_tensor, entry['ligand_graph']
        else:
            return protein_seq_tensor, entry['ligand_graph'], entry['original_index']

def collate_fn(batch):
    if len(batch[0]) == 2:  # Training/validation with targets
        proteins = nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
        batch_ligands = Batch.from_data_list([item[1] for item in batch])
        targets = torch.cat([item[1].y for item in batch]).view(-1, 1)
        return proteins, batch_ligands, targets
    else:  # Test without targets
        proteins, ligands, indices = zip(*batch)
        proteins = nn.utils.rnn.pad_sequence(proteins, batch_first=True, padding_value=0)
        ligands = [lig for lig in ligands if lig is not None]
        batch_ligands = Batch.from_data_list(ligands) if ligands else None
        return proteins, batch_ligands, indices

# ------------------- Model -------------------

class EnhancedProteinLigandModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2, dropout=dropout, batch_first=True),
            num_layers=2
        )
        self.ligand_gnn = AttentiveFP(in_channels=35, hidden_channels=hidden_dim, out_channels=output_dim, edge_dim=10, num_layers=4, num_timesteps=2, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_dim + output_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, protein_seq, ligand_data):
        protein_repr = self.transformer_encoder(self.embedding(protein_seq)).mean(dim=1)
        ligand_repr = self.ligand_gnn(ligand_data.x, ligand_data.edge_index, ligand_data.edge_attr, batch=ligand_data.batch)
        return self.fc(torch.cat([protein_repr, ligand_repr], dim=1)).squeeze(1)

# ------------------- Training / Helpers -------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, num_samples = 0, 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for protein_seq, ligand_data, targets in pbar:
        protein_seq, ligand_data, targets = protein_seq.to(device), ligand_data.to(device), targets.to(device)
        optimizer.zero_grad()
        if Config.USE_MIXED_PRECISION and scaler:
            with torch.cuda.amp.autocast():
                loss = criterion(model(protein_seq, ligand_data), targets.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(protein_seq, ligand_data), targets.view(-1))
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * protein_seq.size(0)
        num_samples += protein_seq.size(0)
        pbar.set_postfix(mae=total_loss/num_samples if num_samples > 0 else 0)
    return total_loss / num_samples

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, num_samples, all_preds, all_targets = 0, 0, [], []
    with torch.no_grad():
        for protein_seq, ligand_data, targets in tqdm(loader, desc="Validating", leave=False):
            protein_seq, ligand_data, targets = protein_seq.to(device), ligand_data.to(device), targets.to(device)
            outputs = model(protein_seq, ligand_data)
            loss = criterion(outputs, targets.view(-1))
            total_loss += loss.item() * protein_seq.size(0)
            num_samples += protein_seq.size(0)
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
    preds, targets = torch.cat(all_preds), torch.cat(all_targets)
    mae = torch.mean(torch.abs(preds - targets)).item()
    r2 = r2_score(targets.numpy(), preds.numpy())
    rmse = np.sqrt(mean_squared_error(targets.numpy(), preds.numpy()))
    return total_loss / num_samples, mae, r2, rmse, preds, targets

def predict_test_set(model, loader, device):
    """Make predictions on test set without targets."""
    model.eval()
    all_preds = []
    all_indices = []
    with torch.no_grad():
        for protein_seq, ligand_data, indices in tqdm(loader, desc="Predicting Test Set", leave=False):
            protein_seq = protein_seq.to(device)
            if ligand_data is not None:
                ligand_data = ligand_data.to(device)
            outputs = model(protein_seq, ligand_data)
            all_preds.append(outputs.cpu())
            all_indices.extend(indices)
    preds = torch.cat(all_preds)
    return preds.numpy(), all_indices

class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True

# --- 3. REPRODUCIBILITY SEEDING AND WORKERS ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError: pass
    print(f"Random seed set to {seed} for strict reproducibility")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ------------------- Protein-Specific Metrics -------------------

def analyze_protein_specific_metrics(df, predictions, targets, dataset_name="Validation"):
    """Calculate and display metrics for each protein separately."""
    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()} - PROTEIN-SPECIFIC METRICS")
    print(f"{'='*60}")
    
    # Add this verification check
    if len(predictions) != len(df):
        print(f"ERROR: Predictions length ({len(predictions)}) != DataFrame length ({len(df)})")
        print("Cannot proceed with protein-specific analysis.")
        return pd.DataFrame()
    
    # Get unique proteins
    unique_proteins = df['protein_sequence'].unique()
    print(f"Number of unique proteins: {len(unique_proteins)}")
    
    protein_metrics = []
    
    for i, protein_seq in enumerate(unique_proteins):
        # Find indices for this protein
        protein_mask = df['protein_sequence'] == protein_seq
        protein_indices = protein_mask[protein_mask].index.tolist()
        
        if len(protein_indices) == 0:
            continue
        
        # Get predictions and targets for this protein
        protein_preds = predictions[protein_indices]
        protein_targets = targets[protein_indices]
        
        # Calculate metrics
        mae = mean_absolute_error(protein_targets, protein_preds)
        rmse = np.sqrt(mean_squared_error(protein_targets, protein_preds))
        r2 = r2_score(protein_targets, protein_preds)
        
        # Identify protein (first 50 chars for display)
        protein_id = f"Protein {i+1}"
        if len(protein_seq) > 50:
            protein_display = protein_seq[:50] + "..."
        else:
            protein_display = protein_seq
        
        # Check if it's SARS-CoV-2 or MERS-CoV
        if protein_seq == "RKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQC":
            protein_name = "SARS-CoV-2 MPro (6Y2E)"
        elif protein_seq == "VKMSHPSGDVEACMVQVTCGSMTLNGLWLDNTVWCPRHVMCPADQLSDPNYDALLISMTNHSFSVQKHIGAPANLRVVGHAMQGTLLKLTVDVANPSTPAYTFTTVKPGAAFSVLACYNGRPTGTFTVVMRPNYTIKGSFLCGSCGSVGYTKEGSVINFCYMHQMELANGTHTGSAFDGTMYGAFMDKQVHQVQLTDKYCSVNVVAWLYAAILNGCAWFVKPNRTSVVSFNEWALANQFTEFVGTQSVDMLAVKTGVAIEQLLYAIQQLYTGFQGKQILGSTMLEDEFTPEDVNMQI":
            protein_name = "MERS-CoV MPro (5C3N)"
        else:
            protein_name = protein_id
        
        print(f"\n{protein_name}")
        print(f"  Sequence: {protein_display}")
        print(f"  Number of samples: {len(protein_indices)}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Target range: [{protein_targets.min():.2f}, {protein_targets.max():.2f}]")
        print(f"  Prediction range: [{protein_preds.min():.2f}, {protein_preds.max():.2f}]")
        
        protein_metrics.append({
            'protein_name': protein_name,
            'protein_sequence': protein_seq,
            'n_samples': len(protein_indices),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'target_min': protein_targets.min(),
            'target_max': protein_targets.max(),
            'target_mean': protein_targets.mean(),
            'target_std': protein_targets.std(),
            'pred_min': protein_preds.min(),
            'pred_max': protein_preds.max(),
            'pred_mean': protein_preds.mean(),
            'pred_std': protein_preds.std()
        })
    
    print(f"\n{'='*60}\n")
    
    return pd.DataFrame(protein_metrics)

def plot_protein_specific_performance(df, predictions, targets, save_dir=None, dataset_name="Validation"):
    """Create plots showing performance for each protein."""
    if save_dir is None:
        save_dir = Config.OUTPUT_DIR
    
    unique_proteins = df['protein_sequence'].unique()
    n_proteins = len(unique_proteins)
    
    # Create figure with subplots for each protein
    fig, axes = plt.subplots(1, n_proteins, figsize=(10*n_proteins, 8))
    if n_proteins == 1:
        axes = [axes]
    
    for i, protein_seq in enumerate(unique_proteins):
        # Find indices for this protein
        protein_mask = df['protein_sequence'] == protein_seq
        protein_indices = protein_mask[protein_mask].index.tolist()
        
        if len(protein_indices) == 0:
            continue
        
        # Get predictions and targets for this protein
        protein_preds = predictions[protein_indices]
        protein_targets = targets[protein_indices]
        
        # Identify protein
        if protein_seq == "RKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQC":
            protein_name = "SARS-CoV-2 MPro"
        elif protein_seq == "VKMSHPSGDVEACMVQVTCGSMTLNGLWLDNTVWCPRHVMCPADQLSDPNYDALLISMTNHSFSVQKHIGAPANLRVVGHAMQGTLLKLTVDVANPSTPAYTFTTVKPGAAFSVLACYNGRPTGTFTVVMRPNYTIKGSFLCGSCGSVGYTKEGSVINFCYMHQMELANGTHTGSAFDGTMYGAFMDKQVHQVQLTDKYCSVNVVAWLYAAILNGCAWFVKPNRTSVVSFNEWALANQFTEFVGTQSVDMLAVKTGVAIEQLLYAIQQLYTGFQGKQILGSTMLEDEFTPEDVNMQI":
            protein_name = "MERS-CoV MPro"
        else:
            protein_name = f"Protein {i+1}"
        
        # Scatter plot
        axes[i].scatter(protein_targets, protein_preds, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
        
        # Perfect prediction line
        min_val = min(protein_targets.min(), protein_preds.min())
        max_val = max(protein_targets.max(), protein_preds.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate metrics
        r2 = r2_score(protein_targets, protein_preds)
        mae = mean_absolute_error(protein_targets, protein_preds)
        rmse = np.sqrt(mean_squared_error(protein_targets, protein_preds))
        
        # Add metrics text box
        textstr = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\nn = {len(protein_indices)}'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        axes[i].text(0.05, 0.95, textstr, transform=axes[i].transAxes, fontsize=11,
                    verticalalignment='top', bbox=props)
        
        axes[i].set_xlabel("Actual pIC50", fontsize=12, fontweight='bold')
        axes[i].set_ylabel("Predicted pIC50", fontsize=12, fontweight='bold')
        axes[i].set_title(f"{protein_name}\n({dataset_name})", fontsize=13, fontweight='bold')
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{dataset_name.lower()}_protein_specific.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved protein-specific plot to {save_path}")

def save_protein_metrics_to_file(protein_metrics_df, save_dir=None, dataset_name="Validation"):
    """Save protein-specific metrics to a CSV file."""
    if save_dir is None:
        save_dir = Config.OUTPUT_DIR
    
    output_path = os.path.join(save_dir, f"{dataset_name.lower()}_protein_metrics.csv")
    protein_metrics_df.to_csv(output_path, index=False)
    print(f"Saved protein-specific metrics to {output_path}")

def save_comprehensive_metrics_report(overall_metrics, protein_metrics_df, save_dir=None, dataset_name="Validation"):
    """Save a comprehensive text report with all metrics."""
    if save_dir is None:
        save_dir = Config.OUTPUT_DIR
    
    output_path = os.path.join(save_dir, f"{dataset_name.lower()}_comprehensive_metrics.txt")
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"{dataset_name.upper()} SET - COMPREHENSIVE METRICS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall/Aggregated Metrics
        f.write("OVERALL AGGREGATED METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Number of Samples: {overall_metrics['n_samples']}\n")
        f.write(f"Mean Absolute Error (MAE): {overall_metrics['mae']:.6f}\n")
        f.write(f"Root Mean Square Error (RMSE): {overall_metrics['rmse']:.6f}\n")
        f.write(f"R² Score: {overall_metrics['r2']:.6f}\n")
        f.write(f"Mean Squared Error (MSE): {overall_metrics['mse']:.6f}\n\n")
        
        f.write(f"Target Value Statistics:\n")
        f.write(f"  Minimum: {overall_metrics['target_min']:.4f}\n")
        f.write(f"  Maximum: {overall_metrics['target_max']:.4f}\n")
        f.write(f"  Mean: {overall_metrics['target_mean']:.4f}\n")
        f.write(f"  Std Dev: {overall_metrics['target_std']:.4f}\n\n")
        
        f.write(f"Prediction Value Statistics:\n")
        f.write(f"  Minimum: {overall_metrics['pred_min']:.4f}\n")
        f.write(f"  Maximum: {overall_metrics['pred_max']:.4f}\n")
        f.write(f"  Mean: {overall_metrics['pred_mean']:.4f}\n")
        f.write(f"  Std Dev: {overall_metrics['pred_std']:.4f}\n\n")
        
        f.write(f"Error Distribution (Absolute Error):\n")
        f.write(f"  25th Percentile: {overall_metrics['error_p25']:.6f}\n")
        f.write(f"  50th Percentile (Median): {overall_metrics['error_p50']:.6f}\n")
        f.write(f"  75th Percentile: {overall_metrics['error_p75']:.6f}\n")
        f.write(f"  90th Percentile: {overall_metrics['error_p90']:.6f}\n")
        f.write(f"  95th Percentile: {overall_metrics['error_p95']:.6f}\n")
        f.write(f"  99th Percentile: {overall_metrics['error_p99']:.6f}\n\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Protein-Specific Metrics
        f.write("PROTEIN-SPECIFIC METRICS\n")
        f.write("-"*80 + "\n\n")
        
        for idx, row in protein_metrics_df.iterrows():
            f.write(f"Protein: {row['protein_name']}\n")
            f.write("-"*80 + "\n")
            f.write(f"Number of Samples: {row['n_samples']}\n")
            f.write(f"Mean Absolute Error (MAE): {row['mae']:.6f}\n")
            f.write(f"Root Mean Square Error (RMSE): {row['rmse']:.6f}\n")
            f.write(f"R² Score: {row['r2']:.6f}\n\n")
            
            f.write(f"Target Value Range: [{row['target_min']:.4f}, {row['target_max']:.4f}]\n")
            f.write(f"Target Mean: {row['target_mean']:.4f} ± {row['target_std']:.4f}\n")
            f.write(f"Prediction Range: [{row['pred_min']:.4f}, {row['pred_max']:.4f}]\n")
            f.write(f"Prediction Mean: {row['pred_mean']:.4f} ± {row['pred_std']:.4f}\n\n")
            
            f.write(f"Protein Sequence:\n{row['protein_sequence']}\n")
            f.write("\n" + "-"*80 + "\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"Saved comprehensive metrics report to {output_path}")

# ------------------- Protein-Specific Metrics -------------------

def plot_learning_curves(metrics_file, save_dir=None):
    if save_dir is None:
        save_dir = Config.OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(metrics_file)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # MAE curves
    axes[0, 0].plot(df["epoch"], df["train_mae"], 'b-', linewidth=2, label="Train MAE", marker="o", markersize=4)
    axes[0, 0].plot(df["epoch"], df["val_mae"], 'r-', linewidth=2, label="Validation MAE", marker="s", markersize=4)
    axes[0, 0].set_xlabel("Epoch", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("MAE (pIC50)", fontsize=12, fontweight='bold')
    axes[0, 0].set_title("Training and Validation MAE", fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # R2 curve
    axes[0, 1].plot(df["epoch"], df["val_r2"], 'g-', linewidth=2, label="Validation R²", marker="^", markersize=4)
    axes[0, 1].set_xlabel("Epoch", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel("R² Score", fontsize=12, fontweight='bold')
    axes[0, 1].set_title("Validation R² Score", fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
    
    # RMSE curve
    axes[1, 0].plot(df["epoch"], df["val_rmse"], 'm-', linewidth=2, label="Validation RMSE", marker="D", markersize=4)
    axes[1, 0].set_xlabel("Epoch", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("RMSE", fontsize=12, fontweight='bold')
    axes[1, 0].set_title("Validation RMSE", fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # MAE comparison (log scale)
    axes[1, 1].semilogy(df["epoch"], df["train_mae"], 'b-', linewidth=2, label="Train MAE", marker="o", markersize=4)
    axes[1, 1].semilogy(df["epoch"], df["val_mae"], 'r-', linewidth=2, label="Validation MAE", marker="s", markersize=4)
    axes[1, 1].set_xlabel("Epoch", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("MAE (log scale)", fontsize=12, fontweight='bold')
    axes[1, 1].set_title("MAE Comparison (Log Scale)", fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_curves.png"), dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved learning curves to {save_dir}/learning_curves.png")

def plot_actual_vs_predicted(actual, predicted, path=None, title="Actual vs Predicted pIC50", dataset_name="Validation"):
    if path is None:
        path = os.path.join(Config.OUTPUT_DIR, f"{dataset_name.lower()}_predictions.png")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(actual, predicted, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
    
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    textstr = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\nn = {len(actual)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel("Actual pIC50", fontsize=14, fontweight='bold')
    ax.set_ylabel("Predicted pIC50", fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {path}")

def plot_residuals(actual, predicted, path=None, dataset_name="Validation"):
    if path is None:
        path = os.path.join(Config.OUTPUT_DIR, f"{dataset_name.lower()}_residuals.png")
    
    residuals = predicted - actual
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Residuals vs Predicted
    axes[0].scatter(predicted, residuals, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted pIC50', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Residuals (Predicted - Actual)', fontsize=12, fontweight='bold')
    axes[0].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Residuals Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    textstr = f'Mean: {residuals.mean():.4f}\nStd: {residuals.std():.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    axes[1].text(0.70, 0.95, textstr, transform=axes[1].transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved residual plot to {path}")

def plot_error_distribution(actual, predicted, path=None, dataset_name="Validation"):
    if path is None:
        path = os.path.join(Config.OUTPUT_DIR, f"{dataset_name.lower()}_error_distribution.png")
    
    errors = np.abs(predicted - actual)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Error vs Actual
    axes[0].scatter(actual, errors, alpha=0.5, s=30, edgecolors='k', linewidths=0.5, c=errors, cmap='viridis')
    axes[0].set_xlabel('Actual pIC50', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    axes[0].set_title('Absolute Error vs Actual pIC50', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Error distribution with percentiles
    axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    percentiles = [50, 75, 90, 95]
    colors = ['g', 'y', 'orange', 'r']
    for p, c in zip(percentiles, colors):
        val = np.percentile(errors, p)
        axes[1].axvline(x=val, color=c, linestyle='--', linewidth=2, label=f'{p}th percentile: {val:.3f}')
    axes[1].set_xlabel('Absolute Error', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Absolute Error Distribution', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved error distribution plot to {path}")

def plot_comparison(val_actual, val_pred, test_actual, test_pred, path=None):
    """Create comparison plot between validation and test sets."""
    if path is None:
        path = os.path.join(Config.OUTPUT_DIR, "comparison_plot.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Validation
    axes[0].scatter(val_actual, val_pred, alpha=0.5, s=30, edgecolors='k', linewidths=0.5, label='Validation')
    min_val = min(val_actual.min(), val_pred.min())
    max_val = max(val_actual.max(), val_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    val_r2 = r2_score(val_actual, val_pred)
    val_mae = mean_absolute_error(val_actual, val_pred)
    val_rmse = np.sqrt(mean_squared_error(val_actual, val_pred))
    
    textstr = f'R² = {val_r2:.4f}\nMAE = {val_mae:.4f}\nRMSE = {val_rmse:.4f}\nn = {len(val_actual)}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
    
    axes[0].set_xlabel('Actual pIC50', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Predicted pIC50', fontsize=14, fontweight='bold')
    axes[0].set_title('Validation Set', fontsize=16, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Test
    axes[1].scatter(test_actual, test_pred, alpha=0.5, s=30, edgecolors='k', 
                   linewidths=0.5, label='Test', color='orange')
    min_test = min(test_actual.min(), test_pred.min())
    max_test = max(test_actual.max(), test_pred.max())
    axes[1].plot([min_test, max_test], [min_test, max_test], 'r--', linewidth=2)
    
    test_r2 = r2_score(test_actual, test_pred)
    test_mae = mean_absolute_error(test_actual, test_pred)
    test_rmse = np.sqrt(mean_squared_error(test_actual, test_pred))
    
    textstr = f'R² = {test_r2:.4f}\nMAE = {test_mae:.4f}\nRMSE = {test_rmse:.4f}\nn = {len(test_actual)}'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    axes[1].text(0.05, 0.95, textstr, transform=axes[1].transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
    
    axes[1].set_xlabel('Actual pIC50', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Predicted pIC50', fontsize=14, fontweight='bold')
    axes[1].set_title('Test Set (Blind)', fontsize=16, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {path}")

# ------------------- Main Function -------------------

def main():
    set_seed(Config.SEED)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # Load training data
    print("="*60)
    print("LOADING TRAINING DATA")
    print("="*60)
    df = pd.read_csv(Config.TRAIN_DATA_FILE)
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    print(f"Available columns: {list(df.columns)}")
    
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['smiles', 'canonical_smiles']:
            column_mapping[col] = 'canonical_smiles'
        elif col_lower in ['protein_seq', 'protein_sequence']:
            column_mapping[col] = 'protein_sequence'
        elif col_lower in ['pic50', 'pic_50', 'p_ic50']:
            column_mapping[col] = 'pIC50'
    
    df = df.rename(columns=column_mapping)
    print(f"Standardized columns: {list(df.columns)}")
    print(f"Loaded dataset with {len(df)} samples")
    
    # Build vocabulary
    print("\n" + "="*60)
    print("BUILDING VOCABULARY")
    print("="*60)
    tokenizer = ProteinTokenizer()
    valid_sequences = df['protein_sequence'].dropna()
    protein_tokens = [tokenizer.tokenize(seq) for seq in valid_sequences]
    vocab = build_vocab_from_iterator(protein_tokens, specials=["<pad>"])
    vocab.set_default_index(vocab["<pad>"])
    print(f"Vocabulary size: {len(vocab)}")
    
    # Save vocab and tokenizer
    with open(Config.VOCAB_FILE, "wb") as f: pickle.dump(vocab, f)
    with open(Config.TOKENIZER_FILE, "wb") as f: pickle.dump(tokenizer, f)
    print("Saved Vocab and Tokenizer.")

    # Load scaffold split indices
    print("\n" + "="*60)
    print("APPLYING SCAFFOLD SPLIT")
    print("="*60)
    split_indices = load_scaffold_split_indices(Config.SCAFFOLD_SPLIT_FILE)
    train_df, val_df = apply_scaffold_split(df, split_indices, target_col='pIC50')

    # Create datasets
    print("\n" + "="*60)
    print("CREATING DATASETS")
    print("="*60)
    train_dataset = EnhancedProteinLigandDataset(train_df, vocab, tokenizer, has_target=True, target_col='pIC50')
    val_dataset = EnhancedProteinLigandDataset(val_df, vocab, tokenizer, has_target=True, target_col='pIC50')
    
    print(f"Final training samples: {len(train_dataset)}")
    print(f"Final validation samples: {len(val_dataset)}")

    # Create dataloaders
    g = torch.Generator()
    g.manual_seed(Config.SEED)
    
    loader_args = dict(
        batch_size=Config.BATCH_SIZE, 
        collate_fn=collate_fn, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type=='cuda' else False,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    # Initialize model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    model = EnhancedProteinLigandModel(len(vocab), Config.HIDDEN_DIM, Config.OUTPUT_DIM, Config.DROPOUT).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.L1Loss()  # MAE loss
    early_stopper = EarlyStopping(patience=Config.PATIENCE)
    scaler = torch.cuda.amp.GradScaler() if Config.USE_MIXED_PRECISION else None
    
    best_val_mae = float('inf')
    metrics_history = []
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        train_mae = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE, scaler)
        val_mae, val_mae_metric, val_r2, val_rmse, _, _ = evaluate(model, val_loader, criterion, Config.DEVICE)
        
        print(f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | Val R2: {val_r2:.4f} | Val RMSE: {val_rmse:.4f}")
        metrics_history.append({
            "epoch": epoch+1, 
            "train_mae": train_mae, 
            "val_mae": val_mae, 
            "val_r2": val_r2, 
            "val_rmse": val_rmse
        })
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mae': best_val_mae,
                'config': {
                    'vocab_size': len(vocab),
                    'hidden_dim': Config.HIDDEN_DIM,
                    'output_dim': Config.OUTPUT_DIM,
                    'dropout': Config.DROPOUT,
                    'atom_features': 35,
                    'bond_features': 10
                }
            }
            torch.save(checkpoint, Config.MODEL_SAVE)
            print(f"✓ Saved best model with Val MAE: {best_val_mae:.4f}")
            
        early_stopper(val_mae)
        if early_stopper.early_stop:
            print(f"\nEarly stopping triggered after {Config.PATIENCE} epochs of no improvement.")
            break

    # Save metrics
    pd.DataFrame(metrics_history).to_csv(Config.METRICS_FILE, index=False)
    print(f"\nSaved training metrics to {Config.METRICS_FILE}")
    
    # Generate learning curves
    print("\n" + "="*60)
    print("GENERATING LEARNING CURVES")
    print("="*60)
    plot_learning_curves(Config.METRICS_FILE)
    
    # Load best model for final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("="*60)
    checkpoint = torch.load(Config.MODEL_SAVE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_val_mae, _, final_val_r2, final_val_rmse, final_preds, final_targets = evaluate(
        model, val_loader, criterion, Config.DEVICE
    )
    
    print(f"\nBest Enhanced Features Model Performance (Validation):")
    print(f"  MAE: {final_val_mae:.4f}")
    print(f"  R²: {final_val_r2:.4f}")
    print(f"  RMSE: {final_val_rmse:.4f}")
    
    # Generate validation plots
    print("\nGenerating validation plots...")
    val_targets_np = final_targets.numpy().flatten()
    val_preds_np = final_preds.numpy().flatten()
    plot_actual_vs_predicted(val_targets_np, val_preds_np, 
                           title="Validation: Actual vs Predicted pIC50",
                           dataset_name="Validation")
    plot_residuals(val_targets_np, val_preds_np, dataset_name="Validation")
    plot_error_distribution(val_targets_np, val_preds_np, dataset_name="Validation")
    
    # Protein-specific analysis for validation set
    print("\nAnalyzing protein-specific performance on validation set...")
    val_protein_metrics = analyze_protein_specific_metrics(
        val_dataset.df, val_preds_np, val_targets_np, dataset_name="Validation"
    )
    save_protein_metrics_to_file(val_protein_metrics, dataset_name="Validation")
    plot_protein_specific_performance(
        val_dataset.df, val_preds_np, val_targets_np, dataset_name="Validation"
    )
    
    # Save comprehensive metrics report for validation
    errors = np.abs(val_preds_np - val_targets_np)
    val_overall_metrics = {
        'n_samples': len(val_targets_np),
        'mae': final_val_mae,
        'rmse': final_val_rmse,
        'r2': final_val_r2,
        'mse': mean_squared_error(val_targets_np, val_preds_np),
        'target_min': val_targets_np.min(),
        'target_max': val_targets_np.max(),
        'target_mean': val_targets_np.mean(),
        'target_std': val_targets_np.std(),
        'pred_min': val_preds_np.min(),
        'pred_max': val_preds_np.max(),
        'pred_mean': val_preds_np.mean(),
        'pred_std': val_preds_np.std(),
        'error_p25': np.percentile(errors, 25),
        'error_p50': np.percentile(errors, 50),
        'error_p75': np.percentile(errors, 75),
        'error_p90': np.percentile(errors, 90),
        'error_p95': np.percentile(errors, 95),
        'error_p99': np.percentile(errors, 99)
    }
    save_comprehensive_metrics_report(val_overall_metrics, val_protein_metrics, dataset_name="Validation")
    
    # Predict on external test set
    print("\n" + "="*60)
    print("PREDICTING ON EXTERNAL TEST SET")
    print("="*60)
    try:
        test_df = pd.read_csv(Config.TEST_DATA_FILE)
        test_df.columns = test_df.columns.str.strip()
        print(f"Test set columns: {list(test_df.columns)}")
        
        # Standardize test column names
        test_column_mapping = {}
        for col in test_df.columns:
            col_lower = col.lower().strip()
            if col_lower in ['smiles', 'canonical_smiles']:
                test_column_mapping[col] = 'canonical_smiles'
            elif col_lower in ['protein_seq', 'protein_sequence']:
                test_column_mapping[col] = 'protein_sequence'
            elif col_lower in ['pic50', 'pic_50', 'p_ic50']:
                test_column_mapping[col] = 'pIC50'
        
        test_df = test_df.rename(columns=test_column_mapping)
        print(f"Standardized test columns: {list(test_df.columns)}")
        print(f"Original test set size: {len(test_df)}")
        
        # Check if test set has pIC50 column
        has_target = 'pIC50' in test_df.columns
        
        if has_target:
            print(f"\nTest set pIC50 statistics:")
            print(f"  Mean: {test_df['pIC50'].mean():.2f}")
            print(f"  Std: {test_df['pIC50'].std():.2f}")
            print(f"  Min: {test_df['pIC50'].min():.2f}")
            print(f"  Max: {test_df['pIC50'].max():.2f}")
            print(f"  NaN count: {test_df['pIC50'].isna().sum()}")
            
            test_dataset = EnhancedProteinLigandDataset(test_df, vocab, tokenizer, has_target=True, target_col='pIC50')
            print(f"Test samples after filtering: {len(test_dataset)}")
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                collate_fn=collate_fn,
                pin_memory=True
            )
            
            # Evaluate on test set
            test_mae, _, test_r2, test_rmse, test_preds, test_targets = evaluate(
                model, test_loader, criterion, Config.DEVICE
            )
            
            print(f"\n{'='*60}")
            print(f"TEST SET RESULTS (WITH TARGETS)")
            print(f"{'='*60}")
            print(f"  MAE: {test_mae:.4f}")
            print(f"  RMSE: {test_rmse:.4f}")
            print(f"  R² Score: {test_r2:.4f}")
            print(f"  Number of samples: {len(test_targets)}")
            print(f"{'='*60}\n")
            
            # Save test predictions with actual values
            test_targets_np = test_targets.numpy().flatten()
            test_preds_np = test_preds.numpy().flatten()
            
            # Create results dataframe with matching length
            test_results = pd.DataFrame({
                'actual_pIC50': test_targets_np,
                'predicted_pIC50': test_preds_np,
                'absolute_error': np.abs(test_preds_np - test_targets_np),
                'residual': test_preds_np - test_targets_np
            })
            
            # Add original data columns from the FILTERED dataset (not the original df)
            # Use only the columns that exist in the processed dataset
            for col in test_dataset.df.columns:
                if col not in test_results.columns:
                    # Match by index - both should have same length now
                    test_results[col] = test_dataset.df[col].reset_index(drop=True)
            
            # Generate all test set plots
            print("Generating test set plots...")
            plot_actual_vs_predicted(test_targets_np, test_preds_np, 
                                   title="Test Set: Actual vs Predicted pIC50",
                                   dataset_name="Test")
            plot_residuals(test_targets_np, test_preds_np, dataset_name="Test")
            plot_error_distribution(test_targets_np, test_preds_np, dataset_name="Test")
            
            # Create comparison plot
            print("Generating validation vs test comparison plot...")
            plot_comparison(val_targets_np, val_preds_np, 
                          test_targets_np, test_preds_np)
            
            # Protein-specific analysis for test set
            print("\nAnalyzing protein-specific performance on test set...")
            # Get the original indices of all processed samples
            test_processed_indices = [item['original_index'] for item in test_dataset.processed_data]
            # Create a dataframe with only the rows that were actually processed
            test_analysis_df = test_dataset.df.loc[test_processed_indices].reset_index(drop=True)

            test_protein_metrics = analyze_protein_specific_metrics(
                test_analysis_df, test_preds_np, test_targets_np, dataset_name="Test"
            )
            save_protein_metrics_to_file(test_protein_metrics, dataset_name="Test")
            plot_protein_specific_performance(
                test_analysis_df, test_preds_np, test_targets_np, dataset_name="Test"
            )
            
            # Save comprehensive metrics report for test set
            test_errors = np.abs(test_preds_np - test_targets_np)
            test_overall_metrics = {
                'n_samples': len(test_targets_np),
                'mae': test_mae,
                'rmse': test_rmse,
                'r2': test_r2,
                'mse': mean_squared_error(test_targets_np, test_preds_np),
                'target_min': test_targets_np.min(),
                'target_max': test_targets_np.max(),
                'target_mean': test_targets_np.mean(),
                'target_std': test_targets_np.std(),
                'pred_min': test_preds_np.min(),
                'pred_max': test_preds_np.max(),
                'pred_mean': test_preds_np.mean(),
                'pred_std': test_preds_np.std(),
                'error_p25': np.percentile(test_errors, 25),
                'error_p50': np.percentile(test_errors, 50),
                'error_p75': np.percentile(test_errors, 75),
                'error_p90': np.percentile(test_errors, 90),
                'error_p95': np.percentile(test_errors, 95),
                'error_p99': np.percentile(test_errors, 99)
            }
            save_comprehensive_metrics_report(test_overall_metrics, test_protein_metrics, dataset_name="Test")
            
            # Save detailed metrics
            metrics_path = os.path.join(Config.OUTPUT_DIR, "test_metrics.txt")
            with open(metrics_path, "w") as f:
                f.write("="*60 + "\n")
                f.write("TEST SET EVALUATION METRICS\n")
                f.write("="*60 + "\n\n")
                f.write(f"Number of samples: {len(test_targets)}\n\n")
                f.write(f"MAE: {test_mae:.6f}\n")
                f.write(f"RMSE: {test_rmse:.6f}\n")
                f.write(f"R² Score: {test_r2:.6f}\n\n")
                f.write("Error Percentiles:\n")
                errors = np.abs(test_preds_np - test_targets_np)
                for p in [25, 50, 75, 90, 95, 99]:
                    f.write(f"  {p}th percentile: {np.percentile(errors, p):.6f}\n")
                f.write("\n" + "="*60 + "\n")
            print(f"Detailed test metrics saved to {metrics_path}")
            
        else:
            # If no targets, just make predictions
            print("\nNo pIC50 targets found in test set. Making predictions only...")
            test_dataset = EnhancedProteinLigandDataset(test_df, vocab, tokenizer, has_target=False)
            print(f"Test samples after filtering: {len(test_dataset)}")
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                collate_fn=collate_fn,
                pin_memory=True
            )
            
            test_preds, test_indices = predict_test_set(model, test_loader, Config.DEVICE)
            print(f"\nGenerated predictions for {len(test_preds)} test samples")
            
            test_results = pd.DataFrame({
                'predicted_pIC50': test_preds,
            })
            
            # Add original data columns from the FILTERED dataset
            for col in test_dataset.df.columns:
                if col not in test_results.columns:
                    test_results[col] = test_dataset.df[col].reset_index(drop=True)
        
        # Save predictions
        output_path = os.path.join(Config.OUTPUT_DIR, "test_predictions.csv")
        test_results.to_csv(output_path, index=False)
        print(f"Test predictions saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing test set: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing without test set predictions...")

    # Save training summary
    summary_path = Config.MODEL_SAVE.replace('.pt', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TRAINING COMPLETE - ENHANCED FEATURES WITH SCAFFOLD SPLIT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best Validation MAE: {best_val_mae:.4f}\n")
        f.write(f"Final Validation R²: {final_val_r2:.4f}\n")
        f.write(f"Final Validation RMSE: {final_val_rmse:.4f}\n\n")
        f.write(f"Model saved to: {Config.MODEL_SAVE}\n")
        f.write(f"Vocab saved to: {Config.VOCAB_FILE}\n")
        f.write(f"Tokenizer saved to: {Config.TOKENIZER_FILE}\n\n")
        f.write("Output files generated:\n")
        f.write("  - learning_curves.png\n")
        f.write("  - validation_predictions.png\n")
        f.write("  - validation_residuals.png\n")
        f.write("  - validation_error_distribution.png\n")
        f.write("  - validation_protein_specific.png\n")
        f.write("  - validation_protein_metrics.csv\n")
        f.write("  - validation_comprehensive_metrics.txt\n")
        if os.path.exists(os.path.join(Config.OUTPUT_DIR, "test_predictions.png")):
            f.write("  - test_predictions.png\n")
            f.write("  - test_residuals.png\n")
            f.write("  - test_error_distribution.png\n")
            f.write("  - test_protein_specific.png\n")
            f.write("  - test_protein_metrics.csv\n")
            f.write("  - test_comprehensive_metrics.txt\n")
            f.write("  - comparison_plot.png\n")
            f.write("  - test_predictions.csv\n")
            f.write("  - test_metrics_legacy.txt\n")
        f.write("\n" + "="*60 + "\n")
    
    print("\n" + "="*60)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to '{Config.OUTPUT_DIR}/' directory")
    print(f"Model and artifacts saved to '{Config.MODEL_DIR}/' directory")
    print(f"\nSummary saved to {summary_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
