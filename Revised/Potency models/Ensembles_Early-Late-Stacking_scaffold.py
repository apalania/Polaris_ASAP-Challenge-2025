# -*- coding: utf-8 -*-
"""
Fusion Strategies: GPFT + XGBoost - 3 Core Strategies
======================================================
Implements Early Fusion, Late Fusion, and Stacking.
Calculates per-protein metrics for SARS-CoV-2 and MERS-CoV.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import AttentiveFP
from rdkit import Chem
from rdkit.Chem import rdPartialCharges, Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from tqdm import tqdm
from typing import List, Tuple
from collections import OrderedDict
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
     # Pretrained Models
    GNN_MODEL = "GNN_best_model.pt"#BEST GNN / GPFT MODEL
    XGB_MODEL = "Random_XGB_MODEL/xgb_target_specific_random_25desc_dock.json"  # BEST XGB MODEL
    XGB_SCALER = "Random_XGB_MODEL/scaler_xgb_target_specific_random_25desc_dock.pkl"
 
   # Vocabularies
    VOCAB_FILE = "vocab.pkl" #GENERATED WITH THE BEST GNN MODEL
    TOKENIZER_FILE = "tokenizer.pkl" #GENERATED WITH THE BEST GNN MODEL
    
    # Data files
    TRAIN_DATA = "polaris_train.csv"
    TEST_DATA = "polaris_unblinded_test.csv"
    SPLIT_JSON = "scaffold_split_indices.json"
    
    # Docking scores
    DOCKING_TRAIN = "train_docking_scores.csv"
    DOCKING_TEST = "test_docking_scores.csv"
    
    # XGBoost descriptors
    DESCRIPTOR_FILE = "descriptors_55.txt" #DESCRIPTOR FILE CORRESPONDING TO THE BEST XGB MODEL
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 64
    SEED = 42
    
    # Fusion training params
    FUSION_EPOCHS = 50
    FUSION_LR = 1e-4
    FUSION_PATIENCE = 10
    
    # Model architecture
    HIDDEN_DIM = 256
    OUTPUT_DIM = 32
    DROPOUT = 0.2
    
    # Fusion settings
    FUSION_HIDDEN = 128
    
    OUTPUT_DIR = "fusion_results_3_strategies"
    
    # Target protein sequences
    SARS_COV2_SEQ = "RKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQC"
    MERS_COV_SEQ = "VKMSHPSGDVEACMVQVTCGSMTLNGLWLDNTVWCPRHVMCPADQLSDPNYDALLISMTNHSFSVQKHIGAPANLRVVGHAMQGTLLKLTVDVANPSTPAYTFTTVKPGAAFSVLACYNGRPTGTFTVVMRPNYTIKGSFLCGSCGSVGYTKEGSVINFCYMHQMELANGTHTGSAFDGTMYGAFMDKQVHQVQLTDKYCSVNVVAWLYAAILNGCAWFVKPNRTSVVSFNEWALANQFTEFVGTQSVDMLAVKTGVAIEQLLYAIQQLYTGFQGKQILGSTMLEDEFTPEDVNMQI"

print(f"Using device: {Config.DEVICE}")

# ==============================================================================
# UTILITIES
# ==============================================================================
def protein_tokenizer(seq): 
    return list(seq.upper())


class Vocab:
    """Vocabulary class for token-to-index mapping"""
    def __init__(self, counter=None, max_size=None, min_freq=1, 
                 specials=('<pad>', '<unk>'), special_first=True):
        from collections import Counter
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


class ProteinTokenizer:
    def __init__(self): 
        self.tokenizer = protein_tokenizer
    
    def tokenize(self, seq): 
        return self.tokenizer(seq)


def load_docking_scores(filepath):
    """Load docking scores from CSV file."""
    if not os.path.exists(filepath):
        print(f"  ⚠️  Docking file not found: {filepath}")
        return {}
    
    df = pd.read_csv(filepath)
    cols = df.columns
    
    smiles_col = next((c for c in cols if 'smile' in c.lower()), None)
    protein_col = next((c for c in cols if 'seq' in c.lower() or 'prot' in c.lower()), None)
    score_col = next((c for c in cols if 'score' in c.lower()), None)
    
    if not (smiles_col and protein_col and score_col):
        return {}
    
    docking_map = {}
    for _, row in df.iterrows():
        score = row[score_col]
        if np.isfinite(score):
            docking_map[(row[smiles_col], row[protein_col])] = score
    
    print(f"  ✅ Loaded {len(docking_map)} docking scores from {filepath}")
    return docking_map


def calculate_per_protein_metrics(y_true, y_pred, protein_seqs, target_seq, protein_name):
    """Calculate metrics for a specific protein target."""
    # Find indices where protein sequence matches target
    indices = [i for i, seq in enumerate(protein_seqs) if seq == target_seq]
    
    if len(indices) == 0:
        return None
    
    y_true_subset = y_true[indices]
    y_pred_subset = y_pred[indices]
    
    mae = mean_absolute_error(y_true_subset, y_pred_subset)
    rmse = np.sqrt(mean_squared_error(y_true_subset, y_pred_subset))
    r2 = r2_score(y_true_subset, y_pred_subset)
    
    try:
        corr = pearsonr(y_true_subset, y_pred_subset)[0]
        r2_corr = corr ** 2
    except:
        corr = 0.0
        r2_corr = 0.0
    
    return {
        'protein': protein_name,
        'n_samples': len(indices),
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': corr,
        'r2_corr': r2_corr
    }

def evaluate_and_store_metrics(strategy_name, y_true, y_pred, protein_seqs, dataset_name):
    """Calculate overall and per-protein metrics for a given dataset."""
    results = []
    
    # Overall metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    try:
        corr = pearsonr(y_true, y_pred)[0]
        r2_corr = corr ** 2
    except:
        corr = 0.0
        r2_corr = 0.0
    
    results.append({
        'strategy': strategy_name,
        'dataset': dataset_name,
        'protein': 'Overall',
        'n_samples': len(y_true),
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': corr,
        'r2_corr': r2_corr
    })
    
    # SARS-CoV-2 metrics
    sars_metrics = calculate_per_protein_metrics(
        y_true, y_pred, protein_seqs, Config.SARS_COV2_SEQ, "SARS-CoV-2 MPro (6Y2E)"
    )
    if sars_metrics:
        results.append({
            'strategy': strategy_name,
            'dataset': dataset_name,
            **sars_metrics
        })
    
    # MERS-CoV metrics
    mers_metrics = calculate_per_protein_metrics(
        y_true, y_pred, protein_seqs, Config.MERS_COV_SEQ, "MERS-CoV MPro (5C3N)"
    )
    if mers_metrics:
        results.append({
            'strategy': strategy_name,
            'dataset': dataset_name,
            **mers_metrics
        })
    
    return results

# ==============================================================================
# FEATURIZATION
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
    electronegativity = {'C': 2.55, 'N': 3.04, 'O': 3.44, 'S': 2.58, 'F': 3.98, 
                         'P': 2.19, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66}.get(atom_type, 2.5) / 4.0
    return np.array(atom_type_enc + degree_enc + h_count_enc + hybrid_enc + 
                    [int(atom.GetIsAromatic()), int(atom.IsInRing())] + 
                    chirality + [atom.GetMass() / 100.0, electronegativity], dtype=np.float32)


def get_bond_features(bond):
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
                  Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    stereo = [int(bond.GetStereo() == s) for s in [
        Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOANY
    ]]
    return np.array([int(bond.GetBondType() == bt) for bt in bond_types] + 
                    [int(bond.GetIsConjugated()), int(bond.IsInRing())] + stereo, dtype=np.float32)


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
            return Data(x=x, edge_index=torch.zeros((2,0), dtype=torch.long), 
                       edge_attr=torch.zeros((0,10), dtype=torch.float))
        return Data(x=x, edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(), 
                   edge_attr=torch.tensor(np.array(edge_features), dtype=torch.float))
    except: return None


def compute_molecular_descriptors(smiles, protein_seq, descriptor_names, docking_dict=None):
    """Compute molecular descriptors for XGBoost features."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        vals = []
        
        for name in descriptor_names:
            try:
                if hasattr(Descriptors, name):
                    v = getattr(Descriptors, name)(mol)
                elif hasattr(rdMolDescriptors, name):
                    v = getattr(rdMolDescriptors, name)(mol)
                else:
                    v = 0.0
            except:
                v = 0.0
            vals.append(float(v) if np.isfinite(v) else 0.0)
        
        if docking_dict:
            key = (smiles, protein_seq)
            vals.extend([
                docking_dict.get(key, 0.0),
                1.0 if key in docking_dict else 0.0
            ])
        
        return np.array(vals, dtype=np.float32)
    except:
        base_len = len(descriptor_names)
        extra = 2 if docking_dict else 0
        return np.zeros(base_len + extra, dtype=np.float32)


# ==============================================================================
# GPFT MODEL ARCHITECTURE
# ==============================================================================
class EnhancedProteinLigandModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, output_dim=32, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
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

    def forward(self, p, l, return_embedding=False):
        p_emb = self.transformer_encoder(self.embedding(p)).mean(dim=1)
        l_emb = self.ligand_gnn(l.x, l.edge_index, l.edge_attr, l.batch)
        combined = torch.cat([p_emb, l_emb], dim=1)
        
        if return_embedding:
            return combined
        
        return self.fc(combined).squeeze(1)


# ==============================================================================
# FUSION MODELS
# ==============================================================================

class EarlyFusionModel(nn.Module):
    """Concatenate GNN embeddings with XGBoost features."""
    
    def __init__(self, gnn_model: nn.Module, xgboost_feat_dim: int = 27, 
                 hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        self.gnn_model = gnn_model
        gnn_emb_dim = gnn_model.hidden_dim + gnn_model.output_dim
        
        combined_dim = gnn_emb_dim + xgboost_feat_dim
        
        self.fusion_network = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, p, l, xgboost_features):
        gnn_embedding = self.gnn_model(p, l, return_embedding=True)
        combined = torch.cat([gnn_embedding, xgboost_features], dim=1)
        out = self.fusion_network(combined)
        return out.squeeze(-1)


class LateFusionEnsemble:
    """Combine predictions from separate GNN and XGBoost models."""
    
    def __init__(self, gnn_model: nn.Module, xgb_model: xgb.XGBRegressor,
                 strategy: str = 'learned', device: str = 'cpu'):
        self.gnn_model = gnn_model
        self.xgb_model = xgb_model
        self.strategy = strategy
        self.device = device
        
        self.gnn_weight = 0.6
        self.xgb_weight = 0.4
        
        self.gnn_model.eval()
    
    def predict(self, protein_seqs, graph_data_list: List[Data], 
                xgboost_features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            batch_graphs = Batch.from_data_list(graph_data_list).to(self.device)
            protein_tensor = protein_seqs.to(self.device)
            gnn_preds = self.gnn_model(protein_tensor, batch_graphs).cpu().numpy()
        
        xgb_preds = self.xgb_model.predict(xgboost_features)
        
        return self.gnn_weight * gnn_preds + self.xgb_weight * xgb_preds
    
    def optimize_weights(self, protein_seqs, graph_data_list: List[Data],
                        xgboost_features: np.ndarray, true_labels: np.ndarray):
        """Learn optimal ensemble weights on validation data."""
        with torch.no_grad():
            batch_graphs = Batch.from_data_list(graph_data_list).to(self.device)
            protein_tensor = protein_seqs.to(self.device)
            gnn_preds = self.gnn_model(protein_tensor, batch_graphs).cpu().numpy()
        
        xgb_preds = self.xgb_model.predict(xgboost_features)
        
        best_mae = float('inf')
        best_gnn_weight = 0.5
        
        for gnn_w in np.linspace(0, 1, 21):
            xgb_w = 1.0 - gnn_w
            ensemble_preds = gnn_w * gnn_preds + xgb_w * xgb_preds
            mae = mean_absolute_error(true_labels, ensemble_preds)
            
            if mae < best_mae:
                best_mae = mae
                best_gnn_weight = gnn_w
        
        self.gnn_weight = best_gnn_weight
        self.xgb_weight = 1.0 - best_gnn_weight
        
        print(f"  Optimized weights: GNN={self.gnn_weight:.3f}, "
              f"XGB={self.xgb_weight:.3f}, MAE={best_mae:.4f}")


class StackingEnsemble:
    """Meta-learner on top of base model predictions."""
    
    def __init__(self, gnn_model: nn.Module, xgb_model: xgb.XGBRegressor,
                 meta_learner: str = 'ridge', device: str = 'cpu'):
        self.gnn_model = gnn_model
        self.xgb_model = xgb_model
        self.device = device
        self.gnn_model.eval()
        
        if meta_learner == 'ridge':
            self.meta_model = Ridge(alpha=1.0)
            self.is_neural = False
        else:
            raise ValueError(f"Unknown meta_learner: {meta_learner}")
        
        self.meta_learner_type = meta_learner
    
    def get_base_predictions(self, protein_seqs, graph_data_list: List[Data],
                            xgboost_features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            batch_graphs = Batch.from_data_list(graph_data_list).to(self.device)
            protein_tensor = protein_seqs.to(self.device)
            gnn_preds = self.gnn_model(protein_tensor, batch_graphs).cpu().numpy().reshape(-1, 1)
        
        xgb_preds = self.xgb_model.predict(xgboost_features).reshape(-1, 1)
        return np.concatenate([gnn_preds, xgb_preds], axis=1)
    
    def fit(self, protein_seqs, graph_data_list: List[Data],
            xgboost_features: np.ndarray, true_labels: np.ndarray):
        base_preds = self.get_base_predictions(protein_seqs, graph_data_list, xgboost_features)
        self.meta_model.fit(base_preds, true_labels)
    
    def predict(self, protein_seqs, graph_data_list: List[Data],
                xgboost_features: np.ndarray) -> np.ndarray:
        base_preds = self.get_base_predictions(protein_seqs, graph_data_list, xgboost_features)
        predictions = self.meta_model.predict(base_preds)
        return predictions


# ==============================================================================
# DATASET
# ==============================================================================
class FusionDataset(Dataset):
    def __init__(self, df, vocab, tokenizer, descriptor_names, docking_dict=None):
        self.data = []
        
        target_col = 'pIC50' if 'pIC50' in df.columns else None
        if target_col is None:
            if 'logIC50' in df.columns: target_col = 'logIC50'
            elif 'IC50' in df.columns: target_col = 'IC50'
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing", leave=False):
            try:
                smi = row.get('SMILES', row.get('canonical_smiles', row.get('ligand_smiles')))
                seq = row.get('PROTEIN_SEQ', row.get('protein_sequence'))
                if pd.isna(smi) or pd.isna(seq): continue

                raw_val = float(row[target_col])
                if np.isnan(raw_val) or np.isinf(raw_val): continue
                
                if target_col == 'pIC50':
                    val = raw_val
                elif target_col == 'IC50':
                    val = -np.log10(raw_val * 1e-9)
                else:
                    val = 9.0 - raw_val

                graph = smiles_to_graph_enhanced(smi)
                if graph is None: continue
                
                xgb_feat = compute_molecular_descriptors(smi, seq, descriptor_names, docking_dict)
                
                tokenized = tokenizer.tokenize(seq)
                seq_tensor = torch.tensor(vocab(tokenized), dtype=torch.long)
                
                # Store protein sequence string for later filtering
                self.data.append((seq_tensor, graph, torch.tensor(xgb_feat, dtype=torch.float), val, seq))
            except Exception as e:
                continue
        
        print(f"  Valid samples: {len(self.data)} / {len(df)}")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


def collate_fn(batch):
    if not batch: return None, None, None, None, None
    p, l, x, t, s = zip(*batch)
    return (nn.utils.rnn.pad_sequence(p, batch_first=True, padding_value=0),
            Batch.from_data_list(l),
            torch.stack(x),
            torch.tensor(t, dtype=torch.float),
            list(s))


# ==============================================================================
# TRAINING & EVALUATION
# ==============================================================================
def train_early_fusion(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, count = 0, 0
    for p, l, x, t, _ in tqdm(loader, desc="Training", leave=False):
        if p is None: continue
        p, l, x, t = p.to(device), l.to(device), x.to(device), t.to(device)
        
        optimizer.zero_grad()
        out = model(p, l, x)
        loss = criterion(out, t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * p.size(0)
        count += p.size(0)
    return total_loss / count if count > 0 else float('inf')


def evaluate_early_fusion(model, loader, criterion, device):
    model.eval()
    total_loss, count = 0, 0
    preds, actuals, protein_seqs = [], [], []
    with torch.no_grad():
        for p, l, x, t, s in tqdm(loader, desc="Eval", leave=False):
            if p is None: continue
            p, l, x, t = p.to(device), l.to(device), x.to(device), t.to(device)
            out = model(p, l, x)
            total_loss += criterion(out, t).item() * p.size(0)
            count += p.size(0)
            preds.extend(out.cpu().numpy())
            actuals.extend(t.cpu().numpy())
            protein_seqs.extend(s)
            
    mae = total_loss / count if count > 0 else float('nan')
    y_true, y_pred = np.array(actuals), np.array(preds)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
    corr = pearsonr(y_true, y_pred)[0]**2 if len(y_true) > 1 and np.std(y_pred) > 1e-9 else 0.0
    
    return mae, r2, corr, y_pred, y_true, protein_seqs


def extract_ensemble_data(dataset):
    """Extract data needed for ensemble predictions."""
    protein_seqs = []
    graphs = []
    xgb_features = []
    targets = []
    protein_strings = []
    
    for seq_tensor, graph, xgb_feat, target, prot_str in dataset:
        protein_seqs.append(seq_tensor)
        graphs.append(graph)
        xgb_features.append(xgb_feat.numpy())
        targets.append(target)
        protein_strings.append(prot_str)
    
    protein_seqs = nn.utils.rnn.pad_sequence(protein_seqs, batch_first=True, padding_value=0)
    xgb_features = np.array(xgb_features)
    targets = np.array(targets)
    
    return protein_seqs, graphs, xgb_features, targets, protein_strings


def plot_predictions(y_true, y_pred, title, save_path):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2)
    
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    plt.text(0.05, 0.95, f'MAE: {mae:.4f}\nR²: {r2:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.xlabel("Actual pIC50")
    plt.ylabel("Predicted pIC50")
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("FUSION STRATEGIES: GPFT + XGBoost - 3 Core Strategies")
    print("="*80)

    # Load vocabularies
    print("\n[1/7] Loading Vocabularies...")
    
    try:
        with open(Config.VOCAB_FILE, "rb") as f: 
            vocab = pickle.load(f)
        print(f"  ✅ Vocab size: {len(vocab)}")
    except (AttributeError, KeyError) as e:
        print(f"  ⚠️  Could not load pickled vocab: {e}")
        raise
    
    try:
        with open(Config.TOKENIZER_FILE, "rb") as f: 
            tokenizer = pickle.load(f)
        print(f"  ✅ Tokenizer loaded successfully")
    except (AttributeError, KeyError) as e:
        print(f"  ⚠️  Creating new tokenizer instance...")
        tokenizer = ProteinTokenizer()
        print(f"  ✅ New tokenizer created")

    # Load descriptor names
    print("\n[2/7] Loading Descriptor Names...")
    descriptor_names = [line.strip() for line in open(Config.DESCRIPTOR_FILE) if line.strip()]
    print(f"  ✅ Using {len(descriptor_names)} descriptors")

    # Load docking scores
    print("\n[3/7] Loading Docking Scores...")
    docking_train = load_docking_scores(Config.DOCKING_TRAIN)
    docking_test = load_docking_scores(Config.DOCKING_TEST)

    # Load data with scaffold split
    print("\n[4/7] Loading Data with Scaffold Split...")
    train_full = pd.read_csv(Config.TRAIN_DATA)
    test_df = pd.read_csv(Config.TEST_DATA)
    
    for df in [train_full, test_df]:
        df.columns = df.columns.str.strip()
    
    # Load split indices
    with open(Config.SPLIT_JSON, 'r') as f:
        split_info = json.load(f)
    
    train_idx = split_info.get('train_indices', split_info.get('train'))
    val_idx = split_info.get('val_indices', split_info.get('val'))
    
    train_df = train_full.iloc[train_idx].reset_index(drop=True)
    val_df = train_full.iloc[val_idx].reset_index(drop=True)
    
    print(f"  ✅ Train: {len(train_df)} samples")
    print(f"  ✅ Val: {len(val_df)} samples")
    print(f"  ✅ Test: {len(test_df)} samples")

    # Create datasets
    print("\n[5/7] Creating Datasets...")
    train_ds = FusionDataset(train_df, vocab, tokenizer, descriptor_names, docking_train)
    val_ds = FusionDataset(val_df, vocab, tokenizer, descriptor_names, docking_train)
    test_ds = FusionDataset(test_df, vocab, tokenizer, descriptor_names, docking_test)

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, 
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn, num_workers=2)

    # Load pretrained models
    print("\n[6/7] Loading Pretrained Models...")
    
    # Load GPFT model
    gnn_model = EnhancedProteinLigandModel(
        len(vocab), Config.HIDDEN_DIM, Config.OUTPUT_DIM, Config.DROPOUT
    ).to(Config.DEVICE)
    
    ckpt = torch.load(Config.GNN_MODEL, map_location=Config.DEVICE)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        gnn_model.load_state_dict(ckpt['model_state_dict'])
    else:
        gnn_model.load_state_dict(ckpt)
    print("  ✅ Loaded GPFT model")
    
    # Load XGBoost model
    print("\n  Loading XGBoost model...")
    xgb_model = None
    
    if Config.XGB_MODEL.endswith('.json'):
        try:
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(Config.XGB_MODEL)
            print("  ✅ Loaded XGBoost model (JSON format)")
        except Exception as e:
            print(f"  ⚠️  Failed to load as JSON: {e}")
    else:
        try:
            with open(Config.XGB_MODEL, 'rb') as f:
                xgb_model = pickle.load(f)
            print("  ✅ Loaded XGBoost model (pickle format)")
        except:
            try:
                xgb_model = xgb.XGBRegressor()
                xgb_model.load_model(Config.XGB_MODEL)
                print("  ✅ Loaded XGBoost model (JSON format)")
            except Exception as e:
                print(f"  ❌ Failed all loading attempts: {e}")
    
    if xgb_model is None:
        raise RuntimeError("Failed to load XGBoost model")
    
    with open(Config.XGB_SCALER, 'rb') as f:
        xgb_scaler = pickle.load(f)
    print("  ✅ Loaded XGBoost scaler")

    # Combined results storage
    all_metrics = []

    # ========================================================================
    # STRATEGY 1: EARLY FUSION
    # ========================================================================
    print("\n" + "="*80)
    print("STRATEGY 1: EARLY FUSION")
    print("="*80)
    
    for param in gnn_model.parameters():
        param.requires_grad = False
    
    early_fusion = EarlyFusionModel(
        gnn_model, 
        xgboost_feat_dim=len(descriptor_names) + 2,
        hidden_dim=Config.FUSION_HIDDEN,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    optimizer = torch.optim.AdamW(early_fusion.parameters(), lr=Config.FUSION_LR)
    criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_mae = float('inf')
    patience_counter = 0
    
    print("\nTraining Early Fusion Model...")
    for epoch in range(Config.FUSION_EPOCHS):
        train_loss = train_early_fusion(early_fusion, train_loader, optimizer, criterion, Config.DEVICE)
        val_mae, val_r2, val_corr, _, _, _ = evaluate_early_fusion(
            early_fusion, val_loader, criterion, Config.DEVICE
        )
        
        print(f"Epoch {epoch+1:3d}/{Config.FUSION_EPOCHS} | "
              f"Train: {train_loss:.4f} | Val MAE: {val_mae:.4f} | R²: {val_r2:.4f}")
        
        scheduler.step(val_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(early_fusion.state_dict(), 
                      f"{Config.OUTPUT_DIR}/early_fusion_best.pt")
        else:
            patience_counter += 1
        
        if patience_counter >= Config.FUSION_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Evaluate on validation set
    early_fusion.load_state_dict(torch.load(f"{Config.OUTPUT_DIR}/early_fusion_best.pt"))
    val_mae, val_r2, val_corr, val_preds, val_true, val_prot_seqs = evaluate_early_fusion(
        early_fusion, val_loader, criterion, Config.DEVICE
    )
    
    print(f"\n✅ Early Fusion Validation Results:")
    print(f"  MAE: {val_mae:.4f} | R²: {val_r2:.4f} | r²: {val_corr:.4f}")
    
    # Store validation metrics
    val_metrics = evaluate_and_store_metrics(
        "Early Fusion", val_true, val_preds, val_prot_seqs, "Validation"
    )
    all_metrics.extend(val_metrics)
    
    # Evaluate on test set
    test_mae, test_r2, test_corr, test_preds, test_true, test_prot_seqs = evaluate_early_fusion(
        early_fusion, test_loader, criterion, Config.DEVICE
    )
    
    print(f"\n✅ Early Fusion Test Results:")
    print(f"  MAE: {test_mae:.4f} | R²: {test_r2:.4f} | r²: {test_corr:.4f}")
    
    # Store test metrics
    test_metrics = evaluate_and_store_metrics(
        "Early Fusion", test_true, test_preds, test_prot_seqs, "Test"
    )
    all_metrics.extend(test_metrics)
    
    # Save predictions
    pd.DataFrame({
        'y_true': test_true,
        'y_pred': test_preds,
        'protein_seq': test_prot_seqs
    }).to_csv(f"{Config.OUTPUT_DIR}/early_fusion_predictions.csv", index=False)
    
    plot_predictions(test_true, test_preds, "Early Fusion - Test Set",
                    f"{Config.OUTPUT_DIR}/early_fusion_plot.png")

    # Extract validation and test data for ensemble methods
    val_proteins, val_graphs, val_xgb_raw, val_targets, val_prot_seqs = extract_ensemble_data(val_ds)
    val_xgb_scaled = xgb_scaler.transform(val_xgb_raw)
    
    test_proteins, test_graphs, test_xgb_raw, test_targets, test_prot_seqs = extract_ensemble_data(test_ds)
    test_xgb_scaled = xgb_scaler.transform(test_xgb_raw)

    # ========================================================================
    # STRATEGY 2: LATE FUSION
    # ========================================================================
    print("\n" + "="*80)
    print("STRATEGY 2: LATE FUSION")
    print("="*80)
    
    gnn_model.eval()
    late_fusion = LateFusionEnsemble(gnn_model, xgb_model, strategy='learned', 
                                    device=Config.DEVICE)
    
    print("Optimizing ensemble weights on validation set...")
    late_fusion.optimize_weights(val_proteins, val_graphs, val_xgb_scaled, val_targets)
    
    # Validation predictions
    val_late_preds = late_fusion.predict(val_proteins, val_graphs, val_xgb_scaled)
    val_mae = mean_absolute_error(val_targets, val_late_preds)
    val_r2 = r2_score(val_targets, val_late_preds)
    val_corr = pearsonr(val_targets, val_late_preds)[0]**2
    
    print(f"\n✅ Late Fusion Validation Results:")
    print(f"  MAE: {val_mae:.4f} | R²: {val_r2:.4f} | r²: {val_corr:.4f}")
    
    # Store validation metrics
    val_metrics = evaluate_and_store_metrics(
        "Late Fusion", val_targets, val_late_preds, val_prot_seqs, "Validation"
    )
    all_metrics.extend(val_metrics)
    
    # Test predictions
    test_late_preds = late_fusion.predict(test_proteins, test_graphs, test_xgb_scaled)
    test_mae = mean_absolute_error(test_targets, test_late_preds)
    test_r2 = r2_score(test_targets, test_late_preds)
    test_corr = pearsonr(test_targets, test_late_preds)[0]**2
    
    print(f"\n✅ Late Fusion Test Results:")
    print(f"  MAE: {test_mae:.4f} | R²: {test_r2:.4f} | r²: {test_corr:.4f}")
    
    # Store test metrics
    test_metrics = evaluate_and_store_metrics(
        "Late Fusion", test_targets, test_late_preds, test_prot_seqs, "Test"
    )
    all_metrics.extend(test_metrics)
    
    pd.DataFrame({
        'y_true': test_targets,
        'y_pred': test_late_preds,
        'protein_seq': test_prot_seqs
    }).to_csv(f"{Config.OUTPUT_DIR}/late_fusion_predictions.csv", index=False)
    
    plot_predictions(test_targets, test_late_preds, "Late Fusion - Test Set",
                    f"{Config.OUTPUT_DIR}/late_fusion_plot.png")

    # ========================================================================
    # STRATEGY 3: STACKING
    # ========================================================================
    print("\n" + "="*80)
    print("STRATEGY 3: STACKING")
    print("="*80)
    
    stacking = StackingEnsemble(gnn_model, xgb_model, meta_learner='ridge',
                               device=Config.DEVICE)
    
    print("Training Ridge meta-learner on validation set...")
    stacking.fit(val_proteins, val_graphs, val_xgb_scaled, val_targets)
    
    # Validation predictions
    val_stack_preds = stacking.predict(val_proteins, val_graphs, val_xgb_scaled)
    val_mae = mean_absolute_error(val_targets, val_stack_preds)
    val_r2 = r2_score(val_targets, val_stack_preds)
    val_corr = pearsonr(val_targets, val_stack_preds)[0]**2
    
    print(f"\n✅ Stacking Validation Results:")
    print(f"  MAE: {val_mae:.4f} | R²: {val_r2:.4f} | r²: {val_corr:.4f}")
    
    # Store validation metrics
    val_metrics = evaluate_and_store_metrics(
        "Stacking", val_targets, val_stack_preds, val_prot_seqs, "Validation"
    )
    all_metrics.extend(val_metrics)
    
    # Test predictions
    test_stack_preds = stacking.predict(test_proteins, test_graphs, test_xgb_scaled)
    test_mae = mean_absolute_error(test_targets, test_stack_preds)
    test_r2 = r2_score(test_targets, test_stack_preds)
    test_corr = pearsonr(test_targets, test_stack_preds)[0]**2
    
    print(f"\n✅ Stacking Test Results:")
    print(f"  MAE: {test_mae:.4f} | R²: {test_r2:.4f} | r²: {test_corr:.4f}")
    
    # Store test metrics
    test_metrics = evaluate_and_store_metrics(
        "Stacking", test_targets, test_stack_preds, test_prot_seqs, "Test"
    )
    all_metrics.extend(test_metrics)
    
    pd.DataFrame({
        'y_true': test_targets,
        'y_pred': test_stack_preds,
        'protein_seq': test_prot_seqs
    }).to_csv(f"{Config.OUTPUT_DIR}/stacking_predictions.csv", index=False)
    
    plot_predictions(test_targets, test_stack_preds, "Stacking - Test Set",
                    f"{Config.OUTPUT_DIR}/stacking_plot.png")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n[7/7] Generating Final Summary...")
    print("\n" + "="*80)
    print("FUSION STRATEGIES - COMBINED RESULTS")
    print("="*80)
    
    # Save all metrics to single CSV
    combined_df = pd.DataFrame(all_metrics)
    combined_df = combined_df[['strategy', 'dataset', 'protein', 'n_samples', 
                               'mae', 'rmse', 'r2', 'pearson_r', 'r2_corr']]
    
    print("\n" + "COMBINED VALIDATION AND TEST METRICS:")
    print("-" * 80)
    print(combined_df.to_string(index=False))
    
    combined_df.to_csv(f"{Config.OUTPUT_DIR}/combined_metrics_val_and_test.csv", index=False)
    
    # Summary tables
    print("\n" + "="*80)
    print("SUMMARY BY STRATEGY (Overall Performance):")
    print("="*80)
    overall_summary = combined_df[combined_df['protein'] == 'Overall'].copy()
    overall_summary = overall_summary.sort_values(['dataset', 'mae'])
    print(overall_summary.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("BEST STRATEGY:")
    print(f"{'='*80}")
    
    # Best on validation
    best_val = overall_summary[overall_summary['dataset'] == 'Validation'].iloc[0]
    print(f"\nValidation Set:")
    print(f"  Strategy: {best_val['strategy']}")
    print(f"  MAE: {best_val['mae']:.4f} | R²: {best_val['r2']:.4f} | r²: {best_val['r2_corr']:.4f}")
    
    # Best on test
    best_test = overall_summary[overall_summary['dataset'] == 'Test'].iloc[0]
    print(f"\nTest Set:")
    print(f"  Strategy: {best_test['strategy']}")
    print(f"  MAE: {best_test['mae']:.4f} | R²: {best_test['r2']:.4f} | r²: {best_test['r2_corr']:.4f}")
    
    print(f"\n✅ Combined metrics saved to: {Config.OUTPUT_DIR}/combined_metrics_val_and_test.csv")
    print(f"✅ All predictions and plots saved to: {Config.OUTPUT_DIR}/")
    print("="*80)


if __name__ == "__main__":
    main()

