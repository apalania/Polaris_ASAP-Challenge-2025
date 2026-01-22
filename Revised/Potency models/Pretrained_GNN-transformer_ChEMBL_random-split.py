import os
# --- 1. STRICT REPRODUCIBILITY SETUP (MUST BE FIRST) ---
# Forces PyTorch to use deterministic algorithms for CUDA convolution/operations
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
# -------------------------------------------------------

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import AttentiveFP
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen, rdPartialCharges
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm
import pickle
from collections import Counter
from typing import Optional

# --- 2. ENABLE SPEED OPTIMIZATIONS FOR RTX A4000 ---
# TensorFloat-32 is deterministic on Ampere GPUs and much faster
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
            # --- DETERMINISTIC SORTING ---
            # Sort by frequency (descending), then by text (ascending) to break ties
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
    DATA_FILE = "protein_ligand.csv"
    BATCH_SIZE = 64  # Optimized for RTX A4000 (16GB VRAM)
    EPOCHS = 200
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Optimized for speed
    HIDDEN_DIM = 256
    OUTPUT_DIM = 32
    DROPOUT = 0.2
    MODEL_SAVE = "potency_enhanced_features_best_model.pt"
    METRICS_FILE = "potency_enhanced_features_training_metrics.csv"
    VOCAB_FILE = "vocab_enhanced_features.pkl"
    TOKENIZER_FILE = "tokenizer_enhanced_features.pkl"
    PATIENCE = 15
    SEED = 42
    USE_MIXED_PRECISION = True 

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

# ------------------- Dataset -------------------

class EnhancedProteinLigandDataset(Dataset):
    def __init__(self, df, vocab, tokenizer):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.processed_data = []

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Preprocessing Ligands"):
            pIC50_val = -math.log10(row['IC50']) if row['IC50'] > 0 else 12
            ligand_graph = smiles_to_graph_enhanced(row['canonical_smiles'])
            if ligand_graph is not None:
                ligand_graph.y = torch.tensor([pIC50_val], dtype=torch.float)
                self.processed_data.append({
                    'protein_sequence': row['protein_sequence'],
                    'ligand_graph': ligand_graph,
                    'pIC50': pIC50_val
                })
        print(f"Final dataset size: {len(self.processed_data)}")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        entry = self.processed_data[idx]
        protein_tokens = self.tokenizer.tokenize(entry['protein_sequence'])
        protein_seq_tensor = torch.tensor(self.vocab(protein_tokens), dtype=torch.long)
        return protein_seq_tensor, entry['ligand_graph']

def collate_fn(batch):
    proteins = nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
    batch_ligands = Batch.from_data_list([item[1] for item in batch])
    targets = torch.cat([item[1].y for item in batch]).view(-1, 1)
    return proteins, batch_ligands, targets

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
    return total_loss / num_samples, r2_score(targets.numpy(), preds.numpy()), np.sqrt(np.mean((targets.numpy() - preds.numpy())**2)), preds, targets

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
# ----------------------------------------------

# ------------------- Plotting -------------------
def plot_learning_curves(metrics_file, save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(metrics_file)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_mae"], label="Train MAE", marker="o", markersize=4)
    plt.plot(df["epoch"], df["val_mae"], label="Validation MAE", marker="s", markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("MAE (pIC50)")
    plt.title("Enhanced Features Model: MAE per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "enhanced_features_learning_curve_mae.png"), dpi=600, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["val_r2"], label="Validation R2", marker="x", markersize=4, color='green')
    plt.xlabel("Epoch")
    plt.ylabel("R2 Score")
    plt.title("Enhanced Features Model: R2 per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "enhanced_features_learning_curve_r2.png"), dpi=600, bbox_inches="tight")
    plt.close()

def plot_results(actual, predicted, save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.6, edgecolors='k')
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    plt.title("Enhanced Features: Actual vs Predicted")
    plt.xlabel("Actual pIC50")
    plt.ylabel("Predicted pIC50")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "enhanced_features_actual_vs_predicted.png"), dpi=600, bbox_inches="tight")
    plt.close()

# ------------------- Main -------------------

def main():
    set_seed(Config.SEED)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    df = pd.read_csv(Config.DATA_FILE)
    print(f"Loaded dataset with {len(df)} samples")

    tokenizer = ProteinTokenizer()
    print("Building vocabulary...")
    protein_tokens = [tokenizer.tokenize(seq) for seq in df['protein_sequence']]
    vocab = build_vocab_from_iterator(protein_tokens, specials=["<pad>"])
    vocab.set_default_index(vocab["<pad>"])
    
    # Save Artifacts 1 & 2: Vocab and Tokenizer
    with open(Config.VOCAB_FILE, "wb") as f: pickle.dump(vocab, f)
    with open(Config.TOKENIZER_FILE, "wb") as f: pickle.dump(tokenizer, f)
    print("Saved Vocab and Tokenizer.")

    dataset = EnhancedProteinLigandDataset(df, vocab, tokenizer)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(Config.SEED)
    )
    
    # --- 4. OPTIMIZED & REPRODUCIBLE DATALOADERS ---
    g = torch.Generator()
    g.manual_seed(Config.SEED)
    
    loader_args = dict(
        batch_size=Config.BATCH_SIZE, 
        collate_fn=collate_fn, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type=='cuda' else False,
        worker_init_fn=seed_worker, # Deterministic worker seeding
        generator=g,                # Deterministic shuffling
        persistent_workers=True,    # Speed optimization (Keeps workers alive)
        prefetch_factor=2           # Speed optimization (Pre-loads batches)
    )
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    # -----------------------------------------------

    model = EnhancedProteinLigandModel(len(vocab), Config.HIDDEN_DIM, Config.OUTPUT_DIM, Config.DROPOUT).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.L1Loss()
    early_stopper = EarlyStopping(patience=Config.PATIENCE)
    scaler = torch.cuda.amp.GradScaler() if Config.USE_MIXED_PRECISION else None
    
    best_val_mae = float('inf')
    metrics_history = []
    
    print("\nStarting Training...")
    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        train_mae = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE, scaler)
        val_mae, val_r2, val_rmse, _, _ = evaluate(model, val_loader, criterion, Config.DEVICE)
        
        print(f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | Val R2: {val_r2:.4f} | Val RMSE: {val_rmse:.4f}")
        metrics_history.append({"epoch": epoch+1, "train_mae": train_mae, "val_mae": val_mae, "val_r2": val_r2, "val_rmse": val_rmse})
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            
            # --- 5. COMPREHENSIVE ARTIFACT SAVING ---
            # Save configuration AND state_dict for easy fine-tuning
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
            # ----------------------------------------
            print(f"✓ Saved best model with Val MAE: {best_val_mae:.4f}")
            
        early_stopper(val_mae)
        if early_stopper.early_stop:
            print(f"Early stopping triggered after {Config.PATIENCE} epochs.")
            break

    pd.DataFrame(metrics_history).to_csv(Config.METRICS_FILE, index=False)
    print("Generating Learning Curves...")
    plot_learning_curves(Config.METRICS_FILE)
    
    # Final Evaluation (Loading from the checkpoint to verify artifact integrity)
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(Config.MODEL_SAVE)
    model.load_state_dict(checkpoint['model_state_dict']) # Load weights
    
    final_val_mae, final_val_r2, final_val_rmse, final_preds, final_targets = evaluate(
        model, val_loader, criterion, Config.DEVICE
    )
    
    print(f"\nFinal Best Enhanced Features Model Performance:")
    print(f"Validation MAE: {final_val_mae:.4f}")
    print(f"Validation R2: {final_val_r2:.4f}")
    print(f"Validation RMSE: {final_val_rmse:.4f}")
    
    plot_results(final_targets.numpy(), final_preds.numpy())
    
    # Save Training Summary
    summary_path = Config.MODEL_SAVE.replace('.pt', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Training Complete.\nBest Val MAE: {best_val_mae:.4f}\nR2: {final_val_r2:.4f}\n")
        f.write(f"Saved Checkpoint: {Config.MODEL_SAVE}\n")
        f.write(f"Contains 'config' dict for automatic fine-tuning initialization.\n")
    
    print(f"Done! Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
