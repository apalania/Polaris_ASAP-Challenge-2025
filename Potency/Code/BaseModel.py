import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data, Batch
from torch_geometric.nn import AttentiveFP
from rdkit import Chem
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from google.cloud import storage
import math

# ----------------- GCP Configuration -----------------
class GCPConfig:
    BUCKET_NAME = "mounikasri"
    MODEL_PREFIX = "model3/ic50_predictor"
    DATA_PATH = "gs://mounikasri/protein.csv"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 200
    PATIENCE = 10
    SEED = 42
    NUM_WORKERS = 0   # <-- changed here from 4 to 0
    ATTENTIVEFP_PARAMS = {
        'in_channels': 3,
        'hidden_channels': 256,
        'out_channels': 32,
        'edge_dim': 1,
        'num_layers': 4,
        'num_timesteps': 4,
        'dropout': 0.25,
    }

storage_client = storage.Client()
bucket = storage_client.bucket(GCPConfig.BUCKET_NAME)

# ----------------- Protein Components -----------------
def protein_tokenizer(seq):
    return list(seq.upper())

class ProteinTokenizer:
    def __init__(self):
        self.tokenizer = protein_tokenizer
    def tokenize(self, seq):
        return self.tokenizer(seq)

class ProteinEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_layers=1, n_heads=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        return self.pool(x).squeeze(-1)

# ----------------- Ligand Components -----------------
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            int(atom.GetIsAromatic()),
            0,0,0,0,0
        ][:GCPConfig.ATTENTIVEFP_PARAMS['in_channels']]
        atom_features.append(features)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        edge_feat = [bond_type, 0.0, 0.0][:GCPConfig.ATTENTIVEFP_PARAMS['edge_dim']]
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([edge_feat, edge_feat])

    return Data(
        x=torch.tensor(atom_features, dtype=torch.float),
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attrs, dtype=torch.float)
    )

class LigandEncoder(nn.Module):
    def __init__(self):
        p = GCPConfig.ATTENTIVEFP_PARAMS
        super().__init__()
        self.attentive_fp = AttentiveFP(
            in_channels=p['in_channels'],
            hidden_channels=p['hidden_channels'],
            out_channels=p['out_channels'],
            edge_dim=p['edge_dim'],
            num_layers=p['num_layers'],
            num_timesteps=p['num_timesteps'],
            dropout=p['dropout']
        )

    def forward(self, data):
        return self.attentive_fp(data.x, data.edge_index, data.edge_attr, data.batch)

# ----------------- Combined Model -----------------
class IC50Predictor(nn.Module):
    def __init__(self, protein_vocab_size):
        super().__init__()
        self.protein_encoder = ProteinEncoder(protein_vocab_size)
        self.ligand_encoder = LigandEncoder()
        self.fc = nn.Sequential(
            nn.Linear(32 + 32, 32),  # protein_emb(32) + ligand_emb(32)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, protein_seq, ligand_data):
        protein_emb = self.protein_encoder(protein_seq)
        ligand_emb = self.ligand_encoder(ligand_data)
        combined = torch.cat([protein_emb, ligand_emb], dim=-1)
        return self.fc(combined).squeeze(-1)

# ----------------- Utility Functions -----------------
def save_to_gcs(local_path, gcs_path):
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{GCPConfig.BUCKET_NAME}/{gcs_path}")

def save_metrics(metrics, path="metrics.txt"):
    with open(path, "w") as file:
        for epoch, data in enumerate(metrics):
            file.write(f"Epoch {epoch+1}: Train Loss: {data['train_loss']:.6f}, Val Loss: {data['val_loss']:.6f}, Val Accuracy: {data['val_accuracy']:.6f}, R2: {data['r2']:.6f}\n")

def save_vocab(vocab, path="vocab.pkl"):
    with open(path, "wb") as f:
        pickle.dump(vocab, f)
    save_to_gcs(path, f"{GCPConfig.MODEL_PREFIX}/{path}")

class EarlyStopping:
    def __init__(self, patience=10):
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
            if self.counter >= self.patience:
                self.early_stop = True

class GCProteinLigandDataset(Dataset):
    def __init__(self, df, vocab, tokenizer):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.tokenizer = tokenizer
        # Apply log transformation to IC50 values
        self.df['IC50'] = self.df['IC50'].apply(lambda x: math.log10(x) if x > 0 else -12)  # -12 as placeholder for zero/negative values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        protein_seq = torch.tensor(self.vocab(self.tokenizer.tokenize(row['protein_sequence'])), dtype=torch.long)
        ligand_data = smiles_to_graph(row['canonical_smiles'])
        target = torch.tensor(row['IC50'], dtype=torch.float)
        return protein_seq, ligand_data, target

def collate_fn(batch):
    proteins, ligands, targets = zip(*batch)
    proteins = nn.utils.rnn.pad_sequence(proteins, batch_first=True, padding_value=0)

    ligands = [lig for lig in ligands if lig is not None]
    batch_ligands = Batch.from_data_list(ligands) if ligands else None
    targets = torch.stack(targets)

    return proteins, batch_ligands, targets

def load_gcs_data():
    local_path = "protein.csv"
    blob = bucket.blob(GCPConfig.DATA_PATH.replace(f"gs://{GCPConfig.BUCKET_NAME}/", ""))
    blob.download_to_filename(local_path)
    df = pd.read_csv(local_path)
    return df

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for protein_seq, ligand_data, targets in loader:
        protein_seq = protein_seq.to(device)
        targets = targets.to(device)
        if ligand_data is not None:
            ligand_data = ligand_data.to(device)
        optimizer.zero_grad()
        outputs = model(protein_seq, ligand_data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * protein_seq.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for protein_seq, ligand_data, targets in loader:
            protein_seq = protein_seq.to(device)
            targets = targets.to(device)
            if ligand_data is not None:
                ligand_data = ligand_data.to(device)
            outputs = model(protein_seq, ligand_data)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * protein_seq.size(0)
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
    avg_loss = total_loss / len(loader.dataset)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    r2 = r2_score(targets.numpy(), preds.numpy())
    accuracy = ((preds.round() == targets).float().mean()).item()
    return avg_loss, accuracy, r2, preds, targets

# ----------------- Plotting -----------------
def plot_actual_vs_predicted(actual, predicted, path="actual_vs_predicted.png"):
    plt.figure(figsize=(8,8))
    plt.scatter(actual, predicted, alpha=0.6)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    plt.xlabel("Actual log(IC50)")
    plt.ylabel("Predicted log(IC50)")
    plt.title("Actual vs Predicted log(IC50)")
    plt.savefig(path)
    plt.close()
    save_to_gcs(path, f"{GCPConfig.MODEL_PREFIX}/{path}")

# ----------------- Main Function -----------------
def main():
    torch.manual_seed(GCPConfig.SEED)
    df = load_gcs_data()

    # Build protein vocab
    tokenizer = ProteinTokenizer()
    protein_tokens = [tokenizer.tokenize(seq) for seq in df['protein_sequence']]
    vocab = build_vocab_from_iterator(protein_tokens, specials=["<pad>"])
    vocab.set_default_index(vocab["<pad>"])

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=GCPConfig.SEED)

    train_dataset = GCProteinLigandDataset(train_df, vocab, tokenizer)
    val_dataset = GCProteinLigandDataset(val_df, vocab, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=GCPConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=GCPConfig.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=GCPConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=GCPConfig.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )

    device = GCPConfig.DEVICE
    model = IC50Predictor(len(vocab)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=GCPConfig.LEARNING_RATE)
    criterion = nn.MSELoss()
    early_stopper = EarlyStopping(patience=GCPConfig.PATIENCE)

    metrics = []
    best_val_loss = float('inf')

    for epoch in range(GCPConfig.MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_r2, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{GCPConfig.MAX_EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val Acc: {val_acc:.4f} - R2: {val_r2:.4f}")
        metrics.append({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "r2": val_r2
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            save_to_gcs("best_model.pt", f"{GCPConfig.MODEL_PREFIX}/best_model.pt")

        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    save_metrics(metrics, "metrics.txt")
    save_to_gcs("metrics.txt", f"{GCPConfig.MODEL_PREFIX}/metrics.txt")
    save_vocab(vocab, "vocab.pkl")

    # Load best model for final evaluation and plotting
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    _, _, _, preds, targets = evaluate(model, val_loader, criterion, device)
    plot_actual_vs_predicted(targets.numpy(), preds.numpy())

if __name__ == "__main__":
    main()