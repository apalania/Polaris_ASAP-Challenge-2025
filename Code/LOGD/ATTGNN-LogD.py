import torch 
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from google.cloud import storage
import os

BUCKET_NAME = "mounikasri"
MODEL_PREFIX = "KSOL"

# ----------- Google Cloud Storage Helper -----------

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}")

# ---------------------- Data Processing ----------------------

def load_data(file_path, smiles_column="SMILES", target_column="LogD"):
    df = pd.read_csv(file_path).dropna(subset=[smiles_column, target_column])
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')  # Convert to numeric, NaNs for invalid
    return df, smiles_column, target_column


def smiles_to_graph(smiles, target):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        one_hot = [0] * 10
        one_hot[min(atomic_num, 9)] = 1
        features = one_hot + [
            atom.GetDegree() / 4.0,
            atom.GetFormalCharge() / 5.0,
            int(atom.GetHybridization()) / 6.0,
            float(atom.GetMass()) / 200.0,
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs() / 4.0,
            atom.GetExplicitValence() / 8.0
        ]
        atom_features.append(features)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_features = [
            int(bond.GetBondType() == Chem.rdchem.BondType.SINGLE),
            int(bond.GetBondType() == Chem.rdchem.BondType.DOUBLE),
            int(bond.GetBondType() == Chem.rdchem.BondType.TRIPLE),
            int(bond.GetBondType() == Chem.rdchem.BondType.AROMATIC),
            int(bond.GetIsConjugated())
        ]
        edge_index.extend([[start, end], [end, start]])
        edge_attr.extend([bond_features, bond_features])

    if not edge_index:
        return None

    return Data(
        x=torch.tensor(atom_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        y=torch.tensor([target], dtype=torch.float)
    )

# ---------------------- Model Definition ----------------------

class EnhancedAttentiveFP(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers=4, num_timesteps=6, hidden_channels=256, dropout=0.25):
        super(EnhancedAttentiveFP, self).__init__()
        self.gnn = AttentiveFP(
            in_channels=node_feat_size,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=edge_feat_size,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout
        )
        self.linear = nn.Linear(hidden_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr, batch)
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu(x)
        return self.output_layer(x).squeeze(1)

# ---------------------- Training Function ----------------------

def train_model(data, epochs=200, batch_size=64, lr=3e-4, patience=15, weight_decay=1e-5, grad_accum_steps=2, bucket_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, drop_last=False)

    num_node_features = data[0].x.size(1)
    num_edge_features = data[0].edge_attr.size(1)

    model = EnhancedAttentiveFP(num_node_features, num_edge_features, num_layers=4, num_timesteps=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.L1Loss()

    best_r2 = -float("inf")
    best_mae = float("inf")
    early_stop_counter = 0
    val_metrics = []

    print("Starting validation phase...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                preds.extend(out.cpu().numpy())
                labels.extend(batch.y.view(-1).cpu().numpy())

        preds = np.array(preds)
        labels = np.array(labels)

        # Filter out any NaN values
        valid_mask = ~np.isnan(preds) & ~np.isnan(labels)
        preds = preds[valid_mask]
        labels = labels[valid_mask]

        if len(preds) == 0 or len(labels) == 0:
            print("Warning: No valid predictions or labels for R²/MAE calculation.")
            continue

        r2 = r2_score(labels, preds)
        mae = mean_absolute_error(labels, preds)



        val_metrics.append({
            'epoch': epoch + 1,
            'train_loss': total_loss / len(train_loader),
            'val_r2': r2,
            'val_mae': mae
        })

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")

        scheduler.step(r2)

        if r2 > best_r2 or (r2 == best_r2 and mae < best_mae):
            best_r2 = r2
            best_mae = mae
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model_val_phase.pth")
            if bucket_name:
                upload_to_gcs(bucket_name, "best_model_val_phase.pth", "models/best_model_val_phase.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping.")
                break

    pd.DataFrame(val_metrics).to_csv("validation_metrics.csv", index=False)
    if bucket_name:
        upload_to_gcs(bucket_name, "validation_metrics.csv", "metrics/validation_metrics.csv")

    print("\nTraining on full dataset...")
    full_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    best_loss = float("inf")
    early_stop_counter = 0
    full_metrics = []

    for epoch in range(epochs // 2):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(full_loader):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or step == len(full_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(full_loader)
        full_metrics.append({'epoch': epoch + 1, 'train_loss': avg_loss})
        print(f"Full Data Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model_LogDfull.pth")
            if bucket_name:
                upload_to_gcs(bucket_name, "best_model_LogDfull.pth", "models/best_model_LogDfull.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping in full training.")
                break

    pd.DataFrame(full_metrics).to_csv("full_training_metrics.csv", index=False)
    if bucket_name:
        upload_to_gcs(bucket_name, "full_training_metrics.csv", "metrics/full_training_metrics.csv")

    # Script and save the trained model
 

    return model, val_data

# ---------------------- Main Execution ----------------------

if __name__ == '__main__':
    # Set your GCS bucket here or None if not using GCS
    GCS_BUCKET_NAME = "mounikasri"

    file_path = "gs://mounikasri/LogD_MERGED.csv"
    df, smiles_col, target_col = load_data(file_path)

    print("Converting SMILES to graph...")
    graph_data = [smiles_to_graph(row[smiles_col], row[target_col]) for _, row in tqdm(df.iterrows(), total=len(df))]
    graph_data = [g for g in graph_data if g is not None]
    print(f"Loaded {len(graph_data)} valid molecules.")

    trained_model, val_data = train_model(graph_data, bucket_name=GCS_BUCKET_NAME)

    # ---------------------- Plotting ----------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.eval()

    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    actuals = []
    predictions = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = trained_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(batch.y.view(-1).cpu().numpy())

    actuals = np.array(actuals)
    predictions = np.array(predictions)
    residuals = actuals - predictions

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=50, color='skyblue', edgecolor='black')
    plt.title("Residuals Histogram (Actual - Predicted)")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("residual_histogram.png")
    plt.show()
    if GCS_BUCKET_NAME:
        upload_to_gcs(GCS_BUCKET_NAME, "residual_histogram.png", "plots/residual_histogram.png")

    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.6, edgecolors='k')
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.title("Actual vs Predicted Scatter Plot")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True)
    plt.savefig("actual_vs_predicted.png")
    plt.show()
    if GCS_BUCKET_NAME:
        upload_to_gcs(GCS_BUCKET_NAME, "actual_vs_predicted.png", "plots/actual_vs_predicted.png")
