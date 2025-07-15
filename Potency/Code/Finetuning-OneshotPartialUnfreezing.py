# -*- coding: utf-8 -*-
"""Fine_tune_potency_with_partial_unfreeze.py"""

# ------------------- Imports -------------------
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import AttentiveFP
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from google.cloud import storage
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
from scipy.stats import pearsonr # <-- Added import for Pearson correlation

# ----------------- GCP Configuration -----------------
class GCPConfig:
    BUCKET_NAME = "idapriya"
    MODEL_PREFIX = "potency_finetune_partial_unfreeze_final" # New prefix for this experiment
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 64
    # 📌 STRATEGY: Use a smaller learning rate for more extensive fine-tuning
    LEARNING_RATE = 1e-5  # Lowered from 2e-5 to avoid destroying pre-trained weights
    MAX_EPOCHS = 100
    PATIENCE = 15 # Increased patience as fine-tuning can be noisier
    SEED = 42
    ATTENTIVEFP_PARAMS = {
        'in_channels': 3, 'hidden_channels': 256, 'out_channels': 32,
        'edge_dim': 1, 'num_layers': 4, 'num_timesteps': 4, 'dropout': 0.25,
    }

storage_client = storage.Client()
bucket = storage_client.bucket(GCPConfig.BUCKET_NAME)

# ----------------- Component Definitions (No changes needed) -----------------
def protein_tokenizer(seq):
    return list(seq.upper())

class ProteinTokenizer:
    def tokenize(self, seq):
        return protein_tokenizer(seq)

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

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    atom_features = [[atom.GetAtomicNum()/100.0, atom.GetDegree()/4.0, int(atom.GetIsAromatic())] for atom in mol.GetAtoms()]
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = [bond.GetBondTypeAsDouble() / 3.0]
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [feat, feat]
    return Data(
        x=torch.tensor(atom_features, dtype=torch.float),
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attrs, dtype=torch.float)
    )

class LigandEncoder(nn.Module):
    def __init__(self):
        p = GCPConfig.ATTENTIVEFP_PARAMS
        super().__init__()
        self.attentive_fp = AttentiveFP(in_channels=p['in_channels'], hidden_channels=p['hidden_channels'], out_channels=p['out_channels'], edge_dim=p['edge_dim'], num_layers=p['num_layers'], num_timesteps=p['num_timesteps'], dropout=p['dropout'])
    def forward(self, data):
        return self.attentive_fp(data.x, data.edge_index, data.edge_attr, data.batch)

class IC50Predictor(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.protein_encoder = ProteinEncoder(vocab_size)
        self.ligand_encoder = LigandEncoder()
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
    def forward(self, protein_seq, ligand_data):
        protein_emb = self.protein_encoder(protein_seq)
        ligand_emb = self.ligand_encoder(ligand_data)
        combined = torch.cat([protein_emb, ligand_emb], dim=-1)
        return self.fc(combined).squeeze(-1)

class GCProteinLigandDataset(Dataset):
    def __init__(self, df, vocab, tokenizer):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles_str = row.get("SMILES", "")
        ligand_data = smiles_to_graph(smiles_str)
        if ligand_data is None:
            print(f"Warning: Could not process SMILES at index {idx}: {smiles_str}. Skipping.")
            return None
        protein_seq = torch.tensor(self.vocab(self.tokenizer.tokenize(row["PROTEIN_SEQ"])), dtype=torch.long)
        target = torch.tensor(row["logIC50"], dtype=torch.float)
        return protein_seq, ligand_data, target

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None, None, None
    proteins, ligands, targets = zip(*batch)
    proteins = nn.utils.rnn.pad_sequence(proteins, batch_first=True, padding_value=0)
    batch_ligands = Batch.from_data_list(ligands)
    targets = torch.stack(targets)
    return proteins, batch_ligands, targets

def save_to_gcs(local_path, gcs_path):
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"✅ Uploaded {local_path} to gs://{GCPConfig.BUCKET_NAME}/{gcs_path}")

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience, self.counter, self.best_loss, self.early_stop = patience, 0, None, False
    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss, self.counter = val_loss, 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, num_samples = 0, 0
    for protein_seq, ligand_data, targets in tqdm(loader, desc="Training", leave=False):
        if protein_seq is None: continue
        protein_seq, targets = protein_seq.to(device), targets.to(device)
        ligand_data = ligand_data.to(device)
        optimizer.zero_grad()
        outputs = model(protein_seq, ligand_data)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        batch_size = protein_seq.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size
    return total_loss / num_samples if num_samples > 0 else 0

def evaluate(model, loader, criterion, device, desc="Evaluating"):
    model.eval()
    total_loss, num_samples = 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for protein_seq, ligand_data, targets in tqdm(loader, desc=desc, leave=False):
            if protein_seq is None: continue
            protein_seq, targets = protein_seq.to(device), targets.to(device)
            ligand_data = ligand_data.to(device)
            outputs = model(protein_seq, ligand_data)
            loss = criterion(outputs, targets)
            batch_size = protein_seq.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            
    if not all_preds or num_samples == 0:
        return float('nan'), float('nan'), float('nan'), torch.tensor([]), torch.tensor([])
        
    avg_loss = total_loss / num_samples
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    if len(targets) > 1:
        # R-squared (Coefficient of Determination)
        r2_cod = r2_score(targets.numpy(), preds.numpy())
        # r-squared (from Pearson correlation)
        corr, _ = pearsonr(targets.numpy(), preds.numpy())
        r2_corr = corr**2
    else:
        r2_cod, r2_corr = 0.0, 0.0
        
    # FIX: Return 5 values
    return avg_loss, r2_cod, r2_corr, preds, targets

def plot_actual_vs_predicted(actual, predicted, path, title_suffix):
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.6)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    plt.xlabel("Actual logIC50")
    plt.ylabel("Predicted logIC50")
    plt.title(f"Actual vs Predicted logIC50 ({title_suffix})")
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    save_to_gcs(path, f"{GCPConfig.MODEL_PREFIX}/{os.path.basename(path)}")

# FIX: Updated function to handle and plot all metrics
def plot_metrics(train_mae, val_mae, val_r2_cod, val_r2_corr, path):
    epochs = range(1, len(train_mae) + 1)
    plt.figure(figsize=(18, 5))

    # Plot 1: MAE
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_mae, label="Train MAE", marker='o')
    plt.plot(epochs, val_mae, label="Val MAE", marker='o')
    plt.title("MAE vs. Epoch")
    plt.xlabel("Epoch"), plt.ylabel("MAE (on logIC50)"), plt.legend(), plt.grid(True)
    
    # Plot 2: R² (Coeff of Det)
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_r2_cod, label="Val R² (Coeff. of Det.)", color='green', marker='o')
    plt.title("R² (Coeff. of Det.) vs. Epoch")
    plt.xlabel("Epoch"), plt.ylabel("R²"), plt.legend(), plt.grid(True)
    
    # Plot 3: r² (Sq. Correlation)
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_r2_corr, label="Val r² (Sq. Correlation)", color='purple', marker='o')
    plt.title("r² (Sq. Correlation) vs. Epoch")
    plt.xlabel("Epoch"), plt.ylabel("r²"), plt.legend(), plt.grid(True)

    plt.tight_layout(), plt.savefig(path), plt.close()
    save_to_gcs(path, f"{GCPConfig.MODEL_PREFIX}/{os.path.basename(path)}")


def main():
    torch.manual_seed(GCPConfig.SEED)
    print(f"Using device: {GCPConfig.DEVICE}")

    # --- Data Loading and Preprocessing ---
    print("Loading and preparing data...")
    train_df = pd.read_csv("polaris_train.csv")
    train_df.columns = train_df.columns.str.strip()
    train_df = train_df.dropna(subset=["SMILES", "PROTEIN_SEQ", "pIC50"])

    test_df = pd.read_csv("polaris_unblinded_test.csv")
    test_df.columns = test_df.columns.str.strip()
    test_df = test_df.dropna(subset=["SMILES", "PROTEIN_SEQ", "pIC50"])

    print("Converting pIC50 to log10(IC50_nM) scale for model consistency...")
    train_df['logIC50'] = 9.0 - train_df['pIC50']
    test_df['logIC50'] = 9.0 - test_df['pIC50']

    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=GCPConfig.SEED)
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")

    # --- Dataset and DataLoader Setup ---
    tokenizer = ProteinTokenizer()
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    train_dataset = GCProteinLigandDataset(train_df, vocab, tokenizer)
    val_dataset = GCProteinLigandDataset(val_df, vocab, tokenizer)
    test_dataset = GCProteinLigandDataset(test_df, vocab, tokenizer) 

    train_loader = DataLoader(train_dataset, batch_size=GCPConfig.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=GCPConfig.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=GCPConfig.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # --- Model Setup ---
    model = IC50Predictor(len(vocab)).to(GCPConfig.DEVICE)
    criterion = nn.L1Loss()
    
    checkpoint = torch.load("best_model.pt", map_location=GCPConfig.DEVICE)
    model.load_state_dict(checkpoint, strict=False)
    print(f"✅ Loaded pretrained weights.")

    # --- Evaluate Base Model (Zero-Shot) ---
    print("\n--- Evaluating Base Model (Zero-Shot) on Test Set (logIC50 scale) ---")
    # FIX: Unpack all 5 return values
    test_loss_base, test_r2_cod_base, test_r2_corr_base, base_preds, base_targets = evaluate(model, test_loader, criterion, GCPConfig.DEVICE, desc="Zero-Shot Test")
    
    submission_base_df = test_df[['SMILES', 'PROTEIN_SEQ']].copy()
    submission_base_df['logIC50_actual'] = base_targets.numpy()
    submission_base_df['logIC50_predicted_base'] = base_preds.numpy()
    submission_base_df.to_csv('predictions_base_model.csv', index=False)
    print("✅ Base model predictions saved to predictions_base_model.csv")
    
    # ==================== KEY CHANGE: PARTIAL UNFREEZING STRATEGY ====================
    print("\n✅ Unfreezing the FC head AND the top layers of encoders for fine-tuning...")
    
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    if hasattr(model.protein_encoder, 'transformer') and model.protein_encoder.transformer.num_layers > 0:
        last_layer_index = model.protein_encoder.transformer.num_layers - 1
        for param in model.protein_encoder.transformer.layers[last_layer_index].parameters():
            param.requires_grad = True
        print("  - Unfroze final protein transformer layer.")

    for param in model.ligand_encoder.attentive_fp.mol_gru.parameters():
        param.requires_grad = True
    for param in model.ligand_encoder.attentive_fp.mol_conv.parameters():
        param.requires_grad = True
    print("  - Unfroze final ligand GNN layers (mol_gru, mol_conv).")
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=GCPConfig.LEARNING_RATE
    )

    # ADDITION: Verify which parameters are trainable
    print("\n--- Verifying Trainable Parameters ---")
    num_trainable = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  -> Training: {name}")
            num_trainable += param.numel()
    print(f"Total trainable parameters: {num_trainable:,}")
    # ==============================================================================
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    early_stopper = EarlyStopping(patience=GCPConfig.PATIENCE)

    # --- Fine-Tuning Loop ---
    print("\n--- Starting Fine-Tuning (Partial Unfreeze on logIC50 scale) ---")
    log_path = "finetune_metrics.csv"
    
    with open(log_path, "w", newline='') as f:
        writer = csv.writer(f)
        # FIX: Updated header for CSV logging
        writer.writerow(["Epoch", "Train_MAE", "Val_MAE", "Val_R2_CoD", "Val_R2_Corr", "LR"])
    
    # FIX: Initialize lists for all metrics
    train_maes, val_maes, val_r2_cods, val_r2_corrs = [], [], [], []
    best_val_loss = float("inf")
    best_val_preds, best_val_targets = None, None
    
    for epoch in range(GCPConfig.MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, GCPConfig.DEVICE)
        # FIX: Unpack all 5 return values from validation
        val_loss, val_r2_cod, val_r2_corr, val_preds_at_epoch, val_targets_at_epoch = evaluate(model, val_loader, criterion, GCPConfig.DEVICE, desc="Validation")
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:03d} -> Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f} | Val R²(CoD): {val_r2_cod:.4f} | Val r²(Corr): {val_r2_corr:.4f} | LR: {current_lr:.1e}")
        
        # FIX: Append all collected metrics to lists
        train_maes.append(train_loss)
        val_maes.append(val_loss)
        val_r2_cods.append(val_r2_cod)
        val_r2_corrs.append(val_r2_corr)

        with open(log_path, "a", newline='') as f:
            writer = csv.writer(f)
            # FIX: Write all new metrics to CSV
            writer.writerow([epoch+1, train_loss, val_loss, val_r2_cod, val_r2_corr, current_lr])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_preds, best_val_targets = val_preds_at_epoch, val_targets_at_epoch
            torch.save(model.state_dict(), "best_finetuned_model.pt")
            print(f"  -> New best model saved with validation MAE: {best_val_loss:.4f}")

        scheduler.step(val_loss)
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("✅ Early stopping triggered.")
            break

    # --- Final Evaluation ---
    print("\n--- Evaluating Fine-Tuned Model on Test Set (logIC50 scale) ---")
    model.load_state_dict(torch.load("best_finetuned_model.pt"))
    # FIX: Unpack all 5 return values
    test_loss_tuned, test_r2_cod_tuned, test_r2_corr_tuned, tuned_preds, tuned_targets = evaluate(model, test_loader, criterion, GCPConfig.DEVICE, desc="Fine-Tuned Test")

    # FIX: Print summary now uses the correctly named and populated variables
    print("\n--- Final Performance Comparison on TEST SET (logIC50 scale) ---")
    print(f"  Metric                   |  Base Model (Zero-Shot) |  Fine-Tuned Model  |  Improvement")
    print(f"  -------------------------|-------------------------|--------------------|---------------")
    print(f"  MAE                      |  {test_loss_base:.4f}                   |  {test_loss_tuned:.4f}           |  {(test_loss_base - test_loss_tuned):.4f}")
    print(f"  R² (Coeff of Det.)       |  {test_r2_cod_base:.4f}                  |  {test_r2_cod_tuned:.4f}            |  {(test_r2_cod_tuned - test_r2_cod_base):.4f}")
    print(f"  r² (Sq. Correlation)     |  {test_r2_corr_base:.4f}                  |  {test_r2_corr_tuned:.4f}            |  {(test_r2_corr_tuned - test_r2_corr_base):.4f}")

    submission_tuned_df = test_df[['SMILES', 'PROTEIN_SEQ']].copy()
    submission_tuned_df['logIC50_actual'] = tuned_targets.numpy()
    submission_tuned_df['logIC50_predicted_tuned'] = tuned_preds.numpy()
    submission_tuned_df['pIC50_actual'] = 9.0 - tuned_targets.numpy()
    submission_tuned_df['pIC50_predicted_tuned'] = 9.0 - tuned_preds.numpy()
    submission_tuned_df.to_csv('predictions_finetuned_model.csv', index=False)
    print("\n✅ Fine-tuned model predictions (in logIC50 and pIC50 scale) saved to predictions_finetuned_model.csv")

    # --- Save Artifacts ---
    print("\nGenerating and saving plots and artifacts...")
    plot_actual_vs_predicted(base_targets.numpy(), base_preds.numpy(), "test_set_base_model_predictions.png", "Test Set - Base Model")
    plot_actual_vs_predicted(tuned_targets.numpy(), tuned_preds.numpy(), "test_set_tuned_model_predictions.png", "Test Set - Fine-Tuned Model")
    if best_val_preds is not None:
        plot_actual_vs_predicted(best_val_targets.numpy(), best_val_preds.numpy(), "validation_set_best_epoch.png", "Validation Set (Best Epoch)")

    # FIX: Call plotting function with the correct metric lists
    if train_maes: # only plot if training happened
        plot_metrics(train_maes, val_maes, val_r2_cods, val_r2_corrs, "finetune_learning_curves.png")

    # Save all relevant files to GCS
    save_to_gcs("predictions_base_model.csv", f"{GCPConfig.MODEL_PREFIX}/predictions_base_model.csv")
    save_to_gcs("predictions_finetuned_model.csv", f"{GCPConfig.MODEL_PREFIX}/predictions_finetuned_model.csv")
    save_to_gcs(log_path, f"{GCPConfig.MODEL_PREFIX}/{os.path.basename(log_path)}")
    save_to_gcs("best_finetuned_model.pt", f"{GCPConfig.MODEL_PREFIX}/best_finetuned_model.pt")

    print("\n✨ Process finished successfully.")


if __name__ == "__main__":
    main()