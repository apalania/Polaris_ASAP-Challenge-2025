# inference.py

"""
=================================================================================
 pIC50 Prediction Inference Script
=================================================================================
Description:
This script predicts pIC50 values for protein-ligand pairs using one of two
pre-trained models. It takes a CSV file as input, processes the protein
sequences and SMILES strings, and prints the predictions to the console.

---------------------------------------------------------------------------------
INSTRUCTIONS FOR USE
---------------------------------------------------------------------------------

1. Dependencies:
   - install PyTorch, Torch-Geometric, RDKit, Pandas, and TQDM.
   - RDKit: pip install rdkit-pypi
   - Torch-Geometric requires a special installation. See the official guide:
     https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

2. File Placement:
   - Place this script (inference.py) in the desired folder.
   - Place the model files (one_shot_partial_unfreezing.pt,
     gradual_partial_unfreezing.pt) and vocab.pkl file in the same folder.
   - Have the input data CSV file ready. It must have the columns
     'protein_sequence' and 'SMILES'.

3. Create the Input CSV (e.g., my_test_data.csv):
   create the input file that contains protein and ligand as shown below:
   protein_sequence,SMILES
   MTEITAAMVKELRESTGAGMMDCKNALSETNGDFDKAVQLLREKGLGKAAKKADRLAAEG,O=C(O)c1ccccc1C(=O)O
   VPSTGEISTATGLTEEKLIKSIVTSIFGK,c1ccccc1
   THISISNOTAREALPROTEINSEQUENCE,CC(=O)Oc1ccccc1C(=O)O
   ....

4. Run from the Command Line:
   - Open the terminal and navigate to the folder where the input csv file, models.pt files, vocab.pkl and inference.py are saved.
   - Run the script, providing the path to input CSV:

     $ python inference.py --input_csv my_test_data.csv

   - specify paths if the files are in different locations:

     $ python inference.py --input_csv path/to/data.csv --vocab_path path/to/vocab.pkl --model_dir path/to/models

---------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import pandas as pd
import argparse
import os
import pickle
from tqdm import tqdm

# --- Dependencies from training scripts ---
from torch_geometric.data import Data, Batch
from torch_geometric.nn import AttentiveFP
from rdkit import Chem

# =============================================================================
# SECTION 1: MODEL AND PREPROCESSING DEFINITIONS
#
# These definitions are copied directly from your training scripts to ensure
# the architecture and data handling are identical.
# =============================================================================

# --- Preprocessing Functions ---

def smiles_to_graph(smiles: str):
    """Converts a SMILES string to a PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features: Atomic Number, Degree, IsAromatic
    atom_features = [
        [atom.GetAtomicNum() / 100.0, atom.GetDegree() / 4.0, float(atom.GetIsAromatic())]
        for atom in mol.GetAtoms()
    ]
    # Edge features: Bond Type
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = [bond.GetBondTypeAsDouble() / 3.0]
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([feat, feat])

    return Data(
        x=torch.tensor(atom_features, dtype=torch.float),
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attrs, dtype=torch.float)
    )

# --- Model Component Definitions ---

class ProteinEncoder(nn.Module):
    """Protein Transformer Encoder."""
    def __init__(self, vocab_size, d_model=32, n_layers=1, n_heads=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1) # [B, D, L]
        return self.pool(x).squeeze(-1) # [B, D]

class LigandEncoder(nn.Module):
    """Ligand GNN Encoder."""
    def __init__(self):
        super().__init__()
        # Parameters hardcoded from your training scripts
        params = {
            'in_channels': 3, 'hidden_channels': 256, 'out_channels': 32,
            'edge_dim': 1, 'num_layers': 4, 'num_timesteps': 4, 'dropout': 0.25,
        }
        self.attentive_fp = AttentiveFP(
            in_channels=params['in_channels'],
            hidden_channels=params['hidden_channels'],
            out_channels=params['out_channels'],
            edge_dim=params['edge_dim'],
            num_layers=params['num_layers'],
            num_timesteps=params['num_timesteps'],
            dropout=params['dropout']
        )

    def forward(self, data):
        # The AttentiveFP model handles batching internally via the 'batch' attribute
        return self.attentive_fp(data.x, data.edge_index, data.edge_attr, data.batch)

class IC50Predictor(nn.Module):
    """The main model combining both encoders."""
    def __init__(self, vocab_size):
        super().__init__()
        self.protein_encoder = ProteinEncoder(vocab_size)
        self.ligand_encoder = LigandEncoder()
        # The final prediction head
        self.fc = nn.Sequential(
            nn.Linear(64, 32), # 32 from protein + 32 from ligand = 64
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, protein_seq, ligand_data):
        protein_emb = self.protein_encoder(protein_seq)
        ligand_emb = self.ligand_encoder(ligand_data)
        combined = torch.cat([protein_emb, ligand_emb], dim=-1)
        return self.fc(combined).squeeze(-1)

# =============================================================================
# SECTION 2: INFERENCE LOGIC
# =============================================================================

def predict(model, vocab, data_df, device):
    """
    Runs predictions for each row in the input DataFrame.
    """
    model.eval()
    results = []

    print("\n--- Starting Predictions ---")

    # The function to tokenize protein sequences using the loaded vocab
    def tokenize_protein(seq):
        return list(seq.upper())

    with torch.no_grad():
        for index, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Predicting"):
            protein_seq_str = row['protein_sequence']
            smiles_str = row['SMILES']

            # --- Preprocess Inputs ---
            # 1. Protein to Tensor
            protein_tokens = vocab(tokenize_protein(protein_seq_str))
            protein_tensor = torch.tensor(protein_tokens, dtype=torch.long).unsqueeze(0).to(device)

            # 2. SMILES to Graph
            ligand_graph = smiles_to_graph(smiles_str)
            if ligand_graph is None:
                print(f"\nWarning: Could not process SMILES at row {index+1}. Skipping.")
                results.append({'protein_sequence': protein_seq_str, 'SMILES': smiles_str, 'predicted_pIC50': 'ERROR'})
                continue
            
            # Use PyG's Batch to create a batch of 1, which the model expects
            ligand_batch = Batch.from_data_list([ligand_graph]).to(device)
            
            # --- Get Model Prediction ---
            # The model predicts logIC50
            logIC50_pred = model(protein_tensor, ligand_batch)

            # Convert logIC50 back to pIC50 (since logIC50 = 9.0 - pIC50)
            pIC50_pred = 9.0 - logIC50_pred.item()
            
            results.append({
                'protein_sequence': protein_seq_str,
                'SMILES': smiles_str,
                'predicted_pIC50': pIC50_pred
            })

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Predict pIC50 values for protein-ligand pairs.",
        formatter_class=argparse.RawTextHelpFormatter # To keep newlines in help text
    )
    parser.add_argument(
        '--input_csv', type=str, required=True,
        help="Path to the input CSV file.\nMust have 'protein_sequence' and 'SMILES' columns."
    )
    parser.add_argument(
        '--vocab_path', type=str, default='vocab.pkl',
        help="Path to the vocab.pkl file.\n(default: 'vocab.pkl' in the current directory)"
    )
    parser.add_argument(
        '--model_dir', type=str, default='.',
        help="Directory where the .pt model files are located.\n(default: current directory)"
    )
    args = parser.parse_args()

    # --- File and Directory Validation ---
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV not found at '{args.input_csv}'")
        return
    if not os.path.exists(args.vocab_path):
        print(f"Error: Vocabulary file not found at '{args.vocab_path}'")
        return
    if not os.path.isdir(args.model_dir):
        print(f"Error: Model directory not found at '{args.model_dir}'")
        return

    # --- User Choice for Model ---
    model_choice = ""
    while model_choice not in ['1', '2']:
        print("\nPlease choose which model to use for inference:")
        print("  1: One Shot Partial Unfreezing")
        print("  2: Gradual Partial Unfreezing")
        model_choice = input("Enter your choice (1 or 2): ")

    one_shot_name = 'one_shot_partial_unfreezing.pt'
    gradual_name = 'gradual_partial_unfreezing.pt' 

    if model_choice == '1':
        model_filename = one_shot_name
        model_path = os.path.join(args.model_dir, model_filename)
        print(f"\nSelected model: One Shot Partial Unfreezing")
    else:
        model_filename = gradual_name
        model_path = os.path.join(args.model_dir, model_filename)
        print(f"\nSelected model: Gradual Partial Unfreezing")

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_filename}' not found in directory '{args.model_dir}'.")
        return

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load vocab
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)
    print(f"Loaded vocabulary with {len(vocab)} tokens.")

    # Initialize model architecture
    model = IC50Predictor(vocab_size=len(vocab)).to(device)
    
    # Load the trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded model weights from '{model_path}'")

    # --- Data Loading ---
    try:
        data_df = pd.read_csv(args.input_csv)
        if 'protein_sequence' not in data_df.columns or 'SMILES' not in data_df.columns:
            print("Error: CSV file must contain 'protein_sequence' and 'SMILES' columns.")
            return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # --- Run Prediction ---
    predictions = predict(model, vocab, data_df, device)

    # --- Print Results ---
    print("\n--- Inference Complete ---")
    print("======================================================")
    print(f"Predicted pIC50 values using '{model_filename}':")
    print("======================================================")
    for i, res in enumerate(predictions):
        prot_short = (res['protein_sequence'][:30] + '...') if len(res['protein_sequence']) > 33 else res['protein_sequence']
        smiles_short = (res['SMILES'][:35] + '...') if len(res['SMILES']) > 38 else res['SMILES']
        
        print(f"\n--- Pair {i+1} ---")
        print(f"  Protein: {prot_short}")
        print(f"  SMILES : {smiles_short}")
        if isinstance(res['predicted_pIC50'], float):
            print(f"  >> Predicted pIC50: {res['predicted_pIC50']:.4f}")
        else:
            print(f"  >> Prediction failed: {res['predicted_pIC50']}")
    print("======================================================")


if __name__ == '__main__':
    main()