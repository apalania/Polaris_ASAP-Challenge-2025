"""
XGBoost Model Training Pipeline with STRATIFIED Split Strategies
==================================================================
Implements:
1. Stratified Random Split - Random split per protein target
2. Stratified Scaffold Split - Scaffold split per protein target

Changes from original:
- Added stratified_random_split() function
- Added stratified_scaffold_split() function
- Updated get_split_data() to use new functions
- Modified Config.SPLIT_STRATEGIES options
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle
import warnings
import os
import random
import json
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    SEED = 42
    TRAIN_DATA = 'polaris_train.csv'
    TEST_DATA = 'polaris_unblinded_test.csv'
    SPLIT_JSON = 'enhanced_scaffold_split_indices.json'  # Use existing JSON from neural network
    DOCKING_TRAIN = 'train_docking_scores.csv'
    DOCKING_TEST = 'test_docking_scores.csv'
    
    # UPDATED: Split strategy options
    # 'target_specific_random' = Random split per protein target
    # 'target_specific_scaffold' = Scaffold split per protein target
    # 'random' = Global random split
    # 'scaffold' = Global scaffold split
    SPLIT_STRATEGIES = ['target_specific_scaffold', 'target_specific_random']
    
    TRAIN_SIZE = 0.9  # Train/val split ratio
    
    # XGBoost parameters
    XGB_PARAMS = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_jobs': 1,
        'tree_method': 'exact',
        'random_state': SEED
    }
    
    # Output directories
    OUTPUT_DIR = 'target_specific_xgb_models'
    PLOTS_DIR = 'target_specific_xgb_plots'
    RESULTS_FILE = 'target_specific_xgb_results.csv'

# Set random seeds
np.random.seed(Config.SEED)
random.seed(Config.SEED)

# ============================================================================
# PLOTTING FUNCTIONS (unchanged from original)
# ============================================================================
def plot_feature_importance(model, config_id, descriptor_names, top_n=20):
    """Plot and save feature importance."""
    print(f"      → Generating feature importance plot...")
    
    importance_dict = model.get_booster().get_score(importance_type='weight')
    
    if not importance_dict:
        print(f"      ⚠️  No feature importance available")
        return
    
    feature_names = descriptor_names.copy()
    if len(feature_names) < len(importance_dict):
        feature_names.extend(['docking_score', 'has_docking'])
    
    importance_list = []
    for i, fname in enumerate(feature_names):
        feat_key = f'f{i}'
        if feat_key in importance_dict:
            importance_list.append({
                'feature': fname,
                'importance': importance_dict[feat_key]
            })
    
    if not importance_list:
        print(f"      ⚠️  Could not extract feature importance")
        return
    
    df_importance = pd.DataFrame(importance_list)
    df_importance = df_importance.sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(df_importance)), df_importance['importance'].values, color='steelblue')
    plt.yticks(range(len(df_importance)), df_importance['feature'].values)
    plt.xlabel('Feature Importance (Weight)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances - {config_id}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path_png = os.path.join(Config.PLOTS_DIR, f'{config_id}_feature_importance.png')
    plot_path_tif = os.path.join(Config.PLOTS_DIR, f'{config_id}_feature_importance.tif')
    plt.savefig(plot_path_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(plot_path_tif, dpi=600, format='tiff', bbox_inches='tight')
    plt.close()
    
    print(f"      ✅ Feature importance saved: {plot_path_png}")


def plot_actual_vs_predicted(y_true, y_pred, config_id, dataset_name='Test'):
    """Plot actual vs predicted values."""
    print(f"      → Generating actual vs predicted plot ({dataset_name})...")
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, color='teal', s=30, edgecolors='black', linewidth=0.5)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    textstr = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    plt.xlabel('Actual pIC50', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted pIC50', fontsize=12, fontweight='bold')
    plt.title(f'Actual vs Predicted - {config_id} ({dataset_name})', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path_png = os.path.join(Config.PLOTS_DIR, f'{config_id}_actual_vs_predicted_{dataset_name.lower()}.png')
    plot_path_tif = os.path.join(Config.PLOTS_DIR, f'{config_id}_actual_vs_predicted_{dataset_name.lower()}.tif')
    plt.savefig(plot_path_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(plot_path_tif, dpi=600, format='tiff', bbox_inches='tight')
    plt.close()
    
    print(f"      ✅ Actual vs predicted saved: {plot_path_png}")


def plot_residuals_density(y_true, y_pred, config_id, dataset_name='Test'):
    """Plot residuals density distribution."""
    print(f"      → Generating residuals density plot ({dataset_name})...")
    
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(residuals, fill=True, color='orange', linewidth=2, alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
    
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    textstr = f'Mean: {mean_res:.4f}\nStd Dev: {std_res:.4f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    plt.text(0.75, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    plt.xlabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    plt.ylabel('Density', fontsize=12, fontweight='bold')
    plt.title(f'Residuals Distribution - {config_id} ({dataset_name})', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path_png = os.path.join(Config.PLOTS_DIR, f'{config_id}_residuals_density_{dataset_name.lower()}.png')
    plot_path_tif = os.path.join(Config.PLOTS_DIR, f'{config_id}_residuals_density_{dataset_name.lower()}.tif')
    plt.savefig(plot_path_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(plot_path_tif, dpi=600, format='tiff', bbox_inches='tight')
    plt.close()
    
    print(f"      ✅ Residuals density saved: {plot_path_png}")


def plot_residuals_scatter(y_true, y_pred, config_id, dataset_name='Test'):
    """Plot residuals scatter plot."""
    print(f"      → Generating residuals scatter plot ({dataset_name})...")
    
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color='purple', s=30, edgecolors='black', linewidth=0.5)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
    
    plt.xlabel('Predicted pIC50', fontsize=12, fontweight='bold')
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    plt.title(f'Residuals vs Predicted - {config_id} ({dataset_name})', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path_png = os.path.join(Config.PLOTS_DIR, f'{config_id}_residuals_scatter_{dataset_name.lower()}.png')
    plot_path_tif = os.path.join(Config.PLOTS_DIR, f'{config_id}_residuals_scatter_{dataset_name.lower()}.tif')
    plt.savefig(plot_path_png, dpi=600, format='png', bbox_inches='tight')
    plt.savefig(plot_path_tif, dpi=600, format='tiff', bbox_inches='tight')
    plt.close()
    
    print(f"      ✅ Residuals scatter saved: {plot_path_png}")


def plot_all_performance(model, y_val, pred_val, y_test, pred_test, config_id, descriptor_names):
    """Generate all performance plots for a model."""
    print(f"\n      📊 Generating Performance Plots for {config_id}...")
    
    plot_feature_importance(model, config_id, descriptor_names, top_n=20)
    
    plot_actual_vs_predicted(y_val, pred_val, config_id, dataset_name='Val')
    plot_residuals_density(y_val, pred_val, config_id, dataset_name='Val')
    plot_residuals_scatter(y_val, pred_val, config_id, dataset_name='Val')
    
    plot_actual_vs_predicted(y_test, pred_test, config_id, dataset_name='Test')
    plot_residuals_density(y_test, pred_test, config_id, dataset_name='Test')
    plot_residuals_scatter(y_test, pred_test, config_id, dataset_name='Test')


# ============================================================================
# SCAFFOLD HELPER FUNCTION
# ============================================================================
def generate_scaffold(smiles):
    """Generate Murcko scaffold for a molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except:
        return None


# ============================================================================
# TARGET-SPECIFIC RANDOM SPLIT
# ============================================================================
def target_specific_random_split(df, smiles_col='ligand_smiles', protein_col='protein_sequence',
                                  train_size=0.9, seed=42, return_indices=False):
    """
    Target-Specific Random Split - performs random split SEPARATELY for each protein target.
    Ensures each target gets proportional representation in train/val.
    
    This is the same as "stratified random split" - splitting per target.
    """
    print(f"  🎯 Performing TARGET-SPECIFIC RANDOM split (train_size={train_size})...")
    
    if protein_col not in df.columns:
        raise ValueError(f"Column '{protein_col}' not found in dataframe")
    
    # Reset index to ensure we track original positions
    df_reset = df.reset_index(drop=True)
    
    unique_proteins = df_reset[protein_col].unique()
    print(f"     Found {len(unique_proteins)} unique protein targets")
    
    train_dfs, val_dfs = [], []
    train_indices_all, val_indices_all = [], []
    
    for i, protein in enumerate(unique_proteins, 1):
        protein_mask = df_reset[protein_col] == protein
        protein_df = df_reset[protein_mask].copy()
        protein_original_indices = df_reset[protein_mask].index.tolist()
        
        if len(protein_df) < 2:
            print(f"     [{i}/{len(unique_proteins)}] {protein[:30]}... → {len(protein_df)} samples (all to train)")
            train_dfs.append(protein_df)
            train_indices_all.extend(protein_original_indices)
            continue
        
        # Random split for this protein target
        train_p, val_p = train_test_split(
            protein_df,
            train_size=train_size,
            random_state=seed,
            shuffle=True
        )
        
        # Track original indices
        train_mask = df_reset.index.isin(train_p.index)
        val_mask = df_reset.index.isin(val_p.index)
        
        train_idx_protein = df_reset[train_mask & protein_mask].index.tolist()
        val_idx_protein = df_reset[val_mask & protein_mask].index.tolist()
        
        train_dfs.append(train_p)
        val_dfs.append(val_p)
        train_indices_all.extend(train_idx_protein)
        val_indices_all.extend(val_idx_protein)
        
        if i <= 5 or i % 10 == 0:  # Show first 5 and every 10th
            print(f"     [{i}/{len(unique_proteins)}] {protein[:30]}... → Train: {len(train_p)}, Val: {len(val_p)}")
    
    # Concatenate all splits
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    
    # Clean after splitting
    initial_train = len(train_df)
    initial_val = len(val_df)
    
    train_df.dropna(subset=['ligand_smiles', 'protein_sequence', 'pIC50'], inplace=True)
    val_df.dropna(subset=['ligand_smiles', 'protein_sequence', 'pIC50'], inplace=True)
    
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    
    print(f"  ✅ Target-specific random split completed and cleaned:")
    print(f"     Train: {initial_train} → {len(train_df)} samples")
    print(f"     Val: {initial_val} → {len(val_df)} samples")
    
    if return_indices:
        return train_df, val_df, train_indices_all, val_indices_all
    return train_df, val_df


# ============================================================================
# TARGET-SPECIFIC SCAFFOLD SPLIT
# ============================================================================
def target_specific_scaffold_split(df, smiles_col='ligand_smiles', protein_col='protein_sequence',
                                   train_size=0.9, seed=42, return_indices=False):
    """
    Target-Specific Scaffold Split - performs scaffold split SEPARATELY for each protein target.
    Ensures no scaffold leakage within each target AND maintains target representation.
    
    This is the same as "stratified scaffold split" - splitting scaffolds per target.
    """
    print(f"  🔬 Performing TARGET-SPECIFIC SCAFFOLD split (train_size={train_size})...")
    
    if protein_col not in df.columns:
        raise ValueError(f"Column '{protein_col}' not found in dataframe")
    
    # Reset index to ensure we track original positions
    df_reset = df.reset_index(drop=True)
    
    unique_proteins = df_reset[protein_col].unique()
    print(f"     Found {len(unique_proteins)} unique protein targets")
    
    train_dfs, val_dfs = [], []
    train_indices_all, val_indices_all = [], []
    
    for i, protein in enumerate(unique_proteins, 1):
        protein_mask = df_reset[protein_col] == protein
        protein_df = df_reset[protein_mask].copy()
        protein_original_indices = df_reset[protein_mask].index.tolist()
        
        if len(protein_df) < 2:
            print(f"     [{i}/{len(unique_proteins)}] {protein[:30]}... → {len(protein_df)} samples (all to train)")
            train_dfs.append(protein_df)
            train_indices_all.extend(protein_original_indices)
            continue
        
        # Scaffold split for THIS protein target
        scaffolds = defaultdict(list)
        
        for idx, row in protein_df.iterrows():
            scaffold = generate_scaffold(row[smiles_col])
            if scaffold:
                scaffolds[scaffold].append(idx)
            else:
                scaffolds['_invalid_'].append(idx)
        
        # Sort scaffolds by size and assign to train/val
        scaffold_sets = sorted(list(scaffolds.values()), key=len, reverse=True)
        train_idx, val_idx = [], []
        train_target = int(train_size * len(protein_df))
        
        for s_set in scaffold_sets:
            if len(train_idx) < train_target:
                train_idx.extend(s_set)
            else:
                val_idx.extend(s_set)
        
        # Ensure we have validation data
        if not val_idx and len(train_idx) > 1:
            split_point = int(train_size * len(train_idx))
            val_idx = train_idx[split_point:]
            train_idx = train_idx[:split_point]
        
        train_p = protein_df.loc[train_idx].copy()
        val_p = protein_df.loc[val_idx].copy() if val_idx else pd.DataFrame()
        
        # Map back to original dataframe indices
        train_idx_original = [protein_original_indices[protein_df.index.get_loc(idx)] for idx in train_idx]
        val_idx_original = [protein_original_indices[protein_df.index.get_loc(idx)] for idx in val_idx] if val_idx else []
        
        train_dfs.append(train_p)
        if not val_p.empty:
            val_dfs.append(val_p)
        
        train_indices_all.extend(train_idx_original)
        val_indices_all.extend(val_idx_original)
        
        if i <= 5 or i % 10 == 0:  # Show first 5 and every 10th
            print(f"     [{i}/{len(unique_proteins)}] {protein[:30]}... → "
                  f"Scaffolds: {len(scaffolds)}, Train: {len(train_p)}, Val: {len(val_p)}")
    
    # Concatenate all splits
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    
    # Clean after splitting
    initial_train = len(train_df)
    initial_val = len(val_df)
    
    train_df.dropna(subset=['ligand_smiles', 'protein_sequence', 'pIC50'], inplace=True)
    val_df.dropna(subset=['ligand_smiles', 'protein_sequence', 'pIC50'], inplace=True)
    
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    
    print(f"  ✅ Target-specific scaffold split completed and cleaned:")
    print(f"     Train: {initial_train} → {len(train_df)} samples")
    print(f"     Val: {initial_val} → {len(val_df)} samples")
    
    if return_indices:
        return train_df, val_df, train_indices_all, val_indices_all
    return train_df, val_df


# ============================================================================
# ORIGINAL SPLIT FUNCTIONS (for backward compatibility)
# ============================================================================
def scaffold_split(df, smiles_col='ligand_smiles', train_size=0.9, seed=42):
    """Global scaffold-based split (original version)."""
    print(f"  🔬 Performing GLOBAL scaffold split (train_size={train_size})...")
    random.seed(seed)
    np.random.seed(seed)
    
    df = df.reset_index(drop=True)
    scaffolds = defaultdict(list)
    
    for idx, row in df.iterrows():
        scaffold = generate_scaffold(row[smiles_col])
        if scaffold:
            scaffolds[scaffold].append(idx)
        else:
            scaffolds['_invalid_'].append(idx)
    
    scaffold_sets = sorted(list(scaffolds.values()), key=len, reverse=True)
    train_idx, val_idx = [], []
    train_target = int(train_size * len(df))
    
    for s_set in scaffold_sets:
        if len(train_idx) < train_target:
            train_idx.extend(s_set)
        else:
            val_idx.extend(s_set)
    
    if not val_idx:
        split_point = int(train_size * len(train_idx))
        val_idx = train_idx[split_point:]
        train_idx = train_idx[:split_point]
    
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)
    
    initial_train = len(train_df)
    initial_val = len(val_df)
    
    train_df.dropna(subset=['ligand_smiles', 'protein_sequence', 'pIC50'], inplace=True)
    val_df.dropna(subset=['ligand_smiles', 'protein_sequence', 'pIC50'], inplace=True)
    
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    
    print(f"  ✅ Global scaffold split completed:")
    print(f"     Train: {initial_train} → {len(train_df)}, Val: {initial_val} → {len(val_df)}")
    print(f"     Unique scaffolds: {len(scaffolds)}")
    
    return train_df, val_df


def random_split(df, train_size=0.9, seed=42):
    """Global random split (original version)."""
    print(f"  🎲 Performing GLOBAL random split (train_size={train_size})...")
    
    train_df, val_df = train_test_split(
        df, 
        train_size=train_size, 
        random_state=seed,
        shuffle=True
    )
    
    initial_train = len(train_df)
    initial_val = len(val_df)
    
    train_df.dropna(subset=['ligand_smiles', 'protein_sequence', 'pIC50'], inplace=True)
    val_df.dropna(subset=['ligand_smiles', 'protein_sequence', 'pIC50'], inplace=True)
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    print(f"  ✅ Global random split completed:")
    print(f"     Train: {initial_train} → {len(train_df)}, Val: {initial_val} → {len(val_df)}")
    
    return train_df, val_df


# ============================================================================
# JSON LOADING FOR TARGET-SPECIFIC SPLITS
# ============================================================================
def load_split_indices_from_json(df, json_path):
    """
    Load train/val split from JSON file (created by neural network or previous XGBoost run).
    This ensures XGBoost uses the EXACT same split as the neural network models.
    """
    print(f"  📥 Loading split indices from {json_path}...")
    
    try:
        with open(json_path, 'r') as f:
            indices = json.load(f)
        
        # Support multiple JSON formats
        train_idx = indices.get('train_indices', indices.get('train'))
        val_idx = indices.get('val_indices', indices.get('val'))
        
        if train_idx is None or val_idx is None:
            raise ValueError("JSON file must contain 'train_indices' and 'val_indices' keys.")
        
        max_idx = max(max(train_idx), max(val_idx))
        if max_idx >= len(df):
            print(f"  ⚠️  Index mismatch! JSON max index={max_idx}, but DataFrame has only {len(df)} rows")
            print(f"  ⚠️  This usually means the DataFrame was modified after indices were created")
            return None, None
        
        # Split using indices from JSON
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = df.iloc[val_idx].copy().reset_index(drop=True)
        
        # Clean the split data
        initial_train = len(train_df)
        initial_val = len(val_df)
        
        train_df.dropna(subset=['ligand_smiles', 'protein_sequence', 'pIC50'], inplace=True)
        val_df.dropna(subset=['ligand_smiles', 'protein_sequence', 'pIC50'], inplace=True)
        
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        
        print(f"  ✅ Split loaded from JSON and cleaned:")
        print(f"     Train: {initial_train} → {len(train_df)} samples")
        print(f"     Val: {initial_val} → {len(val_df)} samples")
        print(f"  🔗 Using SAME split as neural network model!")
        
        return train_df, val_df
    
    except Exception as e:
        print(f"  ⚠️  Error loading split indices: {e}")
        return None, None


def save_split_indices(train_indices, val_indices, strategy_name, seed=42, 
                       smiles_col='ligand_smiles', protein_col='protein_sequence'):
    """
    Save train/val split indices to JSON file for reproducibility.
    Creates files like: target_specific_scaffold_split_indices.json
    
    Args:
        train_indices: List of indices for training set
        val_indices: List of indices for validation set
        strategy_name: Name of the split strategy
        seed: Random seed used
        smiles_col: Name of SMILES column
        protein_col: Name of protein sequence column
    """
    json_filename = f'{strategy_name}_split_indices.json'
    
    try:
        split_info = {
            'train_indices': train_indices,
            'val_indices': val_indices,
            'seed': seed,
            'strategy': strategy_name,
            'smiles_col': smiles_col,
            'protein_col': protein_col,
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'total_size': len(train_indices) + len(val_indices)
        }
        
        with open(json_filename, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"  💾 Split indices saved to: {json_filename}")
        print(f"     Train: {len(train_indices)} indices")
        print(f"     Val: {len(val_indices)} indices")
        
    except Exception as e:
        print(f"  ⚠️  Could not save split indices: {e}")


# ============================================================================
# UPDATED: GET SPLIT DATA FUNCTION
# ============================================================================
def get_split_data(df, split_strategy, json_path=None, save_indices=True):
    """
    Get train/val split based on strategy.
    
    Args:
        df: Input dataframe
        split_strategy: One of 'target_specific_random', 'target_specific_scaffold', 'random', 'scaffold'
        json_path: Path to JSON file with pre-computed indices (optional)
        save_indices: Whether to save indices to JSON for reproducibility
    
    Returns:
        train_df, val_df: Split dataframes
    """
    
    # Try to load from JSON first if provided
    if json_path and os.path.exists(json_path):
        print(f"  🔍 JSON file found: {json_path}")
        train_df, val_df = load_split_indices_from_json(df, json_path)
        if train_df is not None and val_df is not None:
            return train_df, val_df
        print(f"  ⚠️  JSON loading failed, generating new split...")
    
    # Generate new splits WITH index tracking
    train_indices, val_indices = None, None
    
    if split_strategy == 'target_specific_random':
        train_df, val_df, train_indices, val_indices = target_specific_random_split(
            df, train_size=Config.TRAIN_SIZE, seed=Config.SEED, return_indices=True
        )
    
    elif split_strategy == 'target_specific_scaffold':
        train_df, val_df, train_indices, val_indices = target_specific_scaffold_split(
            df, train_size=Config.TRAIN_SIZE, seed=Config.SEED, return_indices=True
        )
    
    elif split_strategy == 'scaffold':
        train_df, val_df = scaffold_split(df, train_size=Config.TRAIN_SIZE, seed=Config.SEED)
    
    elif split_strategy == 'random':
        train_df, val_df = random_split(df, train_size=Config.TRAIN_SIZE, seed=Config.SEED)
    
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}. "
                        f"Options: 'target_specific_random', 'target_specific_scaffold', 'random', 'scaffold'")
    
    # Save indices for reproducibility (only for target-specific splits)
    if save_indices and train_indices is not None and val_indices is not None:
        save_split_indices(train_indices, val_indices, split_strategy, Config.SEED)
    
    return train_df, val_df


# ============================================================================
# FEATURE EXTRACTION (unchanged)
# ============================================================================
def compute_molecular_descriptors(smiles, protein_seq, descriptor_names, docking_dict=None):
    """Compute molecular descriptors for a SMILES string."""
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
        print(f"  ⚠️  Could not identify columns in {filepath}")
        return {}
    
    docking_map = {}
    for _, row in df.iterrows():
        score = row[score_col]
        if np.isfinite(score):
            docking_map[(row[smiles_col], row[protein_col])] = score
    
    print(f"  ✅ Loaded {len(docking_map)} docking scores from {filepath}")
    return docking_map


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    
    print("=" * 80)
    print("XGBoost Training Pipeline with Target-Specific Split Strategies")
    print("=" * 80)
    print(f"Split strategies: {', '.join(Config.SPLIT_STRATEGIES)}")
    print(f"  - target_specific_random: Random split per protein target")
    print(f"  - target_specific_scaffold: Scaffold split per protein target")
    
    # -----------------------------------------------------------------------
    # 1. LOAD DATA
    # -----------------------------------------------------------------------
    print("\n[1/6] Loading Data...")
    train_pool = pd.read_csv(Config.TRAIN_DATA)
    test_ext = pd.read_csv(Config.TEST_DATA)
    
    for df in [train_pool, test_ext]:
        df.columns = df.columns.str.strip()
        df.rename(columns={
            'SMILES': 'ligand_smiles',
            'canonical_smiles': 'ligand_smiles',
            'PROTEIN_SEQ': 'protein_sequence',
            'PROTEIN': 'protein_sequence',
            'TARGET': 'protein_sequence'
        }, inplace=True)
    
    print(f"  ✅ Train pool: {len(train_pool)} samples")
    print(f"  ✅ External test: {len(test_ext)} samples")
    
    # -----------------------------------------------------------------------
    # 2. CALCULATE pIC50
    # -----------------------------------------------------------------------
    print("\n[2/6] Calculating pIC50...")
    
    if 'pIC50' not in train_pool.columns:
        if 'IC50' in train_pool.columns:
            train_pool['IC50'] = pd.to_numeric(train_pool['IC50'], errors='coerce')
            mask = train_pool['IC50'] > 0
            train_pool.loc[mask, 'pIC50'] = -np.log10(train_pool.loc[mask, 'IC50'] * 1e-9)
            train_pool.loc[~mask, 'pIC50'] = np.nan
        elif 'logIC50' in train_pool.columns:
            train_pool['pIC50'] = 9.0 - train_pool['logIC50']
    
    if 'pIC50' not in test_ext.columns:
        if 'IC50' in test_ext.columns:
            test_ext['IC50'] = pd.to_numeric(test_ext['IC50'], errors='coerce')
            mask = test_ext['IC50'] > 0
            test_ext.loc[mask, 'pIC50'] = -np.log10(test_ext.loc[mask, 'IC50'] * 1e-9)
            test_ext.loc[~mask, 'pIC50'] = np.nan
        elif 'logIC50' in test_ext.columns:
            test_ext['pIC50'] = 9.0 - test_ext['logIC50']
    
    initial_test = len(test_ext)
    test_ext.dropna(subset=['ligand_smiles', 'protein_sequence', 'pIC50'], inplace=True)
    test_ext.reset_index(drop=True, inplace=True)
    print(f"  Test: {initial_test} → {len(test_ext)} samples (after cleaning)")
    print(f"  Train pool: {len(train_pool)} samples (will be split per strategy)")
    
    # -----------------------------------------------------------------------
    # 3. LOAD DOCKING SCORES
    # -----------------------------------------------------------------------
    print("\n[3/6] Loading Docking Scores...")
    docking_train = load_docking_scores(Config.DOCKING_TRAIN)
    docking_test = load_docking_scores(Config.DOCKING_TEST)
    
    # -----------------------------------------------------------------------
    # 4. TRAIN MODELS FOR ALL CONFIGURATIONS
    # -----------------------------------------------------------------------
    print("\n[4/6] Training XGBoost Models...")
    
    all_results = []
    descriptor_files = ['descriptors_55.txt', 'descriptors_25.txt']
    
    for split_strategy in Config.SPLIT_STRATEGIES:
        print(f"\n{'='*80}")
        print(f"SPLIT STRATEGY: {split_strategy.upper()}")
        print(f"{'='*80}")
        
        print(f"\n[Splitting Data - {split_strategy}]")
        
        # IMPORTANT: Use existing JSON for target_specific_scaffold
        json_to_use = None
        if split_strategy == 'target_specific_scaffold' and os.path.exists(Config.SPLIT_JSON):
            json_to_use = Config.SPLIT_JSON
            print(f"  📌 Attempting to use existing split from: {Config.SPLIT_JSON}")
            print(f"     (This ensures XGBoost uses SAME split as neural network)")
        
        train_df, val_df = get_split_data(train_pool, split_strategy, json_path=json_to_use)
        
        for desc_file in descriptor_files:
            if not os.path.exists(desc_file):
                print(f"  ⚠️  Skipping {desc_file} (not found)")
                continue
            
            desc_names = [line.strip() for line in open(desc_file) if line.strip()]
            print(f"\n  📋 Using {len(desc_names)} descriptors from {desc_file}")
            
            for use_docking in [False, True]:
                dock_label = 'dock' if use_docking else 'nodock'
                config_id = f"xgb_{split_strategy}_{len(desc_names)}desc_{dock_label}"
                
                print(f"\n    {'='*70}")
                print(f"    Training: {config_id}")
                print(f"    {'='*70}")
                
                dock_map_train = docking_train if use_docking else None
                dock_map_test = docking_test if use_docking else None
                
                print("      → Computing features...")
                X_train = np.array([
                    compute_molecular_descriptors(
                        row['ligand_smiles'], 
                        row['protein_sequence'], 
                        desc_names, 
                        dock_map_train
                    )
                    for _, row in train_df.iterrows()
                ])
                
                X_val = np.array([
                    compute_molecular_descriptors(
                        row['ligand_smiles'], 
                        row['protein_sequence'], 
                        desc_names, 
                        dock_map_train
                    )
                    for _, row in val_df.iterrows()
                ])
                
                X_test = np.array([
                    compute_molecular_descriptors(
                        row['ligand_smiles'], 
                        row['protein_sequence'], 
                        desc_names, 
                        dock_map_test
                    )
                    for _, row in test_ext.iterrows()
                ])
                
                y_train = train_df['pIC50'].values
                y_val = val_df['pIC50'].values
                y_test = test_ext['pIC50'].values
                
                print("      → Scaling features...")
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                
                scaler_path = os.path.join(Config.OUTPUT_DIR, f'scaler_{config_id}.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                print(f"      ✅ Saved scaler: {scaler_path}")
                
                print("      → Training XGBoost...")
                model = xgb.XGBRegressor(**Config.XGB_PARAMS)
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    verbose=False
                )
                
                # Save as both .json and .pth formats
                model_path_json = os.path.join(Config.OUTPUT_DIR, f'{config_id}.json')
                model_path_pth = os.path.join(Config.OUTPUT_DIR, f'{config_id}.pth')
                
                model.save_model(model_path_json)
                
                # Save as pickle (.pth) for compatibility
                with open(model_path_pth, 'wb') as f:
                    pickle.dump(model, f)
                
                print(f"      ✅ Saved model (JSON): {model_path_json}")
                print(f"      ✅ Saved model (PTH): {model_path_pth}")
                
                print("      → Evaluating...")
                pred_val = model.predict(X_val_scaled)
                pred_test = model.predict(X_test_scaled)
                
                mae_val = mean_absolute_error(y_val, pred_val)
                rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))
                r2_val = r2_score(y_val, pred_val)
                
                mae_test = mean_absolute_error(y_test, pred_test)
                rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
                r2_test = r2_score(y_test, pred_test)
                
                print(f"      📊 Val:  MAE={mae_val:.4f}, RMSE={rmse_val:.4f}, R²={r2_val:.4f}")
                print(f"      📊 Test: MAE={mae_test:.4f}, RMSE={rmse_test:.4f}, R²={r2_test:.4f}")
                
                # Save predictions - separate files for val and test
                pred_val_df = pd.DataFrame({
                    'y_true': y_val,
                    'y_pred': pred_val
                })
                pred_val_path = os.path.join(Config.OUTPUT_DIR, f'predictions_{config_id}_val.csv')
                pred_val_df.to_csv(pred_val_path, index=False)
                
                pred_test_df = pd.DataFrame({
                    'y_true': y_test,
                    'y_pred': pred_test
                })
                pred_test_path = os.path.join(Config.OUTPUT_DIR, f'predictions_{config_id}_test.csv')
                pred_test_df.to_csv(pred_test_path, index=False)
                
                print(f"      ✅ Saved predictions: {pred_val_path}")
                print(f"      ✅ Saved predictions: {pred_test_path}")
                
                plot_all_performance(model, y_val, pred_val, y_test, pred_test, config_id, desc_names)
                
                all_results.append({
                    'split_strategy': split_strategy,
                    'config': config_id,
                    'descriptors': len(desc_names),
                    'docking': dock_label,
                    'val_mae': mae_val,
                    'val_rmse': rmse_val,
                    'val_r2': r2_val,
                    'test_mae': mae_test,
                    'test_rmse': rmse_test,
                    'test_r2': r2_test
                })
    
    # -----------------------------------------------------------------------
    # 5. SAVE SUMMARY
    # -----------------------------------------------------------------------
    print(f"\n[5/6] Saving Summary...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(Config.RESULTS_FILE, index=False)
    print(f"  ✅ Results saved to: {Config.RESULTS_FILE}")
    
    # -----------------------------------------------------------------------
    # 6. DISPLAY RESULTS
    # -----------------------------------------------------------------------
    print("\n[6/6] Final Summary")
    print("=" * 80)
    print("TRAINING COMPLETE - SUMMARY BY SPLIT STRATEGY")
    print("=" * 80)
    
    for strategy in Config.SPLIT_STRATEGIES:
        strategy_results = results_df[results_df['split_strategy'] == strategy]
        print(f"\n{strategy.upper().replace('_', ' ')} SPLIT RESULTS:")
        print("-" * 80)
        print(strategy_results.sort_values('test_mae').to_string(index=False))
    
    # Compare target-specific vs global (if both available)
    if len(Config.SPLIT_STRATEGIES) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON: TARGET-SPECIFIC SPLITS")
        print("=" * 80)
        
        # Group by descriptor count and docking
        comparison_groups = results_df.groupby(['descriptors', 'docking'])
        
        for (desc_count, dock), group in comparison_groups:
            print(f"\n{desc_count} Descriptors, Docking={dock}:")
            print("-" * 60)
            for _, row in group.iterrows():
                print(f"  {row['split_strategy']:30s} | "
                      f"Val MAE: {row['val_mae']:.4f} | "
                      f"Test MAE: {row['test_mae']:.4f} | "
                      f"Test R²: {row['test_r2']:.4f}")
    
    print(f"\n📦 Saved Models Directory: {Config.OUTPUT_DIR}/")
    saved_files = sorted(os.listdir(Config.OUTPUT_DIR))
    for f in saved_files[:10]:
        print(f"  ✅ {f}")
    if len(saved_files) > 10:
        print(f"  ... and {len(saved_files) - 10} more files")
    
    print(f"\n📊 Generated Plots Directory: {Config.PLOTS_DIR}/")
    plot_files = sorted(os.listdir(Config.PLOTS_DIR))
    print(f"  ✅ Total plots generated: {len(plot_files)}")
    print(f"  ✅ Plots per model: 7 (feature importance + 6 performance plots)")
    
    print("\n" + "=" * 80)
    print("BEST MODEL BY TEST MAE:")
    print("=" * 80)
    best_model = results_df.loc[results_df['test_mae'].idxmin()]
    print(f"  Config: {best_model['config']}")
    print(f"  Split Strategy: {best_model['split_strategy']}")
    print(f"  Test MAE: {best_model['test_mae']:.4f}")
    print(f"  Test RMSE: {best_model['test_rmse']:.4f}")
    print(f"  Test R²: {best_model['test_r2']:.4f}")
    
    # Highlight differences between strategies
    if 'target_specific_random' in Config.SPLIT_STRATEGIES and 'target_specific_scaffold' in Config.SPLIT_STRATEGIES:
        print("\n" + "=" * 80)
        print("KEY INSIGHTS:")
        print("=" * 80)
        
        ts_rand = results_df[results_df['split_strategy'] == 'target_specific_random']
        ts_scaf = results_df[results_df['split_strategy'] == 'target_specific_scaffold']
        
        if not ts_rand.empty and not ts_scaf.empty:
            avg_mae_rand = ts_rand['test_mae'].mean()
            avg_mae_scaf = ts_scaf['test_mae'].mean()
            
            print(f"  Average Test MAE:")
            print(f"    Target-Specific Random:   {avg_mae_rand:.4f}")
            print(f"    Target-Specific Scaffold: {avg_mae_scaf:.4f}")
            print(f"    Difference:               {abs(avg_mae_rand - avg_mae_scaf):.4f}")
            
            if avg_mae_scaf > avg_mae_rand:
                diff_pct = ((avg_mae_scaf - avg_mae_rand) / avg_mae_rand) * 100
                print(f"\n  💡 Scaffold split is {diff_pct:.1f}% harder than random split")
                print(f"     This indicates scaffold generalization is more challenging!")
            else:
                print(f"\n  ⚠️  Unexpected: Random split appears harder than scaffold")
    
    print("\n✅ Training pipeline completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
