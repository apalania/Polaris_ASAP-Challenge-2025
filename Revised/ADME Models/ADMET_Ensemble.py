#!/usr/bin/env python3

"""
ENSEMBLE STRATEGIES GENERATOR & EVALUATOR
Implements 3 ensemble strategies with ground truth evaluation

TRANSFORMATION LOGIC (matches training):
- KSOL: clip_log10 → log10(y + 1)
- HLM, MLM, MDR1-MDCK2: log1p → np.log1p(y) = log(1 + y)
- LogD: none → no transformation
"""


import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "ensemble_strategies_output"
ENDPOINTS = ["HLM", "KSOL", "MLM", "LogD", "MDR1-MDCK2"]
GROUND_TRUTH_DIR = "unblind_data"

# Endpoint configuration - maps endpoint to its column name and transformation
ENDPOINT_CONFIG = {
    'HLM': {'target_column': 'HLM', 'transform': 'log1p'},  # np.log1p(y)
    'KSOL': {'target_column': 'KSOL', 'transform': 'clip_log10'},  # log10(y + 1)
    'MLM': {'target_column': 'MLM', 'transform': 'log1p'},  # np.log1p(y)
    'LogD': {'target_column': 'LogD', 'transform': 'none'},  # No transformation
    'MDR1-MDCK2': {'target_column': 'MDR1-MDCK2', 'transform': 'log1p'}  # np.log1p(y)
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def banner(text, char="="):
    """Print a formatted banner"""
    print("\n" + char * 80)
    print(f"# {text}")
    print(char * 80)

def create_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {path}")

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_ground_truth(endpoint):
    """Load and preprocess ground truth data for an endpoint"""
    gt_file = f"{GROUND_TRUTH_DIR}/{endpoint}_unblind.csv"
    
    if not os.path.exists(gt_file):
        print(f"✗ Ground truth file not found: {gt_file}")
        return None
    
    df = pd.read_csv(gt_file)
    print(f"✓ Loaded ground truth: {gt_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    # Standardize column name
    target_col = ENDPOINT_CONFIG[endpoint]['target_column']
    if target_col in df.columns:
        df = df.rename(columns={target_col: 'ground_truth'})
    else:
        print(f"✗ Column '{target_col}' not found in ground truth file")
        return None
    
    # Remove NaN values BEFORE transformation
    n_before = len(df)
    df = df.dropna(subset=['ground_truth'])
    n_after = len(df)
    n_dropped = n_before - n_after
    
    if n_dropped > 0:
        print(f"  ⚠️  Dropped {n_dropped} rows with NaN ground truth values")
        print(f"  Valid samples: {n_after}")
    
    # Apply transformation if needed
    transform = ENDPOINT_CONFIG[endpoint]['transform']
    if transform == 'clip_log10':
        print(f"  ℹ️  Transforming ground truth using CLIP_LOG10: log10(y + 1)")
        df['ground_truth'] = np.log10(df['ground_truth'] + 1)
    elif transform == 'log1p':
        print(f"  ℹ️  Transforming ground truth using LOG1P: np.log1p(y) = log(1 + y)")
        df['ground_truth'] = np.log1p(df['ground_truth'])
    else:
        print(f"  ℹ️  No transformation applied (using original scale)")
    
    return df[['SMILES', 'ground_truth']]

def load_metrics(endpoint):
    """Load holdout metrics for an endpoint"""
    metrics_file = f"results_{endpoint}/{endpoint}_HOLDOUT_LOG10.csv"
    
    if not os.path.exists(metrics_file):
        print(f"✗ Metrics file not found: {metrics_file}")
        return None
    
    df = pd.read_csv(metrics_file)
    print(f"✓ Loaded metrics: {metrics_file}")
    print(f"  Shape: {df.shape}")
    
    # Create unified 'config' column
    df['config'] = df['Dataset'] + '_' + df['Features'] + '_' + df['Strategy']
    
    # Standardize column names
    column_mapping = {
        'Seed': 'split_number',
        'MAE': 'mae',
        'MSE': 'mse',
        'R2': 'r2',
        'PEARSON_R': 'pearson_r',
        'SPEARMAN_R': 'spearman_r'
    }
    df = df.rename(columns=column_mapping)
    
    return df

def get_best_split_for_config(endpoint, config, metrics_df):
    """Find the best performing split for a given configuration"""
    config_data = metrics_df[metrics_df['config'] == config]
    if config_data.empty:
        return None, None
    
    best_idx = config_data['mae'].idxmin()
    best_split = int(config_data.loc[best_idx, 'split_number'])
    best_mae = float(config_data.loc[best_idx, 'mae'])
    
    return best_split, best_mae

def load_prediction(endpoint, config, split):
    """Load prediction file for a specific configuration and split"""
    
    # Parse config into components
    parts = config.split('_')
    dataset = parts[0]
    features = parts[1]
    strategy = '_'.join(parts[2:])
    
    # Search patterns for prediction files
    patterns = [
        f"predictions_{endpoint}_{dataset}_{features}_desc_{strategy}_split_{split}_test.csv",
        f"predictions_{endpoint}_{dataset}_{features}_{strategy}_split_{split}_test.csv",
        f"**/predictions*{endpoint}*{dataset}*{features}*{strategy}*split_{split}*.csv"
    ]
    
    pred_file = None
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            pred_file = matches[0]
            break
    
    if not pred_file:
        print(f"    ✗ Prediction file not found for: {config}, split {split}")
        return None
    
    df = pd.read_csv(pred_file)
    
    # Find and standardize prediction column
    pred_cols = [col for col in df.columns if 'Predicted' in col or 'prediction' in col.lower()]
    if not pred_cols:
        print(f"    ✗ No prediction column found in {pred_file}")
        return None
    
    df = df.rename(columns={pred_cols[0]: 'prediction'})
    
    # Apply transformation to match ground truth scale
    transform = ENDPOINT_CONFIG[endpoint]['transform']
    
    if transform == 'clip_log10':
        # Check for invalid values before transformation
        n_negative = (df['prediction'] < 0).sum()
        if n_negative > 0:
            print(f"    ⚠️  Warning: {n_negative} predictions are negative (will clip to 0)")
            df['prediction'] = df['prediction'].clip(lower=0)
        
        # Apply clip_log10 transformation: log10(y + 1)
        df['prediction'] = np.log10(df['prediction'] + 1)
        
    elif transform == 'log1p':
        # Check for invalid values before transformation
        n_negative = (df['prediction'] < -1).sum()
        if n_negative > 0:
            print(f"    ⚠️  Warning: {n_negative} predictions are < -1 (invalid for log1p)")
            df['prediction'] = df['prediction'].clip(lower=-1 + 1e-10)
        
        # Apply log1p transformation: np.log1p(y) = log(1 + y)
        df['prediction'] = np.log1p(df['prediction'])
    
    # Check for NaN after transformation
    n_nan = df['prediction'].isna().sum()
    if n_nan > 0:
        print(f"    ⚠️  Warning: {n_nan} NaN values after transformation")
    
    return df[['SMILES', 'prediction']]

# ============================================================================
# STRATEGY 1: Best Single Model
# ============================================================================

def strategy2_best_single_model(endpoint, metrics_df, ground_truth_df):
    """
    Strategy 1: Use the single best performing model
    
    Logic:
    - Identify the configuration + split with lowest MAE
    - Use only this model's predictions
    - Note: For production, should retrain on 100% of data
    """
    print(f"\n{'='*80}")
    print(f"STRATEGY 1: Best Single Model - {endpoint}")
    print(f"{'='*80}")
    
    # Find absolute best model
    best_row = metrics_df.loc[metrics_df['mae'].idxmin()]
    best_config = best_row['config']
    best_split = int(best_row['split_number'])
    best_mae = float(best_row['mae'])
    
    print(f"\n🏆 Best Overall Model Identified:")
    print(f"  Configuration: {best_config}")
    print(f"  Split: {best_split}")
    print(f"  MAE: {best_mae:.4f}")
    print(f"\n  💡 NOTE: For production deployment, retrain this model on 100% of training data")
    
    # Load predictions
    pred_df = load_prediction(endpoint, best_config, best_split)
    if pred_df is None:
        print(f"\n✗ Could not load predictions for best model")
        return None
    
    # Merge with ground truth
    result_df = pred_df.merge(ground_truth_df, on='SMILES', how='inner')
    
    print(f"\n✓ Strategy 1 complete: Using single best model")
    print(f"  Final dataset: {len(result_df)} compounds")
    
    return result_df

# ============================================================================
# STRATEGY 2: Top 3 Configs Ensemble
# ============================================================================

def strategy3_top3_ensemble(endpoint, metrics_df, ground_truth_df):
    """
    Strategy 2: Ensemble best split from each of top 3 configurations
    
    Logic:
    - Select top 3 configurations by average MAE (excluding DeepChem)
    - For each, use its best performing split
    - Ensemble using simple mean
    """
    print(f"\n{'='*80}")
    print(f"STRATEGY 2: Top 3 Configs Ensemble - {endpoint}")
    print(f"{'='*80}")
    
    # Get top 3 configs
    config_avg_mae = metrics_df.groupby('config')['mae'].mean().sort_values()
    top3_configs = [c for c in config_avg_mae.index 
                   if isinstance(c, str) and 'deepchem' not in c.lower()][:3]
    
    print("\n📊 Selected Top 3 Configurations:")
    predictions_list = []
    smiles_data = None
    
    for i, config in enumerate(top3_configs, 1):
        best_split, best_mae = get_best_split_for_config(endpoint, config, metrics_df)
        print(f"  {i}. {config}")
        print(f"     Best split: {best_split}, MAE: {best_mae:.4f}")
        
        pred_df = load_prediction(endpoint, config, best_split)
        if pred_df is not None:
            if smiles_data is None:
                smiles_data = pred_df['SMILES'].copy()
            predictions_list.append(pred_df['prediction'].values)
            print(f"     ✓ Loaded predictions")
    
    if not predictions_list:
        print(f"\n✗ No predictions loaded for Strategy 2")
        return None
    
    # Create ensemble using simple mean
    ensemble_pred = np.mean(predictions_list, axis=0)
    
    result_df = pd.DataFrame({
        'SMILES': smiles_data,
        'prediction': ensemble_pred
    })
    
    # Merge with ground truth
    result_df = result_df.merge(ground_truth_df, on='SMILES', how='inner')
    
    print(f"\n✓ Strategy 2 complete: {len(predictions_list)} models ensembled")
    print(f"  Final dataset: {len(result_df)} compounds")
    
    return result_df

# ============================================================================
# STRATEGY 3: Best from Model Groups
# ============================================================================

def strategy4_best_from_groups(endpoint, metrics_df, ground_truth_df):
    """
    Strategy 3: Ensemble best model from each data/feature group
    
    Logic:
    - Group models by dataset type (Polaris_55, Polaris_25, Augmented)
    - Select best model from each group
    - Ensemble using simple mean
    """
    print(f"\n{'='*80}")
    print(f"STRATEGY 3: Best from Model Groups - {endpoint}")
    print(f"{'='*80}")
    
    # Define groups
    groups = {
        'Polaris_55': [],
        'Polaris_25': [],
        'Augmented': []
    }
    
    # Categorize configs into groups
    for config in metrics_df['config'].unique():
        if not isinstance(config, str):
            continue  # Skip NaN or non-string values
        if 'Polaris_55' in config:
            groups['Polaris_55'].append(config)
        elif 'Polaris_25' in config:
            groups['Polaris_25'].append(config)
        elif 'Augmented' in config:
            groups['Augmented'].append(config)
    
    print("\n📊 Selecting Best Model from Each Group:")
    predictions_list = []
    smiles_data = None
    
    for group_name, configs in groups.items():
        if not configs:
            print(f"\n  {group_name}: No models found")
            continue
        
        print(f"\n  {group_name} Group:")
        print(f"    Evaluating {len(configs)} configurations...")
        
        # Find best model in this group
        group_metrics = metrics_df[metrics_df['config'].isin(configs)]
        best_row = group_metrics.loc[group_metrics['mae'].idxmin()]
        best_config = best_row['config']
        best_split = int(best_row['split_number'])
        best_mae = float(best_row['mae'])
        
        print(f"    Winner: {best_config}")
        print(f"    Split: {best_split}, MAE: {best_mae:.4f}")
        
        pred_df = load_prediction(endpoint, best_config, best_split)
        if pred_df is not None:
            if smiles_data is None:
                smiles_data = pred_df['SMILES'].copy()
            predictions_list.append(pred_df['prediction'].values)
            print(f"    ✓ Loaded predictions")
    
    if not predictions_list:
        print(f"\n✗ No predictions loaded for Strategy 3")
        return None
    
    # Create ensemble using simple mean
    ensemble_pred = np.mean(predictions_list, axis=0)
    
    result_df = pd.DataFrame({
        'SMILES': smiles_data,
        'prediction': ensemble_pred
    })
    
    # Merge with ground truth
    result_df = result_df.merge(ground_truth_df, on='SMILES', how='inner')
    
    print(f"\n✓ Strategy 3 complete: {len(predictions_list)} models ensembled")
    print(f"  Final dataset: {len(result_df)} compounds")
    
    return result_df

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    
    # Check for NaN values
    mask_valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    n_invalid = (~mask_valid).sum()
    
    if n_invalid > 0:
        print(f"    ⚠️  Warning: {n_invalid} invalid predictions detected, removing...")
        y_true = y_true[mask_valid]
        y_pred = y_pred[mask_valid]
    
    if len(y_true) == 0:
        print(f"    ✗ Error: No valid predictions remaining after filtering")
        return {
            'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan,
            'Pearson_R': np.nan, 'Pearson_p': np.nan,
            'Spearman_R': np.nan, 'Spearman_p': np.nan,
            'Kendall_Tau': np.nan, 'Kendall_p': np.nan,
            'N_samples': 0
        }
    
    # Basic metrics
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Correlations
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    kendall_tau, kendall_p = stats.kendalltau(y_true, y_pred)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Pearson_R': pearson_r,
        'Pearson_p': pearson_p,
        'Spearman_R': spearman_r,
        'Spearman_p': spearman_p,
        'Kendall_Tau': kendall_tau,
        'Kendall_p': kendall_p,
        'N_samples': len(y_true)
    }

def evaluate_strategies(strategies_results, endpoint):
    """Evaluate and compare all strategies"""
    
    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY - {endpoint}")
    print(f"{'='*80}")
    
    evaluation_results = []
    
    for strategy_name, result_df in strategies_results.items():
        if result_df is None or len(result_df) == 0:
            continue
        
        metrics = calculate_metrics(
            result_df['ground_truth'].values,
            result_df['prediction'].values
        )
        
        metrics['Strategy'] = strategy_name
        evaluation_results.append(metrics)
    
    if not evaluation_results:
        print("✗ No strategies to evaluate")
        return None
    
    # Create results DataFrame
    results_df = pd.DataFrame(evaluation_results)
    results_df = results_df.sort_values('MAE')
    
    # Display results
    print(f"\n📊 Performance Comparison (sorted by MAE):")
    print("-" * 110)
    print(f"{'Strategy':<30} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Pearson':>8} {'Spearman':>8} {'Kendall':>8} {'N':>6}")
    print("-" * 110)
    
    for _, row in results_df.iterrows():
        print(f"{row['Strategy']:<30} "
              f"{row['MAE']:>8.4f} "
              f"{row['RMSE']:>8.4f} "
              f"{row['R2']:>8.4f} "
              f"{row['Pearson_R']:>8.4f} "
              f"{row['Spearman_R']:>8.4f} "
              f"{row['Kendall_Tau']:>8.4f} "
              f"{int(row['N_samples']):>6}")
    
    # Identify best strategy
    best_strategy = results_df.iloc[0]
    print(f"\n🏆 BEST STRATEGY: {best_strategy['Strategy']}")
    print(f"   MAE: {best_strategy['MAE']:.4f}")
    print(f"   R²: {best_strategy['R2']:.4f}")
    
    return results_df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_scatter_plots(strategies_results, endpoint, output_dir):
    """Create scatter plots for all strategies"""
    
    n_strategies = len([s for s in strategies_results.values() if s is not None])
    if n_strategies == 0:
        return
    
    # Determine subplot layout based on number of strategies
    if n_strategies == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 7))
        axes = [axes]
    elif n_strategies == 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes = axes.flatten()
    else:  # 3 or more strategies
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
    
    fig.suptitle(f'{endpoint}: Prediction Strategies vs Ground Truth', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, (strategy_name, result_df) in enumerate(strategies_results.items()):
        if result_df is None:
            continue
        
        ax = axes[idx]
        
        y_true = result_df['ground_truth'].values
        y_pred = result_df['prediction'].values
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.4, s=40, edgecolors='none')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                'r--', lw=2, label='Perfect prediction', zorder=10)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Title with metrics
        ax.set_title(f"{strategy_name}\n"
                    f"MAE: {metrics['MAE']:.4f} | "
                    f"R²: {metrics['R2']:.4f} | "
                    f"Pearson: {metrics['Pearson_R']:.4f} | "
                    f"Kendall τ: {metrics['Kendall_Tau']:.4f}",
                    fontsize=10, pad=10)
        
        ax.set_xlabel('Ground Truth', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots if we have 3 strategies
    if n_strategies == 3:
        axes[3].set_visible(False)
    
    plt.tight_layout()
    output_file = f"{output_dir}/{endpoint}_strategy_scatter_plots.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()

def create_comparison_barplot(results_df, endpoint, output_dir):
    """Create bar plot comparing strategy performance"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{endpoint}: Strategy Performance Comparison', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: MAE and RMSE
    x = range(len(results_df))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], results_df['MAE'], width, 
            label='MAE', alpha=0.8, color='steelblue')
    ax1.bar([i + width/2 for i in x], results_df['RMSE'], width, 
            label='RMSE', alpha=0.8, color='coral')
    
    ax1.set_xlabel('Strategy', fontsize=11)
    ax1.set_ylabel('Error', fontsize=11)
    ax1.set_title('Error Metrics Comparison', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['Strategy'], rotation=15, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: R² and Correlations
    ax2.bar([i - width*1.5 for i in x], results_df['R2'], width, 
            label='R²', alpha=0.8, color='forestgreen')
    ax2.bar([i - width*0.5 for i in x], results_df['Pearson_R'], width, 
            label='Pearson R', alpha=0.8, color='purple')
    ax2.bar([i + width*0.5 for i in x], results_df['Spearman_R'], width, 
            label='Spearman R', alpha=0.8, color='orange')
    ax2.bar([i + width*1.5 for i in x], results_df['Kendall_Tau'], width, 
            label='Kendall Tau', alpha=0.8, color='steelblue')
    
    ax2.set_xlabel('Strategy', fontsize=11)
    ax2.set_ylabel('Correlation / R²', fontsize=11)
    ax2.set_title('Correlation Metrics Comparison', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df['Strategy'], rotation=15, ha='right', fontsize=9)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    output_file = f"{output_dir}/{endpoint}_strategy_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()

def create_residual_plots(strategies_results, endpoint, output_dir):
    """Create residual plots for all strategies"""
    
    n_strategies = len([s for s in strategies_results.values() if s is not None])
    if n_strategies == 0:
        return
    
    # Determine subplot layout based on number of strategies
    if n_strategies == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 7))
        axes = [axes]
    elif n_strategies == 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes = axes.flatten()
    else:  # 3 or more strategies
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
    
    fig.suptitle(f'{endpoint}: Residual Plots', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, (strategy_name, result_df) in enumerate(strategies_results.items()):
        if result_df is None:
            continue
        
        ax = axes[idx]
        
        y_true = result_df['ground_truth'].values
        y_pred = result_df['prediction'].values
        residuals = y_true - y_pred
        
        # Residual scatter plot
        ax.scatter(y_pred, residuals, alpha=0.4, s=40, edgecolors='none')
        ax.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero residual')
        
        # Add horizontal lines at ±1 MAE
        mae = np.mean(np.abs(residuals))
        ax.axhline(y=mae, color='orange', linestyle=':', lw=1.5, alpha=0.7, label=f'±1 MAE ({mae:.3f})')
        ax.axhline(y=-mae, color='orange', linestyle=':', lw=1.5, alpha=0.7)
        
        ax.set_title(f"{strategy_name}", fontsize=11, pad=10)
        ax.set_xlabel('Predicted Value', fontsize=10)
        ax.set_ylabel('Residual (True - Predicted)', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots if we have 3 strategies
    if n_strategies == 3:
        axes[3].set_visible(False)
    
    plt.tight_layout()
    output_file = f"{output_dir}/{endpoint}_residual_plots.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("="*80)
    print("SOPHISTICATED ENSEMBLE STRATEGIES GENERATOR & EVALUATOR")
    print("="*80)
    print("\n🎯 Three Advanced Ensemble Strategies:")
    print("  1. Best Single Model")
    print("  2. Top 3 Configs Ensemble")
    print("  3. Best from Model Groups (Polaris_55/25, Augmented)")
    print("\n📊 Evaluation:")
    print("  • KSOL: CLIP_LOG10 transformation → log10(y + 1)")
    print("  • HLM, MLM, MDR1-MDCK2: LOG1P transformation → np.log1p(y)")
    print("  • LogD: No transformation (original scale)")
    print("  • Matches training transformation exactly")
    print("  • Comprehensive metrics (MAE, RMSE, R², Pearson, Spearman, Kendall Tau)")
    print("  • Multiple visualization types")
    print("="*80)
    
    # Create output directory
    create_directory(OUTPUT_DIR)
    
    # Process each endpoint
    for endpoint in ENDPOINTS:
        banner(f"PROCESSING ENDPOINT: {endpoint}", "#")
        
        endpoint_dir = f"{OUTPUT_DIR}/{endpoint}"
        create_directory(endpoint_dir)
        
        # Load data
        ground_truth_df = load_ground_truth(endpoint)
        if ground_truth_df is None:
            print(f"✗ Skipping {endpoint} - no ground truth")
            continue
        
        metrics_df = load_metrics(endpoint)
        if metrics_df is None:
            print(f"✗ Skipping {endpoint} - no metrics")
            continue
        
        # Execute all strategies
        strategies_results = {}
        
        strategies_results['Strategy 1: Best Single'] = \
            strategy2_best_single_model(endpoint, metrics_df, ground_truth_df)
        
        strategies_results['Strategy 2: Top3 Ensemble'] = \
            strategy3_top3_ensemble(endpoint, metrics_df, ground_truth_df)
        
        strategies_results['Strategy 3: Best from Groups'] = \
            strategy4_best_from_groups(endpoint, metrics_df, ground_truth_df)
        
        # Save predictions for each strategy
        print(f"\n💾 Saving Strategy Predictions...")
        for strategy_name, result_df in strategies_results.items():
            if result_df is not None:
                safe_name = strategy_name.replace(' ', '_').replace(':', '').replace('+', 'plus')
                output_file = f"{endpoint_dir}/{safe_name}_{endpoint}_predictions.csv"
                result_df.to_csv(output_file, index=False)
                print(f"  ✓ {strategy_name}: {output_file}")
        
        # Evaluate all strategies
        results_df = evaluate_strategies(strategies_results, endpoint)
        
        if results_df is not None:
            # Save evaluation results
            eval_file = f"{endpoint_dir}/{endpoint}_strategy_evaluation.csv"
            results_df.to_csv(eval_file, index=False)
            print(f"\n💾 Saved evaluation: {eval_file}")
            
            # Create visualizations
            print(f"\n📊 Creating Visualizations...")
            create_scatter_plots(strategies_results, endpoint, endpoint_dir)
            create_comparison_barplot(results_df, endpoint, endpoint_dir)
            create_residual_plots(strategies_results, endpoint, endpoint_dir)
            
            # Summary statistics
            print(f"\n📈 Summary Statistics for {endpoint}:")
            print("-" * 80)

    
    print("\n" + "="*80)
    print(f"✅ ALL PROCESSING COMPLETE")
    print(f"📁 Results saved in: {OUTPUT_DIR}")
    print("="*80)
    print("\n📋 Output Summary:")
    print("  • Prediction CSV files for each strategy")
    print("  • Evaluation metrics CSV")
    print("  • Scatter plots (predictions vs ground truth)")
    print("  • Comparison bar plots (all metrics)")
    print("  • Residual plots (error distribution)")
    print("="*80)

if __name__ == "__main__":
    main()
