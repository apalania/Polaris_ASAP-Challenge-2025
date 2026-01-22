#!/usr/bin/env python3

# XGBoost Regression Pipeline 

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import kendalltau, spearmanr, pearsonr
from xgboost import XGBRegressor
import joblib
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback

# --- REPRODUCIBILITY SETTINGS ---

def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    import random
    
    # Python's random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Environment variables for additional libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # XGBoost specific - ensures deterministic multi-threading
    os.environ['OMP_NUM_THREADS'] = '1'
    
    print(f"✓ All random seeds set to {seed} for reproducibility")

# Call this immediately
set_all_seeds(42)

# --- CHECKPOINT MANAGEMENT ---

class CheckpointManager:
    """Manages experiment progress tracking and resumption."""
    
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, "progress.json")
        ensure_directory(checkpoint_dir)
        self.progress = self.load_progress()
    
    def load_progress(self):
        """Load existing progress or create new tracking structure."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                progress = json.load(f)
                print(f"✓ Loaded checkpoint from {self.checkpoint_file}")
                print(f"  Last updated: {progress.get('last_updated', 'Unknown')}")
                return progress
        return {
            "completed_experiments": [],
            "failed_experiments": [],
            "last_updated": None,
            "total_experiments": 0,
            "completed_count": 0
        }
    
    def save_progress(self):
        """Save current progress to disk."""
        self.progress['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.progress, f, indent=4)
    
    def is_completed(self, experiment_key):
        """Check if an experiment has been completed."""
        return experiment_key in self.progress['completed_experiments']
    
    def is_failed(self, experiment_key):
        """Check if an experiment has failed."""
        return any(exp['key'] == experiment_key for exp in self.progress['failed_experiments'])
    
    def mark_completed(self, experiment_key):
        """Mark an experiment as completed."""
        if experiment_key not in self.progress['completed_experiments']:
            self.progress['completed_experiments'].append(experiment_key)
            self.progress['completed_count'] = len(self.progress['completed_experiments'])
            self.save_progress()
            print(f"✓ Marked as completed: {experiment_key}")
    
    def mark_failed(self, experiment_key, error_msg):
        """Mark an experiment as failed with error details."""
        # Remove from failed list if already there
        self.progress['failed_experiments'] = [
            exp for exp in self.progress['failed_experiments'] 
            if exp['key'] != experiment_key
        ]
        # Add new failure record
        self.progress['failed_experiments'].append({
            "key": experiment_key,
            "error": error_msg,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        self.save_progress()
        print(f"✗ Marked as failed: {experiment_key}")
    
    def set_total_experiments(self, total):
        """Set the total number of experiments to run."""
        self.progress['total_experiments'] = total
        self.save_progress()
    
    def print_summary(self):
        """Print a summary of the current progress."""
        total = self.progress['total_experiments']
        completed = self.progress['completed_count']
        failed = len(self.progress['failed_experiments'])
        remaining = total - completed
        
        print("\n" + "="*80)
        print("CHECKPOINT SUMMARY")
        print("="*80)
        print(f"Total Experiments: {total}")
        print(f"Completed: {completed} ({100*completed/total if total > 0 else 0:.1f}%)")
        print(f"Failed: {failed}")
        print(f"Remaining: {remaining}")
        print(f"Last Updated: {self.progress.get('last_updated', 'Never')}")
        
        if self.progress['failed_experiments']:
            print("\nFailed Experiments:")
            for exp in self.progress['failed_experiments']:
                print(f"  - {exp['key']}")
                print(f"    Error: {exp['error'][:100]}...")
                print(f"    Time: {exp['timestamp']}")
        
        print("="*80 + "\n")
    
    def clear_checkpoint(self):
        """Clear all checkpoint data (start fresh)."""
        self.progress = {
            "completed_experiments": [],
            "failed_experiments": [],
            "last_updated": None,
            "total_experiments": 0,
            "completed_count": 0
        }
        self.save_progress()
        print("✓ Checkpoint cleared. Starting fresh.")


# --- LogD SPECIFIC CONFIGURATION ---
# Fixed file paths for LogD endpoint
LogD_POLARIS_FILE = 'LogD_POLARIS.csv'
LogD_AUGMENTED_FILE = 'LogD_MERGED.csv'
LogD_TEST_FILE = 'polaris-test.csv'
LogD_TARGET_COLUMN = 'LogD'

# LogD requires log transformation and no negative values
LogD_NEEDS_LOG_TRANSFORM = False
LogD_ALLOW_NEGATIVE = True

# --- Utility Functions ---

def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def parse_mixed_smiles(smiles):
    """Parse SMILES/CXSMILES with enhanced stereo handling"""
    parser_params = Chem.SmilesParserParams()
    parser_params.allowCXSMILES = True
    parser_params.strictCXSMILES = False
    mol = Chem.MolFromSmiles(smiles, parser_params)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
    return mol

def calculate_descriptors(smiles, descriptor_list):
    """Calculate a list of RDKit descriptors for a given SMILES string."""
    mol = parse_mixed_smiles(smiles)
    if mol:
        try:
            Chem.SanitizeMol(mol)
            return {desc: getattr(Descriptors, desc)(mol) for desc in descriptor_list}
        except Exception as e:
            print(f"Error calculating descriptors for {smiles}: {e}")
            return {desc: np.nan for desc in descriptor_list}
    return {desc: np.nan for desc in descriptor_list}

def load_descriptors_list(descriptor_file):
    """Load descriptor names from a text file."""
    with open(descriptor_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def compute_and_save_descriptors(input_file, output_file, descriptor_file):
    """Compute and save descriptors for a given dataset if the output file doesn't exist."""
    if os.path.exists(output_file):
        print(f"Descriptor file {output_file} already exists. Loading...")
        return pd.read_csv(output_file)
    
    print(f"Computing descriptors for {input_file}...")
    df = pd.read_csv(input_file)
    descriptor_list = load_descriptors_list(descriptor_file)
    descriptors = df['SMILES'].apply(lambda x: calculate_descriptors(x, descriptor_list))
    descriptor_df = pd.DataFrame(descriptors.tolist())
    descriptor_df.insert(0, "SMILES", df["SMILES"])
    descriptor_df.to_csv(output_file, index=False)
    return descriptor_df

def load_and_prepare_data(descriptor_df, target_file):
    """Load and merge descriptor and target data with LogD-specific preprocessing."""
    
    target_data_full = pd.read_csv(target_file)
    print(f"Available columns in {target_file}: {target_data_full.columns.tolist()}")
    
    target_data = target_data_full[["SMILES", LogD_TARGET_COLUMN]]
    data = pd.merge(descriptor_df, target_data, on="SMILES", how="inner")
    data.dropna(inplace=True)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data[LogD_TARGET_COLUMN] = pd.to_numeric(data[LogD_TARGET_COLUMN], errors='coerce')
    
    # LogD-specific preprocessing
    if LogD_NEEDS_LOG_TRANSFORM:
        if not LogD_ALLOW_NEGATIVE:
            data = data[data[LogD_TARGET_COLUMN] > 0]
        data[LogD_TARGET_COLUMN] = np.log1p(data[LogD_TARGET_COLUMN])
    else:
        data = data[~data[LogD_TARGET_COLUMN].isna()]
    
    data.reset_index(drop=True, inplace=True)
    return data

# --- Splitting Functions ---

def get_split_indices(data, split_strategy, seed):
    """Get train/test split indices based on the chosen strategy."""
    
    if split_strategy == "random":
        train_indices, test_indices = train_test_split(
            np.arange(len(data)), test_size=0.1, random_state=seed
        )
    
    elif split_strategy == "scaffold":
        smiles = data['SMILES'].values
        original_indices = np.arange(len(data))
        
        scaffold_to_indices = defaultdict(list)
        for idx, smi in enumerate(smiles):
            mol = parse_mixed_smiles(smi)
            if mol:
                try:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
                    scaffold_to_indices[scaffold].append(idx)
                except:
                    scaffold_to_indices[smi].append(idx)
            else:
                scaffold_to_indices[smi].append(idx)
        
        scaffold_groups = list(scaffold_to_indices.values())
        rng = np.random.RandomState(seed)
        rng.shuffle(scaffold_groups)
        
        total_size = len(data)
        train_size = int(0.9 * total_size)
        
        train_indices = []
        test_indices = []
        current_train_size = 0
        
        for group in scaffold_groups:
            if current_train_size < train_size:
                train_indices.extend(group)
                current_train_size += len(group)
            else:
                test_indices.extend(group)
        
        train_indices = np.array(train_indices, dtype=int)
        test_indices = np.array(test_indices, dtype=int)
    
    else:
        raise ValueError("Invalid split strategy. Choose 'random' or 'scaffold'.")
        
    return train_indices, test_indices

# --- Core Modeling Functions ---

def optimize_hyperparameters(X_train, y_train, random_state=42):
    """Perform GridSearchCV with controlled randomness for reproducibility."""
    model = XGBRegressor(random_state=random_state, n_jobs=1)
    
    # Optimized grid for your system
    param_grid = {
        "n_estimators": [500, 700, 1000],
        "max_depth": [5, 7, 9],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.9],
        "min_child_weight": [1, 5]
    }
    
    print("Starting GridSearchCV...")
    print(f"Total parameter combinations: {3*3*2*3*2*2} = 216")
    
    # Create stratified folds with fixed random state for reproducibility
    cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
    
    # Use n_jobs=1 for perfect reproducibility
    grid_search = GridSearchCV(
        model, param_grid, 
        cv=cv, 
        scoring="neg_mean_absolute_error", 
        n_jobs=1,  # Single-threaded for reproducibility
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best params from CV: {grid_search.best_params_}")
    return grid_search.best_params_

def evaluate_model(y_true, y_pred):
    """Calculate and return a dictionary of performance metrics."""
    
    # Convert back from log space for LogD
    if LogD_NEEDS_LOG_TRANSFORM:
        y_true_original = np.expm1(y_true)
        y_pred_original = np.expm1(y_pred)
    else:
        y_true_original = y_true
        y_pred_original = y_pred

    metrics = {
        "mae": mean_absolute_error(y_true_original, y_pred_original),
        "mse": mean_squared_error(y_true_original, y_pred_original),
        "r2": r2_score(y_true_original, y_pred_original),
        "pearson_r": pearsonr(y_true_original, y_pred_original)[0],
        "spearman_r": spearmanr(y_true_original, y_pred_original)[0],
        "kendall_tau": kendalltau(y_true_original, y_pred_original)[0]
    }
    return metrics

def run_experiment(data, split_strategy, n_splits=10, output_dir=None, config_name=None):
    """Run the main experiment loop for n_splits, returning detailed results."""
    all_split_results = []
    all_split_models = []
    hyperparams_votes = defaultdict(int)

    X = data.drop(columns=["SMILES", LogD_TARGET_COLUMN])
    y = data[LogD_TARGET_COLUMN]

    for i in range(n_splits):
        print(f"\n--- Starting Split {i+1}/{n_splits} (Seed={i}) ---")
        train_idx, test_idx = get_split_indices(data, split_strategy, seed=i)
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        best_params_for_split = optimize_hyperparameters(X_train, y_train, random_state=42)
        hyperparams_votes[json.dumps(best_params_for_split, sort_keys=True)] += 1

        model = XGBRegressor(**best_params_for_split, random_state=42)
        model.fit(X_train, y_train)
        
        if output_dir and config_name:
            model_path = os.path.join(output_dir, f"model_{config_name}_split_{i}.pth")
            joblib.dump(model, model_path)
            print(f"Model for split {i+1} saved to {model_path}")
        
        all_split_models.append(model)
        
        y_pred = model.predict(X_test)
        
        metrics = evaluate_model(y_test, y_pred)
        print(f"Metrics for Split {i+1}: {metrics}")

        split_info = {
            "split_seed": i,
            "split_number": i+1,
            "best_hyperparameters": best_params_for_split,
            "metrics": metrics
        }
        all_split_results.append(split_info)

    best_hyperparams_str = max(hyperparams_votes, key=hyperparams_votes.get)
    final_best_params = json.loads(best_hyperparams_str)
    
    return all_split_results, final_best_params, all_split_models

def plot_error_bars(split_results_list, config_name, output_dir):
    """Create error bar plots for all metrics across splits."""
    metrics_list = [result['metrics'] for result in split_results_list]
    results_df = pd.DataFrame(metrics_list)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    
    metrics = list(results_df.columns)
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        means = results_df[metric].mean()
        stds = results_df[metric].std()
        
        ax.bar(0, means, yerr=stds, capsize=10, color='steelblue', alpha=0.7, ecolor='black')
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.upper()}: {means:.4f} ± {stds:.4f}', fontsize=11)
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3)
    
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"error_bars_{config_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Error bar plot saved to {plot_path}")

def analyze_and_save_results(split_results_list, best_params, config_name, output_dir):
    """Calculate summary stats and save results."""
    metrics_list = [result['metrics'] for result in split_results_list]
    results_df = pd.DataFrame(metrics_list)
    summary = {
        "mean": results_df.mean().to_dict(),
        "std_dev (error_bars)": results_df.std().to_dict()
    }

    print("\n--- Aggregated Results Over Splits ---")
    for metric, mean_val in summary['mean'].items():
        std_val = summary['std_dev (error_bars)'][metric]
        print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    mae_values = [result['metrics']['mae'] for result in split_results_list]
    best_split_idx = np.argmin(mae_values)
    best_split_info = split_results_list[best_split_idx]
    
    json_output = {
        "experiment_config": config_name,
        "best_hyperparameters": best_params,
        "performance_summary_splits": summary,
        "best_split": {
            "split_number": best_split_info['split_number'],
            "split_seed": best_split_info['split_seed'],
            "metrics": best_split_info['metrics'],
            "hyperparameters": best_split_info['best_hyperparameters']
        }
    }
    
    json_filename = os.path.join(output_dir, f"results_{config_name}.json")
    with open(json_filename, 'w') as f:
        json.dump(json_output, f, indent=4)
    print(f"\nSummary results saved to {json_filename}")

    txt_filename = os.path.join(output_dir, f"detailed_splits_{config_name}.txt")
    with open(txt_filename, 'w') as f:
        f.write(f"Detailed Split-by-Split Results for Experiment: {config_name}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"*** BEST MODEL (Split {best_split_info['split_number']}, Seed {best_split_info['split_seed']}) ***\n")
        f.write(f"*** LOWEST MAE: {best_split_info['metrics']['mae']:.4f} ***\n")
        f.write("=" * 80 + "\n\n")

        for result in split_results_list:
            is_best = (result['split_number'] == best_split_info['split_number'])
            marker = ">>> BEST MODEL <<<" if is_best else ""
            
            f.write(f"--- Split {result['split_number']} (Seed: {result['split_seed']}) {marker} ---\n")
            f.write("Best Hyperparameters for this split:\n")
            for key, value in result['best_hyperparameters'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\nPerformance Metrics for this split:\n")
            for key, value in result['metrics'].items():
                highlight = " ***BEST***" if is_best and key == 'mae' else ""
                f.write(f"  {key.upper()}: {value:.4f}{highlight}\n")
            f.write("\n" + "-" * 80 + "\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY STATISTICS ACROSS ALL SPLITS\n")
        f.write("=" * 80 + "\n\n")
        for metric in summary['mean'].keys():
            f.write(f"{metric.upper()}: {summary['mean'][metric]:.4f} ± {summary['std_dev (error_bars)'][metric]:.4f}\n")
            
    print(f"Detailed split-by-split results saved to {txt_filename}")
    print(f"Best model: Split {best_split_info['split_number']} with MAE = {best_split_info['metrics']['mae']:.4f}")
    
    plot_error_bars(split_results_list, config_name, output_dir)

def ensemble_predict_on_test(split_models, test_file, descriptor_file, config_name, output_dir, X_full_columns):
    """Use all split models to make ensemble predictions on the blind test set."""
    print(f"\n--- Making Ensemble Predictions on Blind Test Set: {test_file} ---")
    
    descriptor_output_path = os.path.join(output_dir, f"descriptors_{config_name}_blind_test.csv")
    test_descriptors = compute_and_save_descriptors(test_file, descriptor_output_path, descriptor_file)
    X_test = test_descriptors.drop(columns=["SMILES"])
    
    X_test = X_test[X_full_columns]
    
    all_predictions = []
    for i, model in enumerate(split_models):
        predictions_transformed = model.predict(X_test)
        all_predictions.append(predictions_transformed)
        print(f"Predictions from split {i+1} model collected")
    
    ensemble_predictions_transformed = np.mean(all_predictions, axis=0)
    
    # LogD-specific inverse transformation and clipping
    if LogD_NEEDS_LOG_TRANSFORM:
        ensemble_predictions = np.expm1(ensemble_predictions_transformed)
        if not LogD_ALLOW_NEGATIVE:
            ensemble_predictions = np.maximum(0, ensemble_predictions)
    else:
        ensemble_predictions = ensemble_predictions_transformed
    
    results_df = pd.DataFrame({
        "SMILES": test_descriptors["SMILES"], 
        f"Predicted_{LogD_TARGET_COLUMN}": ensemble_predictions
    })
    prediction_path = os.path.join(output_dir, f"predictions_{config_name}_ensemble_blind_test.csv")
    results_df.to_csv(prediction_path, index=False)
    print(f"Ensemble predictions saved to {prediction_path}")
    
    # Save individual split predictions
    for i, preds_transformed in enumerate(all_predictions):
        if LogD_NEEDS_LOG_TRANSFORM:
            preds = np.expm1(preds_transformed)
            if not LogD_ALLOW_NEGATIVE:
                preds = np.maximum(0, preds)
        else:
            preds = preds_transformed
            
        individual_df = pd.DataFrame({
            "SMILES": test_descriptors["SMILES"],
            f"Predicted_{LogD_TARGET_COLUMN}_split_{i}": preds
        })
        individual_path = os.path.join(output_dir, f"predictions_{config_name}_split_{i}_blind_test.csv")
        individual_df.to_csv(individual_path, index=False)
    
    print(f"Individual split predictions also saved")
    return ensemble_predictions

# --- Main Execution Block with Checkpointing ---
if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("LogD ENDPOINT PIPELINE - REPRODUCIBLE VERSION")
    print("="*80)
    
    # Initialize checkpoint manager
    checkpoint = CheckpointManager()
    
    # Ask user if they want to clear checkpoints
    checkpoint.print_summary()
    
    if checkpoint.progress['completed_count'] > 0:
        user_input = input("\nDo you want to (C)ontinue from checkpoint, (R)estart fresh, or (Q)uit? [C/R/Q]: ").strip().upper()
        if user_input == 'R':
            checkpoint.clear_checkpoint()
        elif user_input == 'Q':
            print("Exiting...")
            exit(0)
        # else continue (default)
    
    # Configuration - LogD specific
    FEATURE_OPTIONS = {"55_desc": "descriptors_55.txt", "25_desc": "descriptors_25.txt"}
    SPLIT_OPTIONS = ["random", "scaffold"]
    
    data_options = {
        "Polaris": LogD_POLARIS_FILE,
        "Augmented": LogD_AUGMENTED_FILE
    }
    
    # Calculate total experiments
    total_experiments = len(data_options) * len(FEATURE_OPTIONS) * len(SPLIT_OPTIONS)
    checkpoint.set_total_experiments(total_experiments)
    checkpoint.print_summary()
    
    # Create output directory for LogD
    output_dir = "results_LogD"
    ensure_directory(output_dir)
    
    print(f"\n{'#'*80}")
    print(f"STARTING PIPELINE FOR LogD ENDPOINT")
    print(f"Total experiments to run: {total_experiments}")
    print(f"{'#'*80}\n")

    # Main loop with error handling
    for data_name, data_file in data_options.items():
        for feature_name, feature_file in FEATURE_OPTIONS.items():
            for split_strategy in SPLIT_OPTIONS:
                
                # Create unique experiment key
                config_name = f"LogD_{data_name}_{feature_name}_{split_strategy}"
                
                # Check if already completed
                if checkpoint.is_completed(config_name):
                    print(f"\n✓ SKIPPING (already completed): {config_name}")
                    continue
                
                # Check if previously failed
                if checkpoint.is_failed(config_name):
                    retry = input(f"\n⚠ Previously failed: {config_name}\nRetry? [Y/N]: ").strip().upper()
                    if retry != 'Y':
                        print(f"Skipping {config_name}")
                        continue
                
                print(f"\n\n{'='*80}")
                print(f"RUNNING EXPERIMENT: {config_name}")
                print(f"{'='*80}")
                
                try:
                    # Compute descriptors
                    descriptor_output_file = os.path.join(
                        output_dir, 
                        f"descriptors_{data_name}_{feature_name}.csv"
                    )
                    descriptor_df = compute_and_save_descriptors(
                        data_file, descriptor_output_file, feature_file
                    )
                    
                    # Load and prepare data
                    full_data = load_and_prepare_data(descriptor_df, data_file)
                    
                    # Run experiment
                    split_results, best_hyperparams, split_models = run_experiment(
                        full_data, split_strategy, 
                        n_splits=10, output_dir=output_dir, 
                        config_name=config_name
                    )
                    
                    # Analyze and save results
                    analyze_and_save_results(
                        split_results, best_hyperparams, config_name, output_dir
                    )
                    
                    # Get column names for test data alignment
                    X_full = full_data.drop(columns=["SMILES", LogD_TARGET_COLUMN])
                    
                    # Make ensemble predictions
                    ensemble_predict_on_test(
                        split_models=split_models,
                        test_file=LogD_TEST_FILE,
                        descriptor_file=feature_file,
                        config_name=config_name,
                        output_dir=output_dir,
                        X_full_columns=X_full.columns
                    )
                    
                    # Mark as completed
                    checkpoint.mark_completed(config_name)
                    checkpoint.print_summary()
                    
                except Exception as e:
                    # Log the error
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    print(f"\n{'!'*80}")
                    print(f"ERROR in {config_name}")
                    print(f"{'!'*80}")
                    print(error_msg)
                    print("\nFull traceback:")
                    traceback.print_exc()
                    print(f"{'!'*80}\n")
                    
                    # Mark as failed
                    checkpoint.mark_failed(config_name, error_msg)
                    
                    # Ask user what to do
                    user_choice = input("\n(C)ontinue to next experiment, (R)etry this one, or (Q)uit? [C/R/Q]: ").strip().upper()
                    if user_choice == 'Q':
                        print("Exiting... Progress has been saved.")
                        checkpoint.print_summary()
                        exit(0)
                    elif user_choice == 'R':
                        print(f"Retrying {config_name}...")
                        # Don't mark as failed, let it retry in next iteration
                        continue
                    # else continue to next experiment
    
    print("\n\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED FOR LogD ENDPOINT!")
    print("="*80)
    checkpoint.print_summary()
    
    # Save final summary report
    final_report = {
        "endpoint": "LogD",
        "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_experiments": checkpoint.progress['total_experiments'],
        "completed": checkpoint.progress['completed_count'],
        "failed": len(checkpoint.progress['failed_experiments']),
        "completed_experiments": checkpoint.progress['completed_experiments'],
        "failed_experiments": checkpoint.progress['failed_experiments']
    }
    
    with open(os.path.join(checkpoint.checkpoint_dir, "final_report_LogD.json"), 'w') as f:
        json.dump(final_report, f, indent=4)
    
    print("\nFinal report saved to checkpoints/final_report_LogD.json")
