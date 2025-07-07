#!/usr/bin/env python3

# XGBoost Regression Pipeline for KSOL Prediction

import pandas as pd
import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from xgboost import plot_importance
import matplotlib.pyplot as plt
import joblib
import json
import os
import seaborn as sns

# Molecular Parsing Function
def parse_mixed_smiles(smiles):
    """Parse SMILES/CXSMILES with enhanced stereo handling"""
    parser_params = Chem.SmilesParserParams()
    parser_params.allowCXSMILES = True
    parser_params.strictCXSMILES = False
    mol = Chem.MolFromSmiles(smiles, parser_params)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
    return mol

# Compute Descriptors
def calculate_descriptors(smiles, descriptor_list):
    mol = parse_mixed_smiles(smiles)
    if mol:
        try:
            Chem.SanitizeMol(mol)
            return {desc: getattr(Descriptors, desc)(mol) for desc in descriptor_list}
        except Exception as e:
            print(f"Error calculating descriptors for {smiles}: {e}")
            return {desc: np.nan for desc in descriptor_list}
    return {desc: np.nan for desc in descriptor_list}

# Load Descriptors List
def load_descriptors(descriptor_file):
    with open(descriptor_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# Compute and Save Descriptors
def compute_and_save_descriptors(input_file, output_file, descriptor_file):
    df = pd.read_csv(input_file)
    descriptor_list = load_descriptors(descriptor_file)
    descriptors = df['SMILES'].apply(lambda x: calculate_descriptors(x, descriptor_list))
    descriptor_df = pd.DataFrame(descriptors.tolist())
    descriptor_df.insert(0, "SMILES", df["SMILES"])
    descriptor_df.to_csv(output_file, index=False)
    return descriptor_df

# Load and prepare data
def load_and_prepare_data(descriptor_file, target_file, target_col):
    descriptors_df = pd.read_csv(descriptor_file)
    target_data = pd.read_csv(target_file)[["SMILES", target_col]]
    data = pd.merge(descriptors_df, target_data, on="SMILES", how="inner")
    data.dropna(inplace=True)
    data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
    data = data[data[target_col] > 0]
    return data

#plots for the data
def plot_model_performance(final_model, y_test, y_pred, best_params, mae, r2):
    import os
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # Print performance
    print(f"\nOptimized Model Performance:\nMAE: {mae:.4f}, R²: {r2:.4f}")
    print(f"Best Hyperparameters: {best_params}")

    # Plot Feature Importance
    xgb.plot_importance(final_model, max_num_features=20, importance_type='weight')
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/KSOL_feature_importance.png", dpi=600, format='png')
    plt.savefig(f"{output_dir}/KSOL_feature_importance.tif", dpi=600, format='tiff')
    plt.show()

    # Plot Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(np.expm1(y_test), np.expm1(y_pred), alpha=0.6, color='teal')
    plt.plot([min(np.expm1(y_test)), max(np.expm1(y_test))],
             [min(np.expm1(y_test)), max(np.expm1(y_test))], color='red', linestyle='--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted (Original Scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/KSOL_actual_vs_predicted.png", dpi=600, format='png')
    plt.savefig(f"{output_dir}/KSOL_actual_vs_predicted.tif", dpi=600, format='tiff')
    plt.show()

    # Plot density
    # Calculate Residuals
    residuals = np.expm1(y_test) - np.expm1(y_pred)

    # Plot Density Plot
    plt.figure(figsize=(6, 4))
    sns.kdeplot(residuals, fill=True, color='orange', linewidth=2)
    plt.title("Residuals Density Plot")
    plt.xlabel("Residuals")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/KSOL_residuals_density.png", dpi=600, format='png')
    plt.savefig(f"{output_dir}/KSOL_residuals_density.tif", dpi=600, format='tiff')
    plt.show()


# Train and evaluate model
def train_and_evaluate_model(data, target_col):
    X = data.drop(columns=["SMILES", target_col])
    y = np.log1p(data[target_col])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(random_state=42)
    param_grid = {
        "n_estimators": [250,500, 700, 800, 900, 1000],
        "max_depth": [3, 5, 7, 9, 11],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.2,0.4,0.6,0.8, 1.0]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    final_model = XGBRegressor(
        **best_params,
        random_state=42,
        eval_metric=["rmse", "mae"],
        early_stopping_rounds=10
    )

    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    y_pred = final_model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    best_iteration = final_model.best_iteration + 1  

    # Calculate Adjusted R²
    def adjusted_r2_score(r2, n, p):
    	return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    n = X_val.shape[0]
    p = X_val.shape[1]
    adjusted_r2 = adjusted_r2_score(r2, n, p)

    print(f"MAE: {mae:.4f}, R²: {r2:.4f}, Adjusted R²: {adjusted_r2:.4f}")
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Iteration (from early stopping): {best_iteration}")

    return final_model, mae, r2, adjusted_r2, best_params, best_iteration, y_val, y_pred

def train_final_model_on_all_data(data, target_col, best_params, best_n_estimators):
    X_final = data.drop(columns=["SMILES", target_col])
    y_final = np.log1p(data[target_col])

    # Copy and update best_params to avoid mutating the original
    final_params = best_params.copy()
    final_params["n_estimators"] = best_n_estimators

    print("\nTraining FINAL model on full dataset...")
    final_model = XGBRegressor(
        **final_params,
        random_state=42
    )
    final_model.fit(X_final, y_final)
    return final_model

def save_model_and_info(final_model, mae, r2, adjusted_r2, best_params):
    joblib.dump(final_model, "KSOL_final_model_mae.joblib")

    with open("KSOL_model_metrics_mae.txt", 'w') as f:
        f.write(f"Final Model Performance:\nMAE: {mae:.4f}\nR2 Score: {r2:.4f}\nAdjusted R2 Score: {adjusted_r2:.4f}\n")

    config = {
        "model_path": "KSOL_final_model_mae.joblib",
        "metrics_path": "KSOL_model_metrics_mae.txt",
        "best_params": best_params,
        "descriptor_file": "descriptors.txt",
        "descriptor_csv": "computed_descriptors_KSOL_mae.csv",
        "prediction_file": "polaris_predictions_KSOL_mae.csv"
    }

    with open("model_config_eval_KSOL.json", 'w') as f:
        json.dump(config, f, indent=4)

def predict_on_polaris_test(final_model, polaris_file, descriptor_file):
    polaris_descriptors = compute_and_save_descriptors(polaris_file, "polaris_computed_KSOL_descriptors_mae.csv", descriptor_file)
    X_polaris = polaris_descriptors.drop(columns=["SMILES"])
    predictions = np.expm1(final_model.predict(X_polaris))

    results = pd.DataFrame({"SMILES": polaris_descriptors["SMILES"], "Predicted_KSOL": predictions})
    results.to_csv("polaris_predictions_KSOL_mae.csv", index=False)

# Main Execution
if __name__ == "__main__":
    descriptor_output = "computed_descriptors_KSOL_mae.csv"
    KSOL_file = "KSOL_MERGED.csv"
    descriptor_list_file = "descriptors.txt"
    polaris_file = "polaris-test.csv"
    target_column = "KSOL"

    compute_and_save_descriptors(KSOL_file, descriptor_output, descriptor_list_file)
    data = load_and_prepare_data(descriptor_output, KSOL_file, target_column)
    
    final_model_temp, mae, r2, adjusted_r2, best_params, best_n_estimators, y_val, y_pred = train_and_evaluate_model(data, target_column)

    # Add performance plots
    plot_model_performance(final_model_temp, y_val, y_pred, best_params, mae, r2)

    final_model = train_final_model_on_all_data(data, target_column, best_params, best_n_estimators)
    save_model_and_info(final_model, mae, r2, adjusted_r2, best_params)
    predict_on_polaris_test(final_model, polaris_file, descriptor_list_file)
