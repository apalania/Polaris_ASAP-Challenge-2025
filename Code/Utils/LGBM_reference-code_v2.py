# -*- coding: utf-8 -*-

!pip install rdkit

import pandas as pd
import numpy as np
import time
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from lightgbm import LGBMRegressor, plot_importance
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

start = time.time()

# === Parse SMILES with support for CXSMILES ===
def parse_mixed_smiles(smiles):
    parser_params = Chem.SmilesParserParams()
    parser_params.allowCXSMILES = True
    parser_params.strictCXSMILES = False
    mol = Chem.MolFromSmiles(smiles, parser_params)
    return mol or Chem.MolFromSmiles(smiles)

# === Compute descriptors for a molecule ===
def calculate_descriptors(smiles, descriptor_list):
    mol = parse_mixed_smiles(smiles)
    if mol:
        try:
            Chem.SanitizeMol(mol)
            return {desc: getattr(Descriptors, desc)(mol) for desc in descriptor_list}
        except:
            return {desc: np.nan for desc in descriptor_list}
    return {desc: np.nan for desc in descriptor_list}

# === Load descriptors list from file ===
def load_descriptors(descriptor_file):
    with open(descriptor_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# === Compute descriptors and save ===
def compute_and_save_descriptors(input_file, output_file, descriptor_file):
    df = pd.read_csv(input_file)
    descriptor_list = load_descriptors(descriptor_file)
    descriptors = df['SMILES'].apply(lambda x: calculate_descriptors(x, descriptor_list))
    descriptor_df = pd.DataFrame(descriptors.tolist())
    descriptor_df.insert(0, "SMILES", df["SMILES"])
    descriptor_df.to_csv(output_file, index=False)
    return descriptor_df

# === Load descriptors and merge with target ===
def load_and_prepare_data(descriptor_file, target_file, target_col):
    descriptors_df = pd.read_csv(descriptor_file)
    target_data = pd.read_csv(target_file)[["SMILES", target_col]]
    data = pd.merge(descriptors_df, target_data, on="SMILES", how="inner")
    data.dropna(inplace=True)
    data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
    data = data[data[target_col] > 0]
    return data

# === Model performance plots ===
def plot_model_performance(final_model, y_test, y_pred, best_params, mae, r2):
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nMAE: {mae:.4f}, R²: {r2:.4f}")
    print(f"Best Hyperparameters: {best_params}")

    # Feature importance
    plot_importance(final_model, max_num_features=20, importance_type='gain')
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/KSOL_feature_importance.png", dpi=300)
    plt.show()

    # Actual vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(np.expm1(y_test), np.expm1(y_pred), alpha=0.6)
    plt.plot([min(np.expm1(y_test)), max(np.expm1(y_test))],
             [min(np.expm1(y_test)), max(np.expm1(y_test))], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/KSOL_actual_vs_predicted.png", dpi=300)
    plt.show()

# === Train + Evaluate Model ===
def train_and_evaluate_model(data, target_col):
    X = data.drop(columns=["SMILES", target_col])
    y = np.log1p(data[target_col])

    # Remove constant features
    selector = VarianceThreshold(threshold=0.0)
    X_filtered = selector.fit_transform(X)
    selected_columns = X.columns[selector.get_support()]
    X = pd.DataFrame(X_filtered, columns=selected_columns)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LGBMRegressor(random_state=42)

    param_grid = {
        "n_estimators": [250, 500, 750],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.6, 0.8, 1.0]
    }

    search = RandomizedSearchCV(
        model, param_distributions=param_grid, n_iter=20,
        scoring="neg_mean_absolute_error", cv=5, n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    best_params = search.best_params_

    final_model = LGBMRegressor(**best_params, random_state=42)
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    return final_model, mae, r2, best_params, selected_columns, X, y

# === Train on Full Dataset ===
def train_final_model(X, y, best_params):
    model = LGBMRegressor(**best_params, random_state=42)
    model.fit(X, y)
    return model

# === Save model and config ===
def save_model_and_info(model, mae, r2, best_params):
    joblib.dump(model, "final_model_LGB_KSOL.joblib")

    with open("model_metrics_LGB_KSOL.txt", 'w') as f:
        f.write(f"MAE: {mae:.4f}\nR2: {r2:.4f}\n")

    config = {
        "model_path": "final_model_LGB_KSOL.joblib",
        "metrics_path": "model_metrics_LGB_KSOL.txt",
        "best_params": best_params
    }

    with open("model_config_LGB_KSOL.json", 'w') as f:
        json.dump(config, f, indent=4)

# === Predict external test ===
def predict_external(model, test_file, descriptor_file, selected_columns):
    test_descriptors = compute_and_save_descriptors(test_file, "test_descriptors.csv", descriptor_file)
    X_test = test_descriptors[selected_columns]
    predictions = np.expm1(model.predict(X_test))
    test_descriptors["Predicted_KSOL"] = predictions
    test_descriptors.to_csv("test_predictions_KSOL.csv", index=False)

# === Main ===
if __name__ == "__main__":
    descriptor_csv = "computed_descriptors.csv"
    merged_data = "KSOL_MERGED.csv"
    test_smiles = "polaris-test.csv"
    descriptor_list_file = "descriptors.txt"
    target_column = "KSOL"

    if not os.path.exists(descriptor_csv):
        compute_and_save_descriptors(merged_data, descriptor_csv, descriptor_list_file)

    data = load_and_prepare_data(descriptor_csv, merged_data, target_column)
    model_temp, mae, r2, best_params, selected_columns, X_all, y_all = train_and_evaluate_model(data, target_column)
    plot_model_performance(model_temp, y_all, model_temp.predict(X_all), best_params, mae, r2)

    final_model = train_final_model(X_all, y_all, best_params)
    save_model_and_info(final_model, mae, r2, best_params)
    predict_external(final_model, test_smiles, descriptor_list_file, selected_columns)

    print(f"\n Completed in {(time.time() - start)/60:.2f} minutes.")