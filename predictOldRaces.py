import numpy as np
import json
import os
import torch
import torch.optim as optim
import argparse
from predictBNeural import predict_race
from trainModelBNeural import process_by_race, load_data, preprocess_race_data, SimpleNN
from functions.R2 import compute_public_r2, pseudo_r_squared, odds_to_probabilities

# Load the saved model and encoders
# === Paths (adjust folder name as needed) ===

def load_model(model_folder):
    # Load model checkpoint
    model_path = os.path.join(model_folder, "model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path)

    # Load metadata
    metadata_path = os.path.join(model_folder, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    feature_list = metadata.get("training_variables", [])
    input_dim = metadata.get("input_dim", None)
    if input_dim is None:
        raise ValueError("input_dim is missing from the metadata.")

    # Instantiate the model with the input_dim
    model = SimpleNN(input_dim=input_dim)
    model.load_state_dict(checkpoint)
    model.eval()

    print(f"Model loaded from {model_folder}")
    return model, feature_list

def group_probs_by_race(df):
    grouped_probs = []
    for _, group in df.groupby(['date', 'race']):
        odds = group['odds']
        probs = odds_to_probabilities(odds)
        grouped_probs.append(probs.to_numpy())
    return grouped_probs


def save_model_r2(file_path, model_name, model_r2):
    data = {}

    # Load existing data safely
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"[Warning] Could not parse {file_path}, starting fresh.")

    # Update the R² for this model
    data[model_name] = model_r2

    # Save back to JSON (atomic write)
    tmp_path = file_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, file_path)

# =================== MAIN ===================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str, help="CSV path")
    parser.add_argument("--model", type=str, default="models/neural_net_model_20250502_012700", help="Path to model folder (optional)")
    args = parser.parse_args()

    model_folder = args.model
    model, feature_list = load_model(model_folder)

    # Load and preprocess data
    #df_full = load_data(args.csv) #model B
    df_full, csvPath = load_data(args.csv) #model C
    df = preprocess_race_data(df_full.copy())

    X, y, vs = process_by_race(df)  # y gives winner indices
    #print("haha",X.iloc[[7010, 9272]])
    public_probs = group_probs_by_race(df)

    # Handle missing features
    columns_to_keep = [col for col in df.columns if col in feature_list]
    print("features_list", feature_list)
    df = df[columns_to_keep]

    # Predict win probabilities
    model_probs = predict_race(model, X)
    #print("MM", model_probs)  # Debug


    # Attach predictions back to original dataframe
    #df_full["model_prob"] = model_probs

    if "odds" not in df_full.columns:
        raise ValueError("Missing 'odds' column in the race data!")
 
    publicR2 = compute_public_r2(public_probs, y)
    modelR2 = pseudo_r_squared(model_probs, y)

    print("\nUsing model: " + model_folder)
    print("\nPublic R²:", publicR2)
    print("\nModel R²:", modelR2)

    save_model_r2("r2.json", model_folder, float(modelR2))

    print("\n✅ R² scores updated in r2.json")



