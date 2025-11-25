import sys
import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from trainModelNeuralE import prepare_data_for_races, venue_racetrack_racecourse_key, WinnerPredictor, convert_to_list
from functions.publicOddsProb import create_odds_list_from_df, compute_weights_continuous_from_lists ,logits_to_probs, odds_to_probs

def apply_blended_probs(df, w_model, w_odds):
    blended_probs_list = []

    for _, race_df in df.groupby(['date', 'race']):
        logits = np.vstack(race_df['logits'].tolist()).squeeze(-1).reshape(1, -1)
        odds = race_df['odds'].values

        model_probs = logits_to_probs(logits)  # [1, num_runners]
        odds_probs = odds_to_probs(odds).reshape(1, -1)  # [1, num_runners]

        blended_probs = (w_model * model_probs + w_odds * odds_probs).flatten()  # shape: [num_runners]
        blended_probs_list.extend(blended_probs.tolist())

    df = df.copy()
    df['blended_prob'] = blended_probs_list
    return df


def predictOldRaces():
    if len(sys.argv) != 3:
        print("Usage: python3 predictE.py <data.csv> <model_folder>")
        sys.exit(1)

    csv_path = sys.argv[1]
    model_folder = sys.argv[2]

    obj = torch.load(model_folder + "/model.pt", map_location='cpu')

    if isinstance(obj, dict):
        print("This is a state_dict.")
    elif hasattr(obj, 'state_dict'):
        print("This is a full model.")
    else:
        print("Unknown type:", type(obj))

    # Load CSV
    df = pd.read_csv(csv_path)
    df['lbw'] = df['lbw'].fillna(99)  # Replace NaN with the mean value
    df['runningPosition'] = df['runningPosition'].apply(convert_to_list)
    df["venueRacetrackRacecourse"] = df.apply(venue_racetrack_racecourse_key, axis=1)
    df["final_position"] = df["runningPosition"].apply(lambda rp: rp[-1] if isinstance(rp, list) and len(rp) > 0 else None)

    # Load config
    with open(model_folder + "/config.json", 'r') as f:
        config = json.load(f)

    # Group races by (date, race)
    if not {'date', 'race'}.issubset(df.columns):
        raise ValueError("CSV must contain 'date' and 'race' columns.")
    
    grouped_races = df.groupby(['date', 'race'])
    race_list = list(grouped_races.groups.items())

    # Prepare data
    x_cat_tensor, x_num_tensor, y_tensor, is_dummy_tensor = prepare_data_for_races(grouped_races, race_list, config)
    # Load model
    model = WinnerPredictor(
        config["num_categories_dict"],
        config["embedding_dim_dict"],
        len(config["numerical_columns"]) + 1 # +1 for isDummy
    )
    state_dict = torch.load(model_folder + "/model.pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    # Predict LBWs
    all_preds = []

    model.eval()
    with torch.no_grad():
        predicted_lbws = model(x_cat_tensor, x_num_tensor)  # shape: [num_races, max_runners]

    for i in range(predicted_lbws.shape[0]):
        pred = predicted_lbws[i]
        true = y_tensor[i]
        is_dummy = is_dummy_tensor[i]

        # Mask out dummy runners
        mask = is_dummy == 0
        pred_valid = pred[mask]
        true_valid = true[mask]

        all_preds.extend(pred_valid.tolist())                   # collect predictions for real runners

        # Optional: Print or evaluate per race
        print(f"\nRace {i + 1}:")
        for j in range(len(pred_valid)):
            print(f"  Runner {j + 1}: Predicted LBW = {pred_valid[j]:.2f}, True LBW = {true_valid[j]:.2f}")

        if len(true_valid) > 1:
            race_r2 = r2_score(true_valid.numpy(), pred_valid.numpy())
            print(f"  R² for race {i + 1}: {race_r2:.8f}")
        else:
            print("  Not enough data to compute R² for this race.")

    # Save to DataFrame
    assert len(all_preds) == len(df), f"Mismatch: {len(all_preds)} preds vs {len(df)} rows in df"
    df["logits"] = all_preds
    # Global R²
    preds = predicted_lbws.flatten()
    trues = y_tensor.flatten()
    mask = trues != 99.0
    filtered_preds = preds[mask]
    filtered_trues = trues[mask]
    r2 = r2_score(filtered_trues.numpy(), filtered_preds.numpy())
    print(f"\nGlobal R² score (LBW regression): {r2:.8f}")

    # weights for model and odds
    oddsList = create_odds_list_from_df(df)
    w_model, w_odds = compute_weights_continuous_from_lists(predicted_lbws, y_tensor, oddsList)
    print("weights (model: oddsx)", w_model, w_odds)
    df_with_blended = apply_blended_probs(df, w_model, w_odds)

    print(f"Model weight: {w_model:.3f}, Odds weight: {w_odds:.3f}")
    print(df_with_blended[['horseCode', 'blended_prob']].head())

    # Your existing model path
    model_key = os.path.basename(model_folder)  # just the filename

    r2_file = "r2.json"

    # Load existing R² data if file exists
    if os.path.exists(r2_file):
        with open(r2_file, "r") as f:
            r2_data = json.load(f)
    else:
        r2_data = {}

    # Add or update the R² score for this model
    r2_data[model_key] = round(r2, 8)  # convert tensor to float

    # Save updated data back to file
    with open(r2_file, "w") as f:
        json.dump(r2_data, f, indent=2)

    print(f"Saved R² for model {model_key} to {r2_file}")


if __name__ == '__main__':
    predictOldRaces()
