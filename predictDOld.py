import sys
import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from trainModelNeuralD import prepare_data_for_races, venue_racetrack_racecourse_key, WinnerPredictor, convert_to_list
from functions.publicOddsProb import compute_weights,logits_to_probs, odds_to_probs
# Load user-defined function

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

def add_logits_and_probs_to_df(df, race_list, all_logits, is_dummy_tensor):
    """
    Add logits and softmax probabilities back into df rows for real runners only.

    Parameters:
    - df: original DataFrame with only real runners (grouped by date, race)
    - race_list: list of ((date, race_number), group_indices) from df.groupby
    - all_logits: list of logits tensors for each race (length max_runners)
    - is_dummy_tensor: tensor indicating dummy runners (1 for dummy, 0 for real), shape (num_races, max_runners)

    Returns:
    - df_with_preds: original df with two new columns 'logits' and 'prob' added
    """
    logits_col = []
    prob_col = []

    # We'll build these lists per row to assign later

    for i, ((race_date, race_number), indices) in enumerate(race_list):
        df_race = df.loc[indices].copy()

        logits = all_logits[i]  # shape: (max_runners,)
        is_dummy = is_dummy_tensor[i].numpy()  # (max_runners,)

        # Filter out dummy runners in logits and probs
        mask_real = (is_dummy == 0)
        logits_real = logits[mask_real]

        # Compute probabilities by softmax on real runners
        probs_real = torch.softmax(logits_real, dim=0).numpy()

        # Now logits_real and probs_real length == len(df_race), aligned in order

        # Append values in order for each real runner to lists
        logits_col.extend(logits_real.numpy())
        prob_col.extend(probs_real)

    # Assign new columns to the original df in the same order
    df = df.copy()
    df['logits'] = logits_col
    df['prob'] = prob_col

    return df

def predictOldRaces():
    if len(sys.argv) != 3:
        print("Usage: python3 predictOldRaces.py <data.csv> <model_folder>")
        sys.exit(1)

    csv_path = sys.argv[1]
    model_folder = sys.argv[2]

    # Load config
    with open("models/" + model_folder + "/config.json", 'r') as f:
        config = json.load(f)

    # Load CSV
    df = pd.read_csv(csv_path)
    print(df['horseNumber'])
    df['lbw'] = df['lbw'].fillna(99)
    df['runningPosition'] = df['runningPosition'].apply(convert_to_list)
    df["venueRacetrackRacecourse"] = df.apply(venue_racetrack_racecourse_key, axis=1)
    df["final_position"] = df["runningPosition"].apply(
        lambda rp: rp[-1] if isinstance(rp, list) and len(rp) > 0 else None
    )

    # Group by races
    grouped_races = df.groupby(['date', 'race'])
    race_list = list(grouped_races.groups.items())

    # Prepare data (note: this should prepare y_true as winner index)
    x_cat_tensor, x_num_tensor, y_true_tensor, is_dummy_tensor = prepare_data_for_races(
        grouped_races, race_list, config
    )

    # Load model
    model = WinnerPredictor(
        config["num_categories_dict"],
        config["embedding_dim_dict"],
        len(config["numerical_columns"]) + 1  # +1 for isDummy
    )
    state_dict = torch.load("models/" + model_folder + "/model.pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    print("=== Final layer weights/bias statistics ===")
    for name, param in model.named_parameters():
        if param.requires_grad and ("weight" in name or "bias" in name):
            print(f"{name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")

    # Step 2: Check logits
    model.eval()

    # Predict logit scores (before softmax)
    all_probs = []
    true_winner_indices = []
    all_logits = []
    #all_is_dummy = []


    with torch.no_grad():
        for i in range(len(y_true_tensor)):
            x_cat_batch = {field: x_cat_tensor[field][i].unsqueeze(0) for field in x_cat_tensor.keys()}
            x_num_batch = x_num_tensor[i].unsqueeze(0)
        
            logits = model(x_cat_batch, x_num_batch).squeeze(0)   # [num_runners]
            print("logits", logits)
            print("is dummy", is_dummy_tensor)
            mask = is_dummy_tensor[i] == 0

            all_logits.append(logits)

            print("all_logits", all_logits)
            
            logits_valid = logits[mask]

            # Softmax to get probabilities
            probs = torch.softmax(logits_valid, dim=0)
            all_probs.append(probs.numpy())

            true_winner_indices.append(y_true_tensor[i].item())

            # Print race-level prediction
            print(f"\nRace {i + 1}:")
            for j, prob in enumerate(probs):
                print(f"  Runner {j + 1}: Prob = {prob:.4f}")
            print(f"  True winner index: {y_true_tensor[i].item()}")

    df_with_preds = add_logits_and_probs_to_df(df, race_list, all_logits, is_dummy_tensor)
    print(df_with_preds[['date', 'race', 'horseNumber', 'logits', 'prob']].head())

    # Compute pseudo R²
    epsilon = 1e-15
    L_model = sum(np.log(np.clip(probs[true_idx], epsilon, 1)) for probs, true_idx in zip(all_probs, true_winner_indices))      
    L_uniform = sum(np.log(1 / len(probs)) for probs in all_probs)
    pseudo_r2 = 1 - (L_model / L_uniform)

    print(f"\nPseudo R² (McFadden’s-like): {pseudo_r2:.8f}")

    # Save to file
    model_key = os.path.basename(model_folder)
    r2_file = "r2.json"
    if os.path.exists(r2_file):
        with open(r2_file, "r") as f:
            r2_data = json.load(f)
    else:
        r2_data = {}
    r2_data[model_key] = round(pseudo_r2, 8)

    with open(r2_file, "w") as f:
        json.dump(r2_data, f, indent=2)
    print(f"Saved pseudo R² for model {model_key} to {r2_file}")

    # Optional: compute weights and blended probs
    w_model, w_odds = compute_weights(df_with_preds)
    df_with_blended = apply_blended_probs(df_with_preds, w_model, w_odds)
    print(f"Model weight: {w_model:.3f}, Odds weight: {w_odds:.3f}")
    print(df_with_blended[['horseCode', 'blended_prob']].head())


if __name__ == '__main__':
    predictOldRaces()
