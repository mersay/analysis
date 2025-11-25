import sys
import os
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from sklearn.metrics import r2_score
from trainModelNeuralE import prepare_data_for_races, venue_racetrack_racecourse_key, WinnerPredictor, convert_to_list
from functions.publicOddsProb import create_odds_list_from_df, compute_weights_continuous_from_lists ,logits_to_probs, odds_to_probs
import torch.nn.functional as F

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)  

def predict():
    model_folder = "models/mlp_20250508_225321"
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 predictE.py <race_number> [date: YYYY-MM-DD]")
        sys.exit(1)

    race_number = int(sys.argv[1])
    
    if len(sys.argv) == 3:
        date = sys.argv[2]
    else:
        hk_time = datetime.now()#pytz.timezone("Asia/Hong_Kong"))
        date = hk_time.strftime('%Y-%m-%d')

    prediction_csv_path = f"predictions/runners_{date}.csv"
    odds_csv_path = f"odds/runners_{date}_{race_number}.csv"

    try:
        df_pred = pd.read_csv(prediction_csv_path)
        df_odds = pd.read_csv(odds_csv_path, sep=';')

    except FileNotFoundError as e:
        print(f"CSV file not found: {e}")
        sys.exit(1)

    df_pred = df_pred[df_pred["race"] == race_number].copy()
    df_odds = df_odds[df_odds["type"] == "WIN"].copy()

    df_pred["race"] = df_pred["race"].astype(int)
    df_odds["race"] = df_odds["race"].astype(int)
    df_pred["horseNumber"] = df_pred["horseNumber"].astype(int)
    df_odds["horseNumber"] = df_odds["horseNumber"].astype(int)

    merged_df = pd.merge(df_pred, df_odds[["race", "horseNumber", "odds"]],
                         on=["race", "horseNumber"], how="left")

    merged_df["venueRacetrackRacecourse"] = merged_df.apply(venue_racetrack_racecourse_key, axis=1)

    with open(model_folder + "/config.json", 'r') as f:
        config = json.load(f)

    if not {'date', 'race'}.issubset(merged_df.columns):
        raise ValueError("CSV must contain 'date' and 'race' columns.")
    
    grouped_races = merged_df.groupby(['date', 'race'])
    race_list = list(grouped_races.groups.items())

    x_cat_tensor, x_num_tensor, is_dummy_tensor = prepare_data_for_races(grouped_races, race_list, config, True)

    model = WinnerPredictor(
        config["num_categories_dict"],
        config["embedding_dim_dict"],
        len(config["numerical_columns"]) + 1
    )
    state_dict = torch.load(model_folder + "/model.pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        predicted_lbws = model(x_cat_tensor, x_num_tensor)  # [num_races, max_runners]
        probs_tensor = F.softmax(-predicted_lbws, dim=1)     # Convert to probabilities

    # Flatten and filter only real runners
    is_dummy_flat = is_dummy_tensor.flatten()
    is_real_mask = is_dummy_flat == 0
    probs_flat = probs_tensor.flatten()
    real_probs = probs_flat[is_real_mask].numpy()

    df_real = merged_df.copy()
    df_real["model_prob"] = real_probs

    df_real["odds_prob"] = 1 / df_real["odds"]
    df_real["final_prob"] = 0.5 * df_real["model_prob"] + 0.5 * df_real["odds_prob"]
    df_real["value_bet"] = df_real["model_prob"] > df_real["odds_prob"]
    
    df_real_sorted = df_real.sort_values(by=["final_prob"], ascending=[False])
    # Get indices of top 3 odds probability
    top_3_idx = df_real_sorted.nlargest(3, 'odds_prob').index

    print(merged_df)

    def print_highlighted(df):
    # Print the header row
        print(f"{'Horse#':>6}  {'Horse Name':<20} {'Model':>15} {'Odds':>15} {'Final':>15} {'Value Bet'}")

        # Iterate over the rows to print
        for idx, row in df.iterrows():
            # Check if the current row is one of the top 3
            if idx in top_3_idx:
                # Print in color (Yellow for highlighted)
                print(f"\033[93m{row['horseNumber']:>6}  {row['name_ch']:<20}  {row['model_prob']:>15.5f} {row['odds_prob']:>15.5f} {row['final_prob']:>15.5f} {row['value_bet']}\033[0m")
            else:
                # Regular print for non-highlighted rows
                print(f"{row['horseNumber']:>6}  {row['name_ch']:<20}  {row['model_prob']:>15.5f} {row['odds_prob']:>15.5f} {row['final_prob']:>15.5f} {row['value_bet']}")

    # Print the DataFrame with header and highlighted top 3
    print_highlighted(df_real_sorted)
if __name__ == '__main__':
    predict()
