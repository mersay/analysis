import pandas as pd
import numpy as np
import json
import os
import torch
from scipy.stats import entropy
import torch.optim as optim
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
from datetime import datetime
from datetime import datetime
import argparse
from functions.encodeClass import encodeClass, getAllClassVariables
from functions.encodeGoing import encodeGoing, getAllGoingVariables
from functions.encodeVenue import encodeVenue, getAllVenueVariables
from functions.encodeRacecourse import encodeRacecourse, getAllRacecourseVariables
from functions.encodeSex import encodeSex, getAllSexVariables
from functions.encodeCountry import encodeCountry, getAllCountryVariables
from functions.scaleDistance import scaleDistance
from trainModelBNeural import load_data, preprocess_race_data, process_by_race, SimpleNN

# Load the saved model and encoders
# === Paths (adjust folder name as needed) ===

# =================== CONFIG ===================
MAX_RUNNERS = 14
MAX_ITER=1500
feature_list = []
RACE_FEATURES = ["date", "race", "distance", "going", "venue", "racecourse", "racetrack", "class", "money"]  # shared across all runners
RUNNER_FEATURES = [
    "results",
    "place",
    "horseNumber",
    "horseCode",
    "jockey",
    "trainer",
    "handicapWeight",
    "horseWeight",
    "draw",
    "lbw",
    "runningPosition",
    "finishTime",
    "odds",
    "country",
    "colour",
    "sex",
    "currentRating",
    "speedRating",
    "daysSinceLastRace",
    "jocWinCount",
    "jocWinPercent",
    "jocMisData",
    "avesprat",
    "avglbw",
    "age",
    "AGEMISDATA",
    "drawBiasScore"
]
MAX_RUNNERS = 14  # pad races up to 14 runners
RACE_CATEGORICAL_COLUMNS = ["going", "venue", "racecourse", "racetrack", "class"]
RACE_CATEGORICAL_VARS = [getAllRacecourseVariables(), getAllVenueVariables(), getAllClassVariables(), getAllGoingVariables()] #need to drop
RUNNER_CATEGORICAL_VARS = [getAllSexVariables(), getAllCountryVariables()]
RUNNER_CATEGORICAL_COLUMNS = ["country", "colour", "sex"]

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

def mergeData(race_num, date=None):
    # Format the date
    if date is None:
        date_str = datetime.today().strftime("%Y-%m-%d")
    elif isinstance(date, datetime):
        date_str = date.strftime("%Y-%m-%d")
    else:
        date_str = str(date)  # assume already in "YYYY-MM-DD" format

    # File paths
    runner_path = os.path.join("predictions", f"runners_{date_str}.csv")
    odds_path = os.path.join("odds", f"runners_{date_str}_{race_num}.csv")

    if not os.path.exists(runner_path):
        raise FileNotFoundError(f"Runner data not found: {runner_path}")
    if not os.path.exists(odds_path):
        raise FileNotFoundError(f"Odds data not found: {odds_path}")

    # Load full predictions CSV
    runner_df = pd.read_csv(runner_path)

    # Filter for the given race number
    race_df = runner_df[runner_df["race"] == race_num].copy()
    if race_df.empty:
        raise ValueError(f"No data found in runner CSV for race {race_num}")

    # Load odds file
    odds_df = pd.read_csv(odds_path, sep=";")

    # Filter for "WIN" type odds
    odds_df = odds_df[odds_df["type"] == "WIN"]

    # Drop the "race" column from odds_df
    odds_df = odds_df.drop(columns=["race"])

    race_df["horseNumber"] = race_df["horseNumber"].astype(str)

    odds_df["horseNumber"] = odds_df["horseNumber"].astype(str)

    # Merge — adjust key as needed
    merged_df = pd.merge(race_df, odds_df, on="horseNumber", how="left")
    with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', False):
        print(merged_df)

    return merged_df

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict(model, X):
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X).squeeze(1)  # Shape (N,)
    return preds.numpy()

def predict_race(model, race_X, feature_names=[]):
    """
    Predict winning probabilities for all horses in a race.
    """

    # if not isinstance(race_X, pd.DataFrame):
    #     # Create a DataFrame if feature_columns are provided
    #     if feature_names is not None:
    #         race_X_df = pd.DataFrame(race_X, columns=feature_names)
    #     else:
    #         race_X_df = pd.DataFrame(race_X)
    # else:
    #     race_X_df = race_X

    # if race_X_df.isna().any().any():
    #     print("NaN found in race_X:")
    #     print(race_X_df[race_X_df.isna().any(axis=1)])

    scores = predict(model, race_X)  # Get predicted scores
    probabilities = softmax(scores)  # Turn scores into win probabilities
    print("prob", probabilities)
    return probabilities


def prediction_entropy(win_probs):
    """
    Calculate entropy of the win probability distribution.
    Lower entropy = more confident (peaky distribution).
    Higher entropy = more uncertain (flat distribution).
    """
    win_probs = np.array(win_probs)
    win_probs = win_probs / win_probs.sum()  # Ensure it's a valid probability distribution
    ent = entropy(win_probs, base=2)  # Use base 2 → entropy in bits
    return ent
# =================== MAIN ===================

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("race", type=int, help="Race number (e.g. 1, 2, 3...)")
    parser.add_argument("date", type=str, help="Race date (YYYY-MM-DD)")
    parser.add_argument("model", type=str, help="Model path (YYYY-MM-DD)")


    args = parser.parse_args()
    #model_folder = "models/neural_net_model_20250428_033334"  # <-- Your latest model folder
    model_folder = args.model if args.model else "models/neural_net_model_20250428_113134" # <- replace with your actual timestamped folder
    #model_path = os.path.join(model_folder, "model.pth")
    model, feature_list = load_model(model_folder)

    # Load and preprocess data
    df_full = mergeData(args.race, args.date)
    df = preprocess_race_data(df_full.copy())


    # Predict win probabilities
    race_X, dy, c = process_by_race(df_full, feature_list)
    win_probs = predict_race(model, race_X)

    # Attach predictions back to the original dataframe
    df_full["win_prob"] = win_probs

    # Now calculate "value bet"
    # Assume lower odds mean higher market probability
    # Typical calculation: if win_prob * market_odds > 1 => value bet
    # You need to have "win_odds" column (or whatever it's called in your data)
    
    if "odds" not in df_full.columns:
        raise ValueError("Missing 'odds' column in the race data!")
    
    entropy_score = prediction_entropy(win_probs)

    df_full["expected_value"] = df_full["win_prob"] * df_full["odds"]
    df_full["is_value_bet"] = df_full["expected_value"] > 1

    # Display only the useful columns
    display_cols = ["horseNumber", "name_ch", "odds", "win_prob", "expected_value", "is_value_bet"]
    final_output = df_full[display_cols].sort_values(by="win_prob", ascending=False)


    print("\nUsing model: : " + model_folder)
    print(f"\nPrediction Entropy: {entropy_score:.20f} bits")
    print("\n=== Value Bets ===")
    print(final_output.to_string(index=False))


