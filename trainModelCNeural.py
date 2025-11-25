import ast
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import datetime
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from functions.encodeClass import encodeClass, getAllClassVariables
from functions.encodeGoing import encodeGoing, getAllGoingVariables
from functions.encodeVenue import encodeVenue, getAllVenueVariables
from functions.encodeRacecourse import encodeRacecourse, getAllRacecourseVariables
from functions.encodeSex import encodeSex, getAllSexVariables
from functions.encodeCountry import encodeCountry, getAllCountryVariables
from functions.scaleDistance import scaleDistance
from functions.isDummy import isDummy
from trainModelBNeural import save_model
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # No limit on line width
pd.set_option('display.max_colwidth', None) # No limit on individual column width


# this is 9 factors only
# =================== CONFIG ===================
MAX_ITER=1500
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

DROP_COLUMNS = RACE_CATEGORICAL_COLUMNS + RUNNER_CATEGORICAL_COLUMNS + [
    "date",
    "race",
    'distance',
    "results",
    #"place",
    "horseNumber",
    "horseCode",
    "jockey",
    "trainer",
    #"handicapWeight",
    #"horseWeight",
    #"draw",
    #"lbw",
    "runningPosition",
    "finishTime",
    "odds",
    #"country",
    #"colour",
    #"sex",
    "currentRating",
    #"speedRating",
    #"daysSinceLastRace",
    "jocWinCount",
    #"jocWinPercent",
    #"jocMisData",
    #"avglbw",
    #"age",
    #"AGEMISDATA",
    #"drawBiasScore"
]

def load_data(path):
    df = pd.read_csv(path)
    return df, str(path)

# Load the average LBW data from the JSON file
def load_avg_lbw_data(filename="averageLbwByYearAndDistance.json"):
    with open(filename, "r") as file:
        return json.load(file)

# Example of loading data
avg_lbw_by_distance = load_avg_lbw_data()

def combined_race_encoder(row):
    results = {}

    # Apply multiple encodings and update results
    results.update(encodeVenue(row['venue'], row['racetrack']))  # Venue + Racetrack encoding
    results.update(encodeGoing(row['going']))               # Going condition encoding
    results.update(encodeClass(row['class']))                # Class encoding, if needed
    results.update(encodeRacecourse(row['racecourse']))                # Class encoding, if needed
    results.update(scaleDistance(row['distance']))

    # You can add even more encoding functions here
    return results

def combined_runner_encoder(row):
    results = {}

    # Apply multiple encodings and update results
    results.update(isDummy(row['horseCode']))
    results.update(encodeSex(row['sex']))  
    results.update(encodeCountry(row['country']))    


    # You can add even more encoding functions here
    return results

def parse_running_position(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return None

def preprocess_race_data(df):
    # Convert runningPosition to list if it's a string
    if "runningPosition" in df.columns:
        df["runningPosition"] = df["runningPosition"].apply(parse_running_position)

    # Apply race-level encoder
    encoded_df = df.apply(lambda row: pd.Series(combined_race_encoder(row)), axis=1)
    df = pd.concat([df, encoded_df], axis=1)
    encoded_df = df.apply(lambda row: pd.Series(combined_runner_encoder(row)), axis=1)
    # Concatenate
    df = pd.concat([df, encoded_df], axis=1)
    return df
    
def create_default_runner(race_meta, place):
    """
    Creates a default runner entry with basic info filled from race_meta.
    Other runner-specific info can be set as needed.
    """
    
    runner = {
        "date": race_meta.date,
        "race": race_meta.race,
        "venue": race_meta.venue,
        "racetrack": race_meta.racetrack,
        "going": race_meta.going,
        "class": race_meta["class"],
        "racecourse": race_meta.racecourse,
        "distance": race_meta.distance,
        #"distanceScaled": race_meta.distanceScaled,
        "money": race_meta.money,
        # runner
        "results": [],
        "place": place,
        "horseNumber": place,
        "horseCode": "dummyID",
        "draw": None,
        "jockey": "AAA",
        "trainer": "BBB",
        "handicapWeight": 1000,
        "horseWeight": 1000,
        "draw": place,
        "lbw": 99,
        "runningPosition": [place, place, place],
        "finishTime": "",
        "odds": 999,
        "country": "ABC",
        "colour": "ABC",
        "sex": "ABC",
        "currentRating": 0,
        "speedRating": None,
        "daysSinceLastRace": 999,
        "jocWinCount":0,
        "jocWinPercent": 0,
        "jocMisData": 1,
        "trainerWinCount":0,
        "trainerWinPercent": 0,
        "trainerMisData": 1,
        "avesprat": -9999,
        "avglbw": 99,
        "age": 0,
        "AGEMISDATA": 1,
        "drawBiasScore": 0,
        "goingBias": 0,
        "raceCount": 0
    }

    # Generate the encoded features
    encoded_race_features = combined_race_encoder(runner)
    encoded_runner_features = combined_runner_encoder(runner)

    # Merge base runner info and encoded features
    runner.update(encoded_race_features)
    runner.update(encoded_runner_features)
    #print("runner", runner)
    return runner

def pad_race_features(group, race_meta, feature_columns):
    """
    Pad race_df with dummy runners until it has target_num_runners.
    """
    num_existing = len(group)
    runners_to_add = MAX_RUNNERS - num_existing

    if runners_to_add > 0:
        dummy_runners = [create_default_runner(race_meta, num_existing + i) for i in range(runners_to_add)]
        dummy_df = pd.DataFrame(dummy_runners)
        group = pd.concat([group, dummy_df], ignore_index=True)

    features = group[feature_columns].values.tolist()
    #print(features)
    return features, group

def soft_label(row):
    # Extract year from datetime object
    # Convert the string date to a datetime object
    race_date = datetime.strptime(row['date'], '%Y-%m-%d')
    year = race_date.year

    # Construct the key for the dictionary based on distance and year
    dist_cat = f"{row['distance']}_{year}_2_3"

    # Retrieve the average LBW for the given distance and year (with fallback to 1.0)
    avg_lbw = avg_lbw_by_distance.get(dist_cat, 1.0)

    # Calculate soft label
    rp = row.get('runningPosition')
    if isinstance(rp, list) and len(rp) > 0:
        final_pos = rp[-1]
    else:
        final_pos = None

    if final_pos == 1:
        return 1.0
    elif final_pos in [2, 3, 4] and row.get('lbw') is not None and avg_lbw:
        return max(0.0, 1.0 - (row['lbw'] / avg_lbw))
    else:
        return 0.0


def process_by_race(df):
    X, y, feature_columns = [], [], []

    for (date, race_num), group in df.groupby(["date", "race"]):
        # No need to sort the group by "duration" if the winner is always in the first row
        #print(f"Processing group for Date: {date}, Race: {race_num}") # Added this line
        print("group", group)
        race_meta = group.iloc[0]  # shared race-level features
        feature_columns = [col for col in df.columns if col not in ['place', 'lbw'] + DROP_COLUMNS]
        features, group = pad_race_features(group, race_meta, feature_columns)

        # winner_row = group[group["runningPosition"][-1] == 1]
        # if winner_row.empty:
        #     continue

        # winner_idx = group.index.get_loc(winner_row.index[0])

        # X.append(features)
        # y.append(winner_idx)

        # Apply the soft_label function to each row in the group
        soft_labels = group.apply(soft_label, axis=1).tolist()
        # Extend the features and labels lists for each horse in the race
        X.extend(features)
        y.extend(soft_labels)
    #print(y)

    return np.array(X), np.array(y), feature_columns

# ============== TRAIN AND EVALUATE ==============
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=MAX_ITER)
    model.fit(X_train, y_train)

    print("Classes in y_train:", np.unique(y_train))
    print("Classes in y_test:", np.unique(y_test))

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)
    print("Shape of y_probs:", y_probs.shape)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_probs, labels=list(range(MAX_RUNNERS)))

    print(f"Accuracy: {acc:.4f}, Log Loss: {loss:.4f}")
    return model, acc

# =================== Define Neural Network ===================
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output a single continuous value
        )

    def forward(self, x):
        return self.model(x)

# =================== Train Function ===================
def train_neural_network(X, y, input_dim, epochs=100, lr=0.001):
    model = SimpleNN(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train).squeeze(1)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            val_preds = model(X_val).squeeze(1)
            val_loss = loss_fn(val_preds, y_val).item()
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f} - Val Loss: {val_loss:.4f}")

    # After training, evaluate
    model.eval()
    final_preds = model(X_val).squeeze(1).detach().numpy()
    y_true = y_val.numpy()
    r2 = r2_score(y_true, final_preds)
    print(f"\nFinal R² score on validation set: {r2:.4f}")

    return model, r2

# def save_model(model, acc, training_variables, csvPath):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     model_folder = f"models/multinomial_logit_model_{timestamp}"
#     os.makedirs(model_folder, exist_ok=True)

    # Save model using joblib (best for scikit-learn models)
    # model_path = os.path.join(model_folder, "model.joblib")
    # joblib.dump(model, model_path)

    # # Save metadata
    # metadata = {
    #     "timestamp": timestamp,
    #     "accuracy": acc,
    #     "model_type": "Multinomial Logistic Regression",
    #     "solver": model.solver,
    #     "multi_class": model.multi_class,
    #     "loss_function": "Log-Likelihood (Cross-Entropy)",
    #     "training_variables": training_variables,
    #     "metric_type": "R2",
    #     "csvTrained": csvPath,
    #     "input_dim": len(training_variables),
    # }

    # print("Metadata before JSON dump:")
    # print(metadata)

    # with open(os.path.join(model_folder, "metadata.json"), "w") as f:
    #     json.dump(metadata, f, indent=4)

    # # Save feature names separately (optional)
    # with open(os.path.join(model_folder, "features.txt"), "w") as f:
    #     for var in training_variables:
    #         f.write(var + "\n")

    # print(f"\n✅ Model and metadata saved at: {model_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str, help="csv path")
    #parser.add_argument("date", type=str, help="Race date (YYYY-MM-DD)")
    args = parser.parse_args()
    df, csvPath = load_data(args.csv)
    df = preprocess_race_data(df)
    columns_with_nan = df.columns[df.isna().any()].tolist()
    print("Columns with NaN values:")
    #print(columns_with_nan)

    X, y, trainingVariables = process_by_race(df)
    print("trainingVariables", trainingVariables)
    model, r2 = train_neural_network(X, y, input_dim=X.shape[1])
    save_model(model, r2, trainingVariables, csvPath)
