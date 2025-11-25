import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime
import json

# =================== CONFIG ===================
RACE_FEATURES = ["distance", "going", "venue", "racecourse", "racetrack", "class", "money"]  # shared across all runners
RUNNER_FEATURES = [
    #'date',
    #'race',
    'place',
    'horseNumber',
    #'horseCode',
    #'jockey',
    #'trainer',
    'handicapWeight',
    'horseWeight',
    'draw',
    #'rawlbw',
    'lbw',
    #'runningPosition',
    #'finishTime',
    'odds',
    'country',
    'colour',
    'sex',
    'currentRating',
    'duration',
    'lifeWin',
    'daysSinceLastRace',
    'newDist',
    'jocWinCount',
    'jocWinPercent',
    'jocMisData',
    'avesprat',
    'avglbw',
    'age',
    'AGEMISDATA',
    'drawBiasScore',
    'daysSinceLastWorkout',
    'WOMISDATA',
    'winRate',
    'placeRate'
]

MAX_RUNNERS = 14  # pad races up to 14 runners
RACE_CATEGORICAL_VARS = ["going", "venue", "racecourse", "racetrack", "class"]
CATEGORICAL_VARS = ["country", "colour", "sex"]

# ============== LOAD AND PREPARE DATA ==============
def get_feature_names(encoder: ColumnTransformer, input_features: list) -> list:
    """
    Extract feature names from a ColumnTransformer.
    Handles OneHotEncoder inside the transformer.
    """
    output_features = []

    for name, transformer, columns in encoder.transformers_:
        if name == 'remainder' and transformer == 'passthrough':
            # Passthrough columns
            passthrough = [col for col in input_features if col not in sum([c if isinstance(c, list) else [c] for _, _, c in encoder.transformers_ if c != 'passthrough'], [])]
            output_features.extend(passthrough)
        elif hasattr(transformer, 'get_feature_names_out'):
            encoded = transformer.get_feature_names_out()
            output_features.extend(encoded)
        else:
            output_features.extend(columns)

    return output_features

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_runner_data(df):
    encoder = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_VARS)
        ],
        remainder="passthrough"
    )
    encoded = encoder.fit_transform(df[CATEGORICAL_VARS])
    return encoded, encoder

def preprocess_race_data(df):
    # Fill missing categorical values with a placeholder
    df = df.copy()
    df[RACE_CATEGORICAL_VARS] = df[RACE_CATEGORICAL_VARS].fillna("unknown")

    encoder = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), RACE_CATEGORICAL_VARS)
        ],
        remainder="passthrough"
    )
    encoded = encoder.fit_transform(df[RACE_FEATURES])
    return encoded, encoder

def get_default_runner_row():
    return {
        col: "unknown" if col in CATEGORICAL_VARS else 0
        for col in RUNNER_FEATURES
    }

def get_default_race_row():
    return {
        col: "unknown" if col in RACE_CATEGORICAL_VARS else 0
        for col in RACE_FEATURES
    }

def pad_race_features(race_df, race_meta, runner_encoder, race_encoder):
    padded_runners = []

    try:
        # Handle missing values in race_meta
        race_meta_filled = race_meta[RACE_FEATURES].copy()
        race_meta_filled = race_meta_filled.fillna("unknown")  # Only works because our encoder handles it

        race_df_for_encoding = pd.DataFrame([race_meta_filled])
        encoded_race = race_encoder.transform(race_df_for_encoding).flatten()

    except Exception as e:
        print("[Error] Failed encoding race meta:")
        print(race_meta[RACE_FEATURES])
        raise e

    for i in range(MAX_RUNNERS):
        try:
            if i < len(race_df):
                runner = race_df.iloc[i]
                runner_df = pd.DataFrame([runner[RUNNER_FEATURES]])
            else:
                runner_df = pd.DataFrame([get_default_runner_row()])

            encoded_runner = runner_encoder.transform(runner_df).flatten()
        except Exception as e:
            print(f"[Error] Failed encoding runner at position {i}")
            print("[Runner DF]")
            print(runner_df)
            raise e

        # Append runner + encoded race features
        runner_with_race_features = encoded_runner.tolist() + encoded_race.tolist()
        padded_runners.extend(runner_with_race_features)

    return padded_runners


def process_by_race(df, runner_encoder, race_encoder):
    X, y = [], []
    for (date, race_num), group in df.groupby(["date", "race"]):
        # No need to sort the group by "duration" if the winner is always in the first row
        race_meta = group.iloc[0]  # shared race-level features
        features = pad_race_features(group, race_meta, runner_encoder, race_encoder)

        winner_row = group[group["place"] == 1]
        if winner_row.empty:
            print(f"[Warning] No winner found for race on {date}, race {race_num}")
            continue

        
        winner_idx = group.index.get_loc(winner_row.index[0])

        X.append(features)
        y.append(winner_idx)
    return np.array(X), np.array(y)

# ============== TRAIN AND EVALUATE ==============
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
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

# ============== SAVE MODEL ==============
def save_model(model, acc, all_feature_names, runner_encoder, race_encoder):
    # Create timestamp and folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = f"models/multinomial_logistic_model_{timestamp}"
    os.makedirs(model_folder, exist_ok=True)

    # === Save the model ===
    model_path = os.path.join(model_folder, "model.pkl")
    joblib.dump(model, model_path)
    joblib.dump(race_encoder, os.path.join(model_folder, "race_encoder.pkl"))
    joblib.dump(runner_encoder, os.path.join(model_folder, "runner_encoder.pkl"))

    # === Save metadata ===
    metadata = {
        "timestamp": timestamp,
        "accuracy": acc,
        "model_type": "Multinomial Logistic Regression",
        "solver": "lbfgs",
        "max_iter": 500,
        "target_variable": "place"
    }
    with open(os.path.join(model_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # === Save feature names ===
    with open(os.path.join(model_folder, "features.txt"), "w") as f:
        for name in all_feature_names:
            f.write(name + "\n")

    

    print(f"Model and metadata saved in: {model_folder}")



# ============== MAIN ==============
if __name__ == "__main__":
    df = load_data("data/race_data_20250420.csv")
    #df = df[RUNNER_FEATURES]
    df = df.drop(columns=['runningPosition', 'results'])  # List unwanted columns here
    _, runner_encoder = preprocess_runner_data(df)
    _, race_encoder = preprocess_race_data(df)
    X, y = process_by_race(df, runner_encoder, race_encoder)
    runner_feature_names = get_feature_names(runner_encoder, RUNNER_FEATURES)
    race_feature_names = get_feature_names(race_encoder, RACE_FEATURES)

    all_feature_names = []
    for i in range(MAX_RUNNERS):
        for name in runner_feature_names:
            all_feature_names.append(f"runner_{i}_{name}")

        for name in race_feature_names:
            all_feature_names.append(f"race_{name}")
    model, acc = train_model(X, y)
    save_model(model, acc, all_feature_names, runner_encoder, race_encoder)
