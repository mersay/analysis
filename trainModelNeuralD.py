import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
import os
import argparse
import random
import ast
import numpy as np
import datetime
from functions.encodeVenueRacetrackRacecourse import venueRacetrackRacecourse, encodeVenueCombo
from functions.encodeGoing import goings
from functions.encodeCountry import countries
from functions.encodeClass import classes
from functions.encodeSex import sexes
from functions.encodeClassChange import classChanges
import datetime, os, json, torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ----- Step 1: Load or Create Config -----
CONFIG_PATH = "model/model_config.json"

# --- Winner Predictor Model ---
class WinnerPredictor(nn.Module):
    def __init__(self, num_categories_dict, embedding_dim_dict, num_numerical):
        super().__init__()

        super().__init__()
        # Create embeddings for each column using the specified embedding dimension
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(num_classes, embedding_dim_dict[col])  # Use the specific embedding_dim for each column
            for col, num_classes in num_categories_dict.items()
        })
        
        # Calculate the total input size (sum of embeddings and numerical features)
        input_size = sum(embedding_dim_dict[col] for col in num_categories_dict) + num_numerical

        self.mlp = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_cat, x_num):
        batch_size, num_runners, _ = x_num.shape

        embedded = [
            self.embeddings[col](x_cat[col]) for col in sorted(x_cat.keys())
        ]

        embedded = torch.cat(embedded, dim=2)
        x = torch.cat([embedded, x_num], dim=2)

        logits = self.mlp(x).squeeze(-1)
        return logits

def prepare_inputs(df, cat_cols, num_cols, mappings, custom_mapping_funcs=None):
    custom_mapping_funcs = custom_mapping_funcs or {}

    def safe_lookup(val, mapping):
        """Return the mapped value or raise an error if not found"""
        if val in mapping:
            return mapping[val]
        else:
            print(f"Warning: Value '{val}' not found in mapping.")
            # Optionally return a default value or raise a KeyError
            return -1  # Default if not found, you could also raise an exception

    x_cat = {}
    for col in cat_cols:
        print(f"Processing column: {col}")

        # Check if column exists in df
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataframe.")

        if col in custom_mapping_funcs:
            # Use custom function for this column
            try:
                x_cat[col] = torch.tensor([custom_mapping_funcs[col](val) for val in df[col]], dtype=torch.long)
            except Exception as e:
                print(f"Error in custom mapping for column '{col}': {e}")
                raise
        else:
            # Use safe lookup for default dictionary-based encoding
            try:
                x_cat[col] = torch.tensor([safe_lookup(val, mappings[col]) for val in df[col]], dtype=torch.long)
            except KeyError as e:
                print(f"KeyError while processing column '{col}': {e}")
                raise

    # Ensure numerical columns are properly formatted and present in df
    for col in num_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataframe.")

    x_num = torch.tensor(df[num_cols].values, dtype=torch.float32)
    return x_cat, x_num

def remove_non_serializable(d):
    clean = {}
    for k, v in d.items():
        if isinstance(v, dict):
            clean[k] = remove_non_serializable(v)
        elif callable(v):
            continue  # Skip functions
        else:
            clean[k] = v
    return clean

def prepare_data_for_races(races, race_list, config, is_future=False):
    max_runners = 14
    dummy_runner = config["dummy"]
    numerical_cols_to_scale = config["numerical_columns"]
    avg_columns = config["average_columns"]

    x_cat_padded_all_races = []
    x_num_padded_all_races = []
    is_dummy_all_races = []
    y = [] if not is_future else None

    for race_key in race_list:
        race_date, race_number = race_key[0]
        race_df = races.get_group((race_date, race_number)).copy()
        num_current_runners = len(race_df)

        # Step 1: Compute group averages (before dummy insertion), fill NaNs with -1 first
        for col in avg_columns:
            if col in race_df.columns:
                race_df[col] = race_df[col].fillna(-1)

        race_avg_values = {
            col: race_df[race_df[col] != -1][col].mean() if not race_df[race_df[col] != -1][col].empty else -1
            for col in avg_columns
            if col in race_df.columns
        }

        # Step 2: Prepare padded numeric features (dummy runners first)
        padded_num_list = []
        for _ in range(max_runners - num_current_runners):
            dummy_vals = [
                race_avg_values.get(col, dummy_runner.get(col, -1.0)) if col in avg_columns else dummy_runner.get(col, -1.0)
                for col in numerical_cols_to_scale
            ]
            dummy_vals.append(1)  # isDummy
            padded_num_list.append(dummy_vals)

        # Step 3: Process real runners
        for col in avg_columns:
            if col in race_df.columns:
                race_df[col] = race_df[col].replace(-1, race_avg_values[col])

        x_num_race = race_df[numerical_cols_to_scale].copy()
        scaler = StandardScaler()
        x_num_scaled = x_num_race.copy()

        if "distance" in x_num_scaled.columns:
            min_dist, max_dist = 1000, 2400
            x_num_scaled["distance"] = (x_num_scaled["distance"] - min_dist) / (max_dist - min_dist)

        cols_to_standard_scale = [col for col in x_num_scaled.columns if col != "distance"]
        if cols_to_standard_scale:
            x_num_scaled[cols_to_standard_scale] = scaler.fit_transform(x_num_scaled[cols_to_standard_scale])

        for i in range(num_current_runners):
            padded_num_list.append(x_num_scaled.iloc[i].to_list() + [0])  # isDummy = 0

        x_num_padded_race = torch.tensor(padded_num_list, dtype=torch.float32)

        # Step 4: Create padded categorical features (dummy runners first)
        padded_cat_dict = {col: [] for col in config["categorical_columns"] + ["isDummy"]}
        race_shared_fields = ["venueRacetrackRacecourse", "going", "money", "class", "distance"]
        shared_values = {col: race_df.iloc[0][col] for col in race_shared_fields if col in race_df.columns}

        for _ in range(max_runners - num_current_runners):
            for col in config["categorical_columns"]:
                padded_cat_dict[col].append(shared_values.get(col, "Unknown") if col in race_shared_fields else dummy_runner.get(col, "Unknown"))
            padded_cat_dict["isDummy"].append(1)

        x_cat_race = race_df[config["categorical_columns"]].copy()
        for i in range(num_current_runners):
            for col in config["categorical_columns"]:
                padded_cat_dict[col].append(x_cat_race.iloc[i][col])
            padded_cat_dict["isDummy"].append(0)

        # Step 5: Encode categorical features
        x_cat_encoded_padded_race = {}
        for col in config["categorical_columns"]:
            if col in config["category_mappings"]:
                mapping = config["category_mappings"][col]
                if "Unknown" not in mapping:
                    mapping["Unknown"] = len(mapping)
                fallback_index = mapping["Unknown"]
                encoded_vals = [mapping.get(val, fallback_index) for val in padded_cat_dict[col]]
                x_cat_encoded_padded_race[col] = torch.tensor(encoded_vals, dtype=torch.long)

            elif col in config.get("custom_mapping_funcs", {}):
                func = config["custom_mapping_funcs"][col]
                encoded_vals = [func(val) for val in padded_cat_dict[col]]
                x_cat_encoded_padded_race[col] = torch.tensor(encoded_vals, dtype=torch.long)

            else:
                print(f"  [WARN] No mapping found for column '{col}', assigning -1 to all.")
                x_cat_encoded_padded_race[col] = torch.full((max_runners,), -1, dtype=torch.long)

        # Save isDummy separately as its own tensor
        is_dummy_tensor_race = torch.tensor(padded_cat_dict["isDummy"], dtype=torch.long)

        # Save final padded features for this race
        x_cat_encoded_padded_race["isDummy"] = is_dummy_tensor_race
        x_cat_padded_all_races.append(x_cat_encoded_padded_race)
        x_num_padded_all_races.append(x_num_padded_race)
        is_dummy_all_races.append(is_dummy_tensor_race)

        # Step 6: Prepare target (y) if needed
        if not is_future:
            try:
                winner_runner_id = race_df[race_df["final_position"] == 1].index[0]
                winner_index_original = race_df.index.get_loc(winner_runner_id)
                y.append(winner_index_original)
            except IndexError:
                print(f"Warning: No winner found for race {race_date}, {race_number}")
                y.append(-1)

    # Step 7: Stack all races into tensors
    cat_fields = config["categorical_columns"]
    x_cat_train_tensor = {
        field: torch.stack([race_data[field] for race_data in x_cat_padded_all_races])
        for field in cat_fields
    }
    x_num_train_tensor = torch.stack(x_num_padded_all_races)
    is_dummy_tensor = torch.stack(is_dummy_all_races)

    y_train_tensor = torch.tensor(y, dtype=torch.long) if not is_future else None
    if y_train_tensor is not None:
        print("Number of valid targets (y != -1):", (y_train_tensor != -1).sum().item())
        print("Total y entries:", len(y_train_tensor))

    return x_cat_train_tensor, x_num_train_tensor, y_train_tensor, is_dummy_tensor

def split_races(races, val_split=0.2):
    # Split races into training and validation sets
    race_list = list(races)
    #random.shuffle(race_list)  # Shuffle races randomly
    
    # Split based on the given validation split
    split_idx = int(len(race_list) * (1 - val_split))
    train_races = race_list[:split_idx]
    val_races = race_list[split_idx:]
    
    return train_races, val_races

def train_model(df, config):
    # Create a timestamped folder, save model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("models", "mlp_"+ timestamp)
    os.makedirs(save_dir, exist_ok=True)

    config_path = os.path.join(save_dir, "config.json")
    model_path = os.path.join(save_dir, "model.pt")

    # Save config
    with open(config_path, "w") as f:
        serializable_config = remove_non_serializable(config)
        json.dump(serializable_config, f, indent=2)

    print(f"Saved config to {config_path}")

    # Group data by race
    races = df.groupby(['date', 'race'])
    print(f"Number of unique races: {len(races)}")

    # Split races into training and validation
    train_races, val_races = split_races(races, val_split=0.2)
    print(f"Number of training races: {len(train_races)}")
    print(f"Number of validation races: {len(val_races)}")

    # Prepare training and validation data
    x_cat_train, x_num_train, y_train, is_dummy_train = prepare_data_for_races(races, train_races, config)
    x_cat_val, x_num_val, y_val, is_dummy_val = prepare_data_for_races(races, val_races, config)

    print(f"Length of x_cat_train keys: {len(x_cat_train.keys()) if isinstance(x_cat_train, dict) else 0}")
    if isinstance(x_cat_train, dict) and x_cat_train:
        cat_fields = list(x_cat_train.keys())
    else:
        print("Warning: x_cat_train is empty or not a dictionary!")
        return # Or handle this case

    # Stack numerical and target tensors
    x_num_train_tensor = x_num_train
    x_num_val_tensor = x_num_val
    y_train_tensor = y_train
    y_val_tensor = y_val

    # Initialize the model
    model = WinnerPredictor(
        config["num_categories_dict"],
        config["embedding_dim_dict"],
        len(config["numerical_columns"]) + 1 # +1 for isDummy
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 101):
        model.train()
        total_loss = 0

        for i in range(len(y_train_tensor)):
            x_cat_batch = {field: x_cat_train[field][i].unsqueeze(0) for field in cat_fields}
            x_num_batch = x_num_train_tensor[i].unsqueeze(0)

            for col, tensor in x_cat_batch.items():
                if (tensor < 0).any():
                    print(f"Invalid index in column '{col}':", tensor)
            y_batch = y_train_tensor[i].unsqueeze(0)

            optimizer.zero_grad()
            output = model(x_cat_batch, x_num_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        for i in range(len(y_val_tensor)):
            x_cat_batch = {field: x_cat_val[field][i].unsqueeze(0) for field in cat_fields}
            x_num_batch = x_num_val_tensor[i].unsqueeze(0)
            y_batch = y_val_tensor[i].unsqueeze(0)

            output = model(x_cat_batch, x_num_batch)  # [1, num_runners]

            # Mask dummy runners by setting their logits to a very low value
            dummy_mask = is_dummy_val[i]  # Boolean tensor: True for dummy runners
            output[0][dummy_mask] = float('-inf')

            pred = output.argmax(dim=1)  # Now only real runners are considered
            correct += (pred == y_batch).sum().item()
            total += 1

        accuracy = correct / total
        print(f"Epoch {epoch:03d} | Train Loss: {total_loss:.4f} | Val Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# ----- Step 5: Prediction -----
def predict(df_race, model, config):
    model.eval()

    max_runners = 14
    dummy_runner = config["dummy"]
    
    avg_columns = config["average_columns"]
    # Fill missing numeric values with race-wise average
    for col in avg_columns:
        if col in df_race.columns:
            df_race[col] = df_race[col].fillna(df_race[col].mean(skipna=True))

    x_num_race = df_race[config["numerical_columns"]].copy()
    x_cat_race = df_race[config["categorical_columns"]].copy()

    num_current_runners = len(x_num_race)

    # Pad numerical features
    padded_num_list = []
    for i in range(num_current_runners):
        padded_num_list.append(x_num_race.iloc[i].to_list() + [0])  # isDummy = 0
    for _ in range(max_runners - num_current_runners):
        padded_num_list.append([dummy_runner.get(col, 0.0) for col in config["numerical_columns"]] + [1])  # isDummy = 1
    x_num_padded = torch.tensor([padded_num_list], dtype=torch.float32)  # Add batch dim

    # Pad categorical features
    padded_cat_dict = {col: [] for col in config["categorical_columns_mappings"]}
    for i in range(num_current_runners):
        for col in config["categorical_columns"]:
            padded_cat_dict[col].append(x_cat_race.iloc[i][col])
        padded_cat_dict["isDummy"].append(0)
    for _ in range(max_runners - num_current_runners):
        for col in config["categorical_columns"]:
            padded_cat_dict[col].append("Unknown")
        padded_cat_dict["isDummy"].append(1)

    # Encode categorical features
    x_cat_encoded = {}
    for col in config["categorical_columns"]:
        if col in config.get("custom_mapping_funcs", {}):
            func = config["custom_mapping_funcs"][col]
            x_cat_encoded[col] = torch.tensor([[func(val) for val in padded_cat_dict[col]]], dtype=torch.long)
        elif col in config["category_mappings"]:
            mapping = config["category_mappings"][col]
            x_cat_encoded[col] = torch.tensor([[mapping.get(val, -1) for val in padded_cat_dict[col]]], dtype=torch.long)
        else:
            print(f"Warning: No mapping found for column '{col}', assigning -1")
            x_cat_encoded[col] = torch.full((1, max_runners), -1, dtype=torch.long)
    #x_cat_encoded["isDummy"] = torch.tensor([[val for val in padded_cat_dict["isDummy"]]], dtype=torch.long)

    # Predict
    with torch.no_grad():
        logits = model(x_cat_encoded, x_num_padded)
        probs = torch.softmax(logits, dim=1)

    return probs.squeeze().tolist()  # shape: (num_runners,)


def pad_features_with_avg(race_df, x_num_df, x_cat_df, max_runners, dummy_runner, avg_columns):
    """
    Pads missing runners using dummy values, but replaces selected features with race-level averages.
    
    race_df: full DataFrame of runners for the current race
    x_num_df: DataFrame of numerical features for runners in the race
    x_cat_df: DataFrame of categorical features for runners in the race
    max_runners: number to pad up to
    dummy_runner: dict of fallback values for missing runners
    avg_columns: list of numerical column names to use race average
    """

    num_current = len(x_num_df)
    padded_num = x_num_df.copy()
    padded_cat = x_cat_df.copy()

    # Compute per-race averages for relevant features
    race_avgs = {}
    for col in avg_columns:
        race_avgs[col] = race_df[col].dropna().astype(float).mean()

    # Pad with dummy/average values
    for _ in range(max_runners - num_current):
        padded_row = []
        for col in x_num_df.columns:
            if col in race_avgs and not np.isnan(race_avgs[col]):
                padded_row.append(race_avgs[col])
            else:
                padded_row.append(dummy_runner.get(col, 0.0))  # fallback if no race avg
        # Use pd.concat to append the padded row
        padded_num = pd.concat([padded_num, pd.DataFrame([padded_row], columns=x_num_df.columns)], ignore_index=True)

        padded_cat = pd.concat([padded_cat, pd.DataFrame([["Unknown"] * len(x_cat_df.columns)], columns=x_cat_df.columns)], ignore_index=True)

    return padded_num, padded_cat


# form raw variable
def venue_racetrack_racecourse_key(row):
    venue = row["venue"] or "Unknown"
    racetrack = row["racetrack"] or "Unknown"
    racecourse = row["racecourse"] or "Unknown"

    # Special case for ST_ALL_WEATHER_TRACK
    if venue == "ST" and racetrack == "ALL_WEATHER_TRACK":
        racecourse = "Unknown"

    return f"{venue}_{racetrack}_{racecourse}"

# Function to safely convert a string to a list if it is a string representation of a list (for cols like runningPosition)
def convert_to_list(val):
    if isinstance(val, str):
        try:
            # Ensure the string has proper list format by removing spaces around commas
            val = val.replace(" ", "")  # Remove spaces between numbers in list-like strings
            return ast.literal_eval(val)
        except Exception as e:
            print(f"Error converting value {val}: {e}")
            return []  # Return empty list on error
    return val

# --- Main Entry ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str, help="CSV path")
    #parser.add_argument("--config", default="model/model_config.json", help="Optional model config path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    
    # if os.path.exists(args.config):
    #     with open(args.config) as f:
    #         config = json.load(f)
    # else:

    #print(df.columns.tolist())

    # Apply the conversion function to the 'runningPosition' column
    df['runningPosition'] = df['runningPosition'].apply(convert_to_list)
    df["venueRacetrackRacecourse"] = df.apply(venue_racetrack_racecourse_key, axis=1)
    df["final_position"] = df["runningPosition"].apply(lambda rp: rp[-1] if isinstance(rp, list) and len(rp) > 0 else None)

    categorical_columns = ['venueRacetrackRacecourse', 'class', 'country', 'going', 'sex', 'classChange']
    numerical_columns = [
        #'money', 
        'distance', 
        'drawBiasScore', 
        #'speedRating',
        #'SRMISDATA',
        #'daysSinceLastRace', 
        'jocMisData',
        'jocWinPercent', 
        #'trainerMisData', 
        #'trainerWinPercent',
        #'avglbw',
        'age',
        'AGEMISDATA',
        #'goingBias',
        'raceCount',
        #'newDist',
        #'finishingKick'
    ]

    num_categories_dict = {
        "venueRacetrackRacecourse": len(venueRacetrackRacecourse.items()),
        "class": len(classes.items()),
        "country": len(countries.items()),
        "going": len(goings.items()),
        "sex": len(sexes.items()),
        "classChange": len(classChanges.items())
    }

    avg_columns = [
        'drawBiasScore',
        'speedRating',
        'daysSinceLastRace',
        'jocWinPercent',
        'trainerWinPercent',
        'avglbw',
        'age',
        'goingBias',
        'raceCount',
        'handicapWeight',
        'horseWeight'
    ]
    
    dummy_runner = {
        "money": 0.0,
        "distance": 0.0,
        "drawBiasScore": 0.0,
        "speedRating": 0.0,
        "SRMISDATA": 1,
        "daysSinceLastRace": 0.0,
        "jocMisData": 1,
        "jocWinPercent": 0.0,
        "trainerMisData": 1,
        "trainerWinPercent": 0.0,
        "avglbw": 0.0,
        "age": 4.0,
        "AGEMISDATA": 1,
        "goingBias": 0.0,
        "raceCount": 0,
        "handicapWeight": 0.0,
        "horseWeight": 0.0,
        "isDummy": 1,
        "classChange": "noChange",
        "newDist": 0,
        "finishingKick": 0
    }
    config = {
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "average_columns": avg_columns,
        "dummy": dummy_runner,
        "num_categories_dict": num_categories_dict, #number of unique categories
        "embedding_dim_dict": { #output size of the embedding
            "venueRacetrackRacecourse": len(venueRacetrackRacecourse.items()),
            "class": len(classes.items()),
            "country": 4,
            "going": 4,
            "sex": 3,
            "classChange": 3
        },
        "embedding_dim": 4,
        "target_variable": "finish_position",
        "category_mappings": {
            "venueRacetrackRacecourse": venueRacetrackRacecourse,
            "country": countries,
            "going": goings,
            "class": classes,
            "sex": sexes,
            "classChange": classChanges
        }, 
        "custom_mapping_funcs": {
            "venueRacetrackRacecourse": encodeVenueCombo
        }
    }
    

    train_model(df, config)


