import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from functions.publicOddsProb import calculate_win_probabilities_from_odds
import argparse
import os
import csv

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent breaking across multiple lines

# === Step 0: Parse command-line arguments ===
parser = argparse.ArgumentParser(description="Filter race data by race number.")
parser.add_argument('--race', type=int, required=True, help='Race number to filter for')
args = parser.parse_args()

# === Step 1: Load the saved model ===
model_path = "models/multinomial_logistic_model_20250415_143416/model.pkl"  # Update this to your actual filename
model = joblib.load(model_path)

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
file_1 = os.path.join(script_dir, 'predictions/runners_2025-04-16.csv')
odds_file = os.path.join(script_dir, 'odds/runners_2025-04-16_' + str(args.race) +'.csv')
# Example usage:
file1 = 'predictions/runners_2025-04-16.csv'  # Path to the first CSV file

# Read the first CSV file into a DataFrame
df1 = pd.read_csv(file_1)
print(f"Data from {file_1}:")
print(df1.head())  # Print first few rows of df1 for inspection

# Read the second CSV file into a DataFrame
df2 = pd.read_csv(odds_file)
print(f"Data from {odds_file}:")
print(df2.head())  # Print first few rows of df2 for inspection
# Step 1: Calculate implied probability
df2['implied_prob'] = 1 / df2['odds']

# Step 2: Normalize within each race so probabilities sum to 1
df2['normalized_prob'] = df2.groupby('race')['implied_prob'].transform(lambda x: x / x.sum())

# Merge the two DataFrames on 'race' and 'horseNumber'
merged_df = pd.merge(df1, df2, on=['race', 'horseNumber'], how='inner')  # 'inner' join by default
# Merge runners and odds data

merged_df = merged_df[merged_df['race'] == args.race]

def calculate_win_probabilities(horse_scores):
    """
    Given a list or array of scores (logits) for horses in a race,
    return the probability of each horse winning using softmax.

    Parameters:
        horse_scores (list or np.array): Scores for each horse in a single race

    Returns:
        list of float: Win probabilities summing to 1
    """
    scores = np.array(horse_scores, dtype=float)
    exp_scores = np.exp(scores - np.max(scores))  # for numerical stability
    probabilities = exp_scores / np.sum(exp_scores)
    return probabilities.tolist()


csv_columns = [
  'AGEMISDATA',
  'WFA',
  'WOMISDATA',
  '_id',
  'age',
  'allowance',
  'avesprat',
  'avglbw',
  'bestTime',
  'birthYear',
  'class',
  'code',
  'colour',
  'country',
  'currentRating',
  'dam',
  'damsire',
  'date',
  'daysSinceLastRace',
  'daysSinceLastWorkout',
  'distance',
  'draw',
  'drawBiasScore',
  'gear',
  'going',
  'handicapWeight',
  'horseCode',
  'horseId',
  'horseNumber',
  'horseWeight',
  'horseWeightChange',
  'importCategory',
  'internationalRating',
  'isStandBy',
  'jocMisData',
  'jocWinCount',
  'jocWinPercent',
  'jockey',
  'lastSixRuns',
  'lifeWin',
  'money',
  'name',
  'name_ch',
  'name_en',
  'newDist',
  'overweight',
  'owner',
  'placeRate',
  'priority',
  'race',
  'racecourse',
  'racetrack',
  'rating',
  'ratingChange',
  'ratingLower',
  'ratingUpper',
  'seasonStakes',
  'sex',
  'sire',
  'startOfSeasonRating',
  'time',
  'totalStakes',
  'trainer',
  'trainerPreference',
  'trumpCard',
  'venue',
  'winRate'
]

# Function to transform categorical variables into corresponding dummy variables
def fill_feature_columns(df):
    # Racecourse-related features
    df['racecourse_A+3'] = (df['racecourse'] == 'A+3').astype(int)
    df['racecourse_B'] = (df['racecourse'] == 'B').astype(int)
    df['racecourse_B+2'] = (df['racecourse'] == 'B+2').astype(int)
    df['racecourse_C'] = (df['racecourse'] == 'C').astype(int)
    df['racecourse_C+3'] = (df['racecourse'] == 'C+3').astype(int)
    df['racecourse_Unknown'] = (df['racecourse'] == 'Unknown').astype(int)
    df['racecourse_nan'] = df['racecourse'].isna().astype(int)

    # Venue-related features
    df['venue_ST'] = (df['venue'] == 'ST').astype(int)
    df['venue_nan'] = df['venue'].isna().astype(int)

    # Racetrack-related features
    df['racetrack_TURF'] = (df['racetrack'] == 'TURF').astype(int)
    df['racetrack_nan'] = df['racetrack'].isna().astype(int)

    # Going-related features
    df['going_GOOD'] = (df['going'] == 'GOOD').astype(int)
    df['going_GOOD TO FIRM'] = (df['going'] == 'GOOD TO FIRM').astype(int)
    df['going_GOOD TO YIELDING'] = (df['going'] == 'GOOD TO YIELDING').astype(int)
    df['going_WET FAST'] = (df['going'] == 'WET FAST').astype(int)
    df['going_WET SLOW'] = (df['going'] == 'WET SLOW').astype(int)
    df['going_YIELDING'] = (df['going'] == 'YIELDING').astype(int)
    df['going_nan'] = df['going'].isna().astype(int)

    # Class-related features
    df['class_2'] = (df['class'] == 2).astype(int)
    df['class_3'] = (df['class'] == 3).astype(int)
    df['class_4'] = (df['class'] == 4).astype(int)
    df['class_5'] = (df['class'] == 5).astype(int)
    df['class_GROUP'] = (df['class'] == 'GROUP').astype(int)
    df['class_RESTRICTED'] = (df['class'] == 'RESTRICTED').astype(int)
    df['class_nan'] = df['class'].isna().astype(int)

    # Sex-related features
    df['sex_Gelding'] = (df['sex'] == 'Gelding').astype(int)
    df['sex_Horse'] = (df['sex'] == 'Horse').astype(int)
    df['sex_Mare'] = (df['sex'] == 'Mare').astype(int)
    df['sex_Rig'] = (df['sex'] == 'Rig').astype(int)
    df['sex_nan'] = df['sex'].isna().astype(int)

    return df

# Apply the function to add feature columns to the DataFrame
merged_df = fill_feature_columns(merged_df)

# === Step 3: Preprocess (must match training preprocessing!) ===
# Drop columns not used in training
X = merged_df.drop(columns=[
  #'AGEMISDATA',
  'WFA',
  #'WOMISDATA',
  '_id',
  #'age',
  'allowance',
  #'avesprat',
  #'avglbw',
  'bestTime',
  'birthYear',
  'class',
  'code',
  'colour',
  'country',
  'currentRating',
  'dam',
  'damsire',
  'date',
  #'daysSinceLastRace',
  #'daysSinceLastWorkout',
  #'distance',
  #'draw',
  #'drawBiasScore',
  'gear',
  'going',
  #'handicapWeight',
  'horseCode',
  'horseId',
  'horseNumber',
  #'horseWeight',
  'horseWeightChange',
  'importCategory',
  'internationalRating',
  'isStandBy',
  #'jocMisData',
  #'jocWinCount',
  #'jocWinPercent',
  'jockey',
  'lastSixRuns',
  #'lifeWin',
  'money',
  'name',
  'name_ch',
  'name_en',
  #'newDist',
  'overweight',
  'owner',
  #'placeRate',
  'priority',
  'race',
  'racecourse',
  'racetrack',
  'rating',
  'ratingChange',
  'ratingLower',
  'ratingUpper',
  'seasonStakes',
  'sex',
  'sire',
  'startOfSeasonRating',
  'time',
  'totalStakes',
  'trainer',
  'trainerPreference',
  'trumpCard',
  'type',
  'venue',
  #'winRate',
  'normalized_prob',
  'implied_prob'
], errors="ignore")  # Adjust as needed

# need to encode
print(X.columns) 

# Handle NaNs if any
X.fillna(0, inplace=True)

# === Step 4: Make predictions ===
predictions = model.predict(X)
probs = model.predict_proba(X)

print(predictions)
# Add predictions and win probabilities to original DataFrame
merged_df["predicted_rank"] = predictions
merged_df["win_probability"] = probs[:, 0]  # assuming class '0' is 1st place

horse_id_to_check = "HK_2024_K095"
row = merged_df[merged_df["horseId"] == horse_id_to_check]

if not row.empty:
    print("Row for", horse_id_to_check)
    print(row)
else:
    print(f"Horse ID {horse_id_to_check} not found.")

# Print each horse's probability of winning 1st place
print(merged_df[["horseId", "win_probability", "name_ch", "predicted_rank"]].to_string(index=False))

# Calculate raw win probabilities for 1st place (assuming class 0 is 1st place)
merged_df["win_prob"] = calculate_win_probabilities(merged_df["win_probability"])
print(merged_df["win_prob"].tolist())

# Display horseId and their % win chance
print(merged_df[["horseId", "win_probability", "horseNumber", "name_ch", "normalized_prob"]].sort_values(by="win_probability", ascending=False).to_string(index=False))






