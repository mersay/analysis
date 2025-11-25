from pymongo import MongoClient
from datetime import datetime
import os
import pprint
from functions.getRunners import getRunnersByDate
from generateFeatures import generateFeaturesB, generateFeaturesD
import pandas as pd
from zoneinfo import ZoneInfo  # Use backports.zoneinfo if < Python 3.9

def save_predictions_to_csv(data, base_filename="runners"):
    """
    Save a list of dictionaries to a CSV file with Hong Kong's date in the filename under 'predictions' folder.

    :param data: List of dictionaries
    :param base_filename: Base name for the CSV file (default is 'predictions')
    """
    # Create the folder if it doesn't exist
    os.makedirs("predictions", exist_ok=True)

    # Get today's date in Hong Kong time
    hktime = datetime.now(ZoneInfo("Asia/Hong_Kong"))
    today_str = hktime.strftime('%Y-%m-%d')

    filename = f"predictions/{base_filename}_{today_str}.csv"

    # Save the data to CSV
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Saved predictions to {filename}")


client = MongoClient(os.getenv("MONGO_URI"))
db = client["hkjc"]

# you need to run node getTodayRaceInfo.js in crawler FIRST!!!!!!!!!!!!
runners = getRunnersByDate(db)

if not runners:
    print("NO RACE DATA, run node getTodayRaceInfo.js in crawler first!!!!!!!!!!!!") 

for runner in runners:
    print(runner)


# get the variables
# Calculate stats for each horse in the raceResults
runners_with_stats = generateFeaturesD(db, runners)

pprint.pprint(runners)

if (runners):
    save_predictions_to_csv(runners)





