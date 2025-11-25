import pandas as pd
from pymongo import MongoClient
import os
from functions.getRaces import getRaces
from generateFeatures import generateFeaturesForPastRaces, generateFeaturesForPastRacesB, generateFeaturesForPastRacesD
from datetime import datetime


# Example usage:
dbUri = os.getenv("MONGO_URI")
client = MongoClient(dbUri)
db = client["hkjc"]
startYear = 2019
limit = 400

# Retrieve the 50 races you want to analyze (you may already have these from the previous code)
races = getRaces(db, startYear, limit)

# Calculate stats for each horse in the raceResults
races_with_stats = generateFeaturesForPastRacesD(db, races)

flattened_results = []
for race in races_with_stats:
    race_copy = race.copy()
    race_copy.pop("results", None)  # Remove 'results' key if it exists
    for result in race["results"]:
        flattened_result = {**race_copy, **result}
        flattened_results.append(flattened_result)


# Convert cursor to DataFrame
df = pd.DataFrame(list(flattened_results))
print(df)

# Create timestamp
timestamp = datetime.now().strftime("%Y%m%d")

# Save DataFrame to timestamped CSV
df.to_csv(f"data/race_data_{startYear}_{limit}_{timestamp}_D.csv", index=False)