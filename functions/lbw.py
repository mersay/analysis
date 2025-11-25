from pymongo import MongoClient
from datetime import datetime
import numpy as np
import os
import json
# optional: add racecourse

def avgLbwLast3(horse_id, venue, distance, racetrack, going, race_date, db=None):
    if db is None:
        db_uri = os.getenv("MONGO_URI")
        client = MongoClient(db_uri)
        db = client["hkjc"]

    race_result_collection = db["raceResult"]
    race_meta_collection = db["raceMeta"]

    # Find relevant races from raceMeta that match the criteria and are before the race_date
    meta_matches = list(race_meta_collection.find({
        "date": {"$lt": race_date},
        "venue": venue,
        "distance": distance,
        "racetrack": racetrack,
        "going": going
    }, {
        "date": 1,
        "race": 1,
        "_id": 0
    }))

    if not meta_matches:
        print("no matches")
        return 99

    # Create set of (date, race) pairs to join with raceResult
    meta_keys = set((m["date"], m["race"]) for m in meta_matches)

    # Find all relevant results for this horse
    all_results = list(race_result_collection.find({
        "horseCode": horse_id,
        "place": {"$type": "int"},
        "lbw": {"$ne": None, "$not": {"$type": "object"}}
    }, {
        "date": 1,
        "race": 1,
        "lbw": 1,
        "_id": 0
    }))


    # Filter by meta_keys
    filtered_results = [r for r in all_results if (r["date"], r["race"]) in meta_keys]

    # Sort by date descending, take last 3
    filtered_results.sort(key=lambda x: x["date"], reverse=True)
    last_3 = filtered_results[:3]

    # Convert LBW values to float
    lbw_values = []
    for r in last_3:
        try:
            lbw_values.append(float(r["lbw"]))
        except (ValueError, TypeError):
            continue  # Skip invalid LBW values like "PU", "UR", etc.

    if not lbw_values:
        return 99

    return np.mean(lbw_values)

def calculate_avg_lbw_by_distance_and_date(race_distance, race_date, db = None):
    if db is None:
        db_uri = os.getenv("MONGO_URI")
        client = MongoClient(db_uri)
        db = client["hkjc"]
    """
    Calculate the average LBW (Last Beaten Weight) for 2nd and 3rd places for races
    that match the provided distance and race year, and save the result to a JSON file.

    :param race_distance: The distance of the race to filter.
    :param race_date: The date of the race to filter, used for filtering by year.
    :param db: The MongoDB database connection.
    """
    
    # Convert race_date to datetime object
    race_date_obj = datetime.strptime(race_date, '%Y-%m-%d')

    # Get the year from the race_date
    race_year = race_date_obj.year

    # Get races from the 'raceMeta' collection that are within the race_year and before
    race_meta_data = db.raceMeta.find({
        "racecourse": {"$exists": True},  # Ensure only valid entries are considered
        "distance": race_distance,
        "date": {"$lte": race_date_obj}
    })
    
    # Prepare a dictionary to store the average LBW values by year and distance
    lbw_by_year_and_distance = {}

    for race_meta in race_meta_data:
        race_id = race_meta.get("_id")
        race_date = race_meta.get("date")
        race_distance = race_meta.get("distance")
        
        # Get the race result data from 'raceResult' collection matching the race_id and filtering by place
        race_results = db.raceResult.find({
            "date": race_date,
            "race": race_meta["race"],
            "place": {"$in": [2, 3]}  # Filter for 2nd and 3rd place only
        })
        
        lbws_2nd_3rd = []
        
        for result in race_results:
            lbws_2nd_3rd.append(result.get("lbw", None))

        # Calculate average LBW for 2nd and 3rd places
        if lbws_2nd_3rd:
            avg_lbw = sum(lbw for lbw in lbws_2nd_3rd if lbw is not None) / len(lbws_2nd_3rd)
            # Create a key for year and distance, and assign the average LBW value
            key = f"{race_year}_{race_distance}_2_3"
            lbw_by_year_and_distance[key] = avg_lbw

    # Load existing data from JSON file if it exists
    file_path = "averageLbwByYearAndDistance.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = {}

    # Update existing data with new data
    existing_data.update(lbw_by_year_and_distance)

    # Save the updated data to the JSON file
    with open(file_path, "w") as json_file:
        json.dump(existing_data, json_file)

    print(f"Saved average LBW values to {file_path}")


# Define the distances and years
distances = [1000, 1200, 1400, 1600, 1650, 1800, 2000, 2200, 2400]
years = range(2015, 2026)  # From 2015 to 2025

# Function to generate the ending date of each year
def get_year_end_date(year):
    return f"{year}-12-31"

# # Loop through each distance and year, and call the function
# for distance in distances:
#     for year in years:
#         race_date = get_year_end_date(year)
#         print(f"Running for distance {distance} and date {race_date}")
        
#         # Call the calculate_avg_lbw_by_distance_and_date function
#         calculate_avg_lbw_by_distance_and_date(distance, race_date)



