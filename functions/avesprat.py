import logging
from pymongo import MongoClient
import sys
from datetime import datetime
import re
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def convert_duration_to_ms(duration_str):
    """ Convert duration from M.SS.ms or M:SS.ms format to milliseconds. """
    match = re.match(r'(\d+)[.:](\d+)[.:](\d+)', duration_str)
    if match:
        minutes, seconds, milliseconds = map(int, match.groups())
        return (minutes * 60 + seconds) * 1000 + milliseconds
    return None

def avesprat(horse_id, race_date, db=None, no_of_races=5, initial_fetch=20):
    if db is None:
        dbUri = os.getenv("MONGO_URI")
        client = MongoClient(dbUri)
        db = client["hkjc"]

    """ Calculate AVESPRAT for a horse based on valid races prior to race_date. """
    
    # Ensure race_date is a datetime object
    if isinstance(race_date, str):
        race_date = datetime.strptime(race_date, "%d-%m-%Y")

    # Fetch the 10 most recent races prior to race_date
    race_results = list(db.raceResult.find({
        "horseCode": horse_id,
        "date": {"$lt": race_date},
        "$or": [
            {"place": {"$type": "int"}},
            {"place": {"$in": ["UR", "PU", "FE", "WXNR", "DNF", "DISQ"]}} # did compete
        ]
    }).sort("date", -1).limit(initial_fetch))  # Fetch 10 races first

    if not race_results:
        print("No race results found")
        return 0

    total_score = 0
    count = 0
    valid_races = []

    for race in race_results:
        race_number = race["race"]

        # Retrieve raceMeta details
        race_meta = db.raceMeta.find_one({"race": race_number, "date": race["date"]})
        if not race_meta:
            continue  # Skip if raceMeta is missing

        racetrack = race_meta["racetrack"]
        distance = race_meta["distance"]
        venue = race_meta["venue"]
        race_class = race_meta.get("class", None)  

        # Retrieve track record
        track_record = db.trackRecord.find_one({
            "venue": venue,
            "distance": distance,
            "racetrack": racetrack,
            "class": str(race_class)  # TR stores class as string
        })

        if not track_record:
            print("No track record for", venue, racetrack, distance, race_class)
            continue  # Skip if no track record found

        horse_finish_time = convert_duration_to_ms(race["finishTime"])
        if horse_finish_time is None:
            print("No finish time for", race["date"], horse_id, race_number)
            continue  # Skip if no finishing time

        # If we reach here, the race is valid
        valid_races.append((race, track_record))

        # Stop once we have 5 valid races
        if len(valid_races) == no_of_races:
            break

    if len(valid_races) < no_of_races:
        print("Not enough valid races")
        return 0 

    # Calculate AVESPRAT score
    for race, track_record in valid_races:
        track_duration = track_record["duration"]
        horse_finish_time = convert_duration_to_ms(race["finishTime"])

        time_diff = (horse_finish_time - track_duration) / 200  # Deduct 1 point per 1/5 second
        score = max(100 - time_diff, 0)  # Ensure score doesn’t go negative

        total_score += score
        count += 1

    print("Valid races count:", count)
    return total_score / count if count > 0 else 0

# Example usage
client = MongoClient("mongodb://localhost:27017/")
db = client["hkjc"]  # Replace with your actual DB name

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python calculate_avesprat.py <horse_id>")
#         sys.exit(1)

#     horse_id = sys.argv[1]  # Read horse_id from command line argument
#     avesprat = avesprat(horse_id, db)
#     #print(f"AVESPRAT for {horse_id}: {avesprat}")
