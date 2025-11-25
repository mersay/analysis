from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def newDist(horse_id, today_race_distance, db = None):
    if db is None:
        dbUri = os.getenv("MONGO_URI")
        client = MongoClient(dbUri)
        db = client["hkjc"]
    
    # Fetch the last 4 races of the horse, sorted by date (most recent first)
    last_four_races = list(db.raceMeta.find(
        {"raceResults.horseCode": horse_id}, 
        {"distance": 1, "date": 1}
    ).sort("date", -1).limit(4))

    # Extract distances
    distances = [race["distance"] for race in last_four_races]

    # Check how many past race distances are greater than or equal to today's race distance
    count = sum(1 for d in distances if d >= today_race_distance)

    # Set new_dist to 1 if 3 or more races meet the condition, else 0
    new_dist = 1 if count >= 3 else 0

    return new_dist


# Example usage:
#horse_id = "HK_2022_H213"  # Example horse ID

# Example usage (assuming a MongoDB connection is established)
if __name__ == "__main__":
    today_race_distance = 1650  # Dummy value for today's race distance
    horse_id = "HK_2022_H213"
    race_date = "25-03-2025"
    
    #print(newDist(horse_id, today_race_distance))
