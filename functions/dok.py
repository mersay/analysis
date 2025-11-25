import math
from pymongo import MongoClient
import os

def dok(horse_code, today_distance, race_date, db=None):
    if db is None:
        db_uri = os.getenv("MONGO_URI")
        client = MongoClient(db_uri)
        db = client["hkjc"]

    race_results = db["raceResult"]
    race_meta = db["raceMeta"]

    # Get all past results for the horse prior to race_date
    past_results = list(race_results.find({
        "horseCode": horse_code,
        "date": {"$lt": race_date},
        "place": {"$type": "int"},
        "lbw": {"$ne": "", "$not": {"$type": "object"}}
    }, {
        "date": 1,
        "race": 1,
        "place": 1,
        "lbw": 1
    }))

    for result in past_results:
        # Get the corresponding race meta to check distance
        meta = race_meta.find_one({
            "date": result["date"],
            "race": result["race"]
        }, {
            "distance": 1
        })

        if not meta or "distance" not in meta:
            continue

        race_distance = meta["distance"]
        if abs(race_distance - today_distance) > 100:
            continue  # Must be within 1/16 mile ≈ 100 meters

        # Count the number of horses in the race to determine field size
        field_size = race_results.count_documents({
            "date": result["date"],
            "race": result["race"],
            "place": {"$type": "int"}
        })

        try:
            lbw = float(result["lbw"])
        except:
            continue

        in_top_half = result["place"] <= math.floor(field_size / 2)
        within_6_25_lengths = lbw <= 6.25

        if in_top_half or within_6_25_lengths:
            return 1  # DOK criteria met

    return 0  # No qualifying race found
