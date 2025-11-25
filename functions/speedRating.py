from pymongo import MongoClient
from datetime import datetime
import numpy as np

def parse_time_str(time_str):
    """Convert a time string like '1:11.01' into milliseconds."""
    if not isinstance(time_str, str):
        return None
    try:
        mins, rest = time_str.split(":")
        secs = float(rest)
        total_ms = (int(mins) * 60 + secs) * 1000
        return total_ms
    except:
        return None

def get_speed_rating(db=None, horse_code=None, race_date=None):
    if db is None:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["hkjc"]  # assuming your db name is "hkjc"

    if race_date is None:
        race_date = datetime.today()

    ratings = []

    # Query race results for the given horse code
    races = db.raceResult.find({
        "horseCode": horse_code,
        "$or": [
            {"place": {"$type": "int"}},
            {"place": {"$in": ["UR", "PU", "FE", "WXNR", "DNF", "DISQ"]}}
        ]
    })

    for race in races:
        if race["date"] >= race_date:
            continue

        race_meta = db.raceMeta.find_one({
            "date": race["date"],
            "race": race["race"]  # race and date are the unique identifier for each race
        })

        if not race_meta:
            continue  # Skip if race meta is missing for the race

        # Extract race metadata
        race_class = race_meta.get("class")
        venue = race_meta.get("venue")
        racetrack = race_meta.get("racetrack")
        distance = race_meta.get("distance")
        finish_time = race.get("finishTime")

        # Skip races with missing class or finish time
        if not race_class or not finish_time:
            continue

        # Convert finish time string to milliseconds
        try:
            mins, rest = finish_time.split(":")
            secs, ms = rest.split(".")
            finish_ms = int(mins) * 60000 + int(secs) * 1000 + int(ms.ljust(3, '0'))
        except:
            continue

        # Find standard time from trackRecord collection
        print(venue, race_class, racetrack, distance)
        standard = db.trackRecord.find_one({
            "venue": venue,
            "class": race_class,
            "racetrack": racetrack,
            "distance": distance
        })

        if not standard:
            continue  # Skip if no standard time is found

        standard_ms = standard["duration"]
        pace_factor = 1  # (optional) Adjust if you want based on other factors

        # Calculate speed rating
        speed_rating = (standard_ms - finish_ms) * pace_factor
        ratings.append(speed_rating)

    speedRatingMisData = 1 if ratings is None else 0

    return {
        "speedRating": np.mean(ratings) if len(ratings) > 0 else None,
        "SRMISDATA": speedRatingMisData
    }


#get_speed_rating(None, "HK_2015_V321")

