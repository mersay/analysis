from pymongo import MongoClient
import os

from datetime import datetime, timedelta
from pymongo import MongoClient
import os

def prevRacesPerformance(horse_id, race_date, db=None):
    if db is None:
        db_uri = os.getenv("MONGO_URI")
        client = MongoClient(db_uri)
        db = client["hkjc"]

    race_result_collection = db["raceResult"]

    if isinstance(race_date, str):
        race_date = datetime.strptime(race_date, "%Y-%m-%d")

    start_date = race_date - timedelta(days=90)

    query = {
        "horseCode": horse_id,
        "date": {"$gte": start_date, "$lt": race_date},
        "place": {"$type": "int"}
    }

    print("Mongo query:", query)

    results = list(race_result_collection.find(query, {
        "_id": 0,
        "place": 1,
        "runningPosition": 1
    }))

    print("Results found:", len(results))

    total_races = len(results)
    print("total_races" , total_races)
    if total_races == 0:
        return {"winRate": 0.0, "placeRate": 0.0, "total": 0}

    wins = sum(1 for r in results if r["runningPosition"][-1] == 1)
    places = sum(1 for r in results if r["place"] in [1, 2, 3])
    win_pct = wins / total_races * 100
    place_pct = places / total_races * 100
    print(horse_id, " :", "winP", win_pct, "placeP", place_pct)

    return {
        "winRate": round(win_pct, 4),
        "placeRate": round(place_pct, 4),
        "total": total_races
    }
