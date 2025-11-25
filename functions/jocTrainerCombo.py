from pymongo import MongoClient
from datetime import datetime, timedelta
import os

def trainerJockeyComboWinRate(jockey, trainer, race_date, db=None):
    if db is None:
        db_uri = os.getenv("MONGO_URI")
        client = MongoClient(db_uri)
        db = client["hkjc"]

    race_result_collection = db["raceResult"]

    if isinstance(race_date, str):
        race_date = datetime.strptime(race_date, "%Y-%m-%d")

    start_date = race_date - timedelta(days=90)

    query = {
        "jockey": jockey,
        "trainer": trainer,
        "date": {"$gte": start_date, "$lt": race_date},
        "place": {"$type": "int"},
        "runningPosition": {"$exists": True, "$ne": []}
    }

    results = list(race_result_collection.find(query, {
        "_id": 0,
        "place": 1,
        "runningPosition": 1
    }))

    print("Results found:", len(results))

    total_races = len(results)
    if total_races == 0:
        return {"winRate": 0.0, "placeRate": 0.0, "total": 0}

    wins = sum(1 for r in results if isinstance(r["runningPosition"], list) and r["runningPosition"][-1] == 1)
    places = sum(1 for r in results if r["place"] in [1, 2, 3])

    win_pct = wins / total_races * 100
    place_pct = places / total_races * 100

    return {
        "winRate": round(win_pct, 4),
        "placeRate": round(place_pct, 4),
        "total": total_races
    }
