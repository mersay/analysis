from datetime import datetime, timedelta
from pymongo import MongoClient
import os
from dotenv import load_dotenv

def tWinPercent(trainer, race_date, db=None):
    if db is None:
        dbUri = os.getenv("MONGO_URI")
        client = MongoClient(dbUri)
        db = client["hkjc"]

    if isinstance(race_date, str):
        race_date = datetime.strptime(race_date, "%Y-%m-%d")

    start_date = race_date - timedelta(days=90)

    past_races = list(db.raceResult.find(
        {
            "trainer": trainer,
            "date": {"$gte": start_date, "$lt": race_date},
            "$or": [
                {"place": {"$type": "int"}},
                {"place": {"$in": ["UR", "PU", "FE", "WXNR", "DNF", "DISQ"]}}
            ]
        }
    ).sort("date", -1))

    if not past_races:
        return {"winRate": 0.0, "placeRate": 0.0, "total": 0}

    wins = 0
    places = 0
    for r in past_races:
        rp = r.get("runningPosition", [])
        if isinstance(rp, list) and rp:
            pos = rp[-1]
            if pos == 1:
                wins += 1
                places += 1
            elif pos in [2, 3]:
                places += 1

    total_races = len(past_races)
    win_rate = wins / total_races * 100
    place_rate = places / total_races * 100

    return {
        "winRate": round(win_rate, 4),
        "placeRate": round(place_rate, 4),
        "total": total_races
    }
