from datetime import datetime
from pymongo import MongoClient
import os

def daysSinceLastRaceJockey(jockey, race_date=None, db=None):
    if db is None:
        dbUri = os.getenv("MONGO_URI")
        client = MongoClient(dbUri)
        db = client["hkjc"]
    
    
    race_result_col = db["raceResult"]
    if race_date is None:
        race_date = datetime.today().strftime("%d-%m-%Y")  # Format as needed   
    # results = {}

    races = list(race_result_col.find(
        {
            "jockey": jockey,
            "date": {"$lt": race_date},
            "finishTime": {"$ne": None}  # Exclude races where finishTime is None/null
        },
        {"date": 1}
    ).sort("date", -1))


    if (len(races) == 0):
        return 4
    last_race_date = max(races, key=lambda x: x["date"])["date"]
    #last_race_date = datetime.strptime(last_race_date, "%d-%m-%Y")
    days_since_last_race = (race_date - last_race_date).days
    return days_since_last_race

