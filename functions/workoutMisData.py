from datetime import datetime
from pymongo import MongoClient
import os
from dotenv import load_dotenv


def workoutMisData(horse_id, race_date, db=None):
    if db is None:
        dbUri = os.getenv("MONGO_URI")
        client = MongoClient(dbUri)
        db = client["hkjc"]

    days_count = daysSinceLastWorkout(horse_id, race_date, db)
    
    workoutMisData = 1 if days_count is None else 0

    #return jocMisData

    return {
        "daysSinceLastWorkout": days_count if days_count is not None else 99,
        "WOMISDATA": workoutMisData
    }


def daysSinceLastWorkout(horse_id, race_date=None, db=None):
    if db is None:
        dbUri = os.getenv("MONGO_URI")
        client = MongoClient(dbUri)
        db = client["hkjc"]
    
    
    collection = db["trackwork"]
    if race_date is None:
        race_date = datetime.today().strftime("%d-%m-%Y")  # Format as needed   
    # results = {}

    races = list(collection.find(
        {
            "horseId": horse_id,
            "date": {"$lt": race_date},
        },
        {"date": 1}
    ).sort("date", -1))


    if (len(races) == 0):
        return 999
    last_wo_date = max(races, key=lambda x: x["date"])["date"]
    #last_race_date = datetime.strptime(last_race_date, "%d-%m-%Y")
    days_since_last_wo = (race_date - last_wo_date).days
    return days_since_last_wo

# Example usage:
#horse_id = "HK_2022_H213"]
#print(daysSinceLastRace(horse_id))
