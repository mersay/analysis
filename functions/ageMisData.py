import os
from datetime import datetime
from pymongo import MongoClient


def ageMisData(horseId, race_date=None, db=None):
    if db is None:
        dbUri = os.getenv("MONGO_URI")
        client = MongoClient(dbUri)
        db = client["hkjc"]

    # Set race_date to today if not provided
    if race_date is None:
        race_date = datetime.today()

    collection = db["horse"]

    # Fetch horse information from the 'horse' collection
    horse = collection.find_one({"horseId": horseId}, {"_id": 0, "age": 1, "birthYear": 1})
    if horse is None:
        horse = {"age": None, "birthYear": None}
        
    birthYear = horse.get("birthYear", None)
    ageVal = None
    ageMisDataFlag = 0  # Default flag value (0)

    # First: try to calculate age from birthYear
    if birthYear is not None:
        currentYear = race_date.year
        ageVal = currentYear - birthYear
        print("ageVal is using birthYear:", ageVal)
    else:
        # Fallback to stored age
        ageVal = horse.get("age", None)
        if ageVal is None:
            # If both birthYear and stored age are missing, flag as missing and fallback to 4
            ageMisDataFlag = 1
            ageVal = 4
            print("no birthYear and no age, fallback to 4")

    return {
        "age": ageVal,
        "AGEMISDATA": ageMisDataFlag
    }
