from datetime import datetime, timedelta
from pymongo import MongoClient
import os

def tWinCount(trainer, race_date, db=None):
    if db is None:
        dbUri = os.getenv("MONGO_URI")
        client = MongoClient(dbUri)
        db = client["hkjc"]

    # Convert race_date from "DD-MM-YYYY" to datetime object
    #race_date_obj = datetime.strptime(race_date, "%d-%m-%Y")

    # Calculate the date 2 years before the given race date
    start_date_obj = race_date - timedelta(days=2*365)
    
    # Query races where the jockey won (place = 1) in the past two years
    win_count = db.raceResult.count_documents({
        "trainer": trainer,
        "date": {
            "$gte": start_date_obj.strftime("%d-%m-%Y"),
            "$lt": race_date
        },
        "runningPosition": { "$exists": True, "$ne": None },
        "$expr": {
            "$eq": [
                { "$arrayElemAt": ["$runningPosition", -1] },
                1
            ]
        }
    })

    return win_count  # Return number of wins
