from pymongo import MongoClient
import os

def raceCount(horse_id, race_date, db=None):
    if db is None:
        db_uri = os.getenv("MONGO_URI")
        client = MongoClient(db_uri)
        db = client["hkjc"]

    collection = db["raceResult"]

    pipeline = [
        {"$match": {
            "horseCode": horse_id,
            "lbw": {"$ne": ""},
            "date": {"$lt": race_date}  # Only races before the input date
        }},
        {"$addFields": {
            "lbw_num": {
                "$convert": {
                    "input": "$lbw",
                    "to": "double",
                    "onError": None,
                    "onNull": None
                }
            }
        }},
        {"$match": {
            "lbw_num": {"$ne": None}
        }},
        {"$count": "num_races"}
    ]

    result = list(collection.aggregate(pipeline))
    return result[0]["num_races"] if result else 0
