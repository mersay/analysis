from pymongo import MongoClient
from datetime import datetime
import os

def get_closest_fixture(db):
    """
    Retrieve the closest race fixture from 'raceTemp' collection to today's date.
    Assumes the 'date' field in MongoDB is stored as an ISODate (Date()) type.

    :param db: MongoDB database object (e.g., db = client["hkjc"])
    :return: Closest race fixture document
    """
    # Get today's date
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Query for fixtures that are greater than or equal to today's date
    closest_race = db.fixture.find({"date": {"$gte": today}}).sort("date", 1).limit(1)
    closest_race_list = list(closest_race)

    # Retrieve the first result (closest fixture)
    #closest_fixture = closest_race[0] if len(list(closest_race)) > 0 else None
    # Convert the cursor to a list and check if it has any results
        
        # Check if the list is not empty
    closest_fixture = closest_race_list[0] if len(closest_race_list) > 0 else None
    return closest_fixture


def getRunnersByDate(db, inputDate=None):
    """
    Retrieve runner documents for a specific date, enriched with race metadata using an outer join on 'date' and 'race'.

    :param db: MongoDB database object
    :param inputDate: The target date in 'YYYY-MM-DD' format
    :return: List of enriched runner documents for that date
    """

    if inputDate:
        try:
            target_date = datetime.strptime(inputDate, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Input date must be in 'YYYY-MM-DD' format.")
    else:
        closest_fixture = get_closest_fixture(db)
        if closest_fixture:
            target_date = closest_fixture['date']
        else:
            raise ValueError("No fixture found and no input date provided.")


    pipeline = [
        {
            "$match": {
                "date": target_date,
                "isStandBy": False,  # Only include non-standby runners
                "draw": { "$gt": 0 }            
            }
        },

        # First join with raceMetaToday
        {
            "$lookup": {
                "from": "raceMetaToday",
                "let": {"runnerDate": "$date", "runnerRace": "$race"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$date", "$$runnerDate"]},
                                    {"$eq": ["$race", "$$runnerRace"]}
                                ]
                            }
                        }
                    }
                ],
                "as": "raceMeta"
            }
        },
        {
            "$unwind": {
                "path": "$raceMeta",
                "preserveNullAndEmptyArrays": True
            }
        },

        # Second join with horse collection
        {
            "$lookup": {
                "from": "horse",
                "localField": "horseId",
                "foreignField": "horseId",
                "as": "horseInfo"
            }
        },
        {
            "$unwind": {
                "path": "$horseInfo",
                "preserveNullAndEmptyArrays": True
            }
        },

        # Flatten everything
        {
            "$replaceRoot": {
                "newRoot": {
                    "$mergeObjects": ["$$ROOT", "$raceMeta", "$horseInfo"]
                }
            }
        },
        {
            "$project": {
                "raceMeta": 0,
                "horseInfo": 0
            }
        }
    ]


    results = list(db.runner.aggregate(pipeline))
    print("rez", results)
    return results

