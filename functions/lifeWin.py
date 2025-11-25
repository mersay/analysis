from datetime import datetime, timedelta
from pymongo import MongoClient
import os

def lifeWin(horse_id, race_date, db = None):
    """
    Calculate the win percentage of a horse in the past 2 years before the given race date.
    
    :param horse_id: Horse's unique ID (string)
    :param race_date: Race date in format "DD-MM-YYYY" (string)
    :param db: MongoDB database connection
    :return: Win percentage (float between 0 and 1)
    """

    if db is None:
        dbUri = os.getenv("MONGO_URI")
        client = MongoClient(dbUri)
        db = client["hkjc"]
    # Convert race_date string to datetime object
    #race_date_dt = datetime.strptime(race_date, "%d-%m-%Y")

    # Calculate the start date (2 years before the race date)
    start_date_dt = race_date - timedelta(days=2*365)
    #print("start date", start_date_dt.strftime("%d-%m-%Y"))

    # Query for all races where this horse participated in the past 2 years
    past_races = list(db.raceResult.find({
        "horseCode": horse_id,
        "date": {"$gte": start_date_dt, "$lt": race_date},
        "$or": [
            {"place": {"$type": "int"}},
            {"place": {"$in": ["UR", "PU", "FE", "WXNR", "DNF", "DISQ"]}} # did compete
        ]
    }, {"place": 1}))

    # Total races in the past 2 years
    total_races = len(past_races)
    print(f"There are {total_races} past races for {horse_id}")
    # Count the number of races the horse won (place == 1)
    total_wins = sum(1 for race in past_races if race["runningPosition"][-1] == 1)

    # Calculate win percentage (avoid division by zero)
    win_percentage = total_wins / total_races if total_races > 0 else 0

    return win_percentage

# Example usage:
horse_id = "HK_2022_H213"  # Dummy horse ID
race_date = "02-02-2017"

#win_pct = lifeWin(horse_id, race_date)
#print(f"Horse {horse_id} win percentage in last 2 years: {win_pct:.2%}")
