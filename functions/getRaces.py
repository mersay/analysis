from pymongo import MongoClient
from datetime import datetime
import os
import csv
from dotenv import load_dotenv

def getRaces(db, start_year=2015, limit=4000):
    if db is None:
        dbUri = os.getenv("MONGO_URI")
        client = MongoClient(dbUri)
        db = client["hkjc"]

    """
    Query MongoDB to get races starting from the given year,
    including race metadata and corresponding race results.
    
    :param db: MongoDB database connection
    :param start_year: Year to start fetching races from (default: 2017)
    :param limit: Number of races to retrieve (default: 50)
    :return: List of races with metadata and results
    """
    start_date = datetime(start_year, 1, 1)  # Convert start_year to datetime

    races = list(db.raceMeta.find(
        {
            "date": {"$gte": start_date},
            "venue": {"$in": ["ST", "HV"]},
            "class": {"$ne": "", "$exists": True}  # Ensure 'class' is not empty and exists, TEMP FIX
        },
        {"_id": 0, "ratingUpper": 0, "ratingLower": 0}  # Exclude these fields
    ).sort("date", 1).limit(limit))

    for race in races:
        race_id = race["race"]
        race_date = race["date"]

        # Get race results
        results = list(db.raceResult.find(
            {
                "race": race_id, 
                "date": race_date,
                 "$or": [
                    {"place": {"$type": "int"}},
                    {"place": {"$in": ["UR", "PU", "FE", "WXNR", "DNF", "DISQ"]}} # did compete
                ]
            },
            {"_id": 0, "rawlbw": 0}
        ).sort("horseNumber"))
        # For each result, enrich it with detailed horse info from the 'horse' collection
        for result in results:
            horseCode = result.get("horseCode")

            if horseCode:
                horse_info = db.horse.find_one(
                    {"horseId": horseCode},
                    {
                        "_id": 0,
                        "country": 1,
                        "colour": 1,
                        "sex": 1,
                        "sire": 1,
                        "dam": 1,
                        "damsire": 1,
                        "currentRating": 1
                    }
                )

                if horse_info:
                    result["country"] = horse_info.get("country")
                    result["colour"] = horse_info.get("colour")
                    result["sex"] = horse_info.get("sex")
                    #result["sire"] = horse_info.get("sire")
                    #result["dam"] = horse_info.get("dam")
                    #result["damsire"] = horse_info.get("damsire")
                    result["currentRating"] = horse_info.get("currentRating")
                else:
                    result["country"] = None
                    result["colour"] = None
                    result["sex"] = None
                    #result["sire"] = None
                    #result["dam"] = None
                    #result["damsire"] = None
                    result["currentRating"] = None

        race["results"] = results

    save_races_to_csv(races, start_year, limit)
    return races



def save_races_to_csv(races, start_year, limit):
    """
    Save the races to a CSV file under the race_data folder.
    
    :param races: The list of races to save
    :param start_year: The start year used in the query
    :param limit: The number of races fetched
    """
    # Get today's date for the file name
    today = datetime.today().strftime('%Y-%m-%d')
    
    # Set the file path
    file_path = f"race_data/{today}_{start_year}_{limit}.csv"
    
    # Make sure the folder exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Write races to the CSV file
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write headers (assuming race metadata + result fields you want)
        headers = ["date", "race", "horseCode", "place", "horseNumber", "country", "colour", "sex", "currentRating"]
        writer.writerow(headers)
        
        # Write each race's data
        for race in races:
            race_date = race["date"]
            race_id = race["race"]
            
            for result in race["results"]:
                row = [
                    race_date,
                    race_id,
                    result.get("horseCode", ""),
                    result.get("place", ""),
                    result.get("horseNumber", ""),
                    result.get("country", ""),
                    result.get("colour", ""),
                    result.get("sex", ""),
                    result.get("currentRating", "")
                ]
                writer.writerow(row)
                
    #print(f"Race data saved to {file_path}")
    # Save the race data to a CSV file


# Example usage:
# if __name__ == "__main__":
#     # Load environment variables from .env file
#     load_dotenv()

#     # Connect to MongoDB and get the races
#     client = MongoClient(os.getenv("MONGO_URI"))
#     db = client["hkjc"]
    
#     races = getRaces(db)
    
