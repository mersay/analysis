from functions.runningStyle import determine_race_style
from datetime import datetime, timedelta
from pymongo import MongoClient

def analyze_race_styles(race_data, num_runners):
    for runner in race_data:
        style = determine_race_style(
            runner["runningPosition"],
            runner["sectionalTime"],
            num_runners
        )
        print(f"{runner['horseNumber']}: {style}")


def get_running_styles_by_horse(horse_id):
    client = MongoClient()
    db = client["hkjc"]
    sectional_col = db["sectionalTime"]
    meta_col = db["raceMeta"]

    horse_races = list(sectional_col.find({"horseId": horse_id}))

    styles = []
    for race in horse_races:
        date = race["date"]
        race_num = race["race"]
        place = race["place"]
        lbws = race.get("lbws")
        times = race.get("sectionalTime")

        if not lbws or not times:
            continue

        # Get number of runners in the same race
        num_runners = sectional_col.count_documents({"date": date, "race": race_num})

        # Get distance from raceMeta
        meta = meta_col.find_one({"date": date, "race": race_num})
        distance = meta.get("distance") if meta else None

        # Determine running style
        style = determine_race_style(lbws, times, num_runners)

        entry = {
            "date": date,
            "race": race_num,
            "style": style,
            "place": place,
            "distance": distance
        }

        print("Style: ", entry)
        styles.append(entry)

    #print("styles", styles)
    return styles

def getStyleByDateAndRace():
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")  # Change if needed
    db = client["hkjc"]
    collection = db["sectionalTime"]

 # Create date range for 2025-05-04
    start_date = datetime(2025, 5, 10)
    end_date = start_date + timedelta(days=1)

    # Query for all races on that date
    cursor = collection.find({
        "date": {
            "$gte": start_date,
            "$lt": end_date
        }
    })

    # Group runners by race
    races = {}
    for doc in cursor:
        race_num = doc["race"]
        if race_num not in races:
            races[race_num] = []
        races[race_num].append({
            "horseNumber": doc["horseNumber"],
            "horse": doc["horseId"],
            "runningPosition": doc.get("lbws", []),
            "sectionalTime": doc.get("sectionalTime", [])
        })

    # Analyze race styles
    for race_num, race_data in races.items():
        print(f"\n=== Race {race_num} ===")
        analyze_race_styles(race_data, num_runners=len(race_data))


get_running_styles_by_horse("HK_2022_H072")