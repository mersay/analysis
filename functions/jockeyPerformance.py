from pymongo import MongoClient

def calculate_jockey_adjusted_performance(db, jockey_id, going, race_date=None):
    """
    Calculate the average adjusted performance for a jockey on a specific going.

    adjusted_performance = actual_result - expected_probability

    Parameters:
        db: MongoDB connection
        jockey_id: str, the jockey's ID
        going: str, target going (e.g., "GOOD", "YIELDING")
        race_date: str (optional), only include races before this date (YYYY-MM-DD)

    Returns:
        float: average adjusted performance for the jockey
    """

    query = {
        "jockeyId": jockey_id,
         "$or": [
            {"place": {"$type": "int"}},
            {"place": {"$in": ["UR", "PU", "FE", "WXNR", "DNF", "DISQ"]}} # did compete
        ]
    }
    if race_date:
        query["date"] = {"$lt": race_date}

    adjusted_scores = []

    race_results = db.raceResult.find(query)

    for result in race_results:
        date = result["date"]
        race_number = result["race"]

        # Filter by going
        meta = db.raceMeta.find_one({"date": date, "race": race_number})
        if not meta or meta.get("going") != going:
            continue

        odds = result["odds"]
        expected = 1 / odds
        actual = 1 if result["runningPosition"][-1] == 1 else 0

        adjusted = actual - expected
        adjusted_scores.append(adjusted)

    return sum(adjusted_scores) / len(adjusted_scores) if adjusted_scores else 0
