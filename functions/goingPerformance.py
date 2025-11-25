from pymongo import MongoClient
from collections import defaultdict

def get_track_adjusted_scores_by_going(db, horse_code, going, race_date):
    """
    Calculate the average performance score for a horse on a specific going,
    based on races before a given date. Uses 'place' (lower is better) as score.
    Non-integer places are scored as 14. If no valid scores are found, returns 14.

    Parameters:
        db: MongoDB database connection
        horse_code: str, the horseCode to look up
        going: str, the going type to filter by (e.g., "GOOD")
        race_date: str, only include races before this date (YYYY-MM-DD)

    Returns:
        float: average place score for the given going and horse, before race_date
                 (lower average place is better performance). Returns 14 if no scores found.
    """

    scores = []

    # Find all relevant race results for this horse before the given race_date
    target_races = db.raceResult.find({
        "horseCode": horse_code,
        "date": {"$lt": race_date}
    })

    for race in target_races:
        date = race["date"]
        race_number = race["race"]
        place = race.get("place")

        # Get going from raceMeta
        meta = db.raceMeta.find_one({"date": date, "race": race_number})
        if not meta or meta.get("going") != going:
            continue

        if isinstance(place, int):
            score = place  # Use place directly as the score (lower is better)
            scores.append(score)
        elif place in ["UR", "PU", "FE", "WXNR", "DNF", "DISQ"]:
            scores.append(14)  # Score non-integer places as 14

    return sum(scores) / len(scores) if scores else 0