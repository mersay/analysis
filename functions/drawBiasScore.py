from pymongo import MongoClient
from collections import defaultdict
import json
import os
from datetime import datetime

cache_file = "draw_bias_score.json"
base_dir = os.path.dirname(__file__)  # directory where script is
json_path = os.path.join(base_dir, '..', cache_file)

def load_cached_bias(filename=json_path):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {}

def save_cached_bias(filename=json_path, data = None):
    with open(filename, "w") as f:
        json.dump(data, f)


def get_season_key(race_date: datetime) -> str:
    """
    Given a date, return the season key (e.g., '2024-2025').
    The season starts in September and ends in July.
    """

    year = race_date.year
    month = race_date.month

    if month >= 9:  # September to December
        start_year = year
        end_year = year + 1
    else:  # January to July
        start_year = year - 1
        end_year = year

    return f"{start_year}-{end_year}"

def build_race_key(venue, racetrack, racecourse, distance, going):
    if racecourse:
        return f"{venue}_{racetrack}_{racecourse}_{distance}_{going}"
    else:
        return f"{venue}_{racetrack}_{distance}_{going}"

def drawBiasScore(db, draw, date, venue, distance, racetrack, racecourse, going):
    print("draw", draw)
    if draw is None:
        return None
    try:
        draw = int(draw)
    except (ValueError, TypeError):
        return None

    bias_data = None
    if db is None:
        dbUri = os.getenv("MONGO_URI")
        client = MongoClient(dbUri)
        db = client["hkjc"]

    season_key, season_end = get_prior_season_range(date)
    bias_data = load_cached_bias(json_path)

    bias_score_prior = bias_data.get(season_key, {})
    race_key = build_race_key(venue, racetrack, racecourse, distance, going)

    if race_key in bias_score_prior:
        return bias_score_prior.get(race_key, {}).get(str(draw), None)
    else:
        print("calculating bias obj")
        bias = calculate_accumulated_draw_bias(db, date, venue, distance, racetrack, racecourse, going)
        if not bias:
            print("no bias calculated")
            return None
        return bias.get(str(draw), None)
    

def get_prior_season_range(date: datetime):
    if date.month >= 9:
        # Current season is from Sept this year to July next year
        season_end_year = date.year
        season_key = f"{season_end_year - 1}-{season_end_year}"
    else:
        # Current season is ending in July this year
        season_end_year = date.year - 1
        season_key = f"{season_end_year - 1}-{season_end_year}"

    season_end = datetime(season_end_year, 7, 31)
    return season_key, season_end

def calculate_accumulated_draw_bias(db, date: datetime, venue: str, distance: int, racetrack: str, racecourse: str, going: str):
    print("date", date, " venue:", venue, " distance:", distance, "racetrack:", racetrack, " racecourse:", racecourse, " going", going)
    if db is None:
        dbUri = os.getenv("MONGO_URI")
        client = MongoClient(dbUri)
        db = client["hkjc"]
    season_key, season_end = get_prior_season_range(date)

    season_start = datetime(2011, 1, 1)

    match_query = {
        "racetrack": racetrack,
        "venue": venue,
        "going": going,
        "distance": distance,
        "date": {"$gte": season_start, "$lte": season_end}
    }

    print("racecourse is", racecourse)
    if racecourse is not None:
        match_query["racecourse"] = racecourse

    pipeline = [
        {"$match": match_query},
        {
            "$lookup": {
                "from": "raceResult",
                "let": {"date": "$date", "race": "$race"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$date", "$$date"]},
                                    {"$eq": ["$race", "$$race"]}
                                ]
                            }
                        }
                    },
                    {
                        "$project": {
                            "draw": 1,
                            "runningPosition": 1
                        }
                    }
                ],
                "as": "results"
            }
        },
        {"$unwind": "$results"},
        {
            "$match": {
                "results.draw": {"$ne": None},
                "results.runningPosition.0": {"$exists": True}
            }
        },
        {
            "$project": {
                "draw": "$results.draw",
                "finalPosition": {"$arrayElemAt": ["$results.runningPosition", -1]}
            }
        },
        {
            "$group": {
                "_id": "$draw",
                "avgFinish": {"$avg": "$finalPosition"},
                "count": {"$sum": 1}
            }
        },
        {"$sort": {"_id": 1}}
    ]

    result = list(db.raceMeta.aggregate(pipeline, allowDiskUse=True))

    if not result:
        print("No race result for this race type found.")
        return

    draw_bias = {str(row["_id"]): row["avgFinish"] for row in result}
    race_key = build_race_key(venue, racetrack, racecourse, distance, going)

    # Load or initialize the JSON
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            bias_data = json.load(f)
    else:
        bias_data = {}

    if season_key not in bias_data:
        bias_data[season_key] = {}

    bias_data[season_key][race_key] = draw_bias

    with open(json_path, "w") as f:
        json.dump(bias_data, f, indent=2)

    print(f"✅ Saved draw bias under season {season_key}, key: {race_key}")
    return draw_bias

#calculate_accumulated_draw_bias(None, datetime(2020,1,1), 'ST', 1400, 'TURF', 'B', 'GOOD')
