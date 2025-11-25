from pymongo import MongoClient
import os

def jockey_track_stats(jockey_name, venue, race_date, db=None):
    if db is None:
        db_uri = os.getenv("MONGO_URI")
        client = MongoClient(db_uri)
        db = client["hkjc"]

    race_result = db["raceResult"]
    race_meta = db["raceMeta"]

    # Join raceResult with raceMeta on date and race to get venue and money
    pipeline = [
        {"$match": {
            "jockey": jockey_name,
            "place": {"$type": "int"},
            "date": {"$lt": race_date}
        }},
        {
            "$lookup": {
                "from": "raceMeta",
                "let": {"race_date": "$date", "race_num": "$race"},
                "pipeline": [
                    {"$match": {
                        "$expr": {
                            "$and": [
                                {"$eq": ["$date", "$$race_date"]},
                                {"$eq": ["$race", "$$race_num"]},
                                {"$eq": ["$venue", venue]}
                            ]
                        }
                    }}
                ],
                "as": "meta"
            }
        },
        {"$unwind": "$meta"},
    ]

    results = list(race_result.aggregate(pipeline))

    if not results:
        return {
            "winRate": 0.0,
            "placeRate": 0.0,
            "roi": 0.0,
            "totalRides": 0
        }

    total_rides = len(results)
    wins = sum(1 for r in results if r["runningPosition"][-1] == 1)
    places = sum(1 for r in results if r["place"] in [1, 2, 3])

    total_return = sum(float(r.get("meta", {}).get("money", 0)) for r in results)
    total_staked = total_rides  # Assume 1 unit per ride

    roi = (total_return - total_staked) / total_staked * 100 if total_staked > 0 else 0

    return {
        "winRate": round(wins / total_rides * 100, 2),
        "placeRate": round(places / total_rides * 100, 2),
        "roi": round(roi, 2),
        "totalRides": total_rides
    }
