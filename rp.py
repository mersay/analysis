from pymongo import MongoClient
import os

def get_running_position_lengths_by_distance(db=None):
    """
    Retrieves the lengths of runningPosition arrays from raceResults, grouped by distance
    from raceMeta, based on matching 'date' and 'race'. Skips results where
    'runningPosition' is not a valid array.

    Args:
        mongodb_uri (str): The MongoDB connection URI.
        database_name (str): The name of the MongoDB database.
        race_meta_collection (str): The name of the raceMeta collection.
        race_results_collection (str): The name of the raceResults collection.

    Returns:
        list: A list of dictionaries, where each dictionary contains 'distance' and
              'runningPositionLengths' (a list of the lengths of the runningPosition
              arrays for that distance). Returns an empty list on error.
    """
    try:
        if db is None:
            dbUri = os.getenv("MONGO_URI")
            client = MongoClient(dbUri)
            db = client["hkjc"]
        race_results_collection = "raceResult"
        pipeline = [
            {
                "$lookup": {
                    "from": race_results_collection,
                    "let": {"meta_date": "$date", "meta_race": "$race", "meta_distance": "$distance"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$and": [
                                        {"$eq": ["$date", "$$meta_date"]},
                                        {"$eq": ["$race", "$$meta_race"]},
                                        {"$isArray": "$runningPosition"}
                                    ]
                                }
                            }
                        },
                        {
                            "$project": {
                                "_id": 0,
                                "runningPositionLength": {"$size": "$runningPosition"}
                            }
                        }
                    ],
                    "as": "results"
                }
            },
            {
                "$unwind": "$results"
            },
            {
                "$group": {
                    "_id": {"distance": "$distance", "length": "$results.runningPositionLength"}
                }
            },
            {
                "$group": {
                    "_id": "$_id.distance",
                    "distinctRunningPositionLengths": {"$push": "$_id.length"}
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "distance": "$_id",
                    "distinctRunningPositionLengths": 1
                }
            }
        ]

        aggregation_results = list(db["raceMeta"].aggregate(pipeline))
        client.close()
        return aggregation_results

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == '__main__':
    db = None

    lengths_by_distance = get_running_position_lengths_by_distance()

    if lengths_by_distance:
        print("Running Position Lengths by Distance:")
        for item in lengths_by_distance:
            print(f"Distance: {item['distance']}, Lengths: {item['runningPositionLengths']}")
    else:
        print("Could not retrieve running position lengths.")