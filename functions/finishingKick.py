from pymongo import MongoClient
from datetime import datetime

def finishingKick(horse_code, race_date, db):
    """
    Calculate average finishing kick index for last 5 races of a horse prior to race_date.
    
    horse_code: str, the horse's code (e.g. "HK_2022_H213")
    db: MongoDB database object
    race_date: str, race date in the format "YYYY-MM-DD" (or as stored in MongoDB)
    """
    # 1. Find last 5 races before race_date
    race_results = db.raceResult.find(
        {
            "horseCode": horse_code,
            "date": {"$lt": race_date}
        },
        {"race": 1, "date": 1}
    ).sort([("date", -1)]).limit(5)

    
    finishing_kicks = []
    for race_result in race_results:
        race_id = {
            "date": race_result["date"],
            "race": race_result["race"],
            "horseId": horse_code
        }

        print(race_id)
        
        # 2. Get sectionalTime and sectionalTimeBreaks from sectionalTime collection
        sectional_doc = db.sectionalTime.find_one(
            race_id,
            {"sectionalTime": 1, "sectionalTimeBreaks": 1}
        )
        
        if not sectional_doc or "sectionalTime" not in sectional_doc:
            continue
        
        sectional_times = sectional_doc["sectionalTime"]
        if not sectional_times:
            continue
        
        first_section = sectional_times[0]
        final_section = sectional_times[-1]
        
        # Check for final 400m breakdowns (200m splits)
        if "sectionalTimeBreaks" in sectional_doc:
            final_breaks = sectional_doc["sectionalTimeBreaks"]
            # If final 400m has 200m breakdowns, use their average
            if len(final_breaks) > 0 and final_breaks[-1]:
                final_200m_times = final_breaks[-1]
                final_section = sum(final_200m_times) / len(final_200m_times)
        
        # 3. Calculate finishing kick index
        finishing_kick = final_section - first_section
        finishing_kicks.append(finishing_kick)
    
    if finishing_kicks:
        avg_finishing_kick = sum(finishing_kicks) / len(finishing_kicks)
        return avg_finishing_kick
    else:
        return 0

# Example usage
# client = MongoClient("mongodb://localhost:27017/")
# db = client["hkjc"]
# horse_code = "HK_2020_E486"
# race_date = datetime.strptime("2025-05-30", "%Y-%m-%d")
# result = finishingKick(horse_code, race_date, db)
# print("Average finishing kick index:", result)
# result = finishingKick("HK_2021_G446", race_date, db)
# print("Average finishing kick index:", result)


