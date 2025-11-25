from pymongo import MongoClient
from functions.jocWinCount import jocWinCount
from functions.jocWinPercent import jocWinPercent
import os

def jocMisData(jockey, race_date, db=None):
    if db is None:
        dbUri = os.getenv("MONGO_URI")
        client = MongoClient(dbUri)
        db = client["hkjc"]

    """Determine JMISDATA based on missing jockey data."""
    #win_count = jocWinCount(jockey, race_date, db)
    winData = jocWinPercent(jockey, race_date, db)

    winPercent = winData.get("winRate", 0)
    placePercent = winData.get("placeRate", 0)
    
    # If both win count and win percentage are missing, set JMISDATA = 1
    jocMisData = 1 if placePercent is None and winPercent is None else 0

    return {
        "placePercent": placePercent if placePercent is not None else 0,
        "winPercent": winPercent if winPercent is not None else 0,
        "JMISDATA": jocMisData
    }

