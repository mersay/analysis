from pymongo import MongoClient
from functions.trainerWinCount import tWinCount
from functions.trainerWinPercent import tWinPercent
import os

def tMisData(trainer, race_date, db=None):
    if db is None:
        dbUri = os.getenv("MONGO_URI")
        client = MongoClient(dbUri)
        db = client["hkjc"]

    """Determine TMISDATA based on missing trainer data."""
    win_count = tWinCount(trainer, race_date, db)
    winData = tWinPercent(trainer, race_date, db)

    winPercent = winData.get("winRate", 0)
    placePercent = winData.get("placeRate", 0)
    
    
    # If both win count and win percentage are missing, set JMISDATA = 1
    tMisData = 1 if placePercent is None and winPercent is None else 0

    return {
        "placePercent": placePercent if placePercent is not None else 0,
        "winPercent": winPercent if winPercent is not None else 0,
        "TMISDATA": tMisData
    }

