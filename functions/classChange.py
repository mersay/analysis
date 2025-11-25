from datetime import datetime

def get_class_change(db, horse_code: str, current_date: datetime):
    """
    Returns:
        1 if class upgrade (e.g., from Class 3 to Class 2),
        -1 if class downgrade (e.g., from Class 3 to Class 4),
        0 if no change or insufficient data.
    """
    # Get all races for this horse before the current date
    results = list(db.raceResult.find({
        "horseCode": horse_code,
        "date": {"$lt": current_date}
    }).sort("date", -1))

    if not results:
        return 'noChange'  # No prior races

    last_race = results[0]
    last_race_date = last_race["date"]
    last_race_number = last_race["race"]

    # Fetch class for last race and today's race from raceMeta
    last_meta = db.raceMeta.find_one({"date": last_race_date, "race": last_race_number})
    today_meta = db.raceMeta.find_one({"date": current_date, "race": last_race["race"]})

    if not last_meta or not today_meta:
        return 'noChange'  # Missing metadata

    try:
        last_class = int(last_meta["class"])
        current_class = int(today_meta["class"])
    except (KeyError, ValueError):
        return 'noChange'  # Class info missing or invalid

    if current_class < last_class:
        return 'downgrade'   # Up in class (e.g., class 3 to 2)
    elif current_class > last_class:
        return 'upgrade'  # Down in class (e.g., class 3 to 4)
    else:
        return 'noChange'   # No change