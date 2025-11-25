venues = ["ST", "HV"]

racecourses_by_venue = {
    "ST": ['A', 'A+3', 'B', 'B+2', 'C', 'C+3', 'Unknown'],
    "HV": ['A', 'A+3', 'B', 'B+2', 'C', 'C+3']
}

racetracks_by_venue = {
    "ST": ["ALL_WEATHER_TRACK", "TURF"],
    "HV": ["TURF"]
}

combos = []

for venue in venues:
    for racetrack in racetracks_by_venue[venue]:
        if racetrack == "ALL_WEATHER_TRACK":
            combos.append(f"{venue}_{racetrack}_Unknown")
        else:
            for racecourse in racecourses_by_venue[venue]:
                if racecourse != "Unknown":
                    combos.append(f"{venue}_{racetrack}_{racecourse}")

# Map to integer IDs
venue_racetrack_course_mapping = {name: idx for idx, name in enumerate(combos)}

# Example output
# import json
# print(json.dumps(venue_racetrack_course_mapping, indent=2))

venueRacetrackRacecourse = {
  "ST_ALL_WEATHER_TRACK_Unknown": 0,
  "ST_ALL WEATHER TRACK_Unknown": 0,
  "ST_ALL WEATHER TRACK_nan":0,
  "Unknown": 0,
  "ST_TURF_A": 1,
  "ST_TURF_A+3": 2,
  "ST_TURF_B": 3,
  "ST_TURF_B+2": 4,
  "ST_TURF_C": 5,
  "ST_TURF_C+3": 6,
  "HV_TURF_A": 7,
  "HV_TURF_A+3": 8,
  "HV_TURF_B": 9,
  "HV_TURF_B+2": 10,
  "HV_TURF_C": 11,
  "HV_TURF_C+3": 12
}

def encodeVenueCombo(row):
    venue = row["venue"] or "ST"
    racetrack = row["racetrack"] or "TURF"
    
    if venue == "ST" and racetrack == "ALL_WEATHER_TRACK":
        racecourse = "Unknown"
    else:
        racecourse = row["racecourse"] or "Unknown"

    key = f"{venue}_{racetrack}_{racecourse}"
    val = venueRacetrackRacecourse.get(key, -1)
    if val == -1:
      print(key)
    return  val # return -1 if not found (or raise error/log warning)
