venue_racetrack_combos = {
    ("ST", "ALL WEATHER TRACK"): 1,
    ("ST", "TURF"): 2,
    ("HV", "ALL WEATHER TRACK"): 3,
    ("HV", "TURF"): 4,
    ("CH", "ALL WEATHER TRACK"): 5,
    ("CH", "TURF"): 5,
    ("Unknown", "Unknown"): 5
}

venues = {"ST": "sha_tin", "HV": "happy_valley", "CH": "chengdu", "Unknown": "unknown_venue"}
racetracks = {"ALL WEATHER TRACK": "all_weather", "TURF": "turf", "Unknown": "unknown_track"}

venue_racetrack_combos = {
    (v, r): f"{venues[v]}_{racetracks[r]}"
    for v in venues
    for r in racetracks
}

# Build all possible combinations once
boolean_variables = [f"venue_{name}" for name in venue_racetrack_combos.values()]

def encodeVenue(venue, racetrack):
    if venue not in venues:
        venue = "Unknown"
    if racetrack not in racetracks:
        racetrack = "Unknown"
    key = (venue, racetrack)
    combo_name = venue_racetrack_combos.get(key, venue_racetrack_combos[("Unknown", "Unknown")])
    return {f"venue_{v}": int(f"venue_{v}" == f"venue_{combo_name}") for v in venue_racetrack_combos.values()}


def getAllVenueVariables():
    return boolean_variables