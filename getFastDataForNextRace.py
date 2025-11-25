import http.client
import json
import gzip
import io
import pprint
import os
import pandas as pd
import sys
from datetime import datetime


def save_predictions_to_csv(data, race_no):
    """
    Save a list of dictionaries to a CSV file with today's date in the filename under 'predictions' folder.

    :param data: List of dictionaries
    :param base_filename: Base name for the CSV file (default is 'predictions')
    """
    # Create the folder if it doesn't exist
    os.makedirs("odds", exist_ok=True)

    # Generate filename with today's date
    today_str = datetime.today().strftime('%Y-%m-%d')
    filename = f"odds/runners_{today_str}_{race_no}.csv"

    # Save the data to CSV
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8', sep=';')
    print(f"Saved odds to {filename}")

# Check if race number is provided via command line
race_arg = None
if len(sys.argv) > 1:
    try:
        race_arg = int(sys.argv[1])
        print(f"Race number provided via command line: {race_arg}")
    except ValueError:
        print("Invalid race number provided via command line. Must be an integer.")
        sys.exit(1)


# get next Race number API 
nextRace = None
venueCode = None
conn = http.client.HTTPSConnection("info.cld.hkjc.com")


if race_arg is None:
  payload = json.dumps({
    "variables": {
      "date": "",
      "venueCode": ""
    },
    "query": "fragment raceFragment on Race {\n    id\n    no\n    status\n    raceName_en\n    raceName_ch\n    postTime\n    country_en\n    country_ch\n    distance\n    wageringFieldSize\n    go_en\n    go_ch\n    ratingType\n    raceTrack {\n      description_en\n      description_ch\n    }\n    raceCourse {\n      description_en\n      description_ch\n      displayCode\n    }\n    claCode\n    raceClass_en\n    raceClass_ch\n    judgeSigns {\n      value_en\n    }\n  }\n\nfragment racingBlockFragment on RaceMeeting {\n    jpEsts: pmPools(oddsTypes: [TCE,TRI,FF,QTT,DT,TT,SixUP], filters: [\"jackpot\", \"estimatedDividend\"]) {\n      leg {\n        number\n        races\n      }\n      oddsType\n      jackpot\n      estimatedDividend\n      mergedPoolId\n    }\n    poolInvs: pmPools(oddsTypes: [WIN,PLA,QIN,QPL,CWA,CWB,CWC,IWN,FCT,TCE,TRI,FF,QTT,DBL,TBL,DT,TT,SixUP]) {\n      id\n      leg {\n        races\n      }\n    }\n    penetrometerReadings(filters:[\"first\"]) {\n      reading\n      readingTime\n    }\n    hammerReadings(filters:[\"first\"]) {\n      reading\n      readingTime\n    }\n    changeHistories(filters:[\"top3\"]) {\n      type\n      time\n      raceNo\n      runnerNo\n      horseName_ch\n      horseName_en\n      jockeyName_ch\n      jockeyName_en\n      scratchHorseName_ch\n      scratchHorseName_en\n      handicapWeight\n      scrResvIndicator\n    }\n  }\n\nquery racingBlock {\n    timeOffset {\n      rc\n    }\n    raceMeetings\n    {\n      id\n      status\n      venueCode\n      date\n      totalNumberOfRace\n      currentNumberOfRace\n      dateOfWeek\n      meetingType\n      totalInvestment\n      isSeasonLastMeeting\n      races {\n        ...raceFragment\n        runners {\n            id\n            no\n            standbyNo\n            status\n            name_ch\n            name_en\n            horse {\n              id\n              code\n            }\n        }\n      }\n      pmPools(oddsTypes: [TT]) {\n        id\n        leg {\n          races\n        }\n        status\n        sellStatus\n        oddsType\n        lastUpdateTime\n      }\n      ...racingBlockFragment\n    }\n}"
  })
  headers = {
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.9',
    'content-type': 'application/json',
    'dnt': '1',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
    'Cookie': 'ADRUM_BT=R:0|i:13753|g:ca166564-1ee6-4e4b-a170-187268c84917106288|e:34|n:hkjc_0f64f199-5114-4a48-ba81-d692eac42f36'
  }
  conn = http.client.HTTPSConnection("info.cld.hkjc.com")
  conn.request("POST", "/graphql/base/", payload, headers)
  res = conn.getresponse()
  data = res.read()
  # Example from HTTP response

  # Wrap the data in a buffer and decompress
  decompressed = gzip.GzipFile(fileobj=io.BytesIO(data)).read()

  # Now decode as UTF-8 or parse as JSON
  decoded = decompressed.decode("utf-8")
  json_data = json.loads(decoded)

  # Filter race meetings with venueCode "HV" or "ST"
  valid_meetings = [
      meeting for meeting in json_data['data']['raceMeetings']
      if meeting.get('venueCode') in ['HV', 'ST']
  ]

  # Check if any valid meetings exist
  if valid_meetings:
      selected_meeting = valid_meetings[0]  # You can change logic to pick another meeting if needed
      races = selected_meeting.get('races', [])
      nextRace = selected_meeting.get('currentNumberOfRace', None)
      venueCode = selected_meeting.get("venueCode", "ST")

      print(f"Next race number from API: {nextRace} at venue {venueCode}")

  else:
      print("No race meetings found for HV or ST.")
      sys.exit(1)

else:
    nextRace = race_arg
    venueCode = "ST" 

print(nextRace)

if (nextRace == 0):
  raise Exception("ERROR: No upcoming race!")

# api call to get odds for next Race
#conn = http.client.HTTPSConnection("info.cld.hkjc.com")
payload = json.dumps({
  "operationName": "racing",
  "variables": {
    #"date": "",
    "venueCode": venueCode,
    "oddsTypes": [
      "WIN",
      "PLA",
      "QIN",
      "QPL",
      "IWN",
      "FCT",
      "TCE",
      "TRI",
      "QTT",
      "FF",
      "DBL",
      "TBL",
      "DT",
      "TT",
      "SixUP"
    ],
    "raceNo": nextRace
  },
  "query": "query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {\n  raceMeetings(date: $date, venueCode: $venueCode) {\n    pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {\n      id\n      status\n      sellStatus\n      oddsType\n      lastUpdateTime\n      guarantee\n      minTicketCost\n      name_en\n      name_ch\n      leg {\n        number\n        races\n      }\n      cWinSelections {\n        composite\n        name_ch\n        name_en\n        starters\n      }\n      oddsNodes {\n        combString\n        oddsValue\n        hotFavourite\n        oddsDropValue\n        bankerOdds {\n          combString\n          oddsValue\n        }\n      }\n    }\n  }\n}"
})
headers = {
  'accept': '*/*',
  'accept-language': 'en-US,en;q=0.9',
  'content-type': 'application/json',
  'dnt': '1',
  'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'
}
conn.request("POST", "/graphql/base/", payload, headers)
res = conn.getresponse()
data = res.read()
# Example from HTTP response

# Wrap the data in a buffer and decompress
decompressed = gzip.GzipFile(fileobj=io.BytesIO(data)).read()

# Now decode as UTF-8 or parse as JSON
decoded = decompressed.decode("utf-8")
json_data = json.loads(decoded)

# Define the function to handle race and combString
def handle_race_and_comb_string(pool):
    # Handle 'race' based on the 'leg' field in the pool
    races = pool["leg"]["races"]
    if isinstance(races, list) and len(races) == 1:
        race = races[0]  # If the list has only 1 element, return the single value
    else:
        race = races  # Otherwise, return the entire list of races
    
    # Handle 'combString' to split by either '/' or ','
    combStrings = []
    for odds_node in pool["oddsNodes"]:
        comb_string = odds_node["combString"]
        
        # Split combString by '/' or ',' depending on the delimiter
        if '/' in comb_string:
            combStrings.append(list(map(int, comb_string.split('/'))))  # Split by '/' and convert to integers
        elif ',' in comb_string:
            combStrings.append(list(map(int, comb_string.split(','))))  # Split by ',' and convert to integers
        else:
            # If it's a single value, return it as an integer (not in a list)
            combStrings.append([int(comb_string)])  # Single combString as a list with one integer
    
    return race, combStrings

# Main loop to process the data and save to db
odds = []

# should add all odds to db
for meeting in json_data["data"]["raceMeetings"]:
    for pool in meeting["pmPools"]:
        leg = pool.get("leg", {})
        races = leg.get("races", [])

        if nextRace not in races:
            pool_id = pool.get("id", "")
            print(f"Skipping pool with id {pool_id} as its leg races do not include nextRace {nextRace}")
            continue  # Skip just this pool, keep processing others

        # Proceed with parsing
        race, combStrings = handle_race_and_comb_string(pool)
        odds_type = pool["oddsType"]

        for comb_str_list, node in zip(combStrings, pool["oddsNodes"]):
            horse_numbers = comb_str_list if len(comb_str_list) > 1 else comb_str_list[0]
            entry = {
                "race": race,
                "horseNumber": horse_numbers,
                "odds": float(node["oddsValue"]) if node.get("oddsValue") not in [None, "SCR"] else 0,
                "type": odds_type
            }
            odds.append(entry)

# Now 'odds' contains the processed entries to be added to the database

pprint.pprint(odds)

save_predictions_to_csv(odds, nextRace)

