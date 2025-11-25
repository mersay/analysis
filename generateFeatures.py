from functions.avesprat import avesprat
from functions.daysSinceLastRace import daysSinceLastRace
from functions.daysSinceLastRaceJockey import daysSinceLastRaceJockey
from functions.jocMisData import jocMisData
from functions.trainerMisData import tMisData
from functions.jocTrainerCombo import trainerJockeyComboWinRate
from functions.lifeWin import lifeWin
from functions.newDist import newDist
from functions.ageMisData import ageMisData
from functions.drawBiasScore import drawBiasScore
from functions.lbw import avgLbwLast3
from functions.workoutMisData import workoutMisData
from functions.prevRacesPerformance import prevRacesPerformance
from functions.speedRating import get_speed_rating
from functions.goingPerformance import get_track_adjusted_scores_by_going
from functions.raceCount import raceCount
from functions.encodeVenue import encodeVenue
from functions.classChange import get_class_change
from functions.finishingKick import finishingKick
import re
from datetime import datetime

def convert_duration_to_ms(duration_str):
    """ Convert duration from M.SS.ms or M:SS.ms format to milliseconds. """
    match = re.match(r'(\d+)[.:](\d+)[.:](\d+)', duration_str)
    if match:
        minutes, seconds, milliseconds = map(int, match.groups())
        return (minutes * 60 + seconds) * 1000 + milliseconds
    return None

def generateFeaturesForPastRaces(db, races):
    """
    For each horse in the given raceResult, calculate life win percentage, days since last race,
    and new distance variable.

    :param db: MongoDB database connection
    :param races: List of races (with raceMeta and raceResult data)
    :param limit: Number of races to process (default: 50)
    :return: List of races with horses' calculated stats
    """
    # Process each race and add the calculated stats to each horse's result
    for race in races:
        race_distance = race.get("distance", 0)  # Assuming 'distance' is a field in raceMeta

        # Iterate over horses in raceResult
        for result in race["results"]:
            horse_id = result["horseCode"]
            jockey = result["jockey"]
            result['duration'] = convert_duration_to_ms(result["finishTime"])
            # Ensure 'place' is a number; if not, set to 99
            # if not isinstance(result.get("place"), (int, float)):
            #     result["place"] = 99

            # Calculate the horse's life win percentage in the last 2 years
            result["lifeWin"] = lifeWin(horse_id, race['date'], db)

            # Calculate the number of days since the last race
            result["daysSinceLastRace"] = daysSinceLastRace(horse_id, race['date'], db)

            # Calculate the new distance variable
            result["newDist"] = newDist(horse_id, race_distance, db)

            jocData = jocMisData(jockey, race['date'], db)

            result["jocWinCount"] = jocData["win_count"]

            result["jocWinPercent"] = jocData["win_percentage"]

            result["jocMisData"] = jocData['JMISDATA']

            result["avesprat"] = avesprat(horse_id, race['date'], db)

            result["avglbw"] = avgLbwLast3(horse_id, race['venue'], race['distance'], race['racetrack'], race['going'], race['date'], db)

            ageData = ageMisData(horse_id, db)

            result["age"] = ageData['age']

            result['AGEMISDATA'] = ageData['AGEMISDATA']

            result["drawBiasScore"] = drawBiasScore(db, result['draw'], race['date'], race['venue'], race['distance'], race['racetrack'], race.get('racecourse'), race['going'])

            woData = workoutMisData(horse_id, race['date'], db)

            result['daysSinceLastWorkout'] = woData['daysSinceLastWorkout']

            result['WOMISDATA'] = woData['WOMISDATA']

            performance = prevRacesPerformance(horse_id, race['date'], db)

            result['winRate'] = performance['winRate']
            result['placeRate'] = performance['placeRate']



        race['racecourse'] = race.get("racecourse") or "Unknown"
    
    return races

def generateFeaturesForPastRacesB(db, races):
    """
    For each horse in the given raceResult, calculate life win percentage, days since last race,
    and new distance variable.

    :param db: MongoDB database connection
    :param races: List of races (with raceMeta and raceResult data)
    :param limit: Number of races to process (default: 50)
    :return: List of races with horses' calculated stats
    #	Feature	Description
    1	Horse’s Speed Rating	A proprietary number summarizing past performances, adjusted for class, pace, and track conditions.
    2	Class Rating	A measure of the class level the horse has been running at — often derived from prize money or race grade.
    3	Weight Carried	Horses are handicapped with different weights; heavier weight can reduce winning chance.
    4	Jockey Rating	Historical win % and/or performance adjusted for mount quality.
    5	Trainer Rating	Similar to jockey rating — effectiveness of the trainer.
    6	Horse’s Last Finish Position	Place in the most recent race.
    7	Days Since Last Race	Long layoffs or quick turnarounds can affect performance.
    8	Post Position (Draw)	Some gates are more favorable depending on track and race distance.
    9	Going Condition Suitability	How well the horse performs on fast/firm/soft/wet tracks — going bias.
    """
    # Process each race and add the calculated stats to each horse's result
    for race in races:
        race_distance = race.get("distance", 0)  # Assuming 'distance' is a field in raceMeta

        # Iterate over horses in raceResult
        for result in race["results"]:
            horse_id = result.get("horseCode")
            jockey = result.get("jockey")
            trainer = result['trainer']
            draw = result.get('draw')
            going = result.get('going')
            racecourse = result.get('racecourse')
            racetrack = result.get('racetrack')
            venue = result.get('venue')
            date = result.get('date')

            result["drawBiasScore"] = drawBiasScore(db, draw, date, venue, race_distance, racetrack, racecourse, going)

            speedRatingData = get_speed_rating(db, horse_id, date)

            result['speedRating'] = speedRatingData.get("speedRating")

            result['SRMISDATA'] = speedRatingData.get("SRMISDATA")

            # Calculate the number of days since the last race
            result["daysSinceLastRace"] = daysSinceLastRace(horse_id, date, db)

            # Calculate the new distance variable
            #result["newDist"] = newDist(horse_id, race_distance, db)

            jocData = jocMisData(jockey, date, db)

            result["jocWinCount"] = jocData["win_count"]

            result["jocWinPercent"] = jocData["win_percentage"]

            result["jocMisData"] = jocData['JMISDATA']

            trainerData = tMisData(trainer, date, db)

            result["trainerWinCount"] = trainerData["win_count"]

            result["trainerWinPercent"] = trainerData["win_percentage"]

            result["trainerMisData"] = trainerData['TMISDATA']

            #missing trainer data

            #result["avesprat"] = avesprat(horse_id, race['date'], db)

            result["avglbw"] = avgLbwLast3(horse_id, race['venue'], race_distance, race['racetrack'], race['going'], date, db)

            ageData = ageMisData(horse_id, date, db)

            result["age"] = ageData.get('age')

            result['AGEMISDATA'] = ageData['AGEMISDATA']

            #woData = workoutMisData(horse_id, race['date'], db)

            #result['daysSinceLastWorkout'] = woData['daysSinceLastWorkout']

            #result['WOMISDATA'] = woData['WOMISDATA']

            #performance = prevRacesPerformance(horse_id, race['date'], db)

            #result['winRate'] = performance['winRate']
            #result['placeRate'] = performance['placeRate']
            result['goingBias'] = get_track_adjusted_scores_by_going(db, horse_id, race['going'], date)
            result['raceCount'] = raceCount(horse_id, date, db)



        race['racecourse'] = race.get("racecourse") or "Unknown"
    
    return races

def generateFeaturesForPastRacesD(db, races):
    """
    For each horse in the given raceResult, calculate life win percentage, days since last race,
    and new distance variable.

    :param db: MongoDB database connection
    :param races: List of races (with raceMeta and raceResult data)
    :param limit: Number of races to process (default: 50)
    :return: List of races with horses' calculated stats
    #	Feature	Description
    1	Horse’s Speed Rating	A proprietary number summarizing past performances, adjusted for class, pace, and track conditions.
    2	Class Rating	A measure of the class level the horse has been running at — often derived from prize money or race grade.
    3	Weight Carried	Horses are handicapped with different weights; heavier weight can reduce winning chance.
    4	Jockey Rating	Historical win % and/or performance adjusted for mount quality.
    5	Trainer Rating	Similar to jockey rating — effectiveness of the trainer.
    6	Horse’s Last Finish Position	Place in the most recent race.
    7	Days Since Last Race	Long layoffs or quick turnarounds can affect performance.
    8	Post Position (Draw)	Some gates are more favorable depending on track and race distance.
    9	Going Condition Suitability	How well the horse performs on fast/firm/soft/wet tracks — going bias.
    """
    # Process each race and add the calculated stats to each horse's result
    for race in races:
        race_distance = race.get("distance", 0)  # Assuming 'distance' is a field in raceMeta

        # Iterate over horses in raceResult
        for result in race["results"]:
            horse_id = result.get("horseCode")
            jockey = result.get("jockey")
            trainer = result['trainer']
            draw = result.get('draw')
            going = race.get('going')
            racecourse = race.get('racecourse', None)
            racetrack = race.get('racetrack')
            venue = race.get('venue')
            date = result.get('date')

            result["drawBiasScore"] = drawBiasScore(db, draw, date, venue, race_distance, racetrack, racecourse, going)

            speedRatingData = get_speed_rating(db, horse_id, date)

            result['speedRating'] = speedRatingData.get("speedRating")

            result['SRMISDATA'] = speedRatingData.get("SRMISDATA")

            # Calculate the number of days since the last race
            result["daysSinceLastRace"] = daysSinceLastRace(horse_id, date, db)

            result["daysSinceLastRaceJockey"] = daysSinceLastRaceJockey(jockey, date, db)

            # Calculate the new distance variable
            result["newDist"] = newDist(horse_id, race_distance, db)

            jocData = jocMisData(jockey, date, db)

            result["jocPlacePercent"] = jocData["placePercent"]

            result["jocWinPercent"] = jocData["winPercent"]

            result["jocMisData"] = jocData['JMISDATA']

            trainerData = tMisData(trainer, date, db)

            result["trainerWinPercent"] = trainerData["winPercent"]

            result["trainerPlacePercent"] = trainerData["placePercent"]

            result["trainerMisData"] = trainerData['TMISDATA']

            trainerJockeyData = trainerJockeyComboWinRate(jockey, trainer, date, db)

            result['trainerJockeyWinRate'] = trainerJockeyData['winRate']

            #missing trainer data

            #result["avesprat"] = avesprat(horse_id, race['date'], db)

            result["avglbw"] = avgLbwLast3(horse_id, venue, race_distance, race['racetrack'], race['going'], date, db)

            ageData = ageMisData(horse_id, date, db)

            result["age"] = ageData.get('age')

            result['AGEMISDATA'] = ageData['AGEMISDATA']

            #woData = workoutMisData(horse_id, race['date'], db)

            #result['daysSinceLastWorkout'] = woData['daysSinceLastWorkout']

            #result['WOMISDATA'] = woData['WOMISDATA']

            result['goingBias'] = get_track_adjusted_scores_by_going(db, horse_id, race['going'], date)
            result['raceCount'] = raceCount(horse_id, date, db)
            result['classChange'] = get_class_change(db, horse_id, date)
            performance = prevRacesPerformance(horse_id, race['date'], db)
            result['winRate'] = performance.get('winRate', 0)
            result['placeRate'] = performance.get('placeRate', 0)

            result['finishingKick'] = finishingKick(horse_id, date, db)

        #race['racecourse'] = race.get("racecourse") or "Unknown"
    
    return races

def generateFeatures(db, entries):
    """
    For each horse in the given raceResult, calculate life win percentage, days since last race,
    and new distance variable.

    :param db: MongoDB database connection
    :param races: List of races (with raceMeta and raceResult data)
    :param limit: Number of races to process (default: 50)
    :return: List of races with horses' calculated stats
    """
    # Process each race and add the calculated stats to each horse's result
    for entry in entries:

        race_distance = entry.get("distance", 0)  # Assuming 'distance' is a field in raceMeta
        horse_id = entry["horseId"]
        jockey = entry["jockey"]

        # Calculate the horse's life win percentage in the last 2 years
        entry["lifeWin"] = lifeWin(horse_id, entry['date'], db)

        # Calculate the number of days since the last race
        #entry["daysSinceLastRace"] = daysSinceLastRace(horse_id, entry['date'], db)

        # Calculate the new distance variable
        entry["newDist"] = newDist(horse_id, race_distance, db)

        jocData = jocMisData(jockey, entry['date'], db)

        entry["jocWinCount"] = jocData["win_count"]

        entry["jocWinPercent"] = jocData["win_percentage"]

        entry["jocMisData"] = jocData['JMISDATA']

        entry["avesprat"] = avesprat(horse_id, entry['date'], db)

        entry["avglbw"] = avgLbwLast3(horse_id, entry['venue'], entry['distance'], entry['racetrack'], entry['going'], entry['date'], db)

        entry['AGEMISDATA'] = 0

        entry["drawBiasScore"] = drawBiasScore(db, entry['draw'], entry['date'], entry['venue'], entry['distance'], entry['racetrack'], entry.get('racecourse'), entry['going'])

        woData = workoutMisData(horse_id, entry['date'], db)

        entry['daysSinceLastWorkout'] = woData['daysSinceLastWorkout']

        entry['WOMISDATA'] = woData['WOMISDATA']

        performance = prevRacesPerformance(horse_id, entry['date'], db)

        entry['winRate'] = performance['winRate']
        entry['placeRate'] = performance['placeRate']

        entry['racecourse'] = entry.get("racecourse") or "Unknown"

    
    return entries

def generateFeaturesB(db, entries):

    """
    For each horse in the given raceResult, calculate life win percentage, days since last race,
    and new distance variable.

    :param db: MongoDB database connection
    :param races: List of races (with raceMeta and raceResult data)
    :param limit: Number of races to process (default: 50)
    :return: List of races with horses' calculated stats
    #	Feature	Description
    1	Horse’s Speed Rating	A proprietary number summarizing past performances, adjusted for class, pace, and track conditions.
    2	Class Rating	A measure of the class level the horse has been running at — often derived from prize money or race grade.
    3	Weight Carried	Horses are handicapped with different weights; heavier weight can reduce winning chance.
    4	Jockey Rating	Historical win % and/or performance adjusted for mount quality.
    5	Trainer Rating	Similar to jockey rating — effectiveness of the trainer.
    6	Horse’s Last Finish Position	Place in the most recent race.
    7	Days Since Last Race	Long layoffs or quick turnarounds can affect performance.
    8	Post Position (Draw)	Some gates are more favorable depending on track and race distance.
    9	Going Condition Suitability	How well the horse performs on fast/firm/soft/wet tracks — going bias.
    """
    # Process each race and add the calculated stats to each horse's result
    for entry in entries:
            race_distance = entry.get("distance", 0)  # Assuming 'distance' is a field in raceMeta
            horse_id = entry.get("horseId")
            jockey = entry.get("jockey")
            trainer = entry.get("trainer")
            date = entry.get('date')
    
            # Calculate the horse's life win percentage in the last 2 years
            speedRatingData = get_speed_rating(db, horse_id, date)

            entry['speedRating'] = speedRatingData.get("speedRating")

            entry['SRMISDATA'] = speedRatingData.get("SRMISDATA")

            # Calculate the number of days since the last race
            entry["daysSinceLastRace"] = daysSinceLastRace(horse_id, date, db)

            # Calculate the new distance variable
            #entry["newDist"] = newDist(horse_id, race_distance, db)

            jocData = jocMisData(jockey, date, db)

            entry["jocWinCount"] = jocData["win_count"]

            entry["jocWinPercent"] = jocData["win_percentage"]

            entry["jocMisData"] = jocData['JMISDATA']

            trainerData = tMisData(trainer, date, db)

            entry["trainerWinCount"] = trainerData["win_count"]

            entry["trainerWinPercent"] = trainerData["win_percentage"]

            entry["trainerMisData"] = trainerData['TMISDATA']

            #missing trainer data

            #result["avesprat"] = avesprat(horse_id, race['date'], db)

            entry["avglbw"] = avgLbwLast3(horse_id, entry['venue'], race_distance, entry['racetrack'], entry['going'], date, db)

            ageData = ageMisData(horse_id, date, db)

            entry["age"] = ageData.get('age')

            entry['AGEMISDATA'] = ageData.get('AGEMISDATA')

            entry["drawBiasScore"] = drawBiasScore(db, entry.get('draw'), date, entry['venue'], race_distance, entry['racetrack'], entry.get('racecourse'), entry['going'])

            #woData = workoutMisData(horse_id, race['date'], db)

            #result['daysSinceLastWorkout'] = woData['daysSinceLastWorkout']

            #result['WOMISDATA'] = woData['WOMISDATA']

            #performance = prevRacesPerformance(horse_id, race['date'], db)

            #result['winRate'] = performance['winRate']
            #result['placeRate'] = performance['placeRate']
            entry['goingBias'] = get_track_adjusted_scores_by_going(db, horse_id, entry['going'], date)
            entry['raceCount'] = raceCount(horse_id, date, db)

            entry['racecourse'] = entry.get("racecourse") or "Unknown"
    
    return entries

def generateFeaturesD(db, entries):

    """
    For each horse in the given raceResult, calculate life win percentage, days since last race,
    and new distance variable.

    :param db: MongoDB database connection
    :param races: List of races (with raceMeta and raceResult data)
    :param limit: Number of races to process (default: 50)
    :return: List of races with horses' calculated stats
    #	Feature	Description
    1	Horse’s Speed Rating	A proprietary number summarizing past performances, adjusted for class, pace, and track conditions.
    2	Class Rating	A measure of the class level the horse has been running at — often derived from prize money or race grade.
    3	Weight Carried	Horses are handicapped with different weights; heavier weight can reduce winning chance.
    4	Jockey Rating	Historical win % and/or performance adjusted for mount quality.
    5	Trainer Rating	Similar to jockey rating — effectiveness of the trainer.
    6	Horse’s Last Finish Position	Place in the most recent race.
    7	Days Since Last Race	Long layoffs or quick turnarounds can affect performance.
    8	Post Position (Draw)	Some gates are more favorable depending on track and race distance.
    9	Going Condition Suitability	How well the horse performs on fast/firm/soft/wet tracks — going bias.
    """
    # Process each race and add the calculated stats to each horse's result
    for entry in entries:
        race_distance = entry.get("distance")
        horse_id = entry.get("horseId")
        jockey = entry.get("jockey")
        trainer = entry['trainer']
        draw = entry.get('draw')
        going = entry.get('going')
        racecourse = entry.get('racecourse')
        racetrack = entry.get('racetrack')
        venue = entry.get('venue')
        date = entry.get('date')

        # need to add drawBiasScores MISDATA
        entry["drawBiasScore"] = drawBiasScore(db, draw, date, venue, race_distance, racetrack, racecourse, going)

        speedRatingData = get_speed_rating(db, horse_id, date)

        entry['speedRating'] = speedRatingData.get("speedRating")

        entry['SRMISDATA'] = speedRatingData.get("SRMISDATA")

        # Calculate the number of days since the last race
        entry["daysSinceLastRace"] = daysSinceLastRace(horse_id, date, db)

        # Calculate the new distance variable
        entry["newDist"] = newDist(horse_id, race_distance, db)

        jocData = jocMisData(jockey, date, db)

        entry["jocPlacePercent"] = jocData["placePercent"]

        entry["jocWinPercent"] = jocData["winPercent"]

        entry["jocMisData"] = jocData['JMISDATA']

        trainerData = tMisData(trainer, date, db)

        entry["trainerWinPercent"] = trainerData["winPercent"]

        entry["trainerPlacePercent"] = trainerData["placePercent"]

        entry["trainerMisData"] = trainerData['TMISDATA']

        #entry["avesprat"] = avesprat(horse_id, race['date'], db)

        # need to add MISDATA
        entry["avglbw"] = avgLbwLast3(horse_id, venue, race_distance, racetrack , going , date, db)

        #ageData = ageMisData(horse_id, date, db)

        # there will always be age
        # entry["age"] = ageData.get('age')

        entry['AGEMISDATA'] = 0

        #woData = workoutMisData(horse_id, race['date'], db)

        #entry['daysSinceLastWorkout'] = woData['daysSinceLastWorkout']

        #entry['WOMISDATA'] = woData['WOMISDATA']

        entry['goingBias'] = get_track_adjusted_scores_by_going(db, horse_id, going, date)
        entry['raceCount'] = raceCount(horse_id, date, db)
        entry['classChange'] = get_class_change(db, horse_id, date)
        performance = prevRacesPerformance(horse_id, date, db)
        entry['winRate'] = performance.get('winRate', 0)
        entry['placeRate'] = performance.get('placeRate', 0)


        #entry['racecourse'] = entry.get("racecourse") or "Unknown"