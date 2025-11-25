import { MongoClient } from "mongodb";

const hkjc = "hkjc";
const dbUri = "mongodb://localhost:27017";
const fixtureCollection = "fixture";
const trackwork = "trackwork";
const odds = 'odds';
const raceResult = 'raceResult';
const raceMeta = 'raceMeta';
const horseCollection = "horse";


async function getDifferentCombos() {
    const uri = dbUri; // Replace with your MongoDB connection string
    const client = new MongoClient(uri);
  
    try {
        await client.connect();
        const db = client.db(hkjc); // Change to your DB name
        const collection = db.collection(raceMeta);
  
        // Aggregation pipeline with filtering
        const combos = await collection.aggregate([
            {
              "$group": {
                "_id": {
                  "racetrack": "$racetrack",
                  "racecourse": "$racecourse",
                  "going": "$going",
                  "venue": "$venue",
                  "distance": "$distance"
                }
              }
            },
            {
              "$project": {
                "_id": 0,
                "racetrack": "$_id.racetrack",
                "racecourse": "$_id.racecourse",
                "going": "$_id.going",
                "venue": "$_id.venue",
                "distance": "$_id.distance"
              }
            }
          ]).toArray();
  
        console.log("combos", combos);
        return combos;
  
    } catch (err) {
        console.error("Error:", err);
    } finally {
        await client.close();
    }
}

async function getFullRacesByCriteria(distance, venue, racecourse, racetrack, going) {
    const uri = dbUri; // Replace with your MongoDB connection string
    const client = new MongoClient(uri);
  
    try {
        await client.connect();
        const db = client.db(hkjc); // Change to your DB name
        const collection = db.collection(raceMeta);
  
        // Aggregation pipeline with filtering
        const races = await collection.aggregate([
            {
              "$match": {
                "date": { "$regex": "-2017$" } // Matches races in 2017
              }
            },
            {
              "$group": {
                "_id": {
                  "distance": distance,
                  "venue": venue,
                  "racetrack": racetrack,
                  "racecourse": racecourse,
                  "going": going
                },
                "races": { "$push": "$$ROOT" } // Group races by shared attributes
              }
            },
            {
              "$unwind": "$races" // Flatten grouped races back into separate documents
            },
            {
                "$limit": 1 // Limits the number of races to 10
            },
            {
              "$lookup": {
                "from": "raceResult",
                "let": { "raceDate": "$races.date", "raceNumber": "$races.race" },
                "pipeline": [
                  {
                    "$match": {
                      "$expr": {
                        "$and": [
                          { "$eq": ["$date", "$$raceDate"] },
                          { "$eq": ["$race", "$$raceNumber"] }
                        ]
                      }
                    }
                  }
                ],
                "as": "raceResults"
              }
            },
            {
              "$project": {
                "_id": 0,
                "distance": "$_id.distance",
                "venue": "$_id.venue",
                "racetrack": "$_id.racetrack",
                "racecourse": "$_id.racecourse",
                "going": "$_id.going",
                "races": {
                  "date": "$races.date",
                  "race": "$races.race",
                  "raceResults": "$raceResults"
                }
              }
            }
          ]).toArray();
  

        console.log("races", races);
        return races;
  
    } catch (err) {
        console.error("Error:", err);
    } finally {
        await client.close();
    }
}




// Mapping functions to command names
const functionsMap = {
    combo: getDifferentCombos,
    fullRaces: getFullRacesByCriteria,
};

// Command-line argument processing
const [, , command, ...args] = process.argv;

if (functionsMap[command]) {
    functionsMap[command](...args);
} else {
    console.log("Usage: node index.js <command> [args]");
    console.log("Commands:");
    console.log("  combo           - Gets the different combo of the races'");
    console.log("  races <distance> <venue> <racecourse> <racetrack> <going>");
    console.log("  fullRaces  <distance> <venue> <racecourse> <racetrack> <going>");
}
