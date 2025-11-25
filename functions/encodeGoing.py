# Step 2: Find all unique values
goings = {
  'FAST': 1,
  'GOOD': 2,
  'GOOD TO FIRM': 3,
  'GOOD TO YIELDING': 4,
  'HEAVY':5,
  'SEALED':6,
  'SLOW':7,
  'SOFT':8,
  'WET FAST':9,
  'WET SLOW':10,
  'YIELDING':11,
  'YIELDING TO SOFT':12,
  'Unknown': 2
}

def encodeGoing(going):
    return {f"going_{v}":int(k == going) for k, v in goings.items()}

def getAllGoingVariables():
    return [f"going_{v}" for v in goings.values()]