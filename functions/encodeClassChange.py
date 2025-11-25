# Step 2: Find all unique values
classChanges = {
  'downgrade': 0,
  'noChange': 1,
  'upgrade': 2,
  'Unknown': 1
}

def encodeClassChanges(cc):
    return {f"classChanges_{v}":int(cc == classChanges) for k, v in classChanges.items()}

def getAllClassChangesVariables():
    return [f"classChanges_{v}" for v in classChanges.values()]