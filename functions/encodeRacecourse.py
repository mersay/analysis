racecourses = {
   'A': 1,
   'A+3': 2,
   'B':3, 
   'B+2':4, 
   'C': 5, 
   'C+3': 6,
  'UNDEFINED':7,
  'Unknown': 7  # <-- Add this
}

def encodeRacecourse(rc):
    if rc is None:
        rc = "Unknown"
    if rc not in racecourses:
        rc = "Unknown"
    return {f"racecourse_{v}": int(k == rc) for k, v in racecourses.items()}

def getAllRacecourseVariables():
    return [f"racecourse_{v}" for v in racecourses.values()]

