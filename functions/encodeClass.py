classes = {
  '1': 0,
  '2': 1,
  '3': 2,
  '4': 3,
  '5': 4,
  '4YO': 5,
  'GRIFFIN': 6,
  'GROUP': 7,
  'RESTRICTED': 8,
  'UNDEFINED': 9,
  'Unknown': 3
}

import pandas as pd
def encodeClass(cls):
    if pd.isna(cls):
        cls = "UNDEFINED"
    return {f"class_{c}": int(c == cls) for c in classes.items()}

def getAllClassVariables():
    return [f"class_{c}" for c in classes.items()]

