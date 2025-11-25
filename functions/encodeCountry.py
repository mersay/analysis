countries = {
    'ARG': 0,
    'AUS': 1,
    'BRZ': 2,
    'CAN': 3,
    'CHI': 4,
    'FR': 5,
    'GB': 6,
    'GER': 7,
    'GR': 8,
    'IRE': 9,
    'ITY': 10,
    'JPN': 11,
    'NZ': 12,
    'SAF': 13,
    'SPA': 14,
    'USA': 15,
    'Unknown': 1
}

def encodeCountry(country):
    return {f"country_{c}": int(c == country) for c in countries.items()}

def getAllCountryVariables():
    return [f"country_{c}" for c in countries.items()]