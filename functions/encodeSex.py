sexes = {
    "Colt": 1,
    "Filly": 2,
    "Gelding": 3,
    "Horse": 4,
    "Mare": 5,
    "Rig": 6,
    "Unknown": 3  # include an unknown category
}

def encodeSex(sex):
    if sex not in sexes:
        sex = "Unknown"
    return {f"sex_{v}": int(k == sex) for k, v in sexes.items()}


def getAllSexVariables():
    return [f"sex_{v}" for v in sexes.values()]