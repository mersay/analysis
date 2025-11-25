def scaleDistance(distance, min_value=1000, max_value=2400):
    scaled = (distance - min_value) / (max_value - min_value)
    return { 'distanceScaled' : min(max(scaled, 0), 1) } # clip between 0 and 1