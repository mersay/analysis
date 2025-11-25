import numpy as np

def calculate_running_style(sectional_times, running_positions, race_distance):
    """
    Calculates a horse's running style based on sectional times and running positions.

    Args:
        sectional_times (list): A list of sectional times in seconds.
        running_positions (list): A list of running positions at each sectional time.
        race_distance (float): The total distance of the race in furlongs (e.g., 6 for 6 furlongs).

    Returns:
        str: The calculated running style ('Front-Runner', 'Stalker', 'Mid-Pack', 'Closer', or 'Unclear').
    """
    if not sectional_times or not running_positions or len(sectional_times) != len(running_positions):
        return "Unclear"  # Unable to determine with insufficient data

    num_sections = len(sectional_times)
    early_speed = 0
    late_speed = 0
    position_changes = []

    # Calculate early speed (average speed of first 1/3 of race)
    early_sections = min(num_sections // 3, 2)  # Use at least 2 sections if available
    if early_sections > 0:
        early_speed = sum(sectional_times[:early_sections]) / early_sections

    # Calculate late speed (average speed of last 1/3 of race)
    late_sections = min(num_sections // 3, 2)
    if late_sections > 0:
      late_speed = sum(sectional_times[-late_sections:]) / late_sections
    
    #calculate position changes
    for i in range(num_sections-1):
        position_changes.append(running_positions[i+1] - running_positions[i])
    
    avg_position_change = np.mean(position_changes)
    
    first_position = running_positions[0]
    last_position = running_positions[-1]
    
    # Determine running style based on early/late speed and position
    if first_position <= 2 and early_speed < late_speed * 1.1: #Allow for a little slowing
        return "Front-Runner"
    elif first_position <= 4 and early_speed < late_speed * 1.2 :
        return "Stalker"
    elif first_position > 4 and last_position < num_sections/2 and avg_position_change < 0:
        return "Closer"
    elif first_position > 3 and last_position > num_sections/3:
        return "Mid-Pack"
    else:
        return "Unclear"
    
    

def analyze_running_style(horse_races):
    """
    Analyzes the running style of a horse across multiple races.

    Args:
        horse_races (list): A list of dictionaries, where each dictionary represents a race
                        and contains 'sectional_times', 'running_positions', and 'race_distance'.
                        Example:
                        [
                            {
                                'sectional_times': [24.2, 48.1, 1:12.3, 1:37.1],
                                'running_positions': [3, 2, 1, 1],
                                'race_distance': 8
                            },
                            {
                                'sectional_times': [23.5, 47.0, 1:11.8, 1:36.8],
                                'running_positions': [5, 4, 3, 2],
                                'race_distance': 8
                            }
                        ]

    Returns:
        dict: A dictionary containing the frequency of each running style observed for the horse.
              Example: {'Front-Runner': 1, 'Stalker': 1, 'Closer': 0, 'Mid-Pack': 0, 'Unclear': 0}
    """
    style_counts = {'Front-Runner': 0, 'Stalker': 0, 'Closer': 0, 'Mid-Pack': 0, 'Unclear': 0}
    
    if not horse_races:
        return style_counts

    for race in horse_races:
        style = calculate_running_style(race['sectional_times'], race['running_positions'], race['race_distance'])
        style_counts[style] += 1
    return style_counts

def determine_race_style(running_position, sectional_time, num_runners):
    if not running_position or not sectional_time:
        return "Unknown"

    # Normalize positions: 1 = front, 0 = last
    norm_pos = [(num_runners - pos) / (num_runners - 1) for pos in running_position]

    avg_position = sum(norm_pos) / len(norm_pos)
    position_change = norm_pos[-1] - norm_pos[0]

    pace_trend = sectional_time[0] - sectional_time[-1]  # +ve means speed up

    # Simple rules (can be refined)
    if avg_position > 0.66:
        if pace_trend < -0.2:
            return "Front-runner (faded)"
        return "Front-runner"
    elif avg_position < 0.33:
        if position_change > 0.2 and pace_trend > 0.2:
            return "Closer (strong finish)"
        return "Closer"
    else:
        if abs(position_change) < 0.1:
            return "Stalker"
        elif position_change > 0.1:
            return "Mid-pack Closer"
        else:
            return "Mid-pack Presser"

def get_most_frequent_style(style_counts):
    """
    Returns the most frequent running style.

    Args:
        style_counts (dict): A dictionary containing the frequency of each running style.
            e.g., {'Front-Runner': 1, 'Stalker': 2, 'Closer': 0, 'Mid-Pack': 0, 'Unclear': 0}
    Returns:
        str: the most frequent running style
    """
    most_frequent_style = "Unclear"
    max_count = 0

    for style, count in style_counts.items():
        if count > max_count:
            max_count = count
            most_frequent_style = style
    return most_frequent_style
    
if __name__ == "__main__":
    # Example usage:
    race1 = {
        'sectional_times': [24.2, 48.1, 72.3, 97.1],  # Example sectional times for an 8-furlong race
        'running_positions': [3, 2, 1, 1],
        'race_distance': 8
    }
    race2 = {
        'sectional_times': [23.5, 47.0, 71.8, 96.8],
        'running_positions': [5, 4, 3, 2],
        'race_distance': 8
    }
    race3 = {
        'sectional_times': [25.0, 49.5, 74.0, 98.5],
        'running_positions': [8, 7, 6, 3],
        'race_distance': 8
    }
    race4 = {
        'sectional_times': [24.0, 48.5, 73.0, 97.5],
        'running_positions': [1, 1, 2, 2],
        'race_distance': 8
    }
    
    #Single Race
    style1 = calculate_running_style(race1['sectional_times'], race1['running_positions'], race1['race_distance'])
    print(f"Running style for race 1: {style1}")  # Output: Front-Runner
    
    #Multiple Races
    horse_races = [race1, race2, race3, race4]
    style_counts = analyze_running_style(horse_races)
    print(f"Running style counts: {style_counts}")
    
    most_frequent_style = get_most_frequent_style(style_counts)
    print(f"Most frequent running style: {most_frequent_style}")
