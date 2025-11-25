from itertools import product
from typing import Dict, List, Tuple

def estimate_tierce_probs(
    prob_dict: Dict[int, float],
    first_place: List[int],
    third_place: List[int],
    dividends: Dict[Tuple[int, int, int], float] = None
):
    combinations = []
    results = []

    for first, third in product(first_place, third_place):
        for second in prob_dict.keys():
            if second in (first, third):
                continue
            # Calculate approximate joint probability
            p1 = prob_dict[first]
            p2 = prob_dict[second]
            p3 = prob_dict[third]

            denom1 = 1 - p1
            denom2 = 1 - p1 - p2

            if denom1 <= 0 or denom2 <= 0:
                continue  # skip invalid math

            prob = p1 * (p2 / denom1) * (p3 / denom2)
            tierce = (first, second, third)

            expected = None
            if dividends and tierce in dividends:
                expected = prob * dividends[tierce]

            results.append({
                "tierce": tierce,
                "probability": prob,
                "expected_return": expected
            })

    # Sort by highest probability or expected return
    return sorted(results, key=lambda x: x["expected_return"] or x["probability"], reverse=True)