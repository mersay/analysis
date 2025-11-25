def conservative_kelly_criterion(prob, odds, bankroll, kelly_fraction=0.5):
    """
    Conservative Kelly Criterion bet calculator.

    Parameters:
    - prob (float): Your model's win probability (between 0 and 1)
    - odds (float): Decimal odds from bookmaker
    - bankroll (float): Your current total bankroll
    - kelly_fraction (float): Fraction of full Kelly to use (e.g., 0.5 for half-Kelly)

    Returns:
    - bet_amount (float): Recommended bet size
    """

    edge = (odds * prob) - 1
    full_kelly = edge / (odds - 1)

    if full_kelly <= 0:
        return 0.0

    return bankroll * full_kelly * kelly_fraction
