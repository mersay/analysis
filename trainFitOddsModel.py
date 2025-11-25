import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import expit  # sigmoid function

def blend_model_and_odds(model_probs, odds, outcomes):
    """
    Fits a logistic regression to learn weights for model_probs and odds-derived probs.

    Parameters:
        model_probs (array-like): Model-generated win probabilities.
        odds (array-like): Market odds (decimal odds, e.g., 5.0 for 4/1).
        outcomes (array-like): 1 if the horse won, 0 otherwise.

    Returns:
        final_probs (np.ndarray): Blended win probabilities.
        weights (dict): Weights for model_probs and market_probs.
    """
    model_probs = np.asarray(model_probs)
    odds = np.asarray(odds)
    outcomes = np.asarray(outcomes)

    if np.any(odds <= 1):
        raise ValueError("Odds must be greater than 1.0 for conversion to market probabilities.")

    market_probs = 1 / odds

    X = np.column_stack((model_probs, market_probs))
    y = outcomes

    clf = LogisticRegression(solver='lbfgs')
    clf.fit(X, y)

    # Blended probabilities using the learned weights
    logits = clf.intercept_ + np.dot(X, clf.coef_.T)
    final_probs = expit(logits).flatten()

    weights = {
        "intercept": float(clf.intercept_),
        "model_prob_weight": float(clf.coef_[0][0]),
        "market_prob_weight": float(clf.coef_[0][1])
    }

    return final_probs, weights
