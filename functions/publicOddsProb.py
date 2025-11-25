import json
import os
import numpy as np
import torch
from functions.R2 import compute_log_likelihood
import torch.nn.functional as F

def calculate_win_probabilities_from_odds(odds_list):
    """
    Given a list of decimal odds for all horses in a race, calculate 
    the normalized probability of each horse winning.

    :param odds_list: List of decimal odds for each horse
    :return: List of normalized win probabilities
    """
    # Convert odds to probabilities
    probabilities = [1 / odds for odds in odds_list]

    # Normalize probabilities to sum to 1
    total_prob = sum(probabilities)
    normalized_probabilities = [p / total_prob for p in probabilities]

    return normalized_probabilities

def save_weights(model_path, w_model, w_odds, weights_file="weights.json"):
    # Load existing weights if file exists
    if os.path.exists(weights_file):
        with open(weights_file, "r") as f:
            weights_data = json.load(f)
    else:
        weights_data = {}

    # Use model_path as key
    weights_data[model_path] = {
        "w_model": round(w_model, 8),
        "w_odds": round(w_odds, 8)
    }

    # Save updated dictionary
    with open(weights_file, "w") as f:
        json.dump(weights_data, f, indent=2)
    
    print(f"Saved weights for model {model_path} to {weights_file}")

def logits_to_probs(logits):
    """Convert logits (1D or 2D) to softmax probabilities per race (always returns 2D)."""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    elif isinstance(logits, list):
        logits = torch.tensor(logits)

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)  # [1, num_runners]
    elif logits.ndim == 2 and logits.shape[0] == 1:
        pass  # already [1, num_runners]
    elif logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(-1).unsqueeze(0)  # reshape [num_runners, 1] → [1, num_runners]
    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")

    probs = F.softmax(logits, dim=1)
    return probs.numpy()  # shape: [1, num_runners]

def odds_to_probs(odds):
    """Convert decimal odds to implied probabilities and normalize."""
    inv_odds = np.where(odds > 0, 1 / odds, 0.0)
    total = np.sum(inv_odds)
    return inv_odds / total if total > 0 else np.full_like(inv_odds, 1 / len(inv_odds))

def create_odds_list_from_df(df, max_runners=14, dummy_odds=999.0):
    odds_list = []

    for (date, race), race_df in df.groupby(['date', 'race']):
        odds = race_df['odds'].tolist()
        num_runners = len(odds)

        # Log if any real runner had dummy odds
        for i, o in enumerate(odds):
            if o == dummy_odds:
                horse_code = race_df.iloc[i].get('horseCode', 'UNKNOWN')
                print(f"WARNING: Real runner with dummy odds in race {date} R{race}: horseCode={horse_code}, index={i}")

        # Pad with dummy odds for missing runners
        padded_odds = odds + [dummy_odds] * (max_runners - num_runners)
        odds_list.append(padded_odds)

    return odds_list

def compute_log_likelihood(probs, true_indices):
    """Compute log-likelihood from predicted probabilities and true winner indices."""
    log_likelihood = 0.0
    for p, true_idx in zip(probs, true_indices):
        log_likelihood += np.log(p[true_idx] + 1e-12)  # epsilon to avoid log(0)
    return log_likelihood

def compute_weights_continuous_from_lists(predicted_lbws, true_lbws, odds_list):
    model_squared_error, uniform_squared_error = 0.0, 0.0
    odds_ll, uniform_ll = 0.0, 0.0

    for preds, trues, odds in zip(predicted_lbws, true_lbws, odds_list):
        # mask dummy runners using odds < 999
        mask = [o < 999 for o in odds]
        preds_valid = np.array([p for p, m in zip(preds, mask) if m])
        trues_valid = np.array([t for t, m in zip(trues, mask) if m])
        odds_valid = np.array([o for o in odds if o < 999])

        if len(trues_valid) == 0:
            continue

        # R² (regression-style on lbw)
        race_mean = np.mean(trues_valid)
        model_squared_error += np.sum((preds_valid - trues_valid) ** 2)
        uniform_squared_error += np.sum((race_mean - trues_valid) ** 2)

        # Log-likelihood based on odds (classification-style)
        try:
            winner_idx = trues_valid.tolist().index(0.0)  # winner has lbw == 0.0
            odds_probs = odds_to_probs(odds_valid).reshape(1, -1)
            uniform_probs = np.full_like(odds_probs, 1 / len(odds_valid))
            odds_ll += compute_log_likelihood(odds_probs, [winner_idx])
            uniform_ll += compute_log_likelihood(uniform_probs, [winner_idx])
        except ValueError:
            continue  # skip if no lbw == 0.0 (no winner found)

    model_r2 = 1 - model_squared_error / uniform_squared_error if uniform_squared_error > 0 else 0.0
    odds_r2 = 1 - odds_ll / uniform_ll if uniform_ll > 0 else 0.0
    total = model_r2 + odds_r2
    w_model = model_r2 / total if total > 0 else 0.5
    w_odds = odds_r2 / total if total > 0 else 0.5

    print("Model R²:", model_r2)
    print("Odds R²:", odds_r2)
    return w_model, w_odds

def compute_weights(df):
    model_ll, odds_ll, uniform_ll = 0.0, 0.0, 0.0

    for _, race_df in df.groupby(['date', 'race']):
        logits = np.vstack(race_df['logits'].tolist()).squeeze(-1).reshape(1, -1)  # [1, num_runners]
        odds = race_df['odds'].values
        winner_idx = race_df['final_position'].tolist().index(1)

        model_probs = logits_to_probs(logits)  # [1, num_runners]
        odds_probs = odds_to_probs(odds).reshape(1, -1)  # [1, num_runners]

        print("Race:", race_df.iloc[0]['date'], race_df.iloc[0]['race'])
        print("Raw logits:", logits.flatten())
        logits_max = np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits - logits_max)
        print("Exponentiated logits (exps):", exps.flatten())
        print("Model probabilities:", model_probs.flatten())

        for idx, (horse_num, model_p, odds_p) in enumerate(zip(race_df['horseNumber'], model_probs.flatten(), odds_probs.flatten())):
            print(f"  Horse {horse_num:>3}: model_prob={model_p:.4f}, odds_prob={odds_p:.4f}")

        uniform_probs = np.full_like(model_probs, 1 / len(odds))  # [1, num_runners]

        model_ll += compute_log_likelihood(model_probs, [winner_idx])
        odds_ll += compute_log_likelihood(odds_probs, [winner_idx])
        uniform_ll += compute_log_likelihood(uniform_probs, [winner_idx])

    model_r2 = 1 - model_ll / uniform_ll
    odds_r2 = 1 - odds_ll / uniform_ll
    total = model_r2 + odds_r2
    print("mr2, or2, total", model_r2, odds_r2, total)

    w_model = model_r2 / total if total > 0 else 0.5
    w_odds = odds_r2 / total if total > 0 else 0.5

    return w_model, w_odds

