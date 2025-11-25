import pandas as pd
from pymongo import MongoClient
from sklearn.metrics import r2_score
import joblib
import numpy as np
import os

def compute_log_loss(probs, winners):
    """Log loss over a set of races.
    probs: list of arrays of predicted win probs per race (length = num_horses)
    winners: list of indices of actual winners in each race
    """
    return -np.mean([
        np.log(prob[int(win_idx)])  # ensure index is int
        for prob, win_idx in zip(probs, winners)
        if 0 <= int(win_idx) < len(prob) and prob[int(win_idx)] > 0
    ])

def compute_public_r2(public_probs_list, winners_list):
    L_model = compute_log_loss(public_probs_list, winners_list)
    L_uniform = compute_log_loss([np.ones_like(p)/len(p) for p in public_probs_list], winners_list)
    return 1 - (L_model / L_uniform)

def compute_log_likelihood(probs, true_indices):
    """
    Compute the total log-likelihood given predicted probabilities and true winner indices.
    
    Parameters:
    - probs: np.ndarray of shape [num_races, num_runners], softmaxed probabilities
    - true_indices: list or array of length num_races, where each value is the index of the winning runner

    Returns:
    - log_likelihood: float
    """
    log_likelihood = 0.0
    for p, true_idx in zip(probs, true_indices):
        log_likelihood += np.log(p[true_idx] + 1e-12)  # Add small epsilon to avoid log(0)
    return log_likelihood

def pseudo_r_squared(model_probs, true_indices):
    """
    model_probs: List of np.arrays with predicted win probs for each race
    true_indices: List of the true winner index for each race
    """
    L_model = compute_log_likelihood(model_probs, true_indices)
    L_uniform = sum(np.log(1 / len(p)) for p in model_probs)  # uniform prob per race

    R2 = 1 - (L_model / L_uniform)
    return R2


def odds_to_probabilities(odds):
    inv = 1 / odds
    return inv / inv.sum()