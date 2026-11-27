"""Risk-neutral Monte Carlo lookahead with backward induction.

IMPORTANT: This is an approximate finite-horizon policy, not an exact game optimum.
The boundary of the search uses board EV as a terminal continuation proxy.
"""
import ast
from dataclasses import dataclass
from itertools import combinations
from math import comb, sqrt

import numpy as np
import pandas as pd
from scipy.stats import skew

# Standard reconstructed Australian 20-prize board: offer rounds 1..9.
STANDARD_COUNTS = {1: 20, 2: 15, 3: 11, 4: 8, 5: 6,
                   6: 5, 7: 4, 8: 3, 9: 2, 10: 1}


def parse_board(raw):
    """Parse a serialized board, rejecting invalid or inconsistent values."""
    values = ast.literal_eval(raw) if isinstance(raw, str) else raw
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError("Remaining board must be a list of prize values")
    board = tuple(sorted(float(v) for v in values))
    if not board or not all(np.isfinite(board)) or min(board) < 0:
        raise ValueError("Board must contain finite nonnegative prizes")
    return board


def board_features(board, round_number):
    """Recreate Part A's seven pre-offer features from the board."""
    a = np.asarray(board, dtype=float)
    avg = float(a.mean())
    s = float(skew(a, bias=True)) if len(a) >= 3 and a.std() > 0 else 0.0
    if not np.isfinite(s):
        s = 0.0
    return dict(board_ev=avg, round=int(round_number), n_remaining=len(a),
                board_min=float(a.min()), board_max=float(a.max()),
                prob_above_ev=float(np.mean(a > avg)), board_skew=s)


@dataclass
class OfferPredictor:
    """Part A persisted estimator and its exact feature order."""
    estimator: object
    features: list
    name: str = "saved model"

    def predict(self, board, round_number):
        row = board_features(board, round_number)
        missing = set(self.features) - set(row)
        if missing:
            raise ValueError(f"Unsupported model features: {sorted(missing)}")
        prediction = float(self.estimator.predict(pd.DataFrame([row])[self.features])[0])
        # A nonnegative offer is a modeling constraint, not a learned guarantee.
        return max(0.0, prediction)


class EVRatioPredictor:
    """Simple alternative offer model: offer = learned round-specific EV ratio * EV.

    Ratios must be estimated from development data only, never test shows.
    """
    def __init__(self, ratios):
        self.ratios = {int(k): float(v) for k, v in ratios.items()}
        self.name = "round EV ratio"

    def predict(self, board, round_number):
        if int(round_number) not in self.ratios:
            raise ValueError(f"No development-data EV ratio for round {round_number}")
        return max(0.0, self.ratios[int(round_number)] * float(np.mean(board)))


def next_boards(board, round_number, rng, branches):
    """Sample equally likely subsets of remaining prizes for the next offer.

    For small state spaces enumerate all subsets; otherwise sample without
    replacement among uniformly random elimination combinations.
    """
    next_round = round_number + 1
    if next_round not in STANDARD_COUNTS:
        return []
    next_n = STANDARD_COUNTS[next_round]
    if len(board) != STANDARD_COUNTS[round_number]:
        raise ValueError("Board size does not match standard 20-prize game schedule")
    if next_n >= len(board):
        raise ValueError("Expected fewer remaining cases in the next round")
    n_states = comb(len(board), next_n)
    if n_states <= branches:
        return [tuple(board[i] for i in ids) for ids in combinations(range(len(board)), next_n)]
    # Sampling independent uniform combinations is unbiased for the transition
    # expectation; duplicate states are possible when using replacement.
    return [tuple(sorted(board[i] for i in rng.choice(len(board), next_n, replace=False)))
            for _ in range(branches)]

def _value(board, round_number, predictor, rng, depth, branches):
    """Value at a future offer state, evaluated by backward induction."""
    ev = float(np.mean(board))
    if round_number >= 10 or len(board) == 1:
        return ev
    offer = predictor.predict(board, round_number)
    if depth <= 0:
        # Approximation boundary, NOT an exact continuation value.
        return max(offer, ev)
    children = next_boards(board, round_number, rng, branches)
    continuation = float(np.mean([
        _value(child, round_number + 1, predictor, rng, depth - 1, branches)
        for child in children
    ]))
    return max(offer, continuation)


def evaluate_decision(board, round_number, current_offer, predictor,
                      seed=42, branches=3, lookahead=3, simulations=4):
    """Compare the *observed* offer with approximate continuation value.

    Future offers come from predictor; current offer is never predicted.
    Multiple independent Monte Carlo trees reduce sampling noise.
    """
    board = parse_board(board)
    if round_number not in range(1, 10):
        raise ValueError("Standard-game offer round must be between 1 and 9")
    if len(board) != STANDARD_COUNTS[round_number]:
        raise ValueError("Unsupported or inconsistent board size for this round")
    if branches < 1 or lookahead < 1 or simulations < 1:
        raise ValueError("branches, lookahead, and simulations must be positive")
    if not np.isfinite(current_offer) or current_offer < 0:
        raise ValueError("Current offer must be finite and nonnegative")
    estimates = []
    for trial in range(simulations):
        rng = np.random.default_rng(seed + trial)
        children = next_boards(board, round_number, rng, branches)
        estimates.append(float(np.mean([
            _value(child, round_number + 1, predictor, rng, lookahead - 1, branches)
            for child in children
        ])))
    continuation = float(np.mean(estimates))
    std = float(np.std(estimates, ddof=1)) if simulations > 1 else float("nan")
    se = std / sqrt(simulations) if simulations > 1 else float("nan")
    # Normal approximation to the Monte Carlo mean, not a guarantee of policy optimality.
    lower = continuation - 1.96 * se if simulations > 1 else float("nan")
    upper = continuation + 1.96 * se if simulations > 1 else float("nan")
    return {
        "current_offer": float(current_offer),
        "board_ev": float(np.mean(board)),
        "continuation_value": continuation,
        "recommended_decision": int(current_offer >= continuation),
        "estimated_advantage_of_deal": float(current_offer - continuation),
        "mc_std": std,
        "mc_se": se,
        "mc_ci_lower": lower,
        "mc_ci_upper": upper,
        "decision_uncertain": bool(lower <= current_offer <= upper) if simulations > 1 else True,
    }
