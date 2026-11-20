"""Run: python -m pytest tests/test_optimal_strategy.py -q"""
import numpy as np
import pytest

from src.strategies.optimal_strategy import (
    board_features, parse_board, next_boards, evaluate_decision, STANDARD_COUNTS,
)


class ConstantOffer:
    name = "constant"
    def __init__(self, offer):
        self.offer = offer
    def predict(self, board, round_number):
        return self.offer


def test_board_features():
    f = board_features([1, 2, 9], 8)
    assert f["n_remaining"] == 3
    assert f["board_min"] == 1
    assert f["board_max"] == 9
    assert f["prob_above_ev"] == pytest.approx(1 / 3)


def test_game_transition():
    board = tuple(range(1, 4))
    children = next_boards(board, 8, np.random.default_rng(42), 10)
    assert len(children) == 3
    assert all(len(child) == 2 for child in children)


def test_final_offer_uses_terminal_prize_ev():
    board = [10, 30]  # round 9; after elimination one prize remains
    result = evaluate_decision(board, 9, 21, ConstantOffer(999),
                               branches=10, simulations=1)
    assert result["continuation_value"] == pytest.approx(20)
    assert result["recommended_decision"] == 1


def test_rejects_invalid_board_size():
    with pytest.raises(ValueError, match="board size"):
        evaluate_decision([1, 2, 3], 9, 10, ConstantOffer(10))


def test_reproducible_simulation():
    board = list(range(1, 5))
    a = evaluate_decision(board, 7, 1, ConstantOffer(2), seed=17)
    b = evaluate_decision(board, 7, 1, ConstantOffer(2), seed=17)
    assert a == b


def test_monte_carlo_uncertainty_fields():
    result = evaluate_decision([10, 30], 9, 20, ConstantOffer(5), simulations=4)
    assert result["mc_ci_lower"] == pytest.approx(20)
    assert result["mc_ci_upper"] == pytest.approx(20)
    assert result["decision_uncertain"] is True


def test_never_use_future_offer_at_final_prize():
    result = evaluate_decision([10, 30], 9, 19, ConstantOffer(10**9), simulations=4)
    assert result["continuation_value"] == pytest.approx(20)
    assert result["recommended_decision"] == 0


def test_state_sampling_reproducible():
    import pandas as pd
    from src.pipelines.evaluate_optimal_strategy import choose_states
    entries = [(pd.Series({"show_id": i, "round": i % 9 + 1, "bank_offer": 5.0}), (1.0, 10.0))
               for i in range(50)]
    a = choose_states(entries, 12, 42, 0.25)
    b = choose_states(entries, 12, 42, 0.25)
    assert [(x[0], x[1][0]["show_id"]) for x in a] == [(x[0], x[1][0]["show_id"]) for x in b]
    assert sum(cohort == "representative" for cohort, _ in a) == 12
