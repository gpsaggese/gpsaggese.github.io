
import pandas as pd

from src.strategies.ev_strategy import apply_ev_strategy


def test_ev_strategy():
    """Check that the strategy recommends the correct decisions."""

    sample = pd.DataFrame({
        "bank_offer": [40000, 20000, 35000],
        "board_ev": [35000, 35000, 35000]
    })

    result = apply_ev_strategy(sample)

    # Above EV -> Deal
    # Below EV -> No Deal
    # Equal to EV -> Deal
    assert result["ev_strategy"].tolist() == [1, 0, 1]


def test_original_data_unchanged():
    """Check that the function does not modify the input."""

    sample = pd.DataFrame({
        "bank_offer": [40000],
        "board_ev": [35000]
    })

    apply_ev_strategy(sample)

    assert "ev_strategy" not in sample.columns


def test_missing_columns():
    """Check that missing required columns raise an error."""

    sample = pd.DataFrame({
        "bank_offer": [40000]
    })

    try:
        apply_ev_strategy(sample)
        assert False, "Expected ValueError"
    except ValueError:
        pass
