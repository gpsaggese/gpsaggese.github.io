"""
Predict the direction of the next move and measure how tradable it is, for
the Causal Analysis of Financial Tradability project.

Import as:

import predict_utils as putil
"""

import numpy as np
import pandas as pd
import sklearn.linear_model as sklm
import sklearn.pipeline as skpipe
import sklearn.preprocessing as skprep

import data_utils as dutil
import tradability_utils as tu

# #############################################################################
# Walk-forward split
# #############################################################################


def split_walk_forward(
    index: pd.DatetimeIndex,
    test_year: int,
    horizon: str,
    *,
    interval: str = "5m",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the bars into a training set before a year and a test set in it.

    A label at time t uses the prices up to t + horizon. So the training set
    stops one horizon before the test year starts, and no training label uses
    a price from the test year (purging, as in López de Prado, Ch. 7).

    :param index: bar times on a regular grid
    :param test_year: year to test on, e.g., 2023
    :param horizon: holding period, e.g., "1h"
    :param interval: bar size, e.g., "5m"
    :return: boolean masks for the training and test bars
    """
    test_start = pd.Timestamp(f"{test_year}-01-01", tz="UTC")
    test_end = pd.Timestamp(f"{test_year + 1}-01-01", tz="UTC")
    n_bars = dutil.get_n_bars(horizon, interval)
    step = dutil.to_timedelta(interval)
    # The last training label ends at the close of the last bar before the
    # test year.
    train_end = test_start - n_bars * step
    is_train = np.asarray(index < train_end)
    is_test = np.asarray((index >= test_start) & (index < test_end))
    return is_train, is_test


# #############################################################################
# Models
# #############################################################################


def predict_always_buy(features: pd.DataFrame) -> pd.Series:
    """
    Always be long.

    :param features: features as returned by `compute_features()`
    :return: position +1 for every bar with features
    """
    pos = pd.Series(1.0, index=features.index)
    pos[features.isna().all(axis=1)] = np.nan
    return pos


def predict_momentum(features: pd.DataFrame, column: str) -> pd.Series:
    """
    Bet that the last move continues.

    :param features: features as returned by `compute_features()`
    :param column: past return to follow, e.g., "ret_12" for a 1h horizon
        with 5-minute bars
    :return: position +1 after an up move, -1 after a down move, NaN after no
        move
    """
    pos = np.sign(features[column]).replace(0.0, np.nan)
    return pos


def predict_logistic(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    horizon: str,
    test_years: list[int],
    *,
    interval: str = "5m",
) -> pd.Series:
    """
    Predict the direction with a logistic regression, walking forward by year.

    For each test year, fit on all bars before it (see
    `split_walk_forward()`) and predict the bars in it.

    :param features: features as returned by `compute_features()`
    :param labels: labels as returned by `compute_labels()`
    :param horizon: holding period, e.g., "1h"
    :param test_years: years to predict, e.g., [2022, 2023]
    :param interval: bar size, e.g., "5m"
    :return: position +1 or -1 for the bars in the test years, NaN elsewhere
    """
    label = labels[f"label_{horizon}"]
    has_features = features.notna().all(axis=1).to_numpy()
    has_label = label.notna().to_numpy()
    pos = pd.Series(np.nan, index=features.index)
    for test_year in test_years:
        is_train, is_test = split_walk_forward(
            features.index, test_year, horizon, interval=interval
        )
        is_train &= has_features & has_label
        is_test &= has_features
        model = skpipe.make_pipeline(
            skprep.StandardScaler(), sklm.LogisticRegression(max_iter=1000)
        )
        model.fit(features[is_train], label[is_train])
        prob_up = model.predict_proba(features[is_test])[:, 1]
        pos[is_test] = np.where(prob_up > 0.5, 1.0, -1.0)
    return pos


# #############################################################################
# Evaluation
# #############################################################################


def evaluate_positions(
    pos: pd.Series, fwd_ret: pd.Series, *, cost: float = 0.0
) -> pd.Series:
    """
    Measure the hit rate and PnL of a series of trades.

    Each row is one trade: hold `pos` from t to t + horizon and earn
    `pos * fwd_ret`, minus `cost` for the round trip.

    :param pos: position for each trade, +1 or -1
    :param fwd_ret: return over the holding period of each trade
    :param cost: cost of a round trip, as a return (0.001 = 10 bp)
    :return: metrics:
        - `n`: number of trades with a position and a non-zero return
        - `hit_rate`: share of trades in the right direction
        - `weighted_hit_rate`: share of the total |r| traded in the right
          direction. The PnL before costs is positive only if it is above 0.5
        - `n_eff`: effective number of trades (see `compute_effective_n()`)
        - `p_star`: hit rate needed for a 95% chance of profit, no costs
        - `p_star_cost`: the same with costs, p_star + cost / (2 mean |r|)
        - `gross_bps`, `net_bps`: average PnL per trade before and after costs,
          in basis points
        - `total_net`: sum of the PnL after costs
    """
    df = pd.DataFrame({"pos": pos, "fwd_ret": fwd_ret}).dropna()
    df = df[df["fwd_ret"] != 0]
    pnl = df["pos"] * df["fwd_ret"]
    n_eff = tu.compute_effective_n(df["fwd_ret"])
    p_star = tu.compute_min_hit_rate(n_eff)
    mean_abs_ret = df["fwd_ret"].abs().mean()
    stats = {
        "n": len(df),
        "hit_rate": float((pnl > 0).mean()),
        "weighted_hit_rate": float(pnl[pnl > 0].sum() / pnl.abs().sum()),
        "n_eff": n_eff,
        "p_star": p_star,
        "p_star_cost": p_star + cost / (2 * mean_abs_ret),
        "gross_bps": pnl.mean() * 1e4,
        "net_bps": (pnl.mean() - cost) * 1e4,
        "total_net": (pnl - cost).sum(),
    }
    return pd.Series(stats)
