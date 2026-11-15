"""
Predict the direction of the next move and measure how tradable it is, for
the Causal Analysis of Financial Tradability project.

Import as:

import predict_utils as putil
"""

import data_utils as dutil
import numpy as np
import pandas as pd
import sklearn.linear_model as sklm
import sklearn.pipeline as skpipe
import sklearn.preprocessing as skprep
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
    pos = predict_with_target(
        features, labels, horizon, test_years, target="sign", interval=interval
    )
    return pos


TARGETS = ["sign", "weighted_sign", "large_moves", "return"]


def predict_with_target(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    horizon: str,
    test_years: list[int],
    *,
    target: str = "sign",
    min_move: float = 0.001,
    interval: str = "5m",
) -> pd.Series:
    """
    Predict the direction with a linear model trained on a given target.

    The model is fit walking forward by year, as in `predict_logistic()`.
    Only what the model is trained on changes:
    - `sign`: logistic regression on the sign of the move (step 2)
    - `weighted_sign`: the same, with each bar weighted by |r|, so that a
      big move counts more than a small one
    - `large_moves`: the same, trained only on the bars with |r| above
      `min_move`
    - `return`: ridge regression on the return itself; the position is the
      sign of the predicted return

    :param features: features as returned by `compute_features()`
    :param labels: labels as returned by `compute_labels()`
    :param horizon: holding period, e.g., "1h"
    :param test_years: years to predict, e.g., [2022, 2023]
    :param target: what the model is trained on, one of `TARGETS`
    :param min_move: smallest |r| to train on for `large_moves`, as a return
        (0.001 = 10 bp, the cost of a round trip)
    :param interval: bar size, e.g., "5m"
    :return: position +1 or -1 for the bars in the test years, NaN elsewhere
    """
    if target not in TARGETS:
        raise ValueError(f"Invalid target='{target}'")
    label = labels[f"label_{horizon}"]
    has_features = features.notna().all(axis=1).to_numpy()
    has_label = label.notna().to_numpy()
    if target == "sign":
        abs_ret = None
    else:
        abs_ret = labels[f"fwd_ret_{horizon}"].abs()
    pos = pd.Series(np.nan, index=features.index)
    for test_year in test_years:
        is_train, is_test = split_walk_forward(
            features.index, test_year, horizon, interval=interval
        )
        is_train &= has_features & has_label
        is_test &= has_features
        if target == "large_moves":
            is_train &= (abs_ret > min_move).to_numpy()
        if target == "return":
            model = skpipe.make_pipeline(
                skprep.StandardScaler(), sklm.Ridge(alpha=1.0)
            )
            fwd_ret = labels[f"fwd_ret_{horizon}"]
            model.fit(features[is_train], fwd_ret[is_train])
            pred = model.predict(features[is_test])
        else:
            model = skpipe.make_pipeline(
                skprep.StandardScaler(), sklm.LogisticRegression(max_iter=1000)
            )
            fit_params = {}
            if target == "weighted_sign":
                # Scale the weights to a mean of 1, so that the
                # regularization is the same as without weights.
                weight = abs_ret[is_train].to_numpy()
                fit_params["logisticregression__sample_weight"] = (
                    weight / weight.mean()
                )
            model.fit(features[is_train], label[is_train], **fit_params)
            pred = model.predict_proba(features[is_test])[:, 1] - 0.5
        pos[is_test] = np.where(pred > 0, 1.0, -1.0)
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


def sweep_costs(
    pos: pd.Series, fwd_ret: pd.Series, costs: list[float]
) -> pd.DataFrame:
    """
    Measure the trades at several costs per round trip.

    :param pos: position for each trade, +1 or -1
    :param fwd_ret: return over the holding period of each trade
    :param costs: costs of a round trip, as returns (0.001 = 10 bp)
    :return: one row per cost with the metrics of `evaluate_positions()`
    """
    rows = [evaluate_positions(pos, fwd_ret, cost=cost) for cost in costs]
    df = pd.DataFrame(rows)
    df.insert(0, "cost_bps", [cost * 1e4 for cost in costs])
    return df


def evaluate_by_group(
    pos: pd.Series,
    fwd_ret: pd.Series,
    groups: pd.Series,
    *,
    cost: float = 0.0,
) -> pd.DataFrame:
    """
    Measure the trades separately for each group, e.g., each market regime.

    :param pos: position for each trade, +1 or -1
    :param fwd_ret: return over the holding period of each trade
    :param groups: group of each trade; trades with no group are left out
    :param cost: cost of a round trip, as a return (0.001 = 10 bp)
    :return: one row per group with the metrics of `evaluate_positions()`
        and:
        - `share`: share of the trades in the group
        - `mean_abs_bps`: average |r| of the trades, in basis points
        - `se_bps`: standard error of the PnL per trade, in basis points
    """
    df = pd.DataFrame({"pos": pos, "fwd_ret": fwd_ret, "group": groups})
    df = df.dropna()
    df = df[df["fwd_ret"] != 0]
    rows = {}
    for group, sub in df.groupby("group", observed=True):
        stats = evaluate_positions(sub["pos"], sub["fwd_ret"], cost=cost)
        pnl = sub["pos"] * sub["fwd_ret"]
        stats["share"] = len(sub) / len(df)
        stats["mean_abs_bps"] = sub["fwd_ret"].abs().mean() * 1e4
        stats["se_bps"] = pnl.std() / np.sqrt(len(sub)) * 1e4
        rows[group] = stats
    results = pd.DataFrame(rows).T
    results.index.name = "group"
    return results
