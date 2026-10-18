"""
Utility functions to generate the figures of Lesson 10.01 "Auto DS".

Import as:

import L10_01_auto_ds_utils as l1adsut
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.figure as mfigure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as shier
import scipy.spatial.distance as sdist
import scipy.stats as sstats
import seaborn as sns
import sklearn.linear_model as slm
import sklearn.neighbors as sneigh
import sklearn.tree as stree

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Plot helpers
# #############################################################################


# Colors of the figures of Lessons 01.01 and 02.01, so that the figures of the
# lessons look alike.
_BLUE = "#3C6FB0"
_CORAL = "#C0455B"
_ORANGE = "#D98E2B"
_GREEN = "#4F9A5C"
_GRAY = "#7B8794"
_LIGHT_GRAY = "#D5DBE1"

# Number of trading days in one year.
_DAYS_PER_YEAR = 252

# Euler-Mascheroni constant used by the false strategy theorem.
_EULER_GAMMA = 0.5772156649


def _new_figure(
    figsize: Optional[Tuple[float, float]],
    *,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs: Any,
) -> Tuple[mfigure.Figure, np.ndarray]:
    """
    Create a figure with a clean style and no top / right spines.

    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :param nrows: number of rows of subplots
    :param ncols: number of columns of subplots
    :param kwargs: additional arguments passed to `plt.subplots()`
    :return: figure and array of axes with shape (nrows, ncols)
    """
    if figsize is None:
        figsize = tuple(plt.rcParams["figure.figsize"])
    # Use a white background without the default seaborn grid, so that the
    # figure blends into the slide.
    with sns.axes_style("ticks"):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
            **kwargs,
        )
    sns.despine(fig=fig)
    return fig, axes


def _annualized_sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    """
    Compute the annualized Sharpe ratio of a series of returns.

    :param returns: returns of one period each
    :param periods_per_year: number of periods in one year
    :return: annualized Sharpe ratio, 0 if the returns are constant
    """
    std = np.std(returns, ddof=1)
    if std == 0.0:
        return 0.0
    sharpe = float(np.mean(returns) / std * np.sqrt(periods_per_year))
    return sharpe


def _expected_max_sharpe(n_trials: float, sharpe_std: float) -> float:
    """
    Compute the expected maximum Sharpe ratio of zero-skill trials.

    The false strategy theorem approximates the expected maximum of
    `n_trials` independent Sharpe ratio estimates with true value 0 and
    standard deviation `sharpe_std`.

    :param n_trials: number of independent trials
    :param sharpe_std: standard deviation of the Sharpe ratio estimates
    :return: expected maximum Sharpe ratio
    """
    hdbg.dassert_lte(1.0, n_trials, "Need at least one trial")
    if n_trials < 2.0:
        return 0.0
    z_1 = sstats.norm.ppf(1.0 - 1.0 / n_trials)
    z_2 = sstats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    max_sharpe = sharpe_std * ((1.0 - _EULER_GAMMA) * z_1 + _EULER_GAMMA * z_2)
    return float(max_sharpe)


def _simulate_ar1_features(
    rng: np.random.Generator, *, n_obs: int, n_features: int, phi: float
) -> np.ndarray:
    """
    Simulate persistent features with zero mean and unit variance.

    :param rng: random number generator
    :param n_obs: number of observations
    :param n_features: number of features
    :param phi: autoregressive coefficient of each feature
    :return: array of shape (n_obs, n_features)
    """
    hdbg.dassert_lt(abs(phi), 1.0, "The AR(1) process must be stationary")
    noise = rng.normal(size=(n_obs, n_features))
    features = np.zeros((n_obs, n_features))
    # The first value is drawn from the stationary distribution.
    features[0] = noise[0]
    scale = np.sqrt(1.0 - phi**2)
    for t in range(1, n_obs):
        features[t] = phi * features[t - 1] + scale * noise[t]
    return features


# #############################################################################
# Cell 1.1: Search overfits the validation score
# #############################################################################


def _forward_sum(values: np.ndarray, horizon: int) -> np.ndarray:
    """
    Sum the `horizon` values after each observation.

    The entry `t` is `values[t + 1] + ... + values[t + horizon]`, and it is NaN
    when the window leaves the sample.

    :param values: array of one value per observation
    :param horizon: number of observations in the window
    :return: array with the same length as `values`
    """
    n_obs = len(values)
    csum = np.concatenate([[0.0], np.cumsum(values)])
    out = np.full(n_obs, np.nan)
    idx = np.arange(n_obs - horizon)
    out[idx] = csum[idx + horizon + 1] - csum[idx + 1]
    return out


def _purged_folds(
    *, n_obs: int, n_folds: int, label_horizon: int, embargo: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build purged and embargoed k-fold splits of time-ordered observations.

    The label of observation `i` spans the observations from `i + 1` to
    `i + label_horizon`. The training set drops the observations whose label
    overlaps the test block, and the `embargo` observations after it.

    :param n_obs: number of observations
    :param n_folds: number of folds
    :param label_horizon: number of observations spanned by one label
    :param embargo: number of observations embargoed after the test block
    :return: list of (train indices, test indices), one per fold
    """
    bounds = np.linspace(0, n_obs, n_folds + 1).astype(int)
    all_idx = np.arange(n_obs)
    folds = []
    for fold in range(n_folds):
        start, end = bounds[fold], bounds[fold + 1]
        test_idx = all_idx[start:end]
        drop_from = max(0, start - label_horizon)
        drop_to = min(n_obs, end + label_horizon + embargo)
        train_mask = (all_idx < drop_from) | (all_idx >= drop_to)
        folds.append((all_idx[train_mask], test_idx))
    return folds


def _sample_config(
    rng: np.random.Generator, *, n_features: int
) -> Dict[str, Any]:
    """
    Draw one configuration from the search space of the AutoML run.

    The space holds three model families with their hyperparameters and a
    random subset of the features.

    :param rng: random number generator
    :param n_features: number of candidate features
    :return: configuration as a dictionary
    """
    family = rng.choice(["ridge", "tree", "knn"])
    n_used = int(rng.integers(3, 16))
    config: Dict[str, Any] = {
        "family": str(family),
        "features": np.sort(rng.choice(n_features, size=n_used, replace=False)),
    }
    if family == "ridge":
        config["alpha"] = float(10.0 ** rng.uniform(-2.0, 4.0))
    elif family == "tree":
        config["max_depth"] = int(rng.integers(1, 7))
        config["min_samples_leaf"] = int(rng.integers(5, 101))
    else:
        config["n_neighbors"] = int(rng.integers(5, 201))
    return config


def _build_model(config: Dict[str, Any]) -> Any:
    """
    Build the unfitted scikit-learn model of a configuration.

    :param config: configuration drawn by `_sample_config()`
    :return: model with `fit()` and `predict()`
    """
    family = config["family"]
    if family == "ridge":
        model = slm.Ridge(alpha=config["alpha"])
    elif family == "tree":
        model = stree.DecisionTreeRegressor(
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            random_state=0,
        )
    else:
        model = sneigh.KNeighborsRegressor(n_neighbors=config["n_neighbors"])
    return model


def _information_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute the correlation between a prediction and its target.

    :param pred: predictions
    :param target: realized values
    :return: Pearson correlation, 0 if the prediction is constant
    """
    if np.std(pred) == 0.0:
        return 0.0
    corr = float(np.corrcoef(pred, target)[0, 1])
    return corr


def _run_search(
    rng: np.random.Generator,
    *,
    features: np.ndarray,
    target: np.ndarray,
    n_trials: int,
    n_folds: int,
    label_horizon: int,
    embargo: int,
    holdout_start: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a random search and score each trial on the CV and on the holdout.

    The CV score is the mean correlation over purged and embargoed folds of
    the research period. The holdout score is the correlation on the final
    block of a model fit on the whole research period.

    :param rng: random number generator
    :param features: array of shape (n_obs, n_features)
    :param target: array of one label per observation
    :param n_trials: number of configurations to try
    :param n_folds: number of folds of the CV
    :param label_horizon: number of observations spanned by one label
    :param embargo: number of observations embargoed after each test block
    :param holdout_start: index of the first observation of the holdout
    :return: CV score and holdout score of each trial
    """
    n_features = features.shape[1]
    research_idx = np.arange(holdout_start - label_horizon)
    folds = _purged_folds(
        n_obs=holdout_start,
        n_folds=n_folds,
        label_horizon=label_horizon,
        embargo=embargo,
    )
    cv_scores = np.zeros(n_trials)
    holdout_scores = np.zeros(n_trials)
    for trial in range(n_trials):
        config = _sample_config(rng, n_features=n_features)
        x = features[:, config["features"]]
        # Cross-validation score on the research period.
        fold_scores = []
        for train_idx, test_idx in folds:
            model = _build_model(config)
            model.fit(x[train_idx], target[train_idx])
            pred = model.predict(x[test_idx])
            fold_scores.append(_information_coefficient(pred, target[test_idx]))
        cv_scores[trial] = float(np.mean(fold_scores))
        # Holdout score of the model fit on all the research period.
        model = _build_model(config)
        model.fit(x[research_idx], target[research_idx])
        pred = model.predict(x[holdout_start:])
        holdout_scores[trial] = _information_coefficient(
            pred, target[holdout_start:]
        )
    return cv_scores, holdout_scores


def cell1_1_plot_search_overfitting(
    *,
    n_obs: int,
    n_features: int,
    n_informative: int,
    signal_coef: float,
    n_trials: int,
    n_folds: int,
    label_horizon: int,
    embargo: int,
    holdout_frac: float,
    n_repeats: int,
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the CV score and the holdout score of the best trial of a search.

    The features are persistent and only a few of them predict the return
    with a small coefficient. A random search picks the configuration with
    the best purged CV score. The holdout score of the picked configuration is
    drawn for illustration, since in a real run the holdout is not touched
    during the search.

    :param n_obs: number of observations, research period plus holdout
    :param n_features: number of candidate features
    :param n_informative: number of features with a nonzero coefficient
    :param signal_coef: coefficient of each informative feature
    :param n_trials: number of configurations in one search
    :param n_folds: number of folds of the purged CV
    :param label_horizon: number of days spanned by one label
    :param embargo: number of observations embargoed after each test block
    :param holdout_frac: fraction of the observations of the final holdout
    :param n_repeats: number of independent searches that are averaged
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the CV score and the holdout score against the number
        of trials
    """
    hdbg.dassert_lte(
        n_informative, n_features, "Cannot have more informative features"
    )
    holdout_start = int(n_obs * (1.0 - holdout_frac))
    rng = np.random.default_rng(seed)
    cv_best = np.zeros((n_repeats, n_trials))
    holdout_best = np.zeros((n_repeats, n_trials))
    oracle = np.zeros(n_repeats)
    for rep in range(n_repeats):
        features = _simulate_ar1_features(
            rng, n_obs=n_obs, n_features=n_features, phi=0.8
        )
        coef = np.zeros(n_features)
        coef[:n_informative] = signal_coef
        # The return of the next day depends on the features of today.
        ret = features @ coef + rng.normal(size=n_obs)
        ret = np.concatenate([[0.0], ret[:-1]])
        target = _forward_sum(ret, label_horizon)
        # Drop the end of the sample, where the label is not complete.
        valid = ~np.isnan(target)
        features_v, target_v = features[valid], target[valid]
        cv_scores, holdout_scores = _run_search(
            rng,
            features=features_v,
            target=target_v,
            n_trials=n_trials,
            n_folds=n_folds,
            label_horizon=label_horizon,
            embargo=embargo,
            holdout_start=min(holdout_start, len(target_v) - 1),
        )
        # The incumbent is the trial with the best CV score so far.
        best_idx = np.zeros(n_trials, dtype=int)
        running_best = -np.inf
        current = 0
        for trial in range(n_trials):
            if cv_scores[trial] > running_best:
                running_best = cv_scores[trial]
                current = trial
            best_idx[trial] = current
        cv_best[rep] = cv_scores[best_idx]
        holdout_best[rep] = holdout_scores[best_idx]
        # Correlation of the true signal with the target on the holdout.
        signal = features_v @ coef
        oracle[rep] = _information_coefficient(
            signal[holdout_start:], target_v[holdout_start:]
        )
    # Mean and interquartile band over the independent searches.
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    trials = np.arange(1, n_trials + 1)
    for values, color, label in [
        (cv_best, _BLUE, "best CV score"),
        (holdout_best, _CORAL, "holdout score of the best"),
    ]:
        mean = values.mean(axis=0)
        low, high = np.percentile(values, [25, 75], axis=0)
        ax.plot(trials, mean, color=color, linewidth=2.5, label=label)
        ax.fill_between(trials, low, high, color=color, alpha=0.15, linewidth=0)
    ax.axhline(
        oracle.mean(),
        color=_GRAY,
        linestyle="--",
        linewidth=1.0,
        label="oracle: true signal",
    )
    ax.set_xscale("log")
    ax.set_xlabel("number of trials")
    ax.set_ylabel("correlation with the label")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    return fig


# #############################################################################
# Cell 2.1: Triple barrier method
# #############################################################################


def _triple_barrier_label(
    log_price: np.ndarray,
    *,
    start: int,
    horizon: int,
    width: float,
) -> Tuple[int, int]:
    """
    Label one event with the triple barrier method.

    :param log_price: log price path
    :param start: index of the event
    :param horizon: number of bars until the vertical barrier
    :param width: distance of each horizontal barrier from the start, in log
        price units
    :return: label (+1 upper, -1 lower, 0 vertical) and index of the touch
    """
    end = min(start + horizon, len(log_price) - 1)
    path = log_price[start + 1 : end + 1] - log_price[start]
    hit_up = np.flatnonzero(path >= width)
    hit_dn = np.flatnonzero(path <= -width)
    first_up = hit_up[0] if len(hit_up) > 0 else np.inf
    first_dn = hit_dn[0] if len(hit_dn) > 0 else np.inf
    if first_up < first_dn:
        return 1, start + 1 + int(first_up)
    if first_dn < first_up:
        return -1, start + 1 + int(first_dn)
    return 0, end


def cell2_1_plot_triple_barrier(
    *,
    n_bars: int,
    volatility: float,
    horizon: int,
    barrier_mult: float,
    starts: List[int],
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot a price path with the triple barriers and the labels of some events.

    Each horizontal barrier sits `barrier_mult` standard deviations of the
    return over the horizon away from the entry price. The label is +1 when the
    upper barrier is touched first, -1 for the lower one, and 0 when the
    vertical barrier is reached first.

    :param n_bars: number of bars of the path
    :param volatility: volatility of the return of one bar
    :param horizon: number of bars until the vertical barrier
    :param barrier_mult: width of the horizontal barriers, in standard
        deviations of the return over the horizon
    :param starts: indices of the events to label
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the path and one barrier box per event
    """
    rng = np.random.default_rng(seed)
    log_price = np.cumsum(rng.normal(0.0, volatility, size=n_bars))
    price = 100.0 * np.exp(log_price)
    width = barrier_mult * volatility * np.sqrt(horizon)
    names = {1: "+1: profit taking", -1: "-1: stop loss", 0: "0: time out"}
    colors = {1: _GREEN, -1: _CORAL, 0: _ORANGE}
    # Draw the path, then one box of barriers per event.
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    ax.plot(np.arange(n_bars), price, color=_GRAY, linewidth=1.5)
    used = set()
    for start in starts:
        label, touch = _triple_barrier_label(
            log_price, start=start, horizon=horizon, width=width
        )
        p0 = price[start]
        upper, lower = p0 * np.exp(width), p0 * np.exp(-width)
        end = min(start + horizon, n_bars - 1)
        color = colors[label]
        ax.add_patch(
            mpatches.Rectangle(
                (start, lower),
                end - start,
                upper - lower,
                fill=False,
                edgecolor=_LIGHT_GRAY,
                linewidth=1.0,
            )
        )
        # Highlight the barrier that ends the event.
        if label == 1:
            ax.plot([start, end], [upper, upper], color=color, linewidth=2.5)
        elif label == -1:
            ax.plot([start, end], [lower, lower], color=color, linewidth=2.5)
        else:
            ax.plot([end, end], [lower, upper], color=color, linewidth=2.5)
        ax.plot(start, p0, "o", color=_BLUE, markersize=6)
        legend_label = names[label] if label not in used else None
        used.add(label)
        ax.plot(
            touch,
            price[touch],
            "s",
            color=color,
            markersize=7,
            label=legend_label,
        )
    ax.set_xlabel("time (bar)")
    ax.set_ylabel("price")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    return fig


# #############################################################################
# Cell 2.2: Label choice as a hidden trial dimension
# #############################################################################


def _label_choice_sharpes(
    rng: np.random.Generator,
    *,
    n_obs: int,
    n_features: int,
    volatility: float,
    horizons: List[int],
    thresholds: List[float],
    train_frac: float,
    valid_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backtest one model per label configuration on a return series with no skill.

    Each configuration builds the label as the sign of the return over the
    horizon, with a dead zone around 0 that is dropped from the training set.
    A logistic regression trained on the first block trades the sign of its
    score the next day, on the validation block and on the holdout block.

    :param rng: random number generator
    :param n_obs: number of days
    :param n_features: number of features
    :param volatility: daily volatility of the returns
    :param horizons: label horizons, in days
    :param thresholds: dead zone of the label, in multiples of the volatility
        over the horizon
    :param train_frac: fraction of the days used to train the models
    :param valid_frac: fraction of the days of the validation block
    :return: annualized Sharpe ratio of each configuration on the validation
        block and on the holdout block, with shape (n_horizons, n_thresholds)
    """
    ret = rng.normal(0.0, volatility, size=n_obs)
    features = _simulate_ar1_features(
        rng, n_obs=n_obs, n_features=n_features, phi=0.8
    )
    n_train = int(n_obs * train_frac)
    n_valid = int(n_obs * valid_frac)
    valid_sl = slice(n_train, n_train + n_valid)
    hold_sl = slice(n_train + n_valid, n_obs - 1)
    # The position decided at the close of day t earns the return of day t + 1.
    next_ret = np.concatenate([ret[1:], [0.0]])
    shape = (len(horizons), len(thresholds))
    sharpe_valid = np.zeros(shape)
    sharpe_hold = np.zeros(shape)
    for i, horizon in enumerate(horizons):
        fwd = _forward_sum(ret, horizon)
        for j, threshold in enumerate(thresholds):
            cutoff = threshold * volatility * np.sqrt(horizon)
            label = np.sign(fwd)
            label[np.abs(fwd) < cutoff] = 0.0
            # Drop the dead zone and the labels that reach into the next block.
            train_idx = np.arange(n_train - horizon)
            train_idx = train_idx[label[train_idx] != 0.0]
            model = slm.LogisticRegression(C=0.1)
            model.fit(features[train_idx], label[train_idx] > 0.0)
            score = model.decision_function(features)
            position = np.sign(score)
            pnl = position * next_ret
            sharpe_valid[i, j] = _annualized_sharpe(
                pnl[valid_sl], _DAYS_PER_YEAR
            )
            sharpe_hold[i, j] = _annualized_sharpe(pnl[hold_sl], _DAYS_PER_YEAR)
    return sharpe_valid, sharpe_hold


def cell2_2_plot_label_choice(
    *,
    n_obs: int,
    n_features: int,
    volatility: float,
    horizons: List[int],
    thresholds: List[float],
    train_frac: float,
    valid_frac: float,
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the validation Sharpe ratio of one model per label configuration.

    The returns are pure noise, so every configuration has zero skill. The
    best configuration on the validation block is marked, together with its
    Sharpe ratio on the holdout block.

    :param n_obs: number of days
    :param n_features: number of features
    :param volatility: daily volatility of the returns
    :param horizons: label horizons, in days
    :param thresholds: dead zone of the label, in multiples of the volatility
        over the horizon
    :param train_frac: fraction of the days used to train the models
    :param valid_frac: fraction of the days of the validation block
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the Sharpe ratio of each label configuration
    """
    rng = np.random.default_rng(seed)
    sharpe_valid, sharpe_hold = _label_choice_sharpes(
        rng,
        n_obs=n_obs,
        n_features=n_features,
        volatility=volatility,
        horizons=horizons,
        thresholds=thresholds,
        train_frac=train_frac,
        valid_frac=valid_frac,
    )
    best_i, best_j = np.unravel_index(
        np.argmax(sharpe_valid), sharpe_valid.shape
    )
    # One group of dots per horizon, one dot per dead zone.
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    n_thr = len(thresholds)
    offsets = np.linspace(-0.25, 0.25, n_thr)
    for i in range(len(horizons)):
        ax.scatter(
            i + offsets,
            sharpe_valid[i],
            color=_BLUE,
            s=28,
            label="validation Sharpe ratio" if i == 0 else None,
        )
    ax.scatter(
        best_i + offsets[best_j],
        sharpe_valid[best_i, best_j],
        s=150,
        facecolor="none",
        edgecolor=_ORANGE,
        linewidth=2.0,
        label="best on validation",
    )
    ax.scatter(
        best_i + offsets[best_j],
        sharpe_hold[best_i, best_j],
        marker="D",
        color=_CORAL,
        s=45,
        label="same configuration on holdout",
    )
    ax.axhline(0.0, color=_GRAY, linewidth=1.0)
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.set_xlabel("label horizon")
    ax.set_ylabel("annualized Sharpe ratio")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    return fig


# #############################################################################
# Cell 3.1: Effective number of trials
# #############################################################################


def _simulate_clustered_trials(
    rng: np.random.Generator,
    *,
    n_clusters: int,
    trials_per_cluster: int,
    intra_corr: float,
    n_days: int,
) -> np.ndarray:
    """
    Simulate zero-skill trial returns that come in correlated clusters.

    Trials of one cluster share a common factor, e.g., variants of one
    strategy with different hyperparameters.

    :param rng: random number generator
    :param n_clusters: number of clusters
    :param trials_per_cluster: number of trials of each cluster
    :param intra_corr: correlation between two trials of the same cluster
    :param n_days: number of days
    :return: array of shape (n_days, n_clusters * trials_per_cluster)
    """
    hdbg.dassert_lte(0.0, intra_corr, "Correlation must be nonnegative")
    hdbg.dassert_lt(intra_corr, 1.0, "Correlation must be below 1")
    common = rng.normal(size=(n_days, n_clusters))
    idio = rng.normal(size=(n_days, n_clusters * trials_per_cluster))
    common_rep = np.repeat(common, trials_per_cluster, axis=1)
    returns = np.sqrt(intra_corr) * common_rep + np.sqrt(1.0 - intra_corr) * idio
    return returns


def _count_effective_trials(
    returns: np.ndarray, *, cut_corr: float
) -> Tuple[int, np.ndarray]:
    """
    Count the clusters of trials with hierarchical clustering.

    The distance between two trials is `sqrt((1 - corr) / 2)`. The tree is cut
    where the correlation between the merged groups falls below `cut_corr`.

    :param returns: array of shape (n_days, n_trials)
    :param cut_corr: correlation threshold of the cut
    :return: number of clusters and the order of the trials along the tree
    """
    corr = np.corrcoef(returns, rowvar=False)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = shier.linkage(sdist.squareform(dist, checks=False), method="average")
    cut_dist = np.sqrt((1.0 - cut_corr) / 2.0)
    cluster_id = shier.fcluster(link, t=cut_dist, criterion="distance")
    order = shier.leaves_list(link)
    return int(len(np.unique(cluster_id))), order


def cell3_1_plot_effective_trials(
    *,
    n_clusters: int,
    trials_per_cluster: int,
    intra_corr: float,
    n_days: int,
    cut_corr: float,
    n_sims: int,
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the correlation of the trials and the expected maximum Sharpe ratio.

    The left panel shows the correlation matrix of the trials ordered by the
    clustering tree. The right panel compares the expected maximum Sharpe
    ratio under three counts of the trials: all the trials as if independent,
    the number of clusters, and the Monte Carlo value of the correlated
    trials.

    :param n_clusters: number of clusters of correlated trials
    :param trials_per_cluster: number of trials in each cluster
    :param intra_corr: correlation between two trials of the same cluster
    :param n_days: number of days of each trial
    :param cut_corr: correlation threshold of the clustering cut
    :param n_sims: number of Monte Carlo repetitions
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with two panels
    """
    rng = np.random.default_rng(seed)
    n_trials = n_clusters * trials_per_cluster
    returns = _simulate_clustered_trials(
        rng,
        n_clusters=n_clusters,
        trials_per_cluster=trials_per_cluster,
        intra_corr=intra_corr,
        n_days=n_days,
    )
    n_eff, order = _count_effective_trials(returns, cut_corr=cut_corr)
    corr = np.corrcoef(returns, rowvar=False)[np.ix_(order, order)]
    # Standard deviation of an annualized Sharpe ratio of zero-skill returns.
    sharpe_std = np.sqrt(_DAYS_PER_YEAR / n_days)
    max_sharpes = np.zeros(n_sims)
    for sim in range(n_sims):
        sim_returns = _simulate_clustered_trials(
            rng,
            n_clusters=n_clusters,
            trials_per_cluster=trials_per_cluster,
            intra_corr=intra_corr,
            n_days=n_days,
        )
        sharpes = sim_returns.mean(axis=0) / sim_returns.std(axis=0, ddof=1)
        max_sharpes[sim] = sharpes.max() * np.sqrt(_DAYS_PER_YEAR)
    values = [
        _expected_max_sharpe(float(n_trials), sharpe_std),
        _expected_max_sharpe(float(n_eff), sharpe_std),
        float(max_sharpes.mean()),
    ]
    labels = [
        f"all trials\nN = {n_trials}",
        f"clusters\nN = {n_eff}",
        "Monte Carlo\nof the trials",
    ]
    # Left panel: correlation matrix. Right panel: expected maximum Sharpe.
    fig, axes = _new_figure(
        figsize, ncols=2, gridspec_kw={"width_ratios": [1, 1]}
    )
    ax = axes[0, 0]
    image = ax.imshow(corr, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xlabel("trial (ordered by tree)")
    ax.set_ylabel("trial")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="correlation")
    # Expected maximum Sharpe ratio under three counts of the trials.
    ax = axes[0, 1]
    bars = ax.bar(
        range(3), values, color=[_LIGHT_GRAY, _BLUE, _ORANGE], width=0.6
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("expected maximum Sharpe ratio")
    ax.set_ylim(0.0, 1.15 * max(values))
    fig.tight_layout(w_pad=2.5)
    return fig


# #############################################################################
# Cell 3.2: Holdout looks
# #############################################################################


def _crossover_sharpe(
    log_price: np.ndarray, ret: np.ndarray, fast: int, slow: int
) -> float:
    """
    Compute the Sharpe ratio of a moving-average crossover rule.

    The position is long when the fast average is above the slow one, and
    short otherwise. It is decided at the close and earns the next return.

    :param log_price: log price path
    :param ret: returns of each day
    :param fast: window of the fast moving average
    :param slow: window of the slow moving average
    :return: annualized Sharpe ratio
    """
    csum = np.concatenate([[0.0], np.cumsum(log_price)])
    n_obs = len(log_price)
    idx = np.arange(slow - 1, n_obs)
    fast_ma = (csum[idx + 1] - csum[idx + 1 - fast]) / fast
    slow_ma = (csum[idx + 1] - csum[idx + 1 - slow]) / slow
    position = np.sign(fast_ma - slow_ma)
    pnl = position[:-1] * ret[idx[1:]]
    return _annualized_sharpe(pnl, _DAYS_PER_YEAR)


def cell3_2_plot_holdout_looks(
    *,
    n_days: int,
    volatility: float,
    n_looks: int,
    max_slow: int,
    n_sims: int,
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the holdout Sharpe ratio of a rule tuned by repeated looks.

    A researcher tunes the two windows of a moving-average crossover on a
    holdout of a random walk. Each look tries a nearby pair of windows and
    keeps it when the holdout Sharpe ratio improves. The rule is also scored on
    fresh data that was never looked at.

    :param n_days: number of days of the holdout and of the fresh data
    :param volatility: daily volatility of the returns
    :param n_looks: number of looks at the holdout
    :param max_slow: largest window of the slow moving average
    :param n_sims: number of independent researchers that are averaged
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the holdout score and the fresh score against the
        number of looks
    """
    rng = np.random.default_rng(seed)
    holdout_score = np.zeros((n_sims, n_looks + 1))
    fresh_score = np.zeros((n_sims, n_looks + 1))
    for sim in range(n_sims):
        ret_h = rng.normal(0.0, volatility, size=n_days)
        ret_f = rng.normal(0.0, volatility, size=n_days)
        lp_h, lp_f = np.cumsum(ret_h), np.cumsum(ret_f)
        fast = int(rng.integers(2, max_slow // 2))
        slow = int(rng.integers(fast + 1, max_slow))
        best = _crossover_sharpe(lp_h, ret_h, fast, slow)
        for look in range(n_looks + 1):
            if look > 0:
                # Try a nearby pair of windows and keep it if it helps.
                new_fast = int(
                    np.clip(fast + rng.integers(-5, 6), 2, max_slow - 2)
                )
                new_slow = int(
                    np.clip(slow + rng.integers(-10, 11), new_fast + 1, max_slow)
                )
                score = _crossover_sharpe(lp_h, ret_h, new_fast, new_slow)
                if score > best:
                    best, fast, slow = score, new_fast, new_slow
            holdout_score[sim, look] = best
            fresh_score[sim, look] = _crossover_sharpe(lp_f, ret_f, fast, slow)
    # Mean and interquartile band over the independent researchers.
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    looks = np.arange(n_looks + 1)
    for values, color, label in [
        (holdout_score, _CORAL, "holdout (looked at)"),
        (fresh_score, _BLUE, "fresh data"),
    ]:
        mean = values.mean(axis=0)
        low, high = np.percentile(values, [25, 75], axis=0)
        ax.plot(looks, mean, color=color, linewidth=2.5, label=label)
        ax.fill_between(looks, low, high, color=color, alpha=0.15, linewidth=0)
    ax.axhline(0.0, color=_GRAY, linewidth=1.0)
    ax.set_xlabel("number of looks at the holdout")
    ax.set_ylabel("annualized Sharpe ratio")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    return fig
