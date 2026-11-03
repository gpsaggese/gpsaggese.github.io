"""
Utility functions for the "Learn Kats in 60 Minutes" tutorial.

The notebooks `kats.API.ipynb` and `kats.example.ipynb` keep the Kats API calls
and the narrative in the cells. Everything that is plumbing lives here: loading
and cleaning prices, thin wrappers that give every model the same output
format, the metrics, the rolling-origin backtest, and the plots.

Import as:

import kats_utils as utils
"""

import contextlib
import logging
import os
import warnings
from typing import Dict, Iterator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.tsa.stattools as stattools
import tqdm
from kats.consts import TimeSeriesData
from kats.models.arima import ARIMAModel, ARIMAParams
from kats.models.prophet import ProphetModel, ProphetParams

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import helpers.hprint as hprint

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Constants
# #############################################################################


# Column names that `TimeSeriesData` expects.
TIME_COL = "time"
VALUE_COL = "value"

# Pandas frequency alias for business days (Monday to Friday).
BUSINESS_DAY = "B"

# Names of the models, used as keys in dictionaries and tables.
NAIVE = "Naive"
ARIMA = "ARIMA"
PROPHET = "Prophet"

# Columns of the forecast frame that every `forecast_*()` function returns.
_FORECAST_COLS = [TIME_COL, "fcst", "fcst_lower", "fcst_upper"]


# #############################################################################
# Data loading and preprocessing
# #############################################################################


def get_default_data_path() -> str:
    """
    Return the path of the Tesla prices that ship with the tutorial.

    :return: absolute path of `data/TSLA.csv`
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    return os.path.join(data_dir, "TSLA.csv")


def load_prices(path: str) -> pd.DataFrame:
    """
    Load a daily price file with a `Date` column and sort it by date.

    The file is the Kaggle "Tesla Stock Data from 2010 to 2020" CSV (it
    actually runs until 2021-10-14), with columns `Date, Open, High, Low,
    Close, Adj Close, Volume`.

    :param path: path of the CSV file
    :return: prices sorted by date, with `Date` parsed as a timestamp
    """
    _LOG.debug(hprint.to_str("path"))
    hdbg.dassert_file_exists(path)
    df = pd.read_csv(path, parse_dates=["Date"])
    hdbg.dassert_is_subset(
        ["Date", "Close"], df.columns.tolist(), "Price file needs these columns"
    )
    hdbg.dassert(df["Date"].is_unique, "Each trading day must appear only once")
    df = df.sort_values("Date").reset_index(drop=True)
    _LOG.info(
        "Loaded '%d' rows from '%s' to '%s'",
        len(df),
        df["Date"].min().date(),
        df["Date"].max().date(),
    )
    return df


def to_business_day_series(
    df: pd.DataFrame, *, column: str = "Close"
) -> pd.Series:
    """
    Put a price column on a regular business-day grid.

    The market is closed on holidays, so the raw file skips some business days.
    ARIMA needs a regular grid, so every missing business day gets the last
    close carried forward (the price did not change while the market was
    closed).

    :param df: prices with a `Date` column, as returned by `load_prices()`
    :param column: price column to use
    :return: series indexed by business day, without gaps
    """
    _LOG.debug(hprint.to_str("column"))
    series = df.set_index("Date")[column]
    business_days = pd.bdate_range(series.index.min(), series.index.max())
    # Reindexing introduces NaNs exactly on the days the market was closed.
    series = series.reindex(business_days)
    num_filled = int(series.isna().sum())
    _LOG.info(
        "Carried the last close forward over '%d' non-trading business days",
        num_filled,
    )
    series = series.ffill()
    series.name = column
    return series


def to_timeseries_data(series: pd.Series) -> TimeSeriesData:
    """
    Convert a pandas series with a datetime index into `TimeSeriesData`.

    :param series: values indexed by timestamp
    :return: Kats time series with columns `time` and `value`
    """
    df = pd.DataFrame({TIME_COL: series.index, VALUE_COL: series.values})
    return TimeSeriesData(df)


def split_train_test(
    series: pd.Series, horizon: int
) -> Tuple[pd.Series, pd.Series]:
    """
    Split a series chronologically, keeping the last `horizon` points as test.

    A random split would leak the future into the training set, so the test
    set is always the most recent block.

    :param series: values indexed by timestamp
    :param horizon: number of points in the test set
    :return: training series and test series
    """
    _LOG.debug(hprint.to_str("horizon"))
    hdbg.dassert_lt(0, horizon, "Horizon must be positive")
    hdbg.dassert_lt(horizon, len(series), "Horizon must leave training data")
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]
    _LOG.info(
        "Train '%s' to '%s' (%d points), test '%s' to '%s' (%d points)",
        train.index[0].date(),
        train.index[-1].date(),
        len(train),
        test.index[0].date(),
        test.index[-1].date(),
        len(test),
    )
    return train, test


def check_stationarity(series: pd.Series) -> pd.DataFrame:
    """
    Run the augmented Dickey-Fuller test on a series and on its differences.

    A p-value below 0.05 rejects the unit root, i.e., the series is stationary.
    ARIMA needs a stationary series, and the differencing order `d` is the
    number of differences needed to get there.

    :param series: values indexed by timestamp
    :return: one row each for the level and the first difference, with the test
        statistic, the p-value, and a flag for stationarity at the 5% level
    """
    rows = []
    for name, values in [
        ("level", series),
        ("first difference", series.diff().dropna()),
    ]:
        statistic, p_value = stattools.adfuller(values)[:2]
        rows.append(
            {
                "series": name,
                "adf_statistic": statistic,
                "p_value": p_value,
                "stationary_at_5pct": p_value < 0.05,
            }
        )
    return pd.DataFrame(rows).set_index("series")


def make_synthetic_series(
    *, num_periods: int = 365, freq: str = "D", seed: int = 0
) -> pd.Series:
    """
    Generate a small series with a trend, a weekly pattern, and noise.

    The API notebook uses it so every Kats model has something to find and the
    notebook runs in seconds.

    :param num_periods: number of points
    :param freq: pandas frequency alias of the index
    :param seed: seed of the noise
    :return: series indexed by timestamp, starting on 2020-01-01
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=num_periods, freq=freq)
    t = np.arange(num_periods)
    trend = 50 + 0.1 * t
    weekly = 3 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 1, num_periods)
    return pd.Series(trend + weekly + noise, index=index, name="value")


# #############################################################################
# Forecasting wrappers
# #############################################################################


@contextlib.contextmanager
def _quiet_model_logs() -> Iterator[None]:
    """
    Silence the convergence warnings and the Stan logs while a model fits.

    Both are expected during a backtest and would bury the results.
    """
    names = ["prophet", "cmdstanpy"]
    previous = {name: logging.getLogger(name).level for name in names}
    for name in names:
        logging.getLogger(name).setLevel(logging.ERROR)
    # `cmdstanpy` resets its own level to DEBUG at every fit when its logger has
    # no handler, so a no-op handler keeps the level we set.
    null_handler = logging.NullHandler()
    logging.getLogger("cmdstanpy").addHandler(null_handler)
    # Restore the log settings even if the fit raises.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        logging.getLogger("cmdstanpy").removeHandler(null_handler)
        for name, level in previous.items():
            logging.getLogger(name).setLevel(level)


def _forecast_index(train: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """
    Return the next `horizon` business days after the end of the training set.

    :param train: training series on a business-day grid
    :param horizon: number of days to forecast
    :return: dates of the forecast
    """
    index = pd.date_range(
        train.index[-1], periods=horizon + 1, freq=BUSINESS_DAY
    )
    # Drop the first date, which is the last training day.
    return index[1:]


def forecast_naive(
    train: pd.Series, horizon: int, *, interval_level: float = 0.95
) -> pd.DataFrame:
    """
    Forecast with the last observed value, i.e., a random walk without drift.

    Stock prices are close to a random walk, so this is the bar that any model
    has to clear. The interval widens as `sigma * sqrt(h)`, where `sigma` is the
    standard deviation of the daily changes in the training set.

    :param train: training series on a business-day grid
    :param horizon: number of days to forecast
    :param interval_level: probability mass of the prediction interval
    :return: frame with columns `time, fcst, fcst_lower, fcst_upper`
    """
    _LOG.debug(hprint.to_str("horizon interval_level"))
    last_value = train.iloc[-1]
    sigma = train.diff().std()
    steps = np.arange(1, horizon + 1)
    z = scipy.stats.norm.ppf(0.5 + interval_level / 2)
    half_width = z * sigma * np.sqrt(steps)
    return pd.DataFrame(
        {
            TIME_COL: _forecast_index(train, horizon),
            "fcst": last_value,
            "fcst_lower": last_value - half_width,
            "fcst_upper": last_value + half_width,
        }
    )[_FORECAST_COLS]


def select_arima_order(
    train: pd.Series, *, d: int = 1, max_p: int = 2, max_q: int = 2
) -> Tuple[int, int, int]:
    """
    Pick the ARIMA order `(p, d, q)` with the lowest AIC on the training set.

    The AIC rewards fit and penalizes the number of parameters, so it guards
    against picking the biggest model. Only the training data is used, so the
    choice cannot see the test set.

    :param train: training series on a business-day grid
    :param d: differencing order, fixed (see `check_stationarity()`)
    :param max_p: largest autoregressive order to try
    :param max_q: largest moving-average order to try
    :return: the order `(p, d, q)` with the lowest AIC
    """
    _LOG.debug(hprint.to_str("d max_p max_q"))
    ts = to_timeseries_data(train)
    best_aic = np.inf
    best_order = (0, d, 0)
    with _quiet_model_logs():
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                model = ARIMAModel(ts, ARIMAParams(p=p, d=d, q=q))
                model.fit()
                aic = model.model.aic
                _LOG.debug("order=%s aic=%.1f", (p, d, q), aic)
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
    _LOG.info("Selected ARIMA order '%s' with AIC '%.1f'", best_order, best_aic)
    return best_order


def forecast_arima(
    train: pd.Series,
    horizon: int,
    *,
    order: Tuple[int, int, int],
    interval_level: float = 0.95,
) -> pd.DataFrame:
    """
    Fit a Kats `ARIMAModel` and forecast the next `horizon` business days.

    :param train: training series on a business-day grid
    :param horizon: number of days to forecast
    :param order: ARIMA order `(p, d, q)`
    :param interval_level: probability mass of the prediction interval
    :return: frame with columns `time, fcst, fcst_lower, fcst_upper`
    """
    _LOG.debug(hprint.to_str("horizon order interval_level"))
    p, d, q = order
    ts = to_timeseries_data(train)
    with _quiet_model_logs():
        model = ARIMAModel(ts, ARIMAParams(p=p, d=d, q=q))
        model.fit()
        # Kats takes the tail probability, i.e., 0.05 for a 95% interval.
        forecast = model.predict(
            steps=horizon, freq=BUSINESS_DAY, alpha=1 - interval_level
        )
    return _check_forecast(forecast, train, horizon)


def forecast_prophet(
    train: pd.Series,
    horizon: int,
    *,
    interval_level: float = 0.95,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
) -> pd.DataFrame:
    """
    Fit a Kats `ProphetModel` and forecast the next `horizon` business days.

    Prophet defaults to an 80% interval and ARIMA to 95%, so the level is set
    explicitly to make the two intervals comparable.

    :param train: training series on a business-day grid
    :param horizon: number of days to forecast
    :param interval_level: probability mass of the prediction interval
    :param yearly_seasonality: fit a yearly seasonal component
    :param weekly_seasonality: fit a weekly seasonal component
    :return: frame with columns `time, fcst, fcst_lower, fcst_upper`
    """
    _LOG.debug(
        hprint.to_str(
            "horizon interval_level yearly_seasonality weekly_seasonality"
        )
    )
    ts = to_timeseries_data(train)
    params = ProphetParams(
        interval_width=interval_level,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
    )
    with _quiet_model_logs():
        model = ProphetModel(ts, params)
        model.fit()
        forecast = model.predict(steps=horizon, freq=BUSINESS_DAY)
    return _check_forecast(forecast, train, horizon)


def _check_forecast(
    forecast: pd.DataFrame, train: pd.Series, horizon: int
) -> pd.DataFrame:
    """
    Check that a Kats forecast covers the expected days and has the right shape.

    :param forecast: output of a Kats `predict()` call
    :param train: training series the model was fitted on
    :param horizon: number of days that were requested
    :return: the forecast with the standard columns
    """
    hdbg.dassert_eq(len(forecast), horizon, "Forecast has the wrong length")
    expected = _forecast_index(train, horizon)
    hdbg.dassert(
        (forecast[TIME_COL].values == expected.values).all(),
        "Forecast dates must follow the training set on the business-day grid",
    )
    return forecast[_FORECAST_COLS].reset_index(drop=True)


def forecast_all(
    train: pd.Series,
    horizon: int,
    *,
    arima_order: Tuple[int, int, int],
    interval_level: float = 0.95,
) -> Dict[str, pd.DataFrame]:
    """
    Forecast the same horizon with the naive baseline, ARIMA, and Prophet.

    :param train: training series on a business-day grid
    :param horizon: number of days to forecast
    :param arima_order: ARIMA order `(p, d, q)`
    :param interval_level: probability mass of the prediction intervals
    :return: forecast frame of each model, keyed by model name
    """
    return {
        NAIVE: forecast_naive(train, horizon, interval_level=interval_level),
        ARIMA: forecast_arima(
            train, horizon, order=arima_order, interval_level=interval_level
        ),
        PROPHET: forecast_prophet(train, horizon, interval_level=interval_level),
    }


# #############################################################################
# Evaluation
# #############################################################################


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute the mean absolute error, in the units of the series (dollars).

    :param actual: observed values
    :param predicted: forecast values
    :return: mean of `|actual - predicted|`
    """
    hdbg.dassert_eq(len(actual), len(predicted), "Lengths must match")
    return float(np.mean(np.abs(actual - predicted)))


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute the root mean squared error, which punishes large misses more.

    :param actual: observed values
    :param predicted: forecast values
    :return: square root of the mean of `(actual - predicted)^2`
    """
    hdbg.dassert_eq(len(actual), len(predicted), "Lengths must match")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_coverage(actual: np.ndarray, forecast: pd.DataFrame) -> float:
    """
    Compute the share of observed values that fall inside the interval.

    A well-calibrated 95% interval covers about 95% of the points.

    :param actual: observed values
    :param forecast: frame with `fcst_lower` and `fcst_upper`
    :return: fraction of points between the lower and the upper bound
    """
    inside = (actual >= forecast["fcst_lower"].values) & (
        actual <= forecast["fcst_upper"].values
    )
    return float(np.mean(inside))


def evaluate_forecasts(
    actual: pd.Series, forecasts: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Compare the forecasts of several models on the same test set.

    :param actual: observed test values indexed by timestamp
    :param forecasts: forecast frame of each model, keyed by model name
    :return: one row per model, sorted by MAE, with MAE, RMSE, the MAE ratio to
        the naive baseline (below 1 means better than the baseline), and the
        interval coverage
    """
    hdbg.dassert_in(NAIVE, forecasts, "The naive baseline is the reference")
    y = actual.values
    naive_mae = compute_mae(y, forecasts[NAIVE]["fcst"].values)
    rows = {}
    for name, forecast in forecasts.items():
        hdbg.dassert(
            (forecast[TIME_COL].values == actual.index.values).all(),
            "Forecast and test dates must match for '%s'",
            name,
        )
        mae = compute_mae(y, forecast["fcst"].values)
        rows[name] = {
            "MAE": mae,
            "RMSE": compute_rmse(y, forecast["fcst"].values),
            "MAE / naive MAE": mae / naive_mae,
            "interval coverage": compute_coverage(y, forecast),
        }
    return pd.DataFrame(rows).T.sort_values("MAE")


def rolling_origin_evaluation(
    series: pd.Series,
    horizon: int,
    *,
    num_origins: int,
    arima_order: Tuple[int, int, int],
    interval_level: float = 0.95,
) -> pd.DataFrame:
    """
    Backtest the models on consecutive test windows, refitting at each origin.

    A single test window is a coin flip: a model that extrapolates an uptrend
    wins when the price rises and loses when it falls. Walking back through
    `num_origins` non-overlapping windows of `horizon` days shows which result
    holds up. At each origin the models see only the data before it.

    :param series: values indexed by timestamp, on a business-day grid
    :param horizon: number of days in each test window
    :param num_origins: number of windows, ending at the end of the series
    :param arima_order: ARIMA order `(p, d, q)`, kept fixed across origins
    :param interval_level: probability mass of the prediction intervals
    :return: one row per origin and model, with the origin, the price change
        over the window, MAE, RMSE, the MAE ratio to the naive baseline, and
        the interval coverage
    """
    _LOG.debug(hprint.to_str("horizon num_origins arima_order"))
    hdbg.dassert_lt(0, num_origins, "Need at least one origin")
    # Index of the first test point of each window, oldest first.
    first_test = [len(series) - horizon * k for k in range(num_origins, 0, -1)]
    hdbg.dassert_lt(
        0, first_test[0], "Series is too short for '%d' windows", num_origins
    )
    rows: List[Dict[str, object]] = []
    for start in tqdm.tqdm(first_test, desc="Origins"):
        train = series.iloc[:start]
        test = series.iloc[start : start + horizon]
        forecasts = forecast_all(
            train,
            horizon,
            arima_order=arima_order,
            interval_level=interval_level,
        )
        metrics = evaluate_forecasts(test, forecasts)
        change_pct = 100 * (test.iloc[-1] / train.iloc[-1] - 1)
        for name, row in metrics.iterrows():
            rows.append(
                {
                    "origin": train.index[-1],
                    "price change (%)": change_pct,
                    "model": name,
                    **row.to_dict(),
                }
            )
    return pd.DataFrame(rows)


def summarize_rolling_evaluation(results: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize a rolling-origin backtest with one row per model.

    The MAE of different origins is not comparable in dollars (Tesla moved from
    about 5 to 800), so the summary uses ratios and counts.

    :param results: output of `rolling_origin_evaluation()`
    :return: for each model, the mean MAE ratio to the naive baseline, the share
        of windows where it beats the baseline, the share of windows where it
        beats ARIMA, and the mean interval coverage
    """
    mae = results.pivot(index="origin", columns="model", values="MAE")
    summary = pd.DataFrame(
        {
            "mean MAE / naive MAE": results.groupby("model")[
                "MAE / naive MAE"
            ].mean(),
            "windows beating naive (%)": 100 * mae.lt(mae[NAIVE], axis=0).mean(),
            "windows beating ARIMA (%)": 100 * mae.lt(mae[ARIMA], axis=0).mean(),
            "mean interval coverage": results.groupby("model")[
                "interval coverage"
            ].mean(),
        }
    )
    # A model cannot beat itself, so the diagonal entries are not meaningful.
    summary.loc[NAIVE, "windows beating naive (%)"] = np.nan
    summary.loc[ARIMA, "windows beating ARIMA (%)"] = np.nan
    return summary.loc[[NAIVE, ARIMA, PROPHET]]


# #############################################################################
# Plots
# #############################################################################


def plot_forecasts(
    train: pd.Series,
    test: pd.Series,
    forecasts: Dict[str, pd.DataFrame],
    *,
    num_history: int = 120,
    figsize: Tuple[float, float] = (16, 4.5),
) -> None:
    """
    Plot history, actual test values, and each forecast with its interval.

    There is one panel per model, all on the same y axis so the panels can be
    compared by eye.

    :param train: training series
    :param test: observed test values
    :param forecasts: forecast frame of each model, keyed by model name
    :param num_history: number of training points to show before the forecast
    :param figsize: figure size as `(width, height)`
    """
    fig, axes = plt.subplots(
        1, len(forecasts), figsize=figsize, sharey=True, squeeze=False
    )
    for ax, (name, forecast) in zip(axes[0], forecasts.items()):
        ax.plot(train.tail(num_history), color="gray", label="history")
        ax.plot(test, color="black", label="actual")
        ax.plot(forecast[TIME_COL], forecast["fcst"], label="forecast")
        ax.fill_between(
            forecast[TIME_COL],
            forecast["fcst_lower"],
            forecast["fcst_upper"],
            alpha=0.25,
            label="prediction interval",
        )
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=30)
    axes[0][0].set_ylabel("Close price ($)")
    # One legend below all panels, so it does not cover the intervals.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    plt.tight_layout(rect=(0, 0.07, 1, 1))


def plot_rolling_evaluation(
    results: pd.DataFrame, *, figsize: Tuple[float, float] = (14, 4.5)
) -> None:
    """
    Plot the MAE ratio to the naive baseline at each origin, one line a model.

    A ratio below 1 (the dashed line) means the model beat the baseline in that
    window.

    :param results: output of `rolling_origin_evaluation()`
    :param figsize: figure size as `(width, height)`
    """
    _, ax = plt.subplots(figsize=figsize)
    for name in [ARIMA, PROPHET]:
        subset = results[results["model"] == name]
        ax.plot(
            subset["origin"], subset["MAE / naive MAE"], marker="o", label=name
        )
    ax.axhline(1.0, color="black", linestyle="--", label="naive baseline")
    ax.set_yscale("log")
    ax.set_xlabel("forecast origin")
    ax.set_ylabel("MAE / naive MAE (log scale)")
    ax.legend()
    plt.tight_layout()
