"""
nevergrad_utils.py

Utility functions shared by the Nevergrad tutorial notebooks
(nevergrad.API.ipynb and nevergrad.example.ipynb).

- Notebooks should call these functions instead of writing raw logic inline,
  to keep the notebooks clean, modular, and easier to debug.
- Reusable logic (data loading, walk-forward splitting, portfolio statistics,
  cost-aware losses, Nevergrad optimization, out-of-sample evaluation, and
  plotting) will be implemented here as the project progresses.

Import as:

import nevergrad_utils as nvutils
"""

import datetime
import hashlib
import logging
import os
import io
import zipfile
import urllib.request
from typing import Iterator, List, Tuple, Union, Optional

import pandas as pd
import numpy as np
import yfinance as yf
import scipy.optimize as spo

import helpers.hdbg as hdbg
import helpers.hio as hio

_LOG = logging.getLogger(__name__)


TICKERS = ['SPY', 'EFA', 'EEM', 'TLT', 'IEF', 'LQD', 'GLD', 'VNQ', 'DBC']
START_DATE = "2006-01-01"
END_DATE = "2026-01-01"

PRICES_PATH = "data/etf_prices.csv"
PROVENANCE_PATH = "data_provenance.json"

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Industry_Portfolios_daily_CSV.zip"
FRENCH_ZIP_PATH = "data/10_Industry_Portfolios_daily_CSV.zip"
FRENCH_PROVENANCE_PATH = "data_provenance_french.json"

# The monthly Fama-French factors file, whose `RF` column is the risk-free rate.
RF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
RF_ZIP_PATH = "data/F-F_Research_Data_Factors_CSV.zip"
RF_PROVENANCE_PATH = "data_provenance_rf.json"

# Each walk-forward fold trains on this many calendar years and tests on the
# following year.
NUM_TRAIN_YEARS = 5

# Valid weights can miss 0 or 1 by this much from floating-point error, e.g.,
# after many days of drift or from a numerical solver.
WEIGHT_TOLERANCE = 1e-6

TRADING_DAYS_PER_YEAR = 252


# #############################################################################
# Data acquisition
# #############################################################################


def download_data() -> pd.DataFrame:
    """
    Download adjusted daily close prices for `TICKERS` from Yahoo Finance.
    """
    df = yf.download(
        tickers = TICKERS,
        start = START_DATE,
        end = END_DATE,
        auto_adjust = True,
        progress = False
    )
    
    prices = df["Close"][TICKERS]
    return prices

def download_french_data(
    *,
    url: str = FRENCH_URL,
    french_path: str = FRENCH_ZIP_PATH,
    provenance_path: str = FRENCH_PROVENANCE_PATH
) -> None:
    """
    Download a zip file from the Ken French Data Library and record its provenance.

    By default, download the 10-industry daily returns (backup if Yahoo rejects).
    """
    hio.create_dir(os.path.dirname(french_path), incremental = True)
    urllib.request.urlretrieve(url, french_path)
    provenance = {
        "source": "Ken French",
        "url": url,
        "fetch_date": datetime.date.today().isoformat(),
        "file": french_path,
        "sha256": _compute_sha256(french_path)
    }
    hio.to_json(provenance_path, provenance)

def align_and_clean_data(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the dates on which every ticker has a price.
    """
    clean = prices.dropna()
    hdbg.dassert_lt(0, len(clean), "No date has a price for every ticker")
    _LOG.info(
        "Dropped '%s' of '%s' rows; data starts on '%s'",
        len(prices) - len(clean),
        len(prices),
        clean.index[0].date()
    )
    return clean

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily simple returns from prices. 
    
    Simple (not log) returns are used because a portfolio's daily return
    is then exactly the weighted sum `w * r` of its assets' returns.
    """
    hdbg.dassert(not prices.isna().any().any(), "Prices must have no NaNs")
    returns = prices.pct_change(fill_method = None).iloc[1:]
    return returns

def parse_french_returns(
    zip_path: str, *, date_format: str = "%Y%m%d"
) -> pd.DataFrame:
    """
    Parse the first table of returns from a Ken French zip.

    For the 10-industry zip, the first table holds the value-weighted daily
    returns; for the monthly factors zip, it holds the monthly factors.

    :param zip_path: path to the zip downloaded from the Ken French library
    :param date_format: format of the integer dates in the first column
        - `"%Y%m%d"` for daily files (e.g., `20060103`)
        - `"%Y%m"` for monthly files (e.g., `200601`, parsed as 2006-01-01)
    :return: returns as decimals, indexed by date
    """
    with zipfile.ZipFile(zip_path) as zf:
        text = zf.read(zf.namelist()[0]).decode()
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(","))
    # The first table ends at the first blank line, or at the end of the file.
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].strip() == ""),
        len(lines)
    )
    section = "\n".join(lines[start:end])
    returns = pd.read_csv(io.StringIO(section), index_col = 0)
    returns.index = pd.to_datetime(returns.index.astype(str), format = date_format)
    returns.index.name = "Date"
    returns = returns.mask(returns.isin([-99.99, -999]))
    returns = returns / 100
    return returns
    
def check_returns(returns: pd.DataFrame, *, max_abs_return: float = 0.25) -> None:
    """
    Check that the returns are complete and have plausible magnitudes.
    """
    hdbg.dassert(not returns.isna().any().any(), "Returns must have no NaNs")
    hdbg.dassert_no_duplicates(returns.index, "Each date must appear once")
    hdbg.dassert(
        returns.index.is_monotonic_increasing, "Dates must be in order"
    )
    largest = float(returns.abs().max().max())
    hdbg.dassert_lte(
        largest,
        max_abs_return,
        "A daily return is too large to be plausible; inspect the data"
    )

def _compute_sha256(path: str) -> str:
    """
    Compute the SHA-256 checksum of the files bytes.
    """
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return digest

def save_data(
    prices: pd.DataFrame,
    *,
    prices_path: str = PRICES_PATH,
    provenance_path: str = PROVENANCE_PATH
) -> None:
    """
    Save prices to CSV and record where they came from.
    """
    hio.create_dir(os.path.dirname(prices_path), incremental = True)
    prices.to_csv(prices_path)
    _LOG.info("Saved prices to '%s'", prices_path)
    provenance = {
        "source": "yfinance",
        "fetch_date": datetime.date.today().isoformat(),
        "tickers": TICKERS,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "first_date": str(prices.index[0].date()),
        "last_date": str(prices.index[-1].date()),
        "num_rows": len(prices),
        "file": prices_path,
        "sha256": _compute_sha256(prices_path)
    }
    hio.to_json(provenance_path, provenance)
    _LOG.info("Saved provenance to %s", provenance_path)

def load_data(
    *,
    prices_path: str = PRICES_PATH,
    provenance_path: str = PROVENANCE_PATH
) -> pd.DataFrame:
    """
    Load the cached prices, downloading them only if no cache exists.
    """
    if not os.path.exists(prices_path):
        _LOG.info("No cache at %s: downloading from Yahoo", prices_path)
        prices = align_and_clean_data(download_data())
        save_data(
            prices, prices_path = prices_path, provenance_path = provenance_path
        )
    provenance = hio.from_json(provenance_path)
    hdbg.dassert_eq(
        _compute_sha256(prices_path),
        provenance["sha256"],
        "Cache '%s' does not match its provenance record; delete it to re-download",
        prices_path
    )
    prices = pd.read_csv(prices_path, index_col = 0, parse_dates = True)
    _LOG.info(
        "Loaded '%s' rows from '%s' (fetched on '%s')",
        len(prices),
        prices_path,
        provenance["fetch_date"]
    )
    return prices

def load_french_returns(
    *,
    url: str = FRENCH_URL,
    french_path: str = FRENCH_ZIP_PATH,
    provenance_path: str = FRENCH_PROVENANCE_PATH,
    date_format: str = "%Y%m%d"
) -> pd.DataFrame:
    """
    Load the cached backup Ken French returns, downloading them only if no cache exists.

    By default, load the 10-industry daily returns; pass `url`, the paths, and
    `date_format` to load another Ken French file (see `load_risk_free_rate()`).
    """
    if not os.path.exists(french_path):
        _LOG.info("No cache at %s; downloading from Web", french_path)
        download_french_data(
            url = url, french_path = french_path, provenance_path = provenance_path
        )
    provenance = hio.from_json(provenance_path)
    hdbg.dassert_eq(
        _compute_sha256(french_path),
        provenance["sha256"],
        "Cache '%s' does not match its provenance record; delete it to re-download",
        french_path
    )
    returns = parse_french_returns(french_path, date_format = date_format)
    returns = returns[(returns.index >= START_DATE) & (returns.index < END_DATE)]
    _LOG.info(
        "Loaded '%s' rows from '%s' (fetched on '%s')",
        len(returns),
        french_path,
        provenance["fetch_date"]
    )
    return returns

def load_returns(data_source: str) -> pd.DataFrame:
    hdbg.dassert_in(
        data_source, 
        ["etf", "french"], 
        "Source '%s' does not match recognized data source",
        data_source)
    if data_source == "etf":
        df = compute_returns(load_data())
    elif data_source == "french":
        df = load_french_returns()
    check_returns(df)
    return df

def load_risk_free_rate(
    *,
    rf_path: str = RF_ZIP_PATH,
    provenance_path: str = RF_PROVENANCE_PATH
) -> pd.Series:
    """
    Load the monthly 1-month T-bill return used as the risk-free rate.
    """
    factors = load_french_returns(
        url = RF_URL,
        french_path = rf_path,
        provenance_path = provenance_path,
        date_format = "%Y%m"
    )
    rf = factors["RF"]
    # A 1-month T-bill return above 1% per month (about 12% per year) has not
    # occurred since the early 1980s, so it signals a parsing error.
    check_returns(rf.to_frame(), max_abs_return = 0.01)
    return rf


# #############################################################################
# Walk-forward folds
# #############################################################################


def get_test_years(
    index: pd.DatetimeIndex, *, num_train_years: int = NUM_TRAIN_YEARS
) -> List[int]:
    """
    Compute the test year of each walk-forward fold.
    """
    hdbg.dassert_isinstance(index, pd.DatetimeIndex)
    years = sorted(set(index.year))
    hdbg.dassert_lt(
        num_train_years,
        len(years),
        "Need more than '%s' calendar years of data to have a test year",
        num_train_years
    )
    hdbg.dassert_eq(
        years,
        list(range(years[0], years[-1] + 1)),
        "Every calendar year between the first and last date must have data"
    )
    test_years = years[num_train_years:]
    return test_years

def split_fold(
    df: Union[pd.DataFrame, pd.Series],
    test_year: int,
    *,
    num_train_years: int = NUM_TRAIN_YEARS
) -> Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
    """
    Split date-indexed data into the training and test slices of one fold.
    """
    hdbg.dassert_isinstance(df.index, pd.DatetimeIndex)
    years = df.index.year
    is_train = (years >= test_year - num_train_years) & (years < test_year)
    train = df.loc[is_train]
    test = df.loc[years == test_year]
    hdbg.dassert_lt(0, len(train), "Fold '%s' has no training data", test_year)
    hdbg.dassert_lt(0, len(test), "Fold '%s' has no test data", test_year)
    hdbg.dassert_lt(
        train.index.max(),
        test.index.min(),
        "Training data must strictly precede test data"
    )
    return train, test

def generate_folds(
    returns: pd.DataFrame, *, num_train_years: int = NUM_TRAIN_YEARS
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Yield the `(train_returns, test_returns)` pair of each walk-forward fold.
    """
    test_years = get_test_years(returns.index, num_train_years = num_train_years)
    for test_year in test_years:
        yield split_fold(returns, test_year, num_train_years = num_train_years)

def get_fold_table(
    returns: pd.DataFrame, *, num_train_years: int = NUM_TRAIN_YEARS
) -> pd.DataFrame:
    """
    Summarize the date boundaries of each walk-forward fold.
    """
    rows = []
    for train, test in generate_folds(returns, num_train_years = num_train_years):
        rows.append(
            {
                "test_year": test.index.min().year,
                "train_start": train.index.min().date(),
                "train_end": train.index.max().date(),
                "train_days": len(train),
                "test_start": test.index.min().date(),
                "test_end": test.index.max().date(),
                "test_days": len(test),
            }
        )
    table = pd.DataFrame(rows).set_index("test_year")
    return table

# #############################################################################
# Portfolio simulation
# #############################################################################

def check_weights(weights: np.ndarray, num_assets: int) -> None:
    """
    Check that weights are long-only and fully invested, with one per asset.
    """
    hdbg.dassert_eq(
        len(weights),
        num_assets,
        "Expected '%s' weights, one per asset, but got '%s'",
        num_assets,
        len(weights)
    )
    hdbg.dassert_lte(
        -WEIGHT_TOLERANCE,
        float(weights.min()),
        "Weights must be non-negative (long-only)"
    )
    hdbg.dassert_lte(
        abs(float(weights.sum()) - 1),
        WEIGHT_TOLERANCE,
        "Weights must sum to 1 (fully invested)"
    )

def compute_turnover(
    target_weights: np.ndarray,
    current_weights: np.ndarray
) -> float:
    hdbg.dassert_eq(
        len(target_weights),
        len(current_weights),
        "Length '%s' of target_weights and length '%s' of current_weights are not equal",
        len(target_weights), 
        len(current_weights)
    )
    return float(np.abs(target_weights - current_weights).sum())

def get_rebalance_days(
    rebal_freq: str,
    index: pd.DatetimeIndex
) -> np.ndarray:
    hdbg.dassert_isinstance(
        index,
        pd.DatetimeIndex,
        "Index must be of type pd.DatetimeIndex"
    )
    freq_map = {"monthly": "M", "quarterly": "Q", "annual": "Y"}
    hdbg.dassert_in(
        rebal_freq,
        freq_map.keys(),
        "Rebalance frequency '%s' is not a possible frequency (only monthly, quarterly, or annual)",
        rebal_freq
    )
    periods = index.to_period(freq_map[rebal_freq])
    return np.r_[True, periods[1:] != periods[:-1]]

def simulate_portfolio(
    returns: pd.DataFrame,
    target_weights: np.ndarray,
    *,
    rebal_freq: str = "monthly",
    cost_bps: float = 10.0,
    initial_weights: Optional[np.ndarray] = None
) -> Tuple[pd.Series, pd.Series, np.ndarray]:
    """
    Simulate daily portfolio changes.
    """
    target_weights = np.asarray(target_weights)
    if initial_weights is None:
        initial_weights = target_weights.copy()
    else:
        initial_weights = np.asarray(initial_weights)
    
    check_weights(target_weights, returns.shape[1])
    check_weights(initial_weights, returns.shape[1])

    rebalance_days = get_rebalance_days(rebal_freq, returns.index)
    
    r = returns.to_numpy()
    w = initial_weights.copy()
    
    net = []
    turnover = []
    
    for t in range(len(r)):
        if rebalance_days[t]:
            turnover.append(compute_turnover(target_weights, w))
            w = target_weights.copy()
        else:
            turnover.append(0)
        gross = w @ r[t]
        net.append(gross - cost_bps / 10000 * turnover[-1])
        w = w * (1 + r[t]) / (1 + gross)
    
    net = pd.Series(net, index = returns.index)
    turnover = pd.Series(turnover, index = returns.index)
    return net, turnover, w
    
def annualized_return(returns: pd.Series) -> float:
    return returns.mean() * TRADING_DAYS_PER_YEAR

def annualized_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

def equal_weights(num_assets: int) -> np.ndarray:
    return np.asarray([1/num_assets for _ in range(num_assets)])

def min_variance_weights(
    train_returns: pd.DataFrame,
    *,
    max_weight: float = 1.0
) -> np.ndarray:
    n = train_returns.shape[1]
    cov = train_returns.cov().to_numpy() * TRADING_DAYS_PER_YEAR
    hdbg.dassert_lte(
        1,
        max_weight * n,
        "No valid portfolio of '%s' assets of maximum weight '%s' exists; weights must sum to 1",
        n,
        max_weight
    )
    result = spo.minimize(
        fun = lambda w: w @ cov @ w,
        x0 = equal_weights(n),
        method = "SLSQP",
        bounds = [(0, max_weight)] * n,
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options = {"ftol": 1e-12}
    )
    hdbg.dassert(
        result.success,
        "Optimizer failed: '%s'",
        result.message
    )
    
    check_weights(result.x, n)
    
    return result.x
