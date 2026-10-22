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

import pandas as pd
import yfinance as yf

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
    french_path: str = FRENCH_ZIP_PATH,
    provenance_path: str = FRENCH_PROVENANCE_PATH
) -> None:
    """
    Download value-weighted daily returns from Ken French (backup if Yahoo rejects).
    """
    hio.create_dir(os.path.dirname(french_path), incremental = True)
    urllib.request.urlretrieve(FRENCH_URL, french_path)
    provenance = {
        "source": "Ken French",
        "url": FRENCH_URL,
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

def parse_french_returns(zip_path: str) -> pd.DataFrame:
    """
    Parse the value-weighted daily returns from the Ken French 10-industry zip.
    """
    with zipfile.ZipFile(zip_path) as zf:
        text = zf.read(zf.namelist()[0]).decode()
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(","))
    end = next(
        i for i in range(start + 1, len(lines)) if lines[i].strip() == ""
    )
    section = "\n".join(lines[start:end])
    returns = pd.read_csv(io.StringIO(section), index_col = 0)
    returns.index = pd.to_datetime(returns.index.astype(str), format = "%Y%m%d")
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
    french_path: str = FRENCH_ZIP_PATH,
    provenance_path: str = FRENCH_PROVENANCE_PATH
) -> pd.DataFrame:
    """
    Load the cached backup Ken French returns, downloading them only if no cache exists.
    """
    if not os.path.exists(french_path):
        _LOG.info("No cache at %s; downloading from Web", french_path)
        download_french_data(
            french_path = french_path, provenance_path = provenance_path
        )
    provenance = hio.from_json(provenance_path)
    hdbg.dassert_eq(
        _compute_sha256(french_path),
        provenance["sha256"],
        "Cache '%s' does not match its provenance record; delete it to re-download",
        french_path
    )
    returns = parse_french_returns(french_path)
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
