"""
Download SPY 1-minute bars from NYSE TAQ on WRDS.

- One query per trading day on the WRDS consolidated trades (`taqmsec.wct_*`),
  which match each trade with the NBBO in force
- The bars are aggregated on the WRDS server, so only ~390 rows per day are
  downloaded
- Each trade is signed by the quote rule: above the NBBO midpoint is a buy,
  below is a sell, at the midpoint is unsigned

Requires the `wrds` package, the WRDS username in the environment variable
`WRDS_USERNAME`, and a `~/.pgpass` file, created by the first
`wrds.Connection()`. WRDS data must not be committed or shared, so the bars are
written only to `DATA_DIR`, which is git-ignored.

Import as:

import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.taq_utils as rltautil
"""

import logging
import os
import re
import time
from typing import Any, List

import pandas as pd

import helpers.hprint as hprint

_LOG = logging.getLogger(__name__)

# Directory with the downloaded data, next to this file.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# WRDS library with the daily TAQ tables of all years.
_TAQ_LIBRARY = "taqmsec"

# Aggregate the SPY trades of one day into 1-minute bars, during regular
# hours. Trades without a valid NBBO and trades reported out of sequence or
# outside regular hours (conditions T, U, Z) are dropped.
# `volume` counts all the remaining trades. Prices, spreads, and trade signs
# use only the regular trades (conditions empty, @, F, I), since block trades
# such as average-price (B) and contingent (7, V) trades print away from the
# NBBO.
_BARS_SQL = """
WITH t AS (
    SELECT date_trunc('minute', date + time_m) AS minute,
           time_m, tr_seqnum, price, size,
           (nbb + nbo) / 2 AS mid,
           nbo - nbb AS spread,
           COALESCE(tr_scond, '') ~ '^[ @FI]*$' AS reg
    FROM {library}.wct_{day}
    WHERE sym_root = 'SPY' AND sym_suffix IS NULL
      AND time_m >= '09:30' AND time_m < '16:00'
      AND price > 0 AND size > 0 AND nbb > 0 AND nbo > nbb
      AND COALESCE(tr_scond, '') !~ '[TUZ]'
)
SELECT minute,
       (array_agg(price ORDER BY time_m, tr_seqnum)
           FILTER (WHERE reg))[1] AS open,
       max(price) FILTER (WHERE reg) AS high,
       min(price) FILTER (WHERE reg) AS low,
       (array_agg(price ORDER BY time_m DESC, tr_seqnum DESC)
           FILTER (WHERE reg))[1] AS close,
       sum(size) AS volume,
       sum(size) FILTER (WHERE reg) AS reg_volume,
       count(*) FILTER (WHERE reg) AS n_trades,
       sum(price * size) FILTER (WHERE reg) AS notional,
       sum(size) FILTER (WHERE reg AND price > mid) AS buy_volume,
       sum(size) FILTER (WHERE reg AND price < mid) AS sell_volume,
       sum(size * abs(price - mid) / mid) FILTER (WHERE reg) AS abs_dev_size,
       avg(spread / mid) FILTER (WHERE reg) AS rel_spread_mean,
       (array_agg(mid ORDER BY time_m DESC, tr_seqnum DESC)
           FILTER (WHERE reg))[1] AS close_mid
FROM t
GROUP BY minute
ORDER BY minute
"""


def _get_taq_path(year: int) -> str:
    """
    Get the path of the TAQ SPY 1-minute bars of one year.

    :param year: calendar year
    :return: parquet file path in `DATA_DIR`
    """
    _LOG.debug(hprint.to_str("year"))
    path = os.path.join(DATA_DIR, "taq", f"spy_1min_{year}.parquet")
    _LOG.debug("return=%s", path)
    return path


def get_trading_days(tables: List[str], year: int) -> List[str]:
    """
    Extract the trading days of one year from the TAQ table names.

    :param tables: table names of `taqmsec`, e.g., `wct_20150102`
    :param year: calendar year
    :return: sorted days as `YYYYMMDD`
    """
    _LOG.debug(hprint.to_str("len(tables) year"))
    # Keep the daily `wct_YYYYMMDD` tables, not the yearly `wct_YYYY` view.
    pattern = re.compile(rf"^wct_({year}\d{{4}})$")
    days = sorted(m.group(1) for t in tables if (m := pattern.match(t)))
    _LOG.debug("return=%s days", len(days))
    return days


def finalize_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the aggregates returned by `_BARS_SQL` into bars.

    :param raw: one row per minute, with the columns of `_BARS_SQL`
    :return: bars indexed by UTC timestamp with columns `open`, `high`, `low`,
        `close`, `volume`, `reg_volume`, `n_trades`, `vwap`, `buy_volume`,
        `sell_volume`, `signed_volume`, `quoted_spread_bps`,
        `effective_spread_bps`, `close_mid`
    """
    _LOG.debug(hprint.to_str("raw.shape"))
    # The database returns `NUMERIC` columns as `Decimal`.
    df = raw.set_index("minute").astype(float)
    # Minutes with no trade on one side return NULL for that side.
    df[["buy_volume", "sell_volume"]] = df[["buy_volume", "sell_volume"]].fillna(
        0.0
    )
    df["signed_volume"] = df["buy_volume"] - df["sell_volume"]
    df["vwap"] = df["notional"] / df["reg_volume"]
    # Average NBBO spread over the trades, in bps of the midpoint.
    df["quoted_spread_bps"] = df["rel_spread_mean"] * 1e4
    # Size-weighted effective spread: twice the distance from the midpoint.
    df["effective_spread_bps"] = (
        2 * df["abs_dev_size"] / df["reg_volume"] * 1e4
    )
    # TAQ times are New York local times; index by UTC like the Alpaca bars.
    index = pd.DatetimeIndex(df.index).tz_localize("America/New_York")
    df.index = index.tz_convert("UTC")
    df.index.name = "timestamp"
    # Keep the bar columns in a fixed order.
    columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "reg_volume",
        "n_trades",
        "vwap",
        "buy_volume",
        "sell_volume",
        "signed_volume",
        "quoted_spread_bps",
        "effective_spread_bps",
        "close_mid",
    ]
    df = df[columns]
    _LOG.debug("return=%s", df.shape)
    return df


def download_spy_taq_year(db: Any, year: int) -> pd.DataFrame:
    """
    Download the SPY 1-minute bars of one year from TAQ.

    :param db: open `wrds.Connection`
    :param year: calendar year, from 2003
    :return: bars as returned by `finalize_bars()`
    """
    _LOG.debug(hprint.to_str("year"))
    days = get_trading_days(db.list_tables(library=_TAQ_LIBRARY), year)
    dfs = []
    t0 = time.time()
    # Aggregate each day on the WRDS server, then finalize it locally.
    for i, day in enumerate(days):
        raw = db.raw_sql(_BARS_SQL.format(library=_TAQ_LIBRARY, day=day))
        dfs.append(finalize_bars(raw))
        # Log progress only, never data values.
        if (i + 1) % 25 == 0 or i + 1 == len(days):
            _LOG.info(
                "Year '%s': '%s/%s' days in '%.0f' s",
                year,
                i + 1,
                len(days),
                time.time() - t0,
            )
    df = pd.concat(dfs)
    _LOG.debug("return=%s", df.shape)
    return df


def download_spy_taq(start_year: int, end_year: int) -> None:
    """
    Download the TAQ SPY 1-minute bars for each year into `DATA_DIR/taq`.

    Years already on disk are skipped.

    :param start_year: first year
    :param end_year: last year, included
    """
    _LOG.debug(hprint.to_str("start_year end_year"))
    # Import here, so that the rest of the module works without `wrds`.
    import wrds

    os.makedirs(os.path.join(DATA_DIR, "taq"), exist_ok=True)
    # Close the connection even if a year fails.
    db = wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
    try:
        for year in range(start_year, end_year + 1):
            path = _get_taq_path(year)
            # Skip the years downloaded by a previous run.
            if os.path.exists(path):
                _LOG.info("Skipping '%s'", path)
                continue
            df = download_spy_taq_year(db, year)
            df.to_parquet(path)
            _LOG.info("Saved '%s' (%s bars)", path, len(df))
    finally:
        db.close()


def load_spy_taq(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Load the TAQ SPY 1-minute bars from `DATA_DIR/taq`.

    :param start_year: first year
    :param end_year: last year, included
    :return: bars indexed by UTC timestamp
    """
    _LOG.debug(hprint.to_str("start_year end_year"))
    # Read one parquet file per year.
    dfs = [
        pd.read_parquet(_get_taq_path(year))
        for year in range(start_year, end_year + 1)
    ]
    df = pd.concat(dfs)
    _LOG.debug("return=%s rows", len(df))
    return df
