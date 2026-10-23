"""
Unit tests for `nevergrad_utils.py`.

The tests never touch the network: `yf.download()` and
`urllib.request.urlretrieve()` are mocked, and every file is written to the
scratch space of the test.

Import as:

import test.test_nevergrad_utils as ttenevut
"""

import contextlib
import datetime
import hashlib
import logging
import os
import pprint
import shutil
import unittest.mock as umock
import zipfile
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

import helpers.hio as hio
import helpers.hprint as hprint
import helpers.hunit_test as hunitest
import nevergrad_utils as nvutils

_LOG = logging.getLogger(__name__)


# #############################################################################
# Fixtures
# #############################################################################


def _get_prices() -> pd.DataFrame:
    """
    Build daily prices for `nvutils.TICKERS` on four trading days.

    Every ticker moves by +10%, +10%, -10%, so the returns are easy to check.
    DBC has no price on the first day, like before its launch in February 2006.
    """
    dates = pd.DatetimeIndex(
        ["2006-02-03", "2006-02-06", "2006-02-07", "2006-02-08"], name="Date"
    )
    growth = np.array([1.0, 1.1, 1.21, 1.089])
    data = {
        ticker: 10.0 * (i + 1) * growth
        for i, ticker in enumerate(nvutils.TICKERS)
    }
    prices = pd.DataFrame(data, index=dates)
    prices.loc[dates[0], "DBC"] = np.nan
    return prices


def _get_yahoo_download(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build a frame shaped like the output of `yf.download()` for `prices`.

    The columns are a (`Price`, `Ticker`) MultiIndex with an extra `Open` level,
    and the tickers are sorted alphabetically rather than in `nvutils.TICKERS`
    order.
    """
    close = prices[sorted(prices.columns)]
    open_ = close * 0.99
    df = pd.concat({"Close": close, "Open": open_}, axis=1)
    df.columns.names = ["Price", "Ticker"]
    return df


def _get_french_daily_csv() -> str:
    """
    Return a small CSV shaped like the Ken French 10-industry daily file.

    It has two industries, dates on both sides of the project window, and a
    second (equal-weighted) table that the parser must ignore.
    """
    csv_text = """
    This file was created using the 202608 CRSP database.
    Missing data are indicated by -99.99 or -999.


      Average Value Weighted Returns -- Daily
    ,NoDur,Durbl
    20051230,   -0.50,    1.00
    20060103,    1.20,   -0.40
    20060104,   -0.30,    0.60
    20251231,    0.10,    0.20
    20260102,    0.30,    0.10

      Average Equal Weighted Returns -- Daily
    ,NoDur,Durbl
    20060103,    9.99,    9.99
    """
    return csv_text


def _get_french_monthly_csv() -> str:
    """
    Return a small CSV shaped like the monthly Fama-French factors file.

    It has months on both sides of the project window and a second (annual)
    table that the parser must ignore.
    """
    csv_text = """
    This file was created using the 202608 CRSP database.
    The 1-month TBill rate data until 202405 are from Ibbotson Associates.

    ,Mkt-RF,SMB,HML,RF
    200512,   0.50,  -0.30,   0.20,   0.32
    200601,   3.03,   5.46,   0.99,   0.35
    200602,  -0.30,  -0.40,   0.10,   0.34
    202512,  -0.36,  -1.04,   2.40,   0.34
    202601,   1.00,   0.50,  -0.20,   0.31

     Annual Factors: January-December
    ,Mkt-RF,SMB,HML,RF
      2006,  10.57,   0.11,  11.72,   4.80
    """
    return csv_text


def _write_french_zip(zip_path: str, csv_text: str) -> None:
    """
    Write `csv_text` as the single CSV file of a zip, like a Ken French download.

    :param zip_path: path of the zip to write
    :param csv_text: content of the CSV file, indented like the code
    """
    csv_text = hprint.dedent(csv_text)
    # Use Windows line endings, like the files from the Ken French library.
    csv_text = csv_text.replace("\n", "\r\n")
    hio.create_dir(os.path.dirname(zip_path), incremental=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data.csv", csv_text)


def _mock_urlretrieve(src_path: str) -> Any:
    """
    Mock `urllib.request.urlretrieve()` to "download" the local file `src_path`.

    :param src_path: file copied to the destination of every download
    :return: patcher to use as a context manager
    """

    def _copy(url: str, dst_path: str) -> None:
        shutil.copyfile(src_path, dst_path)

    patcher = umock.patch.object(
        nvutils.urllib.request, "urlretrieve", side_effect=_copy
    )
    return patcher


def _mock_yf_download(prices: pd.DataFrame) -> Any:
    """
    Mock `yf.download()` to return `prices` shaped like a Yahoo download.

    :param prices: prices that the mocked download returns
    :return: patcher to use as a context manager
    """
    yahoo_df = _get_yahoo_download(prices)
    patcher = umock.patch.object(nvutils.yf, "download", return_value=yahoo_df)
    return patcher


def _compute_sha256(path: str) -> str:
    """
    Compute the SHA-256 checksum of a file, independently of `nvutils`.
    """
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return digest


def _get_daily_returns(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Build constant returns for one asset on every business day in a range.

    :param start_date: first date, included
    :param end_date: last date, included
    :return: returns indexed by date
    """
    dates = pd.bdate_range(start_date, end_date, name="Date")
    returns = pd.DataFrame({"SPY": 0.001}, index=dates)
    return returns


def _summarize_fold(
    train: Union[pd.DataFrame, pd.Series], test: Union[pd.DataFrame, pd.Series]
) -> str:
    """
    Describe the first date, last date, and size of both slices of a fold.

    E.g., `train 2006-01-02..2010-12-31 (1305), test 2011-01-03..2011-12-30 (260)`
    """
    txt = (
        f"train {train.index.min().date()}..{train.index.max().date()}"
        f" ({len(train)}), "
        f"test {test.index.min().date()}..{test.index.max().date()}"
        f" ({len(test)})"
    )
    return txt


def _get_two_asset_returns() -> pd.DataFrame:
    """
    Build the returns of two assets on three trading days across a month end.

    On both January days asset A gains 1% and asset B loses 0.5%, so a 60/40
    portfolio drifts toward A; nothing moves on 2011-02-01, so that day's return
    is only the cost of any trade.
    """
    dates = pd.DatetimeIndex(
        ["2011-01-03", "2011-01-04", "2011-02-01"], name="Date"
    )
    returns = pd.DataFrame(
        {"A": [0.01, 0.01, 0.0], "B": [-0.005, -0.005, 0.0]}, index=dates
    )
    return returns


def _summarize_simulation(
    net: pd.Series, turnover: pd.Series, end_weights: np.ndarray
) -> str:
    """
    Describe the daily net returns, daily turnover, and end weights of a run.

    Values are printed with 7 decimals, enough to check returns to 0.00001%.
    """
    df = pd.DataFrame({"net": net, "turnover": turnover})
    txt = df.to_string(float_format="{:.7f}".format)
    txt += "\nend_weights " + " ".join(f"{w:.7f}" for w in end_weights)
    return txt


def _format_weights(weights: np.ndarray) -> str:
    """
    Print weights with 6 decimals, showing a solver's -1e-17 as 0.

    E.g., `0.800000 0.000000 0.200000`
    """
    # Adding 0.0 turns the -0.0 of a rounded tiny negative into 0.0.
    rounded = np.round(weights, 6) + 0.0
    txt = " ".join(f"{w:.6f}" for w in rounded)
    return txt


# #############################################################################
# Test_download_data
# #############################################################################


class Test_download_data(hunitest.TestCase):
    """
    Test `nevergrad_utils.download_data()`.
    """

    def test1(self) -> None:
        """
        Test that the `Close` prices are selected in `nvutils.TICKERS` order.
        """
        # Prepare inputs.
        prices = _get_prices()
        # Prepare outputs.
        expected_prices = """
        Ticker       SPY   EFA   EEM   TLT   IEF   LQD   GLD   VNQ   DBC
        Date
        2006-02-03 10.00 20.00 30.00 40.00 50.00 60.00 70.00 80.00   NaN
        2006-02-06 11.00 22.00 33.00 44.00 55.00 66.00 77.00 88.00 99.00
        2006-02-07 12.10 24.20 36.30 48.40 60.50 72.60 84.70 96.80 108.90
        2006-02-08 10.89 21.78 32.67 43.56 54.45 65.34 76.23 87.12 98.01
        """
        expected_call = (
            "call(tickers=['SPY', 'EFA', 'EEM', 'TLT', 'IEF', 'LQD', 'GLD',"
            " 'VNQ', 'DBC'], start='2006-01-01', end='2026-01-01',"
            " auto_adjust=True, progress=False)"
        )
        # Run test.
        with _mock_yf_download(prices) as mock_download:
            actual = nvutils.download_data()
        # Check outputs.
        self.assert_equal(
            actual.to_string(), expected_prices, dedent=True, fuzzy_match=True
        )
        self.assert_equal(str(mock_download.call_args), expected_call)


# #############################################################################
# Test_download_french_data
# #############################################################################


class Test_download_french_data(hunitest.TestCase):
    """
    Test `nevergrad_utils.download_french_data()`.
    """

    def test1(self) -> None:
        """
        Test that the zip is downloaded from `url` and its provenance recorded.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        src_path = os.path.join(scratch_dir, "remote", "factors.zip")
        _write_french_zip(src_path, _get_french_monthly_csv())
        url = nvutils.RF_URL
        french_path = os.path.join(scratch_dir, "data", "factors.zip")
        provenance_path = os.path.join(scratch_dir, "provenance.json")
        # Prepare outputs.
        expected_call = f"call('{url}', '{french_path}')"
        expected_provenance = {
            "source": "Ken French",
            "url": url,
            "fetch_date": datetime.date.today().isoformat(),
            "file": french_path,
            "sha256": _compute_sha256(src_path),
        }
        # Run test.
        with _mock_urlretrieve(src_path) as mock_urlretrieve:
            nvutils.download_french_data(
                url=url, french_path=french_path, provenance_path=provenance_path
            )
        # Check outputs.
        actual_provenance = hio.from_json(provenance_path)
        self.assert_equal(str(mock_urlretrieve.call_args), expected_call)
        self.assert_equal(
            pprint.pformat(actual_provenance), pprint.pformat(expected_provenance)
        )


# #############################################################################
# Test_align_and_clean_data
# #############################################################################


class Test_align_and_clean_data(hunitest.TestCase):
    """
    Test `nevergrad_utils.align_and_clean_data()`.
    """

    def test1(self) -> None:
        """
        Test that rows where any ticker has no price are dropped.
        """
        # Prepare inputs.
        dates = pd.DatetimeIndex(
            ["2006-02-03", "2006-02-06", "2006-02-07"], name="Date"
        )
        prices = pd.DataFrame(
            {"SPY": [100.0, 101.0, np.nan], "DBC": [np.nan, 25.0, 25.5]},
            index=dates,
        )
        # Prepare outputs.
        expected = """
                      SPY   DBC
        Date
        2006-02-06  101.0  25.0
        """
        # Run test.
        actual = nvutils.align_and_clean_data(prices)
        # Check outputs.
        self.assert_equal(
            actual.to_string(), expected, dedent=True, fuzzy_match=True
        )

    def test2(self) -> None:
        """
        Test that an error is raised when no date has a price for every ticker.
        """
        # Prepare inputs.
        dates = pd.DatetimeIndex(["2006-02-03", "2006-02-06"], name="Date")
        prices = pd.DataFrame(
            {"SPY": [100.0, np.nan], "DBC": [np.nan, 25.0]}, index=dates
        )
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            nvutils.align_and_clean_data(prices)


# #############################################################################
# Test_compute_returns
# #############################################################################


class Test_compute_returns(hunitest.TestCase):
    """
    Test `nevergrad_utils.compute_returns()`.
    """

    def test1(self) -> None:
        """
        Test that simple returns are computed and the first date is dropped.
        """
        # Prepare inputs.
        dates = pd.DatetimeIndex(
            ["2006-02-06", "2006-02-07", "2006-02-08"], name="Date"
        )
        prices = pd.DataFrame(
            {"SPY": [100.0, 102.0, 96.9], "DBC": [25.0, 25.0, 26.0]},
            index=dates,
        )
        # Prepare outputs.
        expected = """
                     SPY   DBC
        Date
        2006-02-07  0.02  0.00
        2006-02-08 -0.05  0.04
        """
        # Run test.
        actual = nvutils.compute_returns(prices)
        # Check outputs.
        self.assert_equal(
            actual.round(6).to_string(), expected, dedent=True, fuzzy_match=True
        )


# #############################################################################
# Test_parse_french_returns
# #############################################################################


class Test_parse_french_returns(hunitest.TestCase):
    """
    Test `nevergrad_utils.parse_french_returns()`.
    """

    def helper(self, csv_text: str, date_format: str, expected: str) -> None:
        """
        Write `csv_text` into a zip, parse it, and compare with `expected`.

        :param csv_text: content of the CSV file in the zip
        :param date_format: format of the dates in the first column
        :param expected: expected parsed returns
        """
        # Prepare inputs.
        zip_path = os.path.join(self.get_scratch_space(), "french.zip")
        _write_french_zip(zip_path, csv_text)
        # Run test.
        actual = nvutils.parse_french_returns(zip_path, date_format=date_format)
        # Check outputs.
        self.assert_equal(
            actual.to_string(), expected, dedent=True, fuzzy_match=True
        )

    def test1(self) -> None:
        """
        Test that only the first table of a daily file is parsed, as decimals.
        """
        # Prepare inputs.
        csv_text = _get_french_daily_csv()
        date_format = "%Y%m%d"
        # Prepare outputs.
        expected = """
                    NoDur  Durbl
        Date
        2005-12-30 -0.005  0.010
        2006-01-03  0.012 -0.004
        2006-01-04 -0.003  0.006
        2025-12-31  0.001  0.002
        2026-01-02  0.003  0.001
        """
        # Run test.
        self.helper(csv_text, date_format, expected)

    def test2(self) -> None:
        """
        Test that the missing-value codes -99.99 and -999 become NaN.
        """
        # Prepare inputs.
        csv_text = """
        Missing data are indicated by -99.99 or -999.

        ,NoDur,Durbl
        20060103,    1.20,  -99.99
        20060104, -999.00,    0.60

        Copyright 2026 Eugene F. Fama and Kenneth R. French
        """
        date_format = "%Y%m%d"
        # Prepare outputs.
        expected = """
                    NoDur  Durbl
        Date
        2006-01-03  0.012    NaN
        2006-01-04    NaN  0.006
        """
        # Run test.
        self.helper(csv_text, date_format, expected)

    def test3(self) -> None:
        """
        Test that a first table running to the end of the file is parsed.
        """
        # Prepare inputs.
        csv_text = """
        ,NoDur,Durbl
        20060103,    1.20,   -0.40
        """
        date_format = "%Y%m%d"
        # Prepare outputs.
        expected = """
                    NoDur  Durbl
        Date
        2006-01-03  0.012 -0.004
        """
        # Run test.
        self.helper(csv_text, date_format, expected)

    def test4(self) -> None:
        """
        Test that a monthly file is parsed with `YYYYMM` dates.
        """
        # Prepare inputs.
        csv_text = _get_french_monthly_csv()
        date_format = "%Y%m"
        # Prepare outputs.
        expected = """
                    Mkt-RF     SMB     HML      RF
        Date
        2005-12-01  0.0050 -0.0030  0.0020  0.0032
        2006-01-01  0.0303  0.0546  0.0099  0.0035
        2006-02-01 -0.0030 -0.0040  0.0010  0.0034
        2025-12-01 -0.0036 -0.0104  0.0240  0.0034
        2026-01-01  0.0100  0.0050 -0.0020  0.0031
        """
        # Run test.
        self.helper(csv_text, date_format, expected)


# #############################################################################
# Test_check_returns
# #############################################################################


class Test_check_returns(hunitest.TestCase):
    """
    Test `nevergrad_utils.check_returns()`.
    """

    def helper(self, values: list) -> pd.DataFrame:
        """
        Build daily returns for one asset with the given values.

        :param values: one return per business day, starting on 2006-02-07
        :return: returns indexed by date
        """
        dates = pd.bdate_range("2006-02-07", periods=len(values), name="Date")
        returns = pd.DataFrame({"SPY": values}, index=dates)
        return returns

    def test1(self) -> None:
        """
        Test that complete, ordered returns within the limit pass.
        """
        # Prepare inputs.
        returns = self.helper([0.01, -0.02, 0.25])
        # Run test and check outputs.
        nvutils.check_returns(returns)

    def test2(self) -> None:
        """
        Test that a return beyond `max_abs_return` raises an error.
        """
        # Prepare inputs.
        returns = self.helper([0.01, -0.30])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            nvutils.check_returns(returns)

    def test3(self) -> None:
        """
        Test that a custom `max_abs_return` is applied.
        """
        # Prepare inputs.
        returns = self.helper([0.004, 0.012])
        max_abs_return = 0.01
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            nvutils.check_returns(returns, max_abs_return=max_abs_return)


# #############################################################################
# Test_save_data
# #############################################################################


class Test_save_data(hunitest.TestCase):
    """
    Test `nevergrad_utils.save_data()`.
    """

    def test1(self) -> None:
        """
        Test that the prices are cached and the provenance matches the cache.
        """
        # Prepare inputs.
        prices = _get_prices().iloc[1:]
        scratch_dir = self.get_scratch_space()
        prices_path = os.path.join(scratch_dir, "data", "etf_prices.csv")
        provenance_path = os.path.join(scratch_dir, "provenance.json")
        # Run test.
        nvutils.save_data(
            prices, prices_path=prices_path, provenance_path=provenance_path
        )
        # Check outputs.
        actual_provenance = hio.from_json(provenance_path)
        expected_provenance = {
            "source": "yfinance",
            "fetch_date": datetime.date.today().isoformat(),
            "tickers": nvutils.TICKERS,
            "start_date": "2006-01-01",
            "end_date": "2026-01-01",
            "first_date": "2006-02-06",
            "last_date": "2006-02-08",
            "num_rows": 3,
            "file": prices_path,
            "sha256": _compute_sha256(prices_path),
        }
        self.assert_equal(
            pprint.pformat(actual_provenance), pprint.pformat(expected_provenance)
        )


# #############################################################################
# Test_load_data
# #############################################################################


class Test_load_data(hunitest.TestCase):
    """
    Test `nevergrad_utils.load_data()`.
    """

    def helper(self) -> Any:
        """
        Return the cache and provenance paths in the scratch space.

        :return: `(prices_path, provenance_path)`
        """
        scratch_dir = self.get_scratch_space()
        prices_path = os.path.join(scratch_dir, "data", "etf_prices.csv")
        provenance_path = os.path.join(scratch_dir, "provenance.json")
        return prices_path, provenance_path

    def test1(self) -> None:
        """
        Test that an existing cache is loaded without downloading.
        """
        # Prepare inputs.
        prices = _get_prices().iloc[1:]
        prices_path, provenance_path = self.helper()
        nvutils.save_data(
            prices, prices_path=prices_path, provenance_path=provenance_path
        )
        # Prepare outputs.
        expected = prices.to_string()
        # Run test.
        with _mock_yf_download(prices) as mock_download:
            actual = nvutils.load_data(
                prices_path=prices_path, provenance_path=provenance_path
            )
        # Check outputs.
        self.assert_equal(actual.to_string(), expected)
        self.assertEqual(mock_download.call_count, 0)

    def test2(self) -> None:
        """
        Test that a missing cache is downloaded, cleaned, cached, and loaded.
        """
        # Prepare inputs.
        prices = _get_prices()
        prices_path, provenance_path = self.helper()
        # Prepare outputs.
        # The first date is dropped since DBC has no price on it.
        expected = prices.iloc[1:].to_string()
        # Run test.
        with _mock_yf_download(prices) as mock_download:
            actual = nvutils.load_data(
                prices_path=prices_path, provenance_path=provenance_path
            )
        # Check outputs.
        self.assert_equal(actual.to_string(), expected)
        self.assertEqual(mock_download.call_count, 1)
        self.assertTrue(os.path.exists(prices_path))

    def test3(self) -> None:
        """
        Test that a cache that no longer matches its provenance is rejected.
        """
        # Prepare inputs.
        prices = _get_prices().iloc[1:]
        prices_path, provenance_path = self.helper()
        nvutils.save_data(
            prices, prices_path=prices_path, provenance_path=provenance_path
        )
        # Edit the cache after its checksum was recorded.
        txt = hio.from_file(prices_path)
        hio.to_file(prices_path, txt.replace("2006-02-08", "2006-02-09"))
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            nvutils.load_data(
                prices_path=prices_path, provenance_path=provenance_path
            )


# #############################################################################
# Test_load_french_returns
# #############################################################################


class Test_load_french_returns(hunitest.TestCase):
    """
    Test `nevergrad_utils.load_french_returns()`.
    """

    def helper(self) -> Any:
        """
        Write the remote zip and return the paths used by the loader.

        :return: `(src_path, french_path, provenance_path)`
        """
        scratch_dir = self.get_scratch_space()
        src_path = os.path.join(scratch_dir, "remote", "industries.zip")
        _write_french_zip(src_path, _get_french_daily_csv())
        french_path = os.path.join(scratch_dir, "data", "industries.zip")
        provenance_path = os.path.join(scratch_dir, "provenance.json")
        return src_path, french_path, provenance_path

    def test1(self) -> None:
        """
        Test that a missing cache is downloaded and cut to the project window.
        """
        # Prepare inputs.
        src_path, french_path, provenance_path = self.helper()
        # Prepare outputs.
        # 2005-12-30 and 2026-01-02 are outside the project window.
        expected = """
                    NoDur  Durbl
        Date
        2006-01-03  0.012 -0.004
        2006-01-04 -0.003  0.006
        2025-12-31  0.001  0.002
        """
        expected_call = f"call('{nvutils.FRENCH_URL}', '{french_path}')"
        # Run test.
        with _mock_urlretrieve(src_path) as mock_urlretrieve:
            actual = nvutils.load_french_returns(
                french_path=french_path, provenance_path=provenance_path
            )
        # Check outputs.
        self.assert_equal(
            actual.to_string(), expected, dedent=True, fuzzy_match=True
        )
        self.assert_equal(str(mock_urlretrieve.call_args), expected_call)

    def test2(self) -> None:
        """
        Test that an existing cache is loaded without downloading.
        """
        # Prepare inputs.
        src_path, french_path, provenance_path = self.helper()
        with _mock_urlretrieve(src_path):
            nvutils.download_french_data(
                french_path=french_path, provenance_path=provenance_path
            )
        # Prepare outputs.
        expected = """
                    NoDur  Durbl
        Date
        2006-01-03  0.012 -0.004
        2006-01-04 -0.003  0.006
        2025-12-31  0.001  0.002
        """
        # Run test.
        with _mock_urlretrieve(src_path) as mock_urlretrieve:
            actual = nvutils.load_french_returns(
                french_path=french_path, provenance_path=provenance_path
            )
        # Check outputs.
        self.assert_equal(
            actual.to_string(), expected, dedent=True, fuzzy_match=True
        )
        self.assertEqual(mock_urlretrieve.call_count, 0)


# #############################################################################
# Test_load_returns
# #############################################################################


class Test_load_returns(hunitest.TestCase):
    """
    Test `nevergrad_utils.load_returns()`.

    `load_returns()` reads the default cache paths, which are relative, so each
    test runs from the scratch space.
    """

    def test1(self) -> None:
        """
        Test the ETF source end to end: download, clean, cache, and returns.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        prices = _get_prices()
        data_source = "etf"
        # Prepare outputs.
        expected = """
                    SPY  EFA  EEM  TLT  IEF  LQD  GLD  VNQ  DBC
        Date
        2006-02-07  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1
        2006-02-08 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1
        """
        # Run test.
        with contextlib.chdir(scratch_dir), _mock_yf_download(prices):
            actual = nvutils.load_returns(data_source)
        # Check outputs.
        self.assert_equal(
            actual.round(6).to_string(), expected, dedent=True, fuzzy_match=True
        )

    def test2(self) -> None:
        """
        Test the Ken French source end to end: download, cache, and returns.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        src_path = os.path.join(scratch_dir, "remote", "industries.zip")
        _write_french_zip(src_path, _get_french_daily_csv())
        data_source = "french"
        # Prepare outputs.
        expected = """
                    NoDur  Durbl
        Date
        2006-01-03  0.012 -0.004
        2006-01-04 -0.003  0.006
        2025-12-31  0.001  0.002
        """
        # Run test.
        with contextlib.chdir(scratch_dir), _mock_urlretrieve(src_path):
            actual = nvutils.load_returns(data_source)
        # Check outputs.
        self.assert_equal(
            actual.to_string(), expected, dedent=True, fuzzy_match=True
        )


# #############################################################################
# Test_load_risk_free_rate
# #############################################################################


class Test_load_risk_free_rate(hunitest.TestCase):
    """
    Test `nevergrad_utils.load_risk_free_rate()`.
    """

    def test1(self) -> None:
        """
        Test that the monthly `RF` column is downloaded, cut, and converted.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        src_path = os.path.join(scratch_dir, "remote", "factors.zip")
        _write_french_zip(src_path, _get_french_monthly_csv())
        rf_path = os.path.join(scratch_dir, "data", "factors.zip")
        provenance_path = os.path.join(scratch_dir, "provenance.json")
        # Prepare outputs.
        # 2005-12 and 2026-01 are outside the project window.
        expected = """
        Date
        2006-01-01    0.0035
        2006-02-01    0.0034
        2025-12-01    0.0034
        Name: RF, dtype: float64
        """
        expected_call = f"call('{nvutils.RF_URL}', '{rf_path}')"
        # Run test.
        with _mock_urlretrieve(src_path) as mock_urlretrieve:
            actual = nvutils.load_risk_free_rate(
                rf_path=rf_path, provenance_path=provenance_path
            )
        # Check outputs.
        self.assert_equal(str(actual), expected, dedent=True, fuzzy_match=True)
        self.assert_equal(str(mock_urlretrieve.call_args), expected_call)


# #############################################################################
# Test_get_test_years
# #############################################################################


class Test_get_test_years(hunitest.TestCase):
    """
    Test `nevergrad_utils.get_test_years()`.
    """

    def test1(self) -> None:
        """
        Test that the 2006 to 2025 window gives the 15 test years 2011 to 2025.
        """
        # Prepare inputs.
        index = pd.bdate_range("2006-02-07", "2025-12-31")
        # Prepare outputs.
        expected = str(list(range(2011, 2026)))
        # Run test.
        actual = nvutils.get_test_years(index)
        # Check outputs.
        self.assert_equal(str(actual), expected)

    def test2(self) -> None:
        """
        Test a custom `num_train_years` with a single test year.
        """
        # Prepare inputs.
        index = pd.bdate_range("2000-01-03", "2002-12-31")
        num_train_years = 2
        # Prepare outputs.
        expected = "[2002]"
        # Run test.
        actual = nvutils.get_test_years(index, num_train_years=num_train_years)
        # Check outputs.
        self.assert_equal(str(actual), expected)

    def test3(self) -> None:
        """
        Test that data spanning only `num_train_years` years raises an error.
        """
        # Prepare inputs.
        index = pd.bdate_range("2006-01-02", "2010-12-31")
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            nvutils.get_test_years(index)

    def test4(self) -> None:
        """
        Test that a calendar year with no data raises an error.
        """
        # Prepare inputs.
        index = pd.bdate_range("2000-01-03", "2007-12-31")
        index = index[index.year != 2003]
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            nvutils.get_test_years(index)


# #############################################################################
# Test_split_fold
# #############################################################################


class Test_split_fold(hunitest.TestCase):
    """
    Test `nevergrad_utils.split_fold()`.
    """

    def helper(
        self,
        df: Union[pd.DataFrame, pd.Series],
        test_year: int,
        num_train_years: int,
        expected: str,
    ) -> None:
        """
        Split `df` and compare the fold summary with `expected`.

        :param df: data indexed by date
        :param test_year: calendar year of the test slice
        :param num_train_years: number of calendar years in the training slice
        :param expected: expected output of `_summarize_fold()`
        """
        # Run test.
        train, test = nvutils.split_fold(
            df, test_year, num_train_years=num_train_years
        )
        # Check outputs.
        actual = _summarize_fold(train, test)
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test that daily data outside the fold's years is left out of both slices.
        """
        # Prepare inputs.
        df = _get_daily_returns("2000-01-03", "2004-12-31")
        test_year = 2003
        num_train_years = 2
        # Prepare outputs.
        expected = (
            "train 2001-01-01..2002-12-31 (522), "
            "test 2003-01-01..2003-12-31 (261)"
        )
        # Run test.
        self.helper(df, test_year, num_train_years, expected)

    def test2(self) -> None:
        """
        Test that a monthly series is cut at the same year boundaries.
        """
        # Prepare inputs.
        dates = pd.date_range("2006-01-01", "2025-12-01", freq="MS", name="Date")
        df = pd.Series(0.001, index=dates, name="RF")
        test_year = 2011
        num_train_years = 5
        # Prepare outputs.
        expected = (
            "train 2006-01-01..2010-12-01 (60), "
            "test 2011-01-01..2011-12-01 (12)"
        )
        # Run test.
        self.helper(df, test_year, num_train_years, expected)


# #############################################################################
# Test_generate_folds
# #############################################################################


class Test_generate_folds(hunitest.TestCase):
    """
    Test `nevergrad_utils.generate_folds()`.
    """

    def helper(
        self, returns: pd.DataFrame, num_train_years: int, expected: str
    ) -> None:
        """
        Summarize every fold of `returns` and compare with `expected`.

        :param returns: daily returns indexed by date
        :param num_train_years: number of calendar years in each training window
        :param expected: expected summaries, one fold per line
        """
        # Run test.
        folds = nvutils.generate_folds(returns, num_train_years=num_train_years)
        # Check outputs.
        actual = "\n".join(_summarize_fold(train, test) for train, test in folds)
        self.assert_equal(actual, expected, dedent=True)

    def test1(self) -> None:
        """
        Test that the folds are contiguous, non-overlapping, and rolled yearly.
        """
        # Prepare inputs.
        returns = _get_daily_returns("2000-01-03", "2004-12-31")
        num_train_years = 2
        # Prepare outputs.
        expected = """
        train 2000-01-03..2001-12-31 (521), test 2002-01-01..2002-12-31 (261)
        train 2001-01-01..2002-12-31 (522), test 2003-01-01..2003-12-31 (261)
        train 2002-01-01..2003-12-31 (522), test 2004-01-01..2004-12-31 (262)
        """
        # Run test.
        self.helper(returns, num_train_years, expected)

    def test2(self) -> None:
        """
        Test that the project window from 2006-02-07 gives 15 five-year folds.
        """
        # Prepare inputs.
        returns = _get_daily_returns("2006-02-07", "2025-12-31")
        num_train_years = nvutils.NUM_TRAIN_YEARS
        # Prepare outputs.
        # The first training window starts on 2006-02-07, like the ETF returns,
        # so it is about 4.9 years long.
        expected = """
        train 2006-02-07..2010-12-31 (1279), test 2011-01-03..2011-12-30 (260)
        train 2007-01-01..2011-12-30 (1305), test 2012-01-02..2012-12-31 (261)
        train 2008-01-01..2012-12-31 (1305), test 2013-01-01..2013-12-31 (261)
        train 2009-01-01..2013-12-31 (1304), test 2014-01-01..2014-12-31 (261)
        train 2010-01-01..2014-12-31 (1304), test 2015-01-01..2015-12-31 (261)
        train 2011-01-03..2015-12-31 (1304), test 2016-01-01..2016-12-30 (261)
        train 2012-01-02..2016-12-30 (1305), test 2017-01-02..2017-12-29 (260)
        train 2013-01-01..2017-12-29 (1304), test 2018-01-01..2018-12-31 (261)
        train 2014-01-01..2018-12-31 (1304), test 2019-01-01..2019-12-31 (261)
        train 2015-01-01..2019-12-31 (1304), test 2020-01-01..2020-12-31 (262)
        train 2016-01-01..2020-12-31 (1305), test 2021-01-01..2021-12-31 (261)
        train 2017-01-02..2021-12-31 (1305), test 2022-01-03..2022-12-30 (260)
        train 2018-01-01..2022-12-30 (1305), test 2023-01-02..2023-12-29 (260)
        train 2019-01-01..2023-12-29 (1304), test 2024-01-01..2024-12-31 (262)
        train 2020-01-01..2024-12-31 (1305), test 2025-01-01..2025-12-31 (261)
        """
        # Run test.
        self.helper(returns, num_train_years, expected)


# #############################################################################
# Test_get_fold_table
# #############################################################################


class Test_get_fold_table(hunitest.TestCase):
    """
    Test `nevergrad_utils.get_fold_table()`.
    """

    def test1(self) -> None:
        """
        Test that the table lists the boundaries and sizes of every fold.
        """
        # Prepare inputs.
        returns = _get_daily_returns("2000-01-03", "2004-12-31")
        num_train_years = 2
        # Prepare outputs.
        expected = """
                  train_start   train_end  train_days  test_start    test_end  test_days
        test_year
        2002       2000-01-03  2001-12-31         521  2002-01-01  2002-12-31        261
        2003       2001-01-01  2002-12-31         522  2003-01-01  2003-12-31        261
        2004       2002-01-01  2003-12-31         522  2004-01-01  2004-12-31        262
        """
        # Run test.
        actual = nvutils.get_fold_table(returns, num_train_years=num_train_years)
        # Check outputs.
        self.assert_equal(
            actual.to_string(), expected, dedent=True, fuzzy_match=True
        )


# #############################################################################
# Test_check_weights
# #############################################################################


class Test_check_weights(hunitest.TestCase):
    """
    Test `nevergrad_utils.check_weights()`.
    """

    def test1(self) -> None:
        """
        Test that weights off by floating-point error pass.
        """
        # Prepare inputs.
        # Ten weights of 0.1 sum to 0.9999999999999999 in floating point.
        weights = np.full(10, 0.1)
        # Run test and check outputs.
        nvutils.check_weights(weights, 10)

    def test2(self) -> None:
        """
        Test that a tiny negative weight, like a solver's rounding, passes.
        """
        # Prepare inputs.
        weights = np.array([0.5, 0.5 + 1e-12, -1e-12])
        # Run test and check outputs.
        nvutils.check_weights(weights, 3)

    def test3(self) -> None:
        """
        Test that a number of weights different from the assets raises an error.
        """
        # Prepare inputs.
        weights = np.array([0.5, 0.5])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            nvutils.check_weights(weights, 3)

    def test4(self) -> None:
        """
        Test that a short position raises an error, even if the sum is 1.
        """
        # Prepare inputs.
        weights = np.array([1.2, -0.2])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            nvutils.check_weights(weights, 2)

    def test5(self) -> None:
        """
        Test that weights summing to 0.9 raise an error.
        """
        # Prepare inputs.
        weights = np.array([0.5, 0.4])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            nvutils.check_weights(weights, 2)


# #############################################################################
# Test_compute_turnover
# #############################################################################


class Test_compute_turnover(hunitest.TestCase):
    """
    Test `nevergrad_utils.compute_turnover()`.
    """

    def test1(self) -> None:
        """
        Test that moving 10% from one asset to another counts both sides: 0.2.
        """
        # Prepare inputs.
        target_weights = np.array([0.5, 0.3, 0.2])
        current_weights = np.array([0.4, 0.4, 0.2])
        # Prepare outputs.
        expected = "0.2000000"
        # Run test.
        actual = nvutils.compute_turnover(target_weights, current_weights)
        # Check outputs.
        self.assert_equal(f"{actual:.7f}", expected)

    def test2(self) -> None:
        """
        Test that switching every dollar to another asset gives the maximum, 2.
        """
        # Prepare inputs.
        target_weights = np.array([0.0, 1.0])
        current_weights = np.array([1.0, 0.0])
        # Prepare outputs.
        expected = "2.0000000"
        # Run test.
        actual = nvutils.compute_turnover(target_weights, current_weights)
        # Check outputs.
        self.assert_equal(f"{actual:.7f}", expected)

    def test3(self) -> None:
        """
        Test that weight vectors of different lengths raise an error.
        """
        # Prepare inputs.
        target_weights = np.array([0.5, 0.5])
        current_weights = np.array([0.4, 0.3, 0.3])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            nvutils.compute_turnover(target_weights, current_weights)


# #############################################################################
# Test_get_rebalance_days
# #############################################################################


class Test_get_rebalance_days(hunitest.TestCase):
    """
    Test `nevergrad_utils.get_rebalance_days()`.
    """

    def helper(self, rebal_freq: str, expected: str) -> None:
        """
        Flag the rebalance days of five dates and compare with `expected`.

        The dates start a month, stay in it, start another month, stay in it, and
        then start a new quarter.

        :param rebal_freq: rebalancing frequency
        :param expected: expected flags, one per date
        """
        # Prepare inputs.
        index = pd.DatetimeIndex(
            ["2011-01-03", "2011-01-04", "2011-02-01", "2011-02-02", "2011-04-01"]
        )
        # Run test.
        actual = nvutils.get_rebalance_days(rebal_freq, index)
        # Check outputs.
        self.assert_equal(str(actual.tolist()), expected)

    def test1(self) -> None:
        """
        Test that monthly rebalancing trades on the first trading day of a month.
        """
        # The first rebalance is on 2011-01-03, the first trading day of 2011,
        # not on January 1.
        expected = "[True, False, True, False, True]"
        self.helper("monthly", expected)

    def test2(self) -> None:
        """
        Test that quarterly rebalancing skips the start of February.
        """
        expected = "[True, False, False, False, True]"
        self.helper("quarterly", expected)

    def test3(self) -> None:
        """
        Test that annual rebalancing trades only on the first day.
        """
        expected = "[True, False, False, False, False]"
        self.helper("annual", expected)

    def test4(self) -> None:
        """
        Test that an unknown frequency raises an error.
        """
        # Prepare inputs.
        index = pd.DatetimeIndex(["2011-01-03", "2011-01-04"])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            nvutils.get_rebalance_days("daily", index)


# #############################################################################
# Test_simulate_portfolio
# #############################################################################


class Test_simulate_portfolio(hunitest.TestCase):
    """
    Test `nevergrad_utils.simulate_portfolio()`.

    Every test holds a 60/40 target on `_get_two_asset_returns()`. The expected
    values come from tracking the dollars in each asset by hand, not from the
    weight formulas in the code:
    - Day 1: $60 + $40 grows to $60.60 + $39.80 = $100.40, a 0.4% return
    - Day 2: it grows to $61.206 + $39.601 = $100.807, a 0.40538% return, and
      the weights drift to 0.6071602 / 0.3928398
    - 2011-02-01: trading back to 60/40 has turnover 2 * 0.0071602 = 0.0143204,
      which costs 0.0143204 * 10 bps = 0.0000143
    """

    def helper(
        self,
        rebal_freq: str,
        cost_bps: float,
        initial_weights: Optional[np.ndarray],
        expected: str,
    ) -> None:
        """
        Simulate the 60/40 portfolio and compare the run with `expected`.

        :param rebal_freq: rebalancing frequency
        :param cost_bps: cost in basis points per unit of turnover
        :param initial_weights: weights held before the first day
        :param expected: expected output of `_summarize_simulation()`
        """
        # Prepare inputs.
        returns = _get_two_asset_returns()
        target_weights = np.array([0.6, 0.4])
        # Run test.
        net, turnover, end_weights = nvutils.simulate_portfolio(
            returns,
            target_weights,
            rebal_freq=rebal_freq,
            cost_bps=cost_bps,
            initial_weights=initial_weights,
        )
        # Check outputs.
        actual = _summarize_simulation(net, turnover, end_weights)
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test1(self) -> None:
        """
        Test drift, a monthly rebalance, and its cost, starting at the target.
        """
        # With no initial weights the portfolio starts at its target, so the
        # first day has no trade.
        expected = """
                          net  turnover
        Date
        2011-01-03  0.0040000 0.0000000
        2011-01-04  0.0040538 0.0000000
        2011-02-01 -0.0000143 0.0143204
        end_weights 0.6000000 0.4000000
        """
        self.helper("monthly", 10.0, None, expected)

    def test2(self) -> None:
        """
        Test that the first day trades from the initial weights to the target.
        """
        # Moving from 50/50 to 60/40 has turnover 0.2, which costs 0.0002, so
        # day 1 nets 0.004 - 0.0002 = 0.0038. Later days match `test1()`.
        expected = """
                          net  turnover
        Date
        2011-01-03  0.0038000 0.2000000
        2011-01-04  0.0040538 0.0000000
        2011-02-01 -0.0000143 0.0143204
        end_weights 0.6000000 0.4000000
        """
        self.helper("monthly", 10.0, np.array([0.5, 0.5]), expected)

    def test3(self) -> None:
        """
        Test that annual rebalancing lets the weights drift past a month end.
        """
        # No trade on 2011-02-01, so the end weights are the drifted ones that
        # the next fold would start from.
        expected = """
                         net  turnover
        Date
        2011-01-03 0.0040000 0.0000000
        2011-01-04 0.0040538 0.0000000
        2011-02-01 0.0000000 0.0000000
        end_weights 0.6071602 0.3928398
        """
        self.helper("annual", 10.0, None, expected)

    def test4(self) -> None:
        """
        Test that a zero cost leaves the turnover but removes its charge.
        """
        expected = """
                         net  turnover
        Date
        2011-01-03 0.0040000 0.0000000
        2011-01-04 0.0040538 0.0000000
        2011-02-01 0.0000000 0.0143204
        end_weights 0.6000000 0.4000000
        """
        self.helper("monthly", 0.0, None, expected)

    def test5(self) -> None:
        """
        Test that target weights summing to 0.9 raise an error.
        """
        # Prepare inputs.
        returns = _get_two_asset_returns()
        target_weights = np.array([0.5, 0.4])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            nvutils.simulate_portfolio(returns, target_weights)

    def test6(self) -> None:
        """
        Test that initial weights for a different number of assets raise an error.
        """
        # Prepare inputs.
        returns = _get_two_asset_returns()
        target_weights = np.array([0.6, 0.4])
        initial_weights = np.array([0.4, 0.3, 0.3])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            nvutils.simulate_portfolio(
                returns, target_weights, initial_weights=initial_weights
            )


# #############################################################################
# Test_annualized_return
# #############################################################################


class Test_annualized_return(hunitest.TestCase):
    """
    Test `nevergrad_utils.annualized_return()`.
    """

    def test1(self) -> None:
        """
        Test that the average daily return is scaled by 252 trading days.
        """
        # Prepare inputs.
        returns = pd.Series([0.01, -0.005, 0.0, 0.003])
        # Prepare outputs.
        # The average day returns 0.008 / 4 = 0.002, and 0.002 * 252 = 0.504.
        expected = "0.5040000"
        # Run test.
        actual = nvutils.annualized_return(returns)
        # Check outputs.
        self.assert_equal(f"{actual:.7f}", expected)

    def test2(self) -> None:
        """
        Test that a bumpy series with the same average gets the same return.
        """
        # Prepare inputs.
        # Alternating +2.04% and -1.96% averages +0.04% a day, like a steady
        # +0.04% a day; the arithmetic average ignores the volatility drag.
        returns = pd.Series([0.0204, -0.0196] * 126)
        # Prepare outputs.
        expected = "0.1008000"
        # Run test.
        actual = nvutils.annualized_return(returns)
        # Check outputs.
        self.assert_equal(f"{actual:.7f}", expected)


# #############################################################################
# Test_annualized_volatility
# #############################################################################


class Test_annualized_volatility(hunitest.TestCase):
    """
    Test `nevergrad_utils.annualized_volatility()`.
    """

    def test1(self) -> None:
        """
        Test that the sample standard deviation is scaled by the root of 252.
        """
        # Prepare inputs.
        returns = pd.Series([0.01, -0.01])
        # Prepare outputs.
        # The sample variance is (0.01^2 + 0.01^2) / (2 - 1) = 0.0002 a day, so
        # the annual volatility is sqrt(0.0002 * 252) = 0.2244994. The
        # population standard deviation would give 0.1587451 instead.
        expected = "0.2244994"
        # Run test.
        actual = nvutils.annualized_volatility(returns)
        # Check outputs.
        self.assert_equal(f"{actual:.7f}", expected)

    def test2(self) -> None:
        """
        Test that a constant daily return has zero volatility.
        """
        # Prepare inputs.
        returns = pd.Series([0.0004] * 252)
        # Prepare outputs.
        expected = "0.0000000"
        # Run test.
        actual = nvutils.annualized_volatility(returns)
        # Check outputs.
        self.assert_equal(f"{actual:.7f}", expected)


# #############################################################################
# Test_equal_weights
# #############################################################################


class Test_equal_weights(hunitest.TestCase):
    """
    Test `nevergrad_utils.equal_weights()`.
    """

    def test1(self) -> None:
        """
        Test that each of 4 assets gets a quarter.
        """
        # Prepare outputs.
        expected = "0.250000 0.250000 0.250000 0.250000"
        # Run test.
        actual = nvutils.equal_weights(4)
        # Check outputs.
        self.assert_equal(_format_weights(actual), expected)


# #############################################################################
# Test_min_variance_weights
# #############################################################################


class Test_min_variance_weights(hunitest.TestCase):
    """
    Test `nevergrad_utils.min_variance_weights()`.

    The returns are built from the zero-mean patterns `u1 = [1, -1, 1, -1]`,
    `u2 = [1, 1, -1, -1]`, and `u3 = [1, -1, -1, 1]`, which are uncorrelated
    with each other, so the covariances can be computed by hand.
    """

    def helper(
        self, returns: pd.DataFrame, max_weight: float, expected: str
    ) -> None:
        """
        Solve for the minimum-variance weights and compare with `expected`.

        :param returns: daily returns of the assets, one column per asset
        :param max_weight: largest weight allowed for one asset
        :param expected: expected output of `_format_weights()`
        """
        # Run test.
        actual = nvutils.min_variance_weights(returns, max_weight=max_weight)
        # Check outputs.
        self.assert_equal(_format_weights(actual), expected)

    def test1(self) -> None:
        """
        Test that uncorrelated assets get weights proportional to 1 / variance.
        """
        # Prepare inputs.
        # B's variance is 4 times A's, so the weights are 1 : 1/4 = 0.8 : 0.2.
        returns = pd.DataFrame(
            {"A": [0.1, -0.1, 0.1, -0.1], "B": [0.2, 0.2, -0.2, -0.2]}
        )
        # Run test.
        self.helper(returns, 1.0, "0.800000 0.200000")

    def test2(self) -> None:
        """
        Test that `max_weight` caps the weight of the calmest asset.
        """
        # Prepare inputs.
        returns = pd.DataFrame(
            {"A": [0.1, -0.1, 0.1, -0.1], "B": [0.2, 0.2, -0.2, -0.2]}
        )
        # Run test.
        self.helper(returns, 0.6, "0.600000 0.400000")

    def test3(self) -> None:
        """
        Test the long-only optimum when the closed form goes short.
        """
        # Prepare inputs.
        # B = 0.2 * u1 + 0.1 * u2 moves with A = 0.1 * u1 but is more volatile,
        # so the closed form `w ∝ Σ⁻¹ 1` hedges it: [1.333, -0.444, 0.111].
        # Clipping B to 0 and renormalizing gives [0.923, 0, 0.077], with
        # variance 0.00876 (in units of 4/3); the long-only optimum holds no B
        # and splits A and C = 0.2 * u3 by 1 / variance: [0.8, 0, 0.2], with
        # variance 0.008.
        u1 = np.array([1, -1, 1, -1])
        u2 = np.array([1, 1, -1, -1])
        u3 = np.array([1, -1, -1, 1])
        returns = pd.DataFrame(
            {"A": 0.1 * u1, "B": 0.2 * u1 + 0.1 * u2, "C": 0.2 * u3}
        )
        # Run test.
        self.helper(returns, 1.0, "0.800000 0.000000 0.200000")

    def test4(self) -> None:
        """
        Test that a cap too small to reach a total of 1 raises an error.
        """
        # Prepare inputs.
        returns = pd.DataFrame(
            {"A": [0.1, -0.1, 0.1, -0.1], "B": [0.2, 0.2, -0.2, -0.2]}
        )
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            nvutils.min_variance_weights(returns, max_weight=0.4)
