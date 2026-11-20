"""
Unit tests for kats_utils.

Import as:

import class_project.msml610.Fall2026.projects.UmdTask851_MSML610_Fall2026_Kats_Stock_Price_Forecasting.test.test_kats_utils as tkatsut
"""

import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

import class_project.msml610.Fall2026.projects.UmdTask851_MSML610_Fall2026_Kats_Stock_Price_Forecasting.kats_utils as utils
import helpers.hio as hio
import helpers.hprint as hprint
import helpers.hunit_test as hunitest

_LOG = logging.getLogger(__name__)


# #############################################################################
# Helpers shared by the test classes
# #############################################################################


def _make_bday_series(values: List[float], *, start: str) -> pd.Series:
    """
    Build a series on consecutive business days.

    :param values: values of the series
    :param start: first date, e.g., `2021-01-04` (a Monday)
    :return: series indexed by business day
    """
    index = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=index, name="Close")


def _make_forecast(
    start: str, fcst: List[float], lower: List[float], upper: List[float]
) -> pd.DataFrame:
    """
    Build a forecast frame in the format of the `forecast_*()` functions.

    :param start: first forecast date
    :param fcst: forecast values
    :param lower: lower bounds of the interval
    :param upper: upper bounds of the interval
    :return: frame with columns `time, fcst, fcst_lower, fcst_upper`
    """
    return pd.DataFrame(
        {
            "time": pd.bdate_range(start, periods=len(fcst)),
            "fcst": fcst,
            "fcst_lower": lower,
            "fcst_upper": upper,
        }
    )


# #############################################################################
# Test_get_default_data_path
# #############################################################################


class Test_get_default_data_path(hunitest.TestCase):
    """
    Test `kats_utils.get_default_data_path()`.
    """

    def test1(self) -> None:
        """
        Test that the Tesla file ships with the tutorial.
        """
        # Run test.
        actual = utils.get_default_data_path()
        # Check outputs.
        self.assertTrue(os.path.exists(actual))
        self.assertEqual(os.path.basename(actual), "TSLA.csv")


# #############################################################################
# Test_load_prices
# #############################################################################


class Test_load_prices(hunitest.TestCase):
    """
    Test `kats_utils.load_prices()`.
    """

    def helper(self, csv_text: str, expected: str) -> None:
        """
        Write a CSV file, load it, and compare the whole frame.

        :param csv_text: content of the CSV file
        :param expected: expected frame as a string
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "prices.csv")
        hio.to_file(path, hprint.dedent(csv_text))
        # Run test.
        actual = utils.load_prices(path)
        # Check outputs.
        self.assert_equal(actual.to_string(), expected, dedent=True)

    def test1(self) -> None:
        """
        Test that unsorted rows come back sorted by date.
        """
        # Prepare inputs.
        csv_text = """
        Date,Close
        2021-01-06,12.0
        2021-01-04,10.0
        2021-01-05,11.0
        """
        # Prepare outputs.
        expected = """
                Date  Close
        0 2021-01-04   10.0
        1 2021-01-05   11.0
        2 2021-01-06   12.0
        """
        # Run test and check outputs.
        self.helper(csv_text, expected)

    def test2(self) -> None:
        """
        Test a file with a single row.
        """
        # Prepare inputs.
        csv_text = """
        Date,Close
        2021-01-04,10.0
        """
        # Prepare outputs.
        expected = """
                Date  Close
        0 2021-01-04   10.0
        """
        # Run test and check outputs.
        self.helper(csv_text, expected)

    def test3(self) -> None:
        """
        Test the shipped Tesla file: size, columns, and date range.
        """
        # Prepare inputs.
        path = utils.get_default_data_path()
        # Prepare outputs.
        expected = "(2843, 7) 2010-07-01 2021-10-14"
        # Run test.
        df = utils.load_prices(path)
        # Check outputs.
        actual = "%s %s %s" % (
            df.shape,
            df["Date"].min().date(),
            df["Date"].max().date(),
        )
        self.assert_equal(actual, expected)

    def test4(self) -> None:
        """
        Test that a date appearing twice is rejected.
        """
        # Prepare inputs.
        csv_text = """
        Date,Close
        2021-01-04,10.0
        2021-01-04,11.0
        """
        path = os.path.join(self.get_scratch_space(), "prices.csv")
        hio.to_file(path, hprint.dedent(csv_text))
        # Run test and check output.
        with self.assertRaises(AssertionError):
            utils.load_prices(path)


# #############################################################################
# Test_to_business_day_series
# #############################################################################


class Test_to_business_day_series(hunitest.TestCase):
    """
    Test `kats_utils.to_business_day_series()`.
    """

    def helper(
        self, dates: List[str], values: List[float], expected: str
    ) -> None:
        """
        Build a price frame, put it on the business-day grid, and compare.

        :param dates: dates of the observations
        :param values: closing prices
        :param expected: expected series as a string
        """
        # Prepare inputs.
        df = pd.DataFrame({"Date": pd.to_datetime(dates), "Close": values})
        # Run test.
        actual = utils.to_business_day_series(df)
        # Check outputs.
        self.assert_equal(actual.to_frame().to_string(), expected, dedent=True)

    def test1(self) -> None:
        """
        Test that a missing Tuesday and Thursday get the previous close.
        """
        # Prepare inputs.
        dates = ["2021-01-04", "2021-01-06", "2021-01-08"]
        values = [10.0, 12.0, 14.0]
        # Prepare outputs.
        expected = """
                    Close
        2021-01-04   10.0
        2021-01-05   10.0
        2021-01-06   12.0
        2021-01-07   12.0
        2021-01-08   14.0
        """
        # Run test and check outputs.
        self.helper(dates, values, expected)

    def test2(self) -> None:
        """
        Test that a series without gaps is unchanged.
        """
        # Prepare inputs.
        dates = ["2021-01-04", "2021-01-05", "2021-01-06"]
        values = [10.0, 11.0, 12.0]
        # Prepare outputs.
        expected = """
                    Close
        2021-01-04   10.0
        2021-01-05   11.0
        2021-01-06   12.0
        """
        # Run test and check outputs.
        self.helper(dates, values, expected)

    def test3(self) -> None:
        """
        Test that Friday to Monday does not add weekend rows.
        """
        # Prepare inputs.
        dates = ["2021-01-08", "2021-01-11"]
        values = [5.0, 6.0]
        # Prepare outputs.
        expected = """
                    Close
        2021-01-08    5.0
        2021-01-11    6.0
        """
        # Run test and check outputs.
        self.helper(dates, values, expected)

    def test4(self) -> None:
        """
        Test that an observation on a Saturday is dropped.

        Documents a limit of the function: only business days are kept.
        """
        # Prepare inputs.
        dates = ["2021-01-08", "2021-01-09", "2021-01-11"]
        values = [5.0, 7.0, 6.0]
        # Prepare outputs.
        expected = """
                    Close
        2021-01-08    5.0
        2021-01-11    6.0
        """
        # Run test and check outputs.
        self.helper(dates, values, expected)

    def test5(self) -> None:
        """
        Test the shipped Tesla file: 2843 trading days become 2946 days.
        """
        # Prepare inputs.
        df = utils.load_prices(utils.get_default_data_path())
        # Prepare outputs.
        expected = "2946 0"
        # Run test.
        series = utils.to_business_day_series(df)
        # Check outputs.
        actual = "%d %d" % (len(series), series.isna().sum())
        self.assert_equal(actual, expected)


# #############################################################################
# Test_to_timeseries_data
# #############################################################################


class Test_to_timeseries_data(hunitest.TestCase):
    """
    Test `kats_utils.to_timeseries_data()`.
    """

    def test1(self) -> None:
        """
        Test that times and values survive the conversion.
        """
        # Prepare inputs.
        series = _make_bday_series([10.0, 11.0, 12.0], start="2021-01-04")
        # Prepare outputs.
        expected = """
                time  value
        0 2021-01-04   10.0
        1 2021-01-05   11.0
        2 2021-01-06   12.0
        """
        # Run test.
        actual = utils.to_timeseries_data(series).to_dataframe()
        # Check outputs.
        self.assert_equal(actual.to_string(), expected, dedent=True)


# #############################################################################
# Test_split_train_test
# #############################################################################


class Test_split_train_test(hunitest.TestCase):
    """
    Test `kats_utils.split_train_test()`.
    """

    def helper(self, horizon: int, expected: str) -> None:
        """
        Split ten points and compare the values of both parts.

        :param horizon: number of points in the test set
        :param expected: expected train and test values as a string
        """
        # Prepare inputs.
        series = _make_bday_series(list(range(10)), start="2021-01-04")
        # Run test.
        train, test = utils.split_train_test(series, horizon)
        # Check outputs.
        actual = "train=%s test=%s" % (train.tolist(), test.tolist())
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test that the last three points become the test set.
        """
        # Prepare inputs.
        horizon = 3
        # Prepare outputs.
        expected = "train=[0, 1, 2, 3, 4, 5, 6] test=[7, 8, 9]"
        # Run test and check outputs.
        self.helper(horizon, expected)

    def test2(self) -> None:
        """
        Test a horizon of one point.
        """
        # Prepare inputs.
        horizon = 1
        # Prepare outputs.
        expected = "train=[0, 1, 2, 3, 4, 5, 6, 7, 8] test=[9]"
        # Run test and check outputs.
        self.helper(horizon, expected)

    def test3(self) -> None:
        """
        Test the largest horizon that still leaves one training point.
        """
        # Prepare inputs.
        horizon = 9
        # Prepare outputs.
        expected = "train=[0] test=[1, 2, 3, 4, 5, 6, 7, 8, 9]"
        # Run test and check outputs.
        self.helper(horizon, expected)

    def test4(self) -> None:
        """
        Test that the test set starts after the end of the training set.
        """
        # Prepare inputs.
        series = _make_bday_series(list(range(10)), start="2021-01-04")
        horizon = 4
        # Run test.
        train, test = utils.split_train_test(series, horizon)
        # Check outputs.
        self.assertLess(train.index[-1], test.index[0])


# #############################################################################
# Test_check_stationarity
# #############################################################################


class Test_check_stationarity(hunitest.TestCase):
    """
    Test `kats_utils.check_stationarity()`.
    """

    def helper(self, values: np.ndarray, expected: str) -> None:
        """
        Run the test on a series and compare the stationarity flags.

        :param values: values of the series
        :param expected: expected flags as a string
        """
        # Prepare inputs.
        series = _make_bday_series(values.tolist(), start="2020-01-01")
        # Run test.
        actual = utils.check_stationarity(series)
        # Check outputs.
        self.assert_equal(str(actual["stationary_at_5pct"].to_dict()), expected)

    def test1(self) -> None:
        """
        Test that a random walk needs one difference.
        """
        # Prepare inputs.
        rng = np.random.default_rng(0)
        values = np.cumsum(rng.normal(size=500))
        # Prepare outputs.
        expected = "{'level': False, 'first difference': True}"
        # Run test and check outputs.
        self.helper(values, expected)

    def test2(self) -> None:
        """
        Test that white noise is stationary without differencing.
        """
        # Prepare inputs.
        rng = np.random.default_rng(0)
        values = rng.normal(size=500)
        # Prepare outputs.
        expected = "{'level': True, 'first difference': True}"
        # Run test and check outputs.
        self.helper(values, expected)


# #############################################################################
# Test_make_synthetic_series
# #############################################################################


class Test_make_synthetic_series(hunitest.TestCase):
    """
    Test `kats_utils.make_synthetic_series()`.
    """

    def test1(self) -> None:
        """
        Test the default length, start date, and frequency.
        """
        # Prepare outputs.
        expected = "365 2020-01-01 2020-12-30 D"
        # Run test.
        series = utils.make_synthetic_series()
        # Check outputs.
        actual = "%d %s %s %s" % (
            len(series),
            series.index[0].date(),
            series.index[-1].date(),
            series.index.freqstr,
        )
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Test that the same seed gives the same series.
        """
        # Prepare inputs.
        seed = 3
        # Run test.
        first = utils.make_synthetic_series(seed=seed)
        second = utils.make_synthetic_series(seed=seed)
        # Check outputs.
        self.assertTrue(first.equals(second))

    def test3(self) -> None:
        """
        Test that a different seed gives a different series.
        """
        # Run test.
        first = utils.make_synthetic_series(seed=1)
        second = utils.make_synthetic_series(seed=2)
        # Check outputs.
        self.assertFalse(first.equals(second))

    def test4(self) -> None:
        """
        Test a business-day frequency.
        """
        # Prepare inputs.
        num_periods = 10
        freq = "B"
        # Prepare outputs.
        expected = "2020-01-01 2020-01-14"
        # Run test.
        series = utils.make_synthetic_series(num_periods=num_periods, freq=freq)
        # Check outputs.
        actual = "%s %s" % (series.index[0].date(), series.index[-1].date())
        self.assert_equal(actual, expected)


# #############################################################################
# Test_forecast_naive
# #############################################################################


class Test_forecast_naive(hunitest.TestCase):
    """
    Test `kats_utils.forecast_naive()`.
    """

    def helper(
        self, values: List[float], start: str, horizon: int, expected: str
    ) -> None:
        """
        Forecast a short series and compare the whole forecast frame.

        :param values: values of the training series
        :param start: first training date
        :param horizon: number of days to forecast
        :param expected: expected forecast as a string
        """
        # Prepare inputs.
        train = _make_bday_series(values, start=start)
        # Run test.
        actual = utils.forecast_naive(train, horizon)
        # Check outputs.
        self.assert_equal(actual.round(4).to_string(), expected, dedent=True)

    def test1(self) -> None:
        """
        Test the interval widening as sigma * z * sqrt(h).

        The daily changes are 2, -1, 2, so sigma = sqrt(3) and the half width at
        95% is 1.96 * 1.7321 * sqrt(h), i.e., 3.3948, 4.8009, 5.8799.
        """
        # Prepare inputs.
        values = [100.0, 102.0, 101.0, 103.0]
        start = "2021-01-04"
        horizon = 3
        # Prepare outputs.
        expected = """
                time   fcst  fcst_lower  fcst_upper
        0 2021-01-08  103.0     99.6052    106.3948
        1 2021-01-11  103.0     98.1991    107.8009
        2 2021-01-12  103.0     97.1201    108.8799
        """
        # Run test and check outputs.
        self.helper(values, start, horizon, expected)

    def test2(self) -> None:
        """
        Test that a flat series gives a flat forecast with no interval.
        """
        # Prepare inputs.
        values = [50.0, 50.0, 50.0, 50.0]
        start = "2021-01-04"
        horizon = 2
        # Prepare outputs.
        expected = """
                time  fcst  fcst_lower  fcst_upper
        0 2021-01-08  50.0        50.0        50.0
        1 2021-01-11  50.0        50.0        50.0
        """
        # Run test and check outputs.
        self.helper(values, start, horizon, expected)

    def test3(self) -> None:
        """
        Test that the forecast starts on Monday after a Friday.
        """
        # Prepare inputs.
        values = [10.0, 11.0, 12.0, 13.0, 14.0]
        start = "2021-01-04"
        horizon = 3
        # Prepare outputs.
        expected = "['2021-01-11', '2021-01-12', '2021-01-13']"
        # Run test.
        train = _make_bday_series(values, start=start)
        forecast = utils.forecast_naive(train, horizon)
        # Check outputs.
        actual = str([str(t.date()) for t in forecast["time"]])
        self.assert_equal(actual, expected)

    def test4(self) -> None:
        """
        Test a horizon of one day.
        """
        # Prepare inputs.
        values = [100.0, 102.0, 101.0, 103.0]
        start = "2021-01-04"
        horizon = 1
        # Prepare outputs.
        expected = """
                time   fcst  fcst_lower  fcst_upper
        0 2021-01-08  103.0     99.6052    106.3948
        """
        # Run test and check outputs.
        self.helper(values, start, horizon, expected)


# #############################################################################
# Test_select_arima_order
# #############################################################################


class Test_select_arima_order(hunitest.TestCase):
    """
    Test `kats_utils.select_arima_order()`.
    """

    def test1(self) -> None:
        """
        Test that a grid with only `p=0, q=0` returns that order.
        """
        # Prepare inputs.
        train = utils.make_synthetic_series(num_periods=100, freq="B")
        max_p = 0
        max_q = 0
        # Prepare outputs.
        expected = (0, 1, 0)
        # Run test.
        actual = utils.select_arima_order(train, d=1, max_p=max_p, max_q=max_q)
        # Check outputs.
        self.assertEqual(actual, expected)

    def test2(self) -> None:
        """
        Test that the selected order stays inside the grid.
        """
        # Prepare inputs.
        train = utils.make_synthetic_series(num_periods=100, freq="B")
        max_p = 1
        max_q = 1
        # Run test.
        p, d, q = utils.select_arima_order(train, d=1, max_p=max_p, max_q=max_q)
        # Check outputs.
        self.assertTrue(0 <= p <= max_p)
        self.assertTrue(0 <= q <= max_q)
        self.assertEqual(d, 1)


# #############################################################################
# Test_forecast_arima
# #############################################################################


class Test_forecast_arima(hunitest.TestCase):
    """
    Test `kats_utils.forecast_arima()`.
    """

    def helper(self, interval_level: float) -> pd.DataFrame:
        """
        Fit an ARIMA(1, 1, 0) on a synthetic series and forecast five days.

        :param interval_level: probability mass of the prediction interval
        :return: forecast frame
        """
        # Prepare inputs.
        train = utils.make_synthetic_series(num_periods=100, freq="B")
        horizon = 5
        order = (1, 1, 0)
        # Run test.
        return utils.forecast_arima(
            train, horizon, order=order, interval_level=interval_level
        )

    def test1(self) -> None:
        """
        Test the columns, the length, and the order of the interval bounds.
        """
        # Prepare inputs.
        interval_level = 0.95
        # Prepare outputs.
        expected = "['time', 'fcst', 'fcst_lower', 'fcst_upper'] 5 True"
        # Run test.
        forecast = self.helper(interval_level)
        # Check outputs.
        ordered = (
            (forecast["fcst_lower"] <= forecast["fcst"])
            & (forecast["fcst"] <= forecast["fcst_upper"])
        ).all()
        actual = "%s %d %s" % (forecast.columns.tolist(), len(forecast), ordered)
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Test that the forecast dates follow the training set.
        """
        # Prepare inputs.
        interval_level = 0.95
        # Prepare outputs.
        train = utils.make_synthetic_series(num_periods=100, freq="B")
        expected = str([str(t.date()) for t in utils._forecast_index(train, 5)])
        # Run test.
        forecast = self.helper(interval_level)
        # Check outputs.
        actual = str([str(t.date()) for t in forecast["time"]])
        self.assert_equal(actual, expected)

    def test3(self) -> None:
        """
        Test that a 99% interval is wider than an 80% interval.
        """
        # Run test.
        wide = self.helper(0.99)
        narrow = self.helper(0.80)
        # Check outputs.
        wide_width = (wide["fcst_upper"] - wide["fcst_lower"]).mean()
        narrow_width = (narrow["fcst_upper"] - narrow["fcst_lower"]).mean()
        self.assertGreater(wide_width, narrow_width)


# #############################################################################
# Test_forecast_prophet
# #############################################################################


class Test_forecast_prophet(hunitest.TestCase):
    """
    Test `kats_utils.forecast_prophet()`.
    """

    def helper(self, *, interval_level: float = 0.95) -> pd.DataFrame:
        """
        Fit Prophet on a synthetic trending series and forecast five days.

        :param interval_level: probability mass of the prediction interval
        :return: forecast frame
        """
        # Prepare inputs.
        train = utils.make_synthetic_series(num_periods=100, freq="B")
        horizon = 5
        # Run test.
        return utils.forecast_prophet(
            train, horizon, interval_level=interval_level
        )

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test the columns, the length, and the order of the interval bounds.
        """
        # Prepare outputs.
        expected = "['time', 'fcst', 'fcst_lower', 'fcst_upper'] 5 True"
        # Run test.
        forecast = self.helper()
        # Check outputs.
        ordered = (
            (forecast["fcst_lower"] <= forecast["fcst"])
            & (forecast["fcst"] <= forecast["fcst_upper"])
        ).all()
        actual = "%s %d %s" % (forecast.columns.tolist(), len(forecast), ordered)
        self.assert_equal(actual, expected)

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that a series with an upward trend gets a rising forecast.
        """
        # Run test.
        forecast = self.helper()
        # Check outputs.
        self.assertGreater(forecast["fcst"].iloc[-1], forecast["fcst"].iloc[0])

    @pytest.mark.slow
    def test3(self) -> None:
        """
        Test that a 99% interval is wider than an 80% interval.
        """
        # Run test.
        wide = self.helper(interval_level=0.99)
        narrow = self.helper(interval_level=0.80)
        # Check outputs.
        wide_width = (wide["fcst_upper"] - wide["fcst_lower"]).mean()
        narrow_width = (narrow["fcst_upper"] - narrow["fcst_lower"]).mean()
        self.assertGreater(wide_width, narrow_width)


# #############################################################################
# Test_forecast_all
# #############################################################################


class Test_forecast_all(hunitest.TestCase):
    """
    Test `kats_utils.forecast_all()`.
    """

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that all three models forecast the same dates.
        """
        # Prepare inputs.
        train = utils.make_synthetic_series(num_periods=100, freq="B")
        horizon = 5
        order = (1, 1, 0)
        # Prepare outputs.
        expected = "['Naive', 'ARIMA', 'Prophet'] True"
        # Run test.
        forecasts = utils.forecast_all(train, horizon, arima_order=order)
        # Check outputs.
        times = [f["time"].tolist() for f in forecasts.values()]
        same_dates = all(t == times[0] for t in times)
        actual = "%s %s" % (list(forecasts.keys()), same_dates)
        self.assert_equal(actual, expected)


# #############################################################################
# Test_compute_mae
# #############################################################################


class Test_compute_mae(hunitest.TestCase):
    """
    Test `kats_utils.compute_mae()`.
    """

    def helper(
        self, actual: List[float], predicted: List[float], expected: float
    ) -> None:
        """
        Compute the MAE of two lists and compare with the expected value.

        :param actual: observed values
        :param predicted: forecast values
        :param expected: expected MAE
        """
        # Run test.
        result = utils.compute_mae(np.array(actual), np.array(predicted))
        # Check outputs.
        self.assertAlmostEqual(result, expected)

    def test1(self) -> None:
        """
        Test errors of 1, 0, and 2, whose mean is 1.
        """
        # Prepare inputs.
        actual = [1.0, 2.0, 3.0]
        predicted = [2.0, 2.0, 5.0]
        # Prepare outputs.
        expected = 1.0
        # Run test and check outputs.
        self.helper(actual, predicted, expected)

    def test2(self) -> None:
        """
        Test that a perfect forecast has no error.
        """
        # Prepare inputs.
        actual = [1.0, 2.0, 3.0]
        predicted = [1.0, 2.0, 3.0]
        # Prepare outputs.
        expected = 0.0
        # Run test and check outputs.
        self.helper(actual, predicted, expected)

    def test3(self) -> None:
        """
        Test a single point.
        """
        # Prepare inputs.
        actual = [4.0]
        predicted = [1.0]
        # Prepare outputs.
        expected = 3.0
        # Run test and check outputs.
        self.helper(actual, predicted, expected)


# #############################################################################
# Test_compute_rmse
# #############################################################################


class Test_compute_rmse(hunitest.TestCase):
    """
    Test `kats_utils.compute_rmse()`.
    """

    def helper(
        self, actual: List[float], predicted: List[float], expected: float
    ) -> None:
        """
        Compute the RMSE of two lists and compare with the expected value.

        :param actual: observed values
        :param predicted: forecast values
        :param expected: expected RMSE
        """
        # Run test.
        result = utils.compute_rmse(np.array(actual), np.array(predicted))
        # Check outputs.
        self.assertAlmostEqual(result, expected)

    def test1(self) -> None:
        """
        Test squared errors of 1, 0, and 4, so RMSE is sqrt(5 / 3).
        """
        # Prepare inputs.
        actual = [1.0, 2.0, 3.0]
        predicted = [2.0, 2.0, 5.0]
        # Prepare outputs.
        expected = float(np.sqrt(5 / 3))
        # Run test and check outputs.
        self.helper(actual, predicted, expected)

    def test2(self) -> None:
        """
        Test that a perfect forecast has no error.
        """
        # Prepare inputs.
        actual = [1.0, 2.0, 3.0]
        predicted = [1.0, 2.0, 3.0]
        # Prepare outputs.
        expected = 0.0
        # Run test and check outputs.
        self.helper(actual, predicted, expected)

    def test3(self) -> None:
        """
        Test a single point, where RMSE equals the absolute error.
        """
        # Prepare inputs.
        actual = [4.0]
        predicted = [1.0]
        # Prepare outputs.
        expected = 3.0
        # Run test and check outputs.
        self.helper(actual, predicted, expected)


# #############################################################################
# Test_compute_coverage
# #############################################################################


class Test_compute_coverage(hunitest.TestCase):
    """
    Test `kats_utils.compute_coverage()`.
    """

    def helper(
        self,
        actual: List[float],
        lower: List[float],
        upper: List[float],
        expected: float,
    ) -> None:
        """
        Compute the coverage of an interval and compare with the expected value.

        :param actual: observed values
        :param lower: lower bounds of the interval
        :param upper: upper bounds of the interval
        :param expected: expected share of points inside the interval
        """
        # Prepare inputs.
        forecast = pd.DataFrame({"fcst_lower": lower, "fcst_upper": upper})
        # Run test.
        result = utils.compute_coverage(np.array(actual), forecast)
        # Check outputs.
        self.assertAlmostEqual(result, expected)

    def test1(self) -> None:
        """
        Test that three of four points inside the interval give 0.75.
        """
        # Prepare inputs.
        actual = [1.0, 2.0, 3.0, 4.0]
        lower = [0.0, 0.0, 0.0, 5.0]
        upper = [2.0, 4.0, 4.0, 6.0]
        # Prepare outputs.
        expected = 0.75
        # Run test and check outputs.
        self.helper(actual, lower, upper, expected)

    def test2(self) -> None:
        """
        Test that a point exactly on the bound counts as inside.
        """
        # Prepare inputs.
        actual = [2.0]
        lower = [2.0]
        upper = [2.0]
        # Prepare outputs.
        expected = 1.0
        # Run test and check outputs.
        self.helper(actual, lower, upper, expected)

    def test3(self) -> None:
        """
        Test that points outside the interval give zero coverage.
        """
        # Prepare inputs.
        actual = [10.0, 10.0]
        lower = [0.0, 0.0]
        upper = [1.0, 1.0]
        # Prepare outputs.
        expected = 0.0
        # Run test and check outputs.
        self.helper(actual, lower, upper, expected)


# #############################################################################
# Test_evaluate_forecasts
# #############################################################################


class Test_evaluate_forecasts(hunitest.TestCase):
    """
    Test `kats_utils.evaluate_forecasts()`.
    """

    def test1(self) -> None:
        """
        Test the whole table for the naive baseline and a perfect forecast.

        The baseline misses by 0, 10, 20, so MAE is 10, RMSE is sqrt(500 / 3),
        and only the first point is inside its interval.
        """
        # Prepare inputs.
        actual = _make_bday_series([10.0, 20.0, 30.0], start="2021-01-04")
        forecasts = {
            "Naive": _make_forecast(
                "2021-01-04",
                [10.0] * 3,
                [5.0] * 3,
                [15.0] * 3,
            ),
            "Perfect": _make_forecast(
                "2021-01-04",
                [10.0, 20.0, 30.0],
                [9.0, 19.0, 29.0],
                [11.0, 21.0, 31.0],
            ),
        }
        # Prepare outputs.
        expected = """
                  MAE     RMSE  MAE / naive MAE  interval coverage
        Perfect   0.0   0.0000              0.0             1.0000
        Naive    10.0  12.9099              1.0             0.3333
        """
        # Run test.
        actual_table = utils.evaluate_forecasts(actual, forecasts)
        # Check outputs.
        self.assert_equal(
            actual_table.round(4).to_string(), expected, dedent=True
        )

    def test2(self) -> None:
        """
        Test that a forecast dated differently from the test set is rejected.
        """
        # Prepare inputs.
        actual = _make_bday_series([10.0, 20.0], start="2021-01-04")
        shifted = _make_forecast("2021-01-05", [10.0] * 2, [9.0] * 2, [11.0] * 2)
        forecasts = {"Naive": shifted}
        # Run test and check output.
        with self.assertRaises(AssertionError):
            utils.evaluate_forecasts(actual, forecasts)


# #############################################################################
# Test_rolling_origin_evaluation
# #############################################################################


class Test_rolling_origin_evaluation(hunitest.TestCase):
    """
    Test `kats_utils.rolling_origin_evaluation()`.
    """

    def helper(self, num_origins: int) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Backtest a synthetic series of 150 business days with 10-day windows.

        :param num_origins: number of test windows
        :return: the series and the backtest results
        """
        # Prepare inputs.
        series = utils.make_synthetic_series(num_periods=150, freq="B")
        horizon = 10
        order = (1, 1, 0)
        # Run test.
        results = utils.rolling_origin_evaluation(
            series, horizon, num_origins=num_origins, arima_order=order
        )
        return series, results

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that two windows end at the series end and do not overlap.

        The origins are the last training days, at positions 129 and 139.
        """
        # Prepare inputs.
        num_origins = 2
        # Run test.
        series, results = self.helper(num_origins)
        # Prepare outputs.
        expected_origins = [series.index[129], series.index[139]]
        expected_rows = 6
        # Check outputs.
        self.assertEqual(sorted(results["origin"].unique()), expected_origins)
        self.assertEqual(len(results), expected_rows)
        self.assert_equal(
            str(sorted(results["model"].unique())),
            "['ARIMA', 'Naive', 'Prophet']",
        )

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that one window trains on exactly what `split_train_test()` does.
        """
        # Prepare inputs.
        num_origins = 1
        # Run test.
        series, results = self.helper(num_origins)
        # Prepare outputs.
        train, _ = utils.split_train_test(series, 10)
        expected_origin = train.index[-1]
        # Check outputs.
        self.assertEqual(
            [pd.Timestamp(o) for o in results["origin"].unique()],
            [expected_origin],
        )
        self.assertEqual(len(results), 3)


# #############################################################################
# Test_summarize_rolling_evaluation
# #############################################################################


class Test_summarize_rolling_evaluation(hunitest.TestCase):
    """
    Test `kats_utils.summarize_rolling_evaluation()`.
    """

    def test1(self) -> None:
        """
        Test the whole summary table on hand-built results of three origins.

        MAE by origin: Naive 10, 10, 10; ARIMA 8, 12, 9; Prophet 20, 5, 15.
        """
        # Prepare inputs.
        origins = pd.to_datetime(["2021-01-04", "2021-02-01", "2021-03-01"])
        maes = {
            "Naive": [10.0, 10.0, 10.0],
            "ARIMA": [8.0, 12.0, 9.0],
            "Prophet": [20.0, 5.0, 15.0],
        }
        coverages = {
            "Naive": [0.5, 0.5, 0.5],
            "ARIMA": [0.6, 0.8, 0.7],
            "Prophet": [0.1, 0.3, 0.2],
        }
        rows = []
        for model, values in maes.items():
            for i, origin in enumerate(origins):
                rows.append(
                    {
                        "origin": origin,
                        "model": model,
                        "MAE": values[i],
                        "MAE / naive MAE": values[i] / maes["Naive"][i],
                        "interval coverage": coverages[model][i],
                    }
                )
        results = pd.DataFrame(rows)
        # Prepare outputs.
        expected = """
                 mean MAE / naive MAE  windows beating naive (%)  windows beating ARIMA (%)  mean interval coverage
        model
        Naive                  1.0000                        NaN                    33.3333                     0.5
        ARIMA                  0.9667                    66.6667                        NaN                     0.7
        Prophet                1.3333                    33.3333                    33.3333                     0.2
        """
        # Run test.
        actual = utils.summarize_rolling_evaluation(results)
        # Check outputs.
        self.assert_equal(
            actual.round(4).to_string(), expected, dedent=True, fuzzy_match=True
        )


# #############################################################################
# Test_plot_forecasts
# #############################################################################


class Test_plot_forecasts(hunitest.TestCase):
    """
    Test `kats_utils.plot_forecasts()`.
    """

    def test1(self) -> None:
        """
        Test that the plot has one panel per model.
        """
        # Prepare inputs.
        plt = utils.plt
        plt.switch_backend("Agg")
        series = utils.make_synthetic_series(num_periods=60, freq="B")
        train, test = utils.split_train_test(series, 5)
        forecast = utils.forecast_naive(train, 5)
        forecasts = {"A": forecast, "B": forecast, "C": forecast}
        # Prepare outputs.
        expected = 3
        # Run test.
        utils.plot_forecasts(train, test, forecasts)
        # Check outputs.
        actual = len(plt.gcf().axes)
        plt.close("all")
        self.assertEqual(actual, expected)


# #############################################################################
# Test_plot_rolling_evaluation
# #############################################################################


class Test_plot_rolling_evaluation(hunitest.TestCase):
    """
    Test `kats_utils.plot_rolling_evaluation()`.
    """

    def test1(self) -> None:
        """
        Test that the plot has a single panel.
        """
        # Prepare inputs.
        plt = utils.plt
        plt.switch_backend("Agg")
        origins = pd.to_datetime(["2021-01-04", "2021-02-01"])
        rows = [
            {"origin": o, "model": m, "MAE / naive MAE": r}
            for m, r in [("ARIMA", 1.0), ("Prophet", 2.0)]
            for o in origins
        ]
        results = pd.DataFrame(rows)
        # Prepare outputs.
        expected = 1
        # Run test.
        utils.plot_rolling_evaluation(results)
        # Check outputs.
        actual = len(plt.gcf().axes)
        plt.close("all")
        self.assertEqual(actual, expected)
