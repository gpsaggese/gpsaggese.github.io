"""Offline tests for descriptive price context; no application imports or I/O."""

import copy
import json
import math
import unittest

from research.financial_sentiment_using_prices.price_context import build_price_context


def bar(day, open_price, close, timestamp=None):
    return {"t": timestamp or f"2026-10-{day:02d}T04:00:00Z", "o": open_price, "c": close}


def fixture():
    return {"bars": {
        "PLTR": [bar(6, 99, 100), bar(7, 110, 121), bar(8, 120, 125)],
        "SPY": [bar(6, 199, 200), bar(7, 202, 204), bar(8, 205, 206)],
        "IGV": [bar(6, 49, 50), bar(7, 51, 52), bar(8, 52, 53)],
    }}


def context(raw=None, start="2026-10-06", end="2026-10-08", benchmarks=None):
    return build_price_context(fixture() if raw is None else raw, ["PLTR"], start, end,
                               {"PLTR": "IGV"} if benchmarks is None else benchmarks)


class PriceContextTests(unittest.TestCase):
    def test_exact_period_comparisons_and_semantics(self):
        raw = fixture()
        original = copy.deepcopy(raw)
        result = context(raw)
        row = result["companies"]["PLTR"]
        self.assertEqual(row["status"], "Available")
        self.assertAlmostEqual(row["return_pct"], 25)
        self.assertAlmostEqual(row["spy_return_pct"], 3)
        self.assertAlmostEqual(row["sector_return_pct"], 6)
        self.assertAlmostEqual(row["excess_vs_spy_pp"], 22)
        self.assertAlmostEqual(row["excess_vs_sector_pp"], 19)
        self.assertEqual(set(result["market_benchmarks"]), {"SPY", "IGV"})
        self.assertTrue(result["descriptive_only"])
        self.assertFalse(result["is_forecast"])
        self.assertEqual(result["return_type"], "raw_price_close_to_close")
        self.assertIn("not portfolio total returns", result["methodology"]["limitations"])
        self.assertEqual(raw, original)
        json.dumps(result, allow_nan=False)

    def test_overnight_and_intraday_decomposition_compound(self):
        daily = context()["companies"]["PLTR"]["daily_sessions"]
        self.assertEqual([row["date"] for row in daily], ["2026-10-07", "2026-10-08"])
        first = daily[0]
        self.assertEqual((first["previous_close"], first["open"], first["close"]), (100, 110, 121))
        self.assertAlmostEqual(first["gap_pct"], 10)
        self.assertAlmostEqual(first["intraday_pct"], 10)
        self.assertAlmostEqual(first["return_pct"], 21)
        for row in daily:
            self.assertAlmostEqual(1 + row["return_pct"] / 100,
                                   (1 + row["gap_pct"] / 100) * (1 + row["intraday_pct"] / 100))

    def test_missing_exact_endpoints_are_not_replaced_by_nearby_bars(self):
        for missing_day in (6, 8):
            with self.subTest(missing_day=missing_day):
                raw = fixture()
                raw["bars"]["PLTR"] = [bar(5, 79, 80), *[b for b in raw["bars"]["PLTR"] if b["t"] != bar(missing_day, 1, 1)["t"]], bar(9, 130, 140)]
                row = context(raw)["companies"]["PLTR"]
                self.assertIsNone(row["return_pct"])
                self.assertIsNone(row["excess_vs_spy_pp"])
                self.assertEqual(row["status"], "Unavailable")
                self.assertIn(f"2026-10-{missing_day:02d}", row["unavailable_reasons"]["return_pct"])

    def test_future_bars_excluded_before_price_and_duplicate_validation(self):
        raw = fixture()
        expected = context(raw)
        for symbol in raw["bars"]:
            raw["bars"][symbol].extend([bar(9, math.nan, 0), bar(9, -1, -2)])
        self.assertEqual(context(raw), expected)

    def test_missing_interior_bar_does_not_create_multiday_overnight_gap(self):
        raw = fixture()
        del raw["bars"]["PLTR"][1]
        row = context(raw)["companies"]["PLTR"]
        self.assertAlmostEqual(row["return_pct"], 25)
        self.assertEqual(row["daily_sessions_status"], "Unavailable")
        missing, following = row["daily_sessions"]
        self.assertIsNone(missing["close"])
        self.assertEqual(following["previous_session_date"], "2026-10-07")
        self.assertIsNone(following["previous_close"])
        self.assertIsNone(following["gap_pct"])
        self.assertIsNone(following["return_pct"])
        self.assertIsNotNone(following["intraday_pct"])

    def test_missing_benchmark_keeps_company_return_but_not_excess(self):
        raw = fixture()
        del raw["bars"]["SPY"]
        del raw["bars"]["IGV"][-1]
        result = context(raw)
        row = result["companies"]["PLTR"]
        self.assertAlmostEqual(row["return_pct"], 25)
        self.assertEqual(row["status"], "Unavailable")
        for field in ("spy_return_pct", "sector_return_pct", "excess_vs_spy_pp", "excess_vs_sector_pp"):
            self.assertIsNone(row[field])
            self.assertTrue(row["unavailable_reasons"][field].startswith("Unavailable:"))
        self.assertEqual(result["market_benchmarks"]["SPY"]["status"], "Unavailable")

    def test_unconfigured_benchmark_is_not_assumed(self):
        row = context(benchmarks={})["companies"]["PLTR"]
        self.assertIsNone(row["sector_symbol"])
        self.assertIsNone(row["sector_return_pct"])
        self.assertAlmostEqual(row["excess_vs_spy_pp"], 22)

    def test_missing_open_does_not_invent_zero_gap(self):
        raw = fixture()
        del raw["bars"]["PLTR"][1]["o"]
        row = context(raw)["companies"]["PLTR"]["daily_sessions"][0]
        self.assertIsNone(row["gap_pct"])
        self.assertIsNone(row["intraday_pct"])
        self.assertAlmostEqual(row["return_pct"], 21)
        self.assertEqual(row["status"], "Unavailable")

    def test_timezone_conversion_uses_new_york_date_and_dst(self):
        raw = {"bars": {symbol: [
            bar(1, 99, 100, "2026-11-01T23:30:00-08:00"),
            bar(2, 100, 110, "2026-11-03T05:00:00Z"),
            bar(3, 110, 999, "2026-11-04T05:00:00Z"),
        ] for symbol in ("PLTR", "SPY", "IGV")}}
        row = context(raw, "2026-11-02", "2026-11-03")["companies"]["PLTR"]
        self.assertAlmostEqual(row["return_pct"], 10)
        self.assertEqual(row["daily_sessions"][0]["previous_session_date"], "2026-11-02")

    def test_rejects_naive_malformed_or_missing_timestamp(self):
        for timestamp in ("2026-10-07T04:00:00", "2026-10-07", "not-a-date", None):
            with self.subTest(timestamp=timestamp):
                raw = fixture()
                raw["bars"]["PLTR"][1]["t"] = timestamp
                with self.assertRaisesRegex(ValueError, "timezone-aware"):
                    context(raw)

    def test_rejects_duplicate_new_york_dates_even_different_instants(self):
        raw = fixture()
        raw["bars"]["PLTR"].append(bar(7, 110, 121, "2026-10-08T01:00:00Z"))
        with self.assertRaisesRegex(ValueError, "Duplicate New York session"):
            context(raw)

    def test_rejects_supplied_invalid_prices(self):
        for field in ("o", "c", "h", "l"):
            for value in (0, -1, float("nan"), float("inf"), -float("inf"), True, "100"):
                with self.subTest(field=field, value=value):
                    raw = fixture()
                    raw["bars"]["PLTR"][1][field] = value
                    with self.assertRaisesRegex(ValueError, "price"):
                        context(raw)

    def test_missing_close_is_unavailable_and_unsorted_input_is_supported(self):
        raw = fixture()
        raw["bars"]["PLTR"].reverse()
        self.assertEqual(context(raw), context())
        del raw["bars"]["PLTR"][0]["c"]
        self.assertIsNone(context(raw)["companies"]["PLTR"]["return_pct"])

    def test_invalid_date_window_and_input_shape(self):
        for start, end in (("20261006", "2026-10-08"), ("2026-10-09", "2026-10-08")):
            with self.subTest(start=start), self.assertRaises(ValueError):
                context(start=start, end=end)
        with self.assertRaisesRegex(ValueError, "bars.*dictionary"):
            context({})
        with self.assertRaisesRegex(ValueError, "Duplicate requested"):
            build_price_context(fixture(), ["PLTR", "PLTR"], "2026-10-06", "2026-10-08", {})

    def test_same_day_is_zero_only_when_the_exact_close_exists(self):
        row = context(start="2026-10-07", end="2026-10-07")["companies"]["PLTR"]
        self.assertEqual(row["return_pct"], 0)
        self.assertEqual(row["daily_sessions"], [])
        row = context(start="2026-10-10", end="2026-10-10")["companies"]["PLTR"]
        self.assertIsNone(row["return_pct"])


if __name__ == "__main__":
    unittest.main()
