"""Offline rule-calendar and synthetic archive coverage regression tests."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calendar_audit import audit_archive, export_calendar, main


@unittest.skipUnless(importlib.util.find_spec("exchange_calendars"), "Optional exchange_calendars not installed")
class CalendarAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calendar = export_calendar("2026-09-01", "2026-11-30")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "input"
        self.archive.mkdir()
        self.raw = {"bars": {s: [{"t": day["date"] + "T04:00:00Z", "o": 100 + i}
                                for i, day in enumerate(self.calendar["sessions"])
                                if day["date"] <= "2026-10-29"]
                            for s in ("PLTR", "SPY")}, "next_page_token": None}
        self.request = {"file": "daily_bars_raw.json", "status": 200,
                        "route": "/v2/stocks/bars", "params": {"timeframe": "1Day", "adjustment": "raw"},
                        "started_at": "2026-10-30T21:00:00Z", "response_received_at": "2026-10-30T21:01:00Z"}
        self.save()

    def save(self):
        path = self.archive / "daily_bars_raw.json"
        path.write_text(json.dumps(self.raw))
        self.request["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.archive / "manifest.json").write_text(json.dumps({"requests": [self.request]}))

    def audit(self, **changes):
        args = {"month": "2026-09", "as_of": "2026-10-31T23:00:00Z", "symbols": ["PLTR"],
                "calendar": self.calendar}
        args.update(changes)
        return audit_archive(self.archive, **args)

    def test_rules_exclude_labor_day_and_keep_twenty_one_september_sessions(self):
        dates = [s["date"] for s in self.calendar["sessions"]]
        self.assertNotIn("2026-09-07", dates)
        report = self.audit()
        self.assertEqual(report["summary"]["decision_sessions_in_month"], 21)
        self.assertEqual(report["summary"]["coverage"]["PLTR"]["10"]["paired_raw_open_endpoints"], 21)
        self.assertEqual(report["summary"]["strict_eligible_labels_created"], 0)

    def test_dst_and_early_close_are_rule_based(self):
        days = {s["date"]: s for s in self.calendar["sessions"]}
        self.assertEqual(days["2026-10-30"]["open_at"], "2026-10-30T13:30:00Z")
        self.assertEqual(days["2026-11-02"]["open_at"], "2026-11-02T14:30:00Z")
        self.assertNotIn("2026-11-26", days)
        self.assertEqual(days["2026-11-27"]["close_at"], "2026-11-27T18:00:00Z")

    def test_tenth_exit_counts_after_entry_and_september_last_tail(self):
        rows = self.audit()["rows"]
        selected = next(r for r in rows if r["decision_session"] == "2026-09-01" and r["horizon_sessions"] == 10)
        self.assertEqual(selected["exit_session"], "2026-09-16")
        exits = {r["horizon_sessions"]: r["exit_session"] for r in rows if r["decision_session"] == "2026-09-30"}
        self.assertEqual(exits, {5: "2026-10-07", 10: "2026-10-14", 20: "2026-10-28"})

    def test_missing_company_or_spy_endpoint_keeps_denominator_and_exposes_dates(self):
        for ticker in ("PLTR", "SPY"):
            with self.subTest(ticker=ticker):
                saved = deepcopy(self.raw)
                self.raw["bars"][ticker] = [r for r in self.raw["bars"][ticker] if not r["t"].startswith("2026-09-16")]
                self.save()
                report = self.audit()
                row = next(r for r in report["rows"] if r["decision_session"] == "2026-09-01" and r["horizon_sessions"] == 10)
                self.assertFalse(row["paired_raw_open_endpoints_present"])
                self.assertIn({"symbol": ticker, "date": "2026-09-16"}, row["missing_or_invalid_open_endpoints"])
                self.assertEqual(report["summary"]["decision_sessions_in_month"], 21)
                self.raw = saved

    def test_invalid_or_null_open_never_counts_as_coverage(self):
        for value in (None, False, 0, -1, float("nan"), float("inf")):
            self.raw["bars"]["PLTR"][0]["o"] = value
            self.save()
            row = self.audit()["rows"][0]
            self.assertFalse(row["paired_raw_open_endpoints_present"])

    def test_response_after_as_of_or_before_start_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.audit(as_of="2026-10-01T23:00:00Z")
        self.request["started_at"] = "2026-10-31T00:00:00Z"
        self.save()
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.audit()

    def test_calendar_maturity_does_not_imply_received_prices(self):
        self.request["started_at"] = "2026-10-01T12:00:00Z"
        self.request["response_received_at"] = "2026-10-01T12:01:00Z"
        self.save()
        rows = self.audit()["rows"]
        last = next(r for r in rows if r["decision_session"] == "2026-09-30" and r["horizon_sessions"] == 10)
        self.assertTrue(last["calendar_horizon_mature_as_of"])
        self.assertFalse(last["paired_raw_open_endpoints_present"])
        self.assertEqual(last["endpoint_open_after_response_receipt"], ["2026-10-14"])

    def test_tampered_response_is_rejected(self):
        (self.archive / "daily_bars_raw.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash-bound"):
            self.audit()

    def test_duplicate_dates_are_rejected(self):
        self.raw["bars"]["PLTR"].append(self.raw["bars"]["PLTR"][0])
        self.save()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.audit()

    def test_bar_on_holiday_is_reported_and_never_creates_session(self):
        self.raw["bars"]["PLTR"].append({"t": "2026-09-07T04:00:00Z", "o": 100})
        self.save()
        report = self.audit()
        self.assertEqual(report["unexpected_bar_dates"], [{"symbol": "PLTR", "date": "2026-09-07", "reason": "bar_on_non_session"}])
        self.assertNotIn("2026-09-07", [r["decision_session"] for r in report["rows"]])

    def test_raw_basis_and_pagination_require_explicit_evidence(self):
        self.request["params"]["adjustment"] = "all"
        self.save()
        with self.assertRaisesRegex(ValueError, "raw"):
            self.audit()
        self.request["params"]["adjustment"] = "raw"
        self.raw["next_page_token"] = "more"
        self.save()
        with self.assertRaisesRegex(ValueError, "pagination"):
            self.audit()

    def test_short_or_modified_calendar_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "twenty"):
            self.audit(calendar=export_calendar("2026-09-01", "2026-10-09"))
        changed = deepcopy(self.calendar)
        changed["sessions"].pop(4)
        with self.assertRaisesRegex(ValueError, "bind"):
            self.audit(calendar=changed)

    def test_report_is_json_serializable_and_no_mutation_or_performance_claim(self):
        before = deepcopy(self.calendar)
        report = self.audit()
        json.dumps(report, allow_nan=False)
        self.assertEqual(before, self.calendar)
        self.assertTrue(report["calendar"]["provenance"]["independent_of_price_dates"])
        self.assertEqual(report["summary"]["model_evaluation_status"], "Not evaluated")
        self.assertFalse(any(r["formal_label_eligible"] for r in report["rows"]))

    def test_cli_preserves_input_and_existing_run(self):
        output = self.root / "output"
        args = ["--archive", str(self.archive), "--month", "2026-09", "--as-of", "2026-10-31T23:00:00Z",
                "--calendar-end", "2026-11-30", "--symbols", "PLTR", "--output-dir", str(output)]
        before = {p.name: p.read_bytes() for p in self.archive.iterdir()}
        main(args)
        saved = (output / "report.json").read_bytes()
        with self.assertRaises(FileExistsError):
            main(args)
        with self.assertRaisesRegex(ValueError, "separate"):
            main(args[:-1] + [str(self.archive / "bad")])
        self.assertEqual(saved, (output / "report.json").read_bytes())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.archive.iterdir()})


if __name__ == "__main__":
    unittest.main()
