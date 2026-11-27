"""Audit provenance boundaries with fabricated archives, never private market data."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_preflight import audit_archives, main


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def article(identifier, created_at, symbols, *, updated_at=None, content="Saved body"):
    return {
        "id": identifier, "created_at": created_at,
        "updated_at": updated_at or created_at,
        "symbols": symbols, "headline": "Fixture company announcement",
        "content": content, "summary": "Fixture summary",
        "source": "fixture-news", "url": f"https://example.invalid/news/{identifier}",
    }


def normalized(raw, receipt=None):
    row = {key: copy.deepcopy(raw[key]) for key in
           ("id", "created_at", "updated_at", "symbols", "headline", "summary", "source", "url")}
    row["text"] = raw["content"]
    if receipt is not None:
        row["retrieved_at"] = receipt
    return row


class DataPreflightTests(unittest.TestCase):
    """Fixtures have sparse bars and late captures on purpose: they cannot be labels."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.primary = self.root / "primary"
        self.legacy = self.root / "legacy"
        self.primary.mkdir()
        self.legacy.mkdir()
        self.receipt = "2026-10-10T20:00:00Z"
        shared = article(100, "2026-09-15T14:00:00Z", ["PLTR", "MSFT"])
        old_rows = [shared,
                    article(101, "2026-10-01T03:30:00Z", ["PLTR"]),
                    article(102, "2026-09-01T03:30:00Z", ["AMZN"])]
        requests = []
        for index in range(4):
            name = f"news_recent_{index:03d}.json"
            write_json(self.legacy / name, {"news": old_rows[index:index + 1]})
            requests.append({
                "file": name, "http_status": 200, "route": "/v1beta1/news",
                "params": {"start": "2026-09-25T00:00:00Z", "limit": 50},
                "observed_at": "2026-10-07T16:00:00Z", "elapsed_seconds": 4.25,
            })
        write_json(self.legacy / "requests.json", requests)
        write_json(self.legacy / "news_normalized.json", [normalized(row) for row in old_rows])
        self.refresh_legacy_hashes()

        current = copy.deepcopy(shared)
        current.update(updated_at="2026-10-09T15:00:00Z", content="Later version of saved body")
        primary_rows = [current,
                        article(200, "2026-09-20T14:00:00Z", ["AMZN"]),
                        article(201, "2026-10-01T16:00:00Z", ["PLTR"])]
        write_json(self.primary / "news_raw_000.json", {"news": primary_rows, "next_page_token": None})
        write_json(self.primary / "news_normalized.json",
                   [normalized(row, self.receipt) for row in primary_rows])
        write_json(self.primary / "daily_bars_raw.json", {
            "bars": {symbol: [
                {"t": "2026-09-01T04:00:00Z", "o": 100.0},
                {"t": "2026-09-30T04:00:00Z", "o": 103.0},
                {"t": "2026-10-01T04:00:00Z", "o": 102.0},
            ] for symbol in ("PLTR", "AMZN", "MSFT", "SPY")},
            "next_page_token": None,
        })
        write_json(self.primary / "manifest.json", {"requests": [
            {"file": "news_raw_000.json", "status": 200, "route": "/v1beta1/news",
             "response_received_at": self.receipt,
             "params": {"start": "2026-09-01T00:00:00Z", "end": self.receipt,
                        "symbols": "PLTR,AMZN,MSFT"}},
            {"file": "daily_bars_raw.json", "status": 200, "route": "/v2/stocks/bars",
             "response_received_at": self.receipt,
             "params": {"timeframe": "1Day", "adjustment": "raw"}},
        ]})
        self.refresh_primary_hashes()

    def refresh_legacy_hashes(self):
        write_json(self.legacy / "hashes.json", {
            path.name: sha256(path) for path in self.legacy.glob("*.json")
            if path.name != "hashes.json"
        })

    def refresh_primary_hashes(self):
        manifest = load_json(self.primary / "manifest.json")
        for request in manifest["requests"]:
            request["sha256"] = sha256(self.primary / request["file"])
        write_json(self.primary / "manifest.json", manifest)

    def audit(self, **changes):
        args = {"symbols": ["PLTR", "AMZN", "MSFT"], "month": "2026-09",
                "as_of": "2026-10-16T12:00:00Z"}
        args.update(changes)
        return audit_archives(self.primary, self.legacy, **args)

    def cli(self, output):
        with contextlib.redirect_stdout(io.StringIO()):
            return main(["--input-dir", str(self.primary), "--legacy-alpaca-dir", str(self.legacy),
                         "--output-dir", str(output), "--month", "2026-09",
                         "--as-of", "2026-10-16T12:00:00Z"])

    def test_missing_calendar_does_not_invent_denominator_maturity_labels_or_performance(self):
        report = self.audit()
        summary = report["summary"]
        for field in ("decision_session_denominator", "ten_session_matchable_fraction",
                      "whole_month_horizon_mature"):
            self.assertIsNone(summary[field])
        self.assertEqual(summary["strict_eligible_labels_created"], 0)
        self.assertEqual(summary["real_model_evaluation_status"], "Not evaluated")
        self.assertIn("independent_versioned_complete_exchange_calendar", report["missing_evidence"])
        self.assertIn("full_price_history_and_ten_session_label_tail", report["missing_evidence"])
        self.assertNotIn("labels", report)

    def test_legacy_request_start_and_elapsed_time_never_become_receipt(self):
        old = [row for row in self.audit()["selected_capture_audit"] if row["archive"] == "legacy"]
        self.assertEqual(len(old), 2)
        for row in old:
            self.assertIsNone(row["response_received_at"])
            self.assertIsNone(row["capture_available_at"])
            self.assertFalse(row["historical_availability_verified"])
            self.assertIn("request start is not receipt", row["availability_reason"])

    def test_captures_preserve_both_versions_without_backdating_later_text(self):
        captures = [row for row in self.audit()["selected_capture_audit"] if row["article_id"] == "100"]
        self.assertEqual(len(captures), 2)
        self.assertEqual(len({row["version_sha256"] for row in captures}), 2)
        later = next(row for row in captures if row["archive"] == "primary")
        self.assertEqual(later["response_received_at"], self.receipt)
        self.assertEqual(later["updated_at"], "2026-10-09T15:00:00Z")
        self.assertFalse(later["historical_availability_verified"])
        self.assertNotEqual(later["capture_available_at"], later["published_at"])

    def test_identical_recaptures_do_not_inflate_distinct_versions_or_article_ids(self):
        raw = load_json(self.primary / "news_raw_000.json")
        original = load_json(self.legacy / "news_recent_000.json")["news"][0]
        raw["news"][0] = original
        write_json(self.primary / "news_raw_000.json", raw)
        rows = load_json(self.primary / "news_normalized.json")
        rows[0] = normalized(original, self.receipt)
        write_json(self.primary / "news_normalized.json", rows)
        self.refresh_primary_hashes()
        summary = self.audit()["summary"]
        self.assertEqual(summary["selected_capture_rows"], 4)
        self.assertEqual(summary["selected_unique_article_ids"], 3)
        self.assertEqual(summary["selected_distinct_versions"], 3)

    def test_multicompany_source_keeps_one_cross_capture_group_without_claiming_event_identity(self):
        report = self.audit()
        shared = [row for row in report["selected_capture_audit"] if row["article_id"] == "100"]
        self.assertTrue(all(row["symbols"] == ["MSFT", "PLTR"] for row in shared))
        self.assertEqual({row["source_group_id"] for row in shared}, {"alpaca-news:100"})
        self.assertTrue(all("event_group_id" not in row for row in shared))
        self.assertEqual(report["summary"]["selected_article_company_pairs"], 4)
        self.assertIn("event_grouping_across_articles_and_companies_and_independent_review",
                      report["missing_evidence"])

    def test_publication_month_uses_new_york_boundary_not_utc_date(self):
        selected = self.audit()["selected_capture_audit"]
        self.assertIn("101", {row["article_id"] for row in selected})
        self.assertNotIn("102", {row["article_id"] for row in selected})
        self.assertNotIn("201", {row["article_id"] for row in selected})
        self.assertEqual(next(row for row in selected if row["article_id"] == "101")
                         ["publication_date_ny"], "2026-09-30")

    def test_raw_openings_never_become_total_return_indices(self):
        report = self.audit()
        for company in report["companies_and_benchmark"].values():
            self.assertEqual(company["return_basis"], "raw_price")
            self.assertEqual(company["observed_daily_dates_in_month"], ["2026-09-01", "2026-09-30"])
            self.assertEqual(company["observed_daily_dates_after_month"], ["2026-10-01"])
            self.assertIn("not actual regular open", company["bar_timestamp_policy"])
        self.assertIn("corporate_actions_and_consistent_open_total_return_indices", report["missing_evidence"])

    def test_as_of_before_a_capture_is_rejected_even_if_publication_month_is_complete(self):
        with self.assertRaisesRegex(ValueError, "capture|receipt"):
            self.audit(as_of="2026-10-08T12:00:00Z")

    def test_unfinished_month_is_rejected_at_new_york_month_boundary(self):
        with self.assertRaisesRegex(ValueError, "not yet complete"):
            self.audit(as_of="2026-10-01T03:59:59Z")

    def test_primary_raw_response_tampering_is_rejected(self):
        path = self.primary / "news_raw_000.json"
        raw = load_json(path)
        raw["news"][0]["content"] = "Tampered body"
        write_json(path, raw)
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.audit()

    def test_missing_legacy_raw_hash_is_rejected(self):
        hashes = load_json(self.legacy / "hashes.json")
        hashes.pop("news_recent_003.json")
        write_json(self.legacy / "hashes.json", hashes)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.audit()

    def test_missing_raw_response_is_rejected_without_creating_output(self):
        (self.primary / "news_raw_000.json").unlink()
        output = self.root / "not-created"
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.cli(output)
        self.assertFalse(output.exists())

    def test_legacy_normalized_rehash_cannot_replace_original_body(self):
        rows = load_json(self.legacy / "news_normalized.json")
        rows[0]["text"] = "Changed normalization with a freshly recomputed hash"
        write_json(self.legacy / "news_normalized.json", rows)
        self.refresh_legacy_hashes()
        with self.assertRaisesRegex(ValueError, "text differs"):
            self.audit()

    def test_primary_normalized_body_or_receipt_cannot_differ_from_raw_evidence(self):
        path = self.primary / "news_normalized.json"
        original = load_json(path)
        for field, replacement in (("text", "Invented body"),
                                   ("retrieved_at", "2026-09-15T14:00:01Z")):
            with self.subTest(field=field):
                rows = copy.deepcopy(original)
                rows[0][field] = replacement
                write_json(path, rows)
                with self.assertRaisesRegex(ValueError, "body or receipt"):
                    self.audit()

    def test_primary_normalized_summary_and_source_must_match_saved_response(self):
        path = self.primary / "news_normalized.json"
        original = load_json(path)
        for field in ("summary", "source"):
            with self.subTest(field=field):
                rows = copy.deepcopy(original)
                rows[0][field] = "An invented normalized field"
                write_json(path, rows)
                with self.assertRaises(ValueError):
                    self.audit()

    def test_price_hash_tampering_is_rejected_before_any_coverage_is_reported(self):
        path = self.primary / "daily_bars_raw.json"
        raw = load_json(path)
        raw["bars"]["PLTR"][0]["o"] = 9999
        write_json(path, raw)
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.audit()

    def test_two_bar_timestamps_on_same_new_york_date_are_not_two_sessions(self):
        path = self.primary / "daily_bars_raw.json"
        raw = load_json(path)
        raw["bars"]["PLTR"].append({"t": "2026-09-01T13:30:00Z", "o": 101})
        write_json(path, raw)
        self.refresh_primary_hashes()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.audit()

    def test_bar_after_actual_receipt_is_rejected_even_with_valid_hash(self):
        path = self.primary / "daily_bars_raw.json"
        raw = load_json(path)
        raw["bars"]["PLTR"].append({"t": "2026-10-11T04:00:00Z", "o": 101})
        write_json(path, raw)
        self.refresh_primary_hashes()
        with self.assertRaisesRegex(ValueError, "Future"):
            self.audit()

    def test_invalid_price_values_cannot_enter_coverage(self):
        path = self.primary / "daily_bars_raw.json"
        original = load_json(path)
        for value in (True, 0, -1, "100", float("nan"), float("inf")):
            with self.subTest(value=value):
                raw = copy.deepcopy(original)
                raw["bars"]["PLTR"][0]["o"] = value
                write_json(path, raw)
                self.refresh_primary_hashes()
                with self.assertRaisesRegex(ValueError, "opening"):
                    self.audit()

    def test_new_private_output_is_reproducible_and_preserves_every_input_byte(self):
        before = {path: path.read_bytes() for directory in (self.primary, self.legacy)
                  for path in directory.iterdir()}
        output = self.root / "new-run"
        report = self.cli(output)
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual((output / ".gitignore").read_text(), "*\n")
        self.assertEqual(output.stat().st_mode & 0o777, 0o700)
        self.assertEqual(report, load_json(output / "preflight.json"))
        manifest = load_json(output / "run_manifest.json")
        self.assertEqual(manifest["preflight_sha256"], sha256(output / "preflight.json"))
        for source, expected in manifest["input_files_sha256"].items():
            self.assertEqual(sha256(Path(source)), expected)
        serialized = (output / "preflight.json").read_text()
        self.assertNotIn("Later version of saved body", serialized)
        self.assertNotIn("Fixture summary", serialized)

    def test_existing_output_is_preserved(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "previous-result"
        sentinel.write_text("Keep this evidence")
        with self.assertRaises(FileExistsError):
            self.cli(output)
        self.assertEqual(sentinel.read_text(), "Keep this evidence")
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_output_inside_either_archive_is_rejected(self):
        for archive in (self.primary, self.legacy):
            with self.subTest(archive=archive):
                output = archive / "new-run"
                with self.assertRaisesRegex(ValueError, "outside"):
                    self.cli(output)
                self.assertFalse(output.exists())

    def test_direct_output_symlink_and_symlink_alias_into_input_are_rejected(self):
        destination = self.root / "destination"
        direct = self.root / "direct-symlink"
        direct.symlink_to(destination, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.cli(direct)
        self.assertFalse(destination.exists())
        alias = self.root / "archive-alias"
        alias.symlink_to(self.primary, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.cli(alias / "new-run")
        self.assertFalse((self.primary / "new-run").exists())


if __name__ == "__main__":
    unittest.main()
