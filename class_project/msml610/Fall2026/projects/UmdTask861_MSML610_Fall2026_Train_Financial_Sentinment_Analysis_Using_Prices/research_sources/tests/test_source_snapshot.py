"""Bind artificial source snapshots without reading private research archives."""

import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_preflight import _snapshot_source_records, audit_source_records


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cutoff = "2026-10-12T16:00:00Z"
        self.received = "2026-10-12T16:21:10Z"
        self.started = "2026-10-12T16:21:09Z"
        self.raw_news = {
            "id": 1, "headline": "Artificial company announcement",
            "content": "<p>Artificial &amp; independently retained body.</p>",
            "summary": "Artificial summary", "symbols": ["DEMO"],
            "created_at": "2026-10-12T14:00:00Z",
            "updated_at": "2026-10-12T14:10:00Z",
            "url": "https://example.invalid/news/1", "source": "fixture",
        }
        self.news = {key: self.raw_news[key] for key in (
            "id", "headline", "symbols", "created_at", "updated_at", "url", "source")}
        self.news.update(text="Artificial & independently retained body.", received_at=self.received)
        self.comment_publication = "2026-10-12T16:08:08+00:00"
        epoch = str(int(datetime.fromisoformat(self.comment_publication).timestamp()))
        self.raw_post = {"id": "post-1", "title": "<b>Artificial bullish comment</b>",
                         "desc": "<p>Artificial &amp; unverified opinion.</p>", "publish_time": epoch}
        title, excerpt = "Artificial bullish comment", "Artificial & unverified opinion."
        self.futu = {
            "query_symbol": "DEMO", "post_id": "post-1", "title": title, "excerpt": excerpt,
            "raw_publish_time": epoch, "published_at": self.comment_publication,
            "text_sha256": hashlib.sha256((title + " " + excerpt).encode()).hexdigest(),
            "request_started_at": self.started, "response_received_at": self.received,
            "first_seen_in_this_audit_at": self.received,
        }
        self.sec_rows = [
            {"filingDate": "2025-10-29", "acceptanceDateTime": "2025-10-30T00:50:29Z",
             "form": "10-K", "accessionNumber": "fixture-annual-1",
             "primaryDocument": "fy-2026-label-is-not-an-event-date.htm"},
            {"filingDate": "2026-10-09", "acceptanceDateTime": "2026-10-09T20:00:00Z",
             "form": "4", "accessionNumber": "fixture-insider-1", "primaryDocument": "form4.htm"},
        ]
        recent = {key: [row[key] for row in self.sec_rows] for key in self.sec_rows[0]}
        recent["reportDate"] = ["2025-06-30", "2026-10-08"]
        self.response_names = ["news_raw_000.json", "sec_submissions.json",
                               "sec_selected_document.html", "sandisk_ir.html",
                               "cxmt_news.html", "nanya_news.html"]
        write_json(self.root / "source_capture/news_raw_000.json", {"news": [self.raw_news]})
        write_json(self.root / "source_capture/news_normalized.json", [self.news])
        write_json(self.root / "source_capture/sec_submissions.json", {
            "cik": "fixture", "tickers": ["DEMO"], "fiscalYearEnd": "0630",
            "unverified_fiscal_year_label": "FY2026", "filings": {"recent": recent}})
        write_json(self.root / "source_capture/sec_rows.json", self.sec_rows)
        for name in self.response_names[2:]:
            (self.root / "source_capture" / name).write_text(
                "<html>Artificial unparsed response, never an eligible transcript.</html>", encoding="utf-8")
        requests = [{"file": name, "request_started_at": self.started, "received_at": self.received,
                     "http_status": 403 if name in self.response_names[2:4] else 200}
                    for name in self.response_names]
        write_json(self.root / "source_capture/requests.json", requests)
        self.refresh_request_hashes()
        write_json(self.root / "futu_current/SNDK_raw.json", {"code": 0, "data": [self.raw_post]})
        write_json(self.root / "futu_current/SNDK_request.json", {
            "request_started_at": self.started, "response_received_at": self.received,
            "http_status": 200, "response_bytes": (self.root / "futu_current/SNDK_raw.json").stat().st_size,
            "sha256": digest(self.root / "futu_current/SNDK_raw.json")})
        write_json(self.root / "futu_current/all_normalized.json", [self.futu])
        write_json(self.root / "daily.json", {"bars": {}})
        write_json(self.root / "intraday.json", {"bars": {}})
        write_json(self.root / "manifest.json", {"analysis_cutoff": self.cutoff})
        write_json(self.root / "news_context.json", {
            "manual_summary": "MANUAL_TEXT_MUST_NOT_BECOME_REPORT_OR_TRANSCRIPT"})
        write_json(self.root / "context_sources.json", [{"id": "ir", "records": 1}])
        self.generation_names = ["daily.json", "intraday.json", "manifest.json",
                                 "futu_current/all_normalized.json", "news_context.json", "context_sources.json"]
        self.refresh_generation_hashes()

    def refresh_generation_hashes(self):
        write_json(self.root / "generation_audit.json", {
            "analysis_cutoff": self.cutoff,
            "inputs": {name: {"present": True, "bytes": (self.root / name).stat().st_size,
                              "sha256": digest(self.root / name)} for name in self.generation_names}})

    def refresh_request_hashes(self):
        path = self.root / "source_capture/requests.json"
        requests = read_json(path)
        for request in requests:
            source = self.root / "source_capture" / request["file"]
            request.update(bytes=source.stat().st_size, sha256=digest(source))
        write_json(path, requests)

    def snapshot(self):
        return _snapshot_source_records(self.root)

    def test_minimal_fixture_binds_all_inputs_without_exporting_source_text(self):
        records, fingerprints, coverage = self.snapshot()
        self.assertEqual(len(records), 3)
        self.assertEqual(len(fingerprints), 18)
        self.assertEqual({row["source_family"] for row in records},
                         {"news", "investor_comment", "financial_report"})
        for path, expected in fingerprints.items():
            self.assertEqual(digest(Path(path)), expected)
        for row in records:
            self.assertEqual(row["event_ids"], [])
            self.assertIsNone(row["event_identity_evidence_id"])
            self.assertFalse({"text", "title", "excerpt", "headline", "summary"} & row.keys())
            observation = row["captures"][0]
            self.assertEqual(digest(Path(observation["retrieval_evidence_id"])),
                             observation["response_sha256"])
            self.assertIn(observation["request_evidence_id"], fingerprints)
        self.assertNotIn("MANUAL_TEXT_MUST_NOT_BECOME_REPORT_OR_TRANSCRIPT",
                         json.dumps([records, coverage]))

    def test_all_six_generation_inputs_are_hash_bound(self):
        for name in self.generation_names:
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "integrity"):
                    self.snapshot()
                path.write_bytes(original)

    def test_generation_audit_cannot_omit_an_input(self):
        path = self.root / "generation_audit.json"
        audit = read_json(path)
        del audit["inputs"]["daily.json"]
        write_json(path, audit)
        with self.assertRaisesRegex(ValueError, "all six"):
            self.snapshot()

    def test_all_six_source_responses_are_hash_bound(self):
        for name in self.response_names:
            with self.subTest(name=name):
                path = self.root / "source_capture" / name
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "integrity"):
                    self.snapshot()
                path.write_bytes(original)

    def test_news_normalization_and_exact_receipt_must_match_raw(self):
        for field, value in (("text", "Invented body"), ("headline", "Invented headline"),
                             ("received_at", self.started), ("updated_at", "2026-10-12T15:00:00Z")):
            with self.subTest(field=field):
                row = copy.deepcopy(self.news)
                row[field] = value
                write_json(self.root / "source_capture/news_normalized.json", [row])
                with self.assertRaisesRegex(ValueError, "captured text or receipt"):
                    self.snapshot()
        write_json(self.root / "source_capture/news_normalized.json", [self.news])

    def test_rehashed_raw_news_still_must_match_normalized_version(self):
        raw = copy.deepcopy(self.raw_news)
        raw["content"] = "Different later content"
        write_json(self.root / "source_capture/news_raw_000.json", {"news": [raw]})
        self.refresh_request_hashes()
        with self.assertRaisesRegex(ValueError, "captured text or receipt"):
            self.snapshot()

    def test_news_revision_retains_original_time_and_changes_version_hash(self):
        initial = next(row for row in self.snapshot()[0] if row["source_family"] == "news")
        raw, normalized = copy.deepcopy(self.raw_news), copy.deepcopy(self.news)
        raw.update(content="A revised artificial text", updated_at="2026-10-12T16:15:00Z")
        normalized.update(text=raw["content"], updated_at=raw["updated_at"])
        write_json(self.root / "source_capture/news_raw_000.json", {"news": [raw]})
        write_json(self.root / "source_capture/news_normalized.json", [normalized])
        self.refresh_request_hashes()
        revised = next(row for row in self.snapshot()[0] if row["source_family"] == "news")
        self.assertEqual(revised["original_published_at"], self.raw_news["created_at"])
        self.assertEqual(revised["version_published_at"], raw["updated_at"])
        self.assertNotEqual(revised["text_version_sha256"], initial["text_version_sha256"])
        audited = audit_source_records([revised], cutoff=self.cutoff, as_of="2026-10-13T00:00:00Z")
        self.assertIn("version_published_at_after_cutoff", audited["records"][0]["reasons"])

    def test_news_cannot_have_a_version_later_than_its_response(self):
        raw, normalized = copy.deepcopy(self.raw_news), copy.deepcopy(self.news)
        raw["updated_at"] = normalized["updated_at"] = "2026-10-12T17:00:00Z"
        write_json(self.root / "source_capture/news_raw_000.json", {"news": [raw]})
        write_json(self.root / "source_capture/news_normalized.json", [normalized])
        self.refresh_request_hashes()
        with self.assertRaisesRegex(ValueError, "version is later"):
            self.snapshot()

    def test_futu_raw_response_is_independently_hash_bound(self):
        path = self.root / "futu_current/SNDK_raw.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.snapshot()

    def test_futu_text_epoch_and_receipt_fields_remain_bound_after_rehash(self):
        changes = (("title", "Invented title"), ("excerpt", "Invented excerpt"),
                   ("text_sha256", "0" * 64), ("raw_publish_time", "1"),
                   ("published_at", "2026-10-12T14:00:00Z"),
                   ("request_started_at", self.cutoff), ("response_received_at", self.cutoff),
                   ("first_seen_in_this_audit_at", self.cutoff))
        for field, value in changes:
            with self.subTest(field=field):
                row = copy.deepcopy(self.futu)
                row[field] = value
                write_json(self.root / "futu_current/all_normalized.json", [row])
                self.refresh_generation_hashes()
                with self.assertRaisesRegex(ValueError, "Futu text or timing"):
                    self.snapshot()
        write_json(self.root / "futu_current/all_normalized.json", [self.futu])
        self.refresh_generation_hashes()

    def test_futu_unix_seconds_are_utc_and_after_cutoff_stays_after_cutoff(self):
        records, _, coverage = self.snapshot()
        comment = next(row for row in records if row["source_family"] == "investor_comment")
        self.assertEqual(comment["original_published_at"], self.comment_publication)
        self.assertEqual(comment["first_seen_in_this_audit_at"], self.received)
        self.assertFalse(coverage["historical_first_receipt_verified"])
        comment["first_seen_in_this_audit_at"] = "2026-10-12T14:00:00Z"
        result = audit_source_records([comment], cutoff=self.cutoff, as_of="2026-10-13T00:00:00Z")
        row = result["records"][0]
        self.assertEqual(row["first_actual_receipt_at"], self.received)
        self.assertFalse(row["source_contract_eligible"])
        self.assertIn("original_published_at_after_cutoff", row["reasons"])
        self.assertIn("text_version_received_after_cutoff", row["reasons"])

    def test_futu_original_title_and_desc_must_match_even_with_updated_raw_hash(self):
        for field in ("title", "desc"):
            with self.subTest(field=field):
                raw = copy.deepcopy(self.raw_post)
                raw[field] = "Artificial revised text"
                path = self.root / "futu_current/SNDK_raw.json"
                write_json(path, {"code": 0, "data": [raw]})
                request_path = self.root / "futu_current/SNDK_request.json"
                request = read_json(request_path)
                request.update(sha256=digest(path), response_bytes=path.stat().st_size)
                write_json(request_path, request)
                with self.assertRaisesRegex(ValueError, "Futu text or timing"):
                    self.snapshot()

    def test_annual_acceptance_date_report_period_and_fiscal_label_are_separate(self):
        annual = next(row for row in self.snapshot()[0] if row["source_family"] == "financial_report")
        self.assertEqual(annual["original_published_at"], "2025-10-30T00:50:29+00:00")
        self.assertEqual(annual["event_calendar_date"], "2025-10-29")
        self.assertEqual(annual["fiscal_period"], "2025-06-30")
        self.assertNotIn("2026", annual["event_calendar_date"])
        self.assertFalse(annual["text_available"])
        self.assertIsNone(annual["text_version_sha256"])
        self.assertIsNone(annual["captures"][0]["text_version_sha256"])

    def test_other_filings_failed_html_and_industry_capture_are_not_report_text(self):
        records, _, coverage = self.snapshot()
        self.assertEqual(coverage["sec_metadata_records"], 2)
        self.assertEqual(coverage["financial_report_metadata_records"], 1)
        self.assertEqual(coverage["other_sec_metadata_records"], 1)
        self.assertEqual(coverage["financial_report_text_records"], 0)
        self.assertEqual(coverage["earnings_call_records"], 0)
        self.assertEqual(len(coverage["failed_body_retrievals"]), 2)
        self.assertEqual(len(coverage["unparsed_html_retrievals"]), 2)
        self.assertNotIn("sec-accession:fixture-insider-1", {row["source_id"] for row in records})
        self.assertNotIn("earnings_call", {row["source_family"] for row in records})
        for retrieval in coverage["failed_body_retrievals"] + coverage["unparsed_html_retrievals"]:
            self.assertFalse(retrieval["text_available"])
            self.assertIsNone(retrieval["event_calendar_date"])
            self.assertIsNone(retrieval["text_version_sha256"])
        self.assertEqual({row["status"] for row in coverage["failed_body_retrievals"]}, {403})

    def test_sec_selected_metadata_must_match_raw_submission(self):
        rows = copy.deepcopy(self.sec_rows)
        rows[0]["acceptanceDateTime"] = "2026-01-01T00:00:00Z"
        write_json(self.root / "source_capture/sec_rows.json", rows)
        with self.assertRaisesRegex(ValueError, "SEC metadata differs"):
            self.snapshot()


if __name__ == "__main__":
    unittest.main()
