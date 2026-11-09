"""Verify the bounded issuer capture with artificial HTML and no network access.

These fixtures test retention and eligibility contracts, never market behavior.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import issuer_event


RELEASE_TITLE = "Microsoft Cloud and AI Strength Drives First Quarter Results"
CALL_TITLE = "Microsoft FY26 First Quarter Earnings Conference Call"
CUTOFF = "2025-10-30T12:45:00Z"


def release_html(date="October 29, 2025", *, ending=True, incidental=""):
    return ("<!doctype html><html><head><script type='application/ld+json'>"
            '{"datePublished":"2025-10-29T20:00:00Z"}</script></head><body>'
            "<nav>Navigation outside the report</nav><div id='pressreleasecontent'>"
            f"<h1>{RELEASE_TITLE}</h1><p>REDMOND, Wash. — {date} — Quarterly results.</p>"
            "<p>For the quarter ended September 30, 2025</p>"
            "<table><tr><td>Operating income</td><td>Fixture value</td></tr></table>"
            f"<p>{incidental}</p>"
            + ("<p>IMPORTANT NOTICE TO USERS</p>" if ending else "")
            + "</div><footer>Footer outside the report</footer></body></html>").encode()


def call_html(date="October 29, 2025", *, ending=True, incidental=""):
    return ("<html><head><script type='application/ld+json'>"
            '{"datePublished":"2025-10-29T20:01:00Z"}</script></head><body>'
            "<nav>Navigation outside the call</nav>"
            f"<h1>{CALL_TITLE}</h1><p>Wednesday, {date}</p><p>2:30 PM PT</p>"
            "<p>JONATHAN NEILSON: Welcome to our artificial test call.</p>"
            f"<p>Q&amp;A</p><p>{incidental}</p>"
            + ("<p>END</p>" if ending else "")
            + "<footer>Footer outside the call</footer></body></html>").encode()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(encoded):
    return hashlib.sha256(encoded).hexdigest()


class FakeResponse:
    def __init__(self, encoded, *, status=200, headers=None, on_read=None):
        self.encoded = encoded
        self.status = status
        self.headers = headers or {
            "Content-Type": "text/html; charset=utf-8",
            "Date": "Sat, 17 Oct 2026 16:00:00 GMT",
            "Last-Modified": "Wed, 29 Oct 2025 20:01:00 GMT",
            "ETag": '"fixture-etag"',
            "Set-Cookie": "unneeded-cookie-not-retained",
        }
        self.on_read = on_read
        self.read_limits = []
        self.closed = False

    def read(self, limit):
        self.read_limits.append(limit)
        if self.on_read:
            self.on_read()
        return self.encoded[:limit]

    def getcode(self):
        return self.status

    def close(self):
        self.closed = True


class CaptureClock:
    def __init__(self, start=None):
        self.current = start or datetime(2026, 10, 17, 16, 0, tzinfo=timezone.utc)
        self.events = []

    def __call__(self):
        self.events.append("clock")
        self.current += timedelta(seconds=1)
        return self.current


class IssuerEventTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def capture(self, *, name="capture", responses=None, clock=None):
        clock = clock or CaptureClock()
        responses = responses or [FakeResponse(release_html()), FakeResponse(call_html())]
        fetched = []

        def fetch(url):
            fetched.append(url)
            response = responses[len(fetched) - 1]
            if isinstance(response, Exception):
                raise response
            return response

        output = self.root / name
        manifest = issuer_event.capture_event(output, fetch=fetch, clock=clock)
        return output, manifest, fetched, responses

    def audit(self, capture, *, name="audit"):
        output = self.root / name
        report = issuer_event.audit_capture(
            capture, output, cutoff=CUTOFF,
            clock=CaptureClock(datetime(2026, 10, 17, 17, 0, tzinfo=timezone.utc)))
        return output, report

    def alter_manifest(self, capture, change):
        path = capture / "manifest.json"
        value = load_json(path)
        change(value)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def test_capture_requests_exactly_two_fixed_documents_once_and_retains_original_bytes(self):
        output, manifest, fetched, responses = self.capture()
        self.assertEqual(fetched, list(issuer_event.SOURCES.values()))
        self.assertEqual(manifest["status"], "complete")
        self.assertFalse(manifest["historical_pipeline_receipt_verified"])
        self.assertEqual((output / ".gitignore").read_text(), "*\n")
        for row, response in zip(manifest["requests"], responses):
            self.assertEqual(row["method"], "GET")
            raw = (output / row["raw_file"]).read_bytes()
            self.assertEqual(raw, response.encoded)
            self.assertEqual(row["bytes"], len(raw))
            self.assertEqual(row["sha256"], digest(raw))
            self.assertEqual(row["http_status"], 200)
            self.assertEqual(row["headers"], {key: response.headers[key]
                                             for key in issuer_event.HEADER_ALLOWLIST})
            self.assertEqual(response.read_limits, [issuer_event.MAX_RESPONSE_BYTES + 1])
            self.assertTrue(response.closed)
        self.assertEqual(load_json(output / "manifest.json"), manifest)

    def test_receipt_clock_is_sampled_after_body_read_and_differs_from_request_start(self):
        clock = CaptureClock()
        responses = [FakeResponse(release_html(), on_read=lambda: clock.events.append("read-release")),
                     FakeResponse(call_html(), on_read=lambda: clock.events.append("read-call"))]
        _, manifest, _, _ = self.capture(responses=responses, clock=clock)
        self.assertEqual(clock.events,
                         ["clock", "clock", "read-release", "clock", "clock", "read-call", "clock", "clock"])
        for row in manifest["requests"]:
            self.assertGreater(row["response_received_at"], row["request_started_at"])

    def test_default_fetch_uses_get_without_authorization_or_redirects(self):
        response = FakeResponse(b"artificial response")
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(issuer_event, "build_opener", return_value=opener) as builder:
            returned = issuer_event._fetch(issuer_event.SOURCES["financial_report"])
        self.assertIs(returned, response)
        self.assertIsInstance(builder.call_args.args[0], issuer_event._NoRedirect)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.full_url, issuer_event.SOURCES["financial_report"])
        self.assertFalse(any(key.lower() in {"authorization", "cookie", "x-api-key"}
                             for key, _ in request.header_items()))
        self.assertIsNone(issuer_event._NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.invalid"))

    def test_partial_network_failure_is_not_retried_and_other_source_is_attempted(self):
        output, manifest, fetched, _ = self.capture(responses=[OSError("fixture unavailable"), FakeResponse(call_html())])
        self.assertEqual(fetched, list(issuer_event.SOURCES.values()))
        self.assertEqual(manifest["status"], "partial")
        failed = manifest["requests"][0]
        self.assertIsNone(failed["response_received_at"])
        self.assertIsNone(failed["sha256"])
        self.assertIsNone(failed["raw_file"])
        self.assertEqual(failed["error_type"], "OSError")
        self.assertFalse((output / "raw/financial_report.html").exists())
        _, report = self.audit(output)
        self.assertEqual(report["capture_attempts"], 2)
        self.assertEqual(report["verified_body_documents"], 1)
        self.assertIn("financial_report", report["body_failures"])

    def test_http_failure_body_and_status_are_retained_but_not_eligible_text(self):
        encoded = b"<html>Artificial access denied page</html>"
        capture, manifest, _, responses = self.capture(responses=[FakeResponse(encoded, status=403), FakeResponse(call_html())])
        row = manifest["requests"][0]
        self.assertEqual(row["http_status"], 403)
        self.assertEqual((capture / row["raw_file"]).read_bytes(), encoded)
        self.assertTrue(responses[0].closed)
        _, report = self.audit(capture)
        self.assertEqual(report["verified_body_documents"], 1)
        self.assertIn("HTTP failure", report["body_failures"]["financial_report"])

    def test_truncated_response_is_bounded_retained_and_refused_as_body(self):
        with mock.patch.object(issuer_event, "MAX_RESPONSE_BYTES", 48):
            capture, manifest, _, responses = self.capture()
        self.assertEqual(manifest["status"], "partial")
        for row, response in zip(manifest["requests"], responses):
            self.assertTrue(row["truncated"])
            self.assertEqual(row["bytes"], 48)
            self.assertEqual(response.read_limits, [49])
            self.assertEqual((capture / row["raw_file"]).read_bytes(), response.encoded[:48])
        _, report = self.audit(capture)
        self.assertEqual(report["verified_body_documents"], 0)

    def test_release_dateline_is_date_precision_only_and_http_metadata_is_not_publication(self):
        text, evidence = issuer_event.extract_official_body(release_html(), "financial_report")
        self.assertTrue(text.startswith(RELEASE_TITLE + "\n"))
        self.assertNotIn("Navigation", text)
        self.assertNotIn("Footer", text)
        self.assertNotIn("datePublished", text)
        self.assertEqual(evidence["original_publication_date"], "2025-10-29")
        self.assertEqual(evidence["original_publication_precision"], "date")
        self.assertEqual(evidence["date_role"], "release_dateline")
        self.assertIsNone(evidence["original_published_at"])
        self.assertIsNone(evidence["version_published_at"])
        self.assertIsNone(evidence["historical_pipeline_response_received_at"])
        hit = evidence["date_evidence"]
        self.assertEqual(text[hit["start"]:hit["end"]], hit["text"])

    def test_call_event_date_and_webcast_time_are_not_transcript_publication(self):
        text, evidence = issuer_event.extract_official_body(call_html(), "earnings_call")
        self.assertTrue(text.startswith(CALL_TITLE + "\n"))
        self.assertTrue(text.endswith("END\n"))
        self.assertNotIn("Navigation", text)
        self.assertNotIn("Footer", text)
        self.assertIsNone(evidence["original_publication_date"])
        self.assertEqual(evidence["original_publication_precision"], "unknown")
        self.assertEqual(evidence["date_role"], "call_event_date_only")
        self.assertIsNone(evidence["original_published_at"])
        self.assertEqual(evidence["event_calendar_date"], "2025-10-29")
        self.assertEqual(evidence["fiscal_period"], "FY2026 Q1")
        hit = evidence["date_evidence"]
        self.assertEqual(text[hit["start"]:hit["end"]], hit["text"])

    def test_release_title_split_by_a_line_break_keeps_body_and_date_offsets(self):
        encoded = release_html().replace(
            RELEASE_TITLE.encode(), b"Microsoft Cloud and AI Strength<br>Drives First Quarter Results")
        text, evidence = issuer_event.extract_official_body(encoded, "financial_report")
        self.assertTrue(text.startswith("Microsoft Cloud and AI Strength\nDrives First Quarter Results\n"))
        hit = evidence["date_evidence"]
        self.assertEqual(text[hit["start"]:hit["end"]], "October 29, 2025")

    def test_fiscal_label_cannot_supply_missing_calendar_date(self):
        for family, encoded in (("financial_report", release_html("Date unavailable")),
                                ("earnings_call", call_html("Date unavailable"))):
            with self.subTest(family=family):
                with self.assertRaises(ValueError):
                    issuer_event.extract_official_body(encoded, family)

    def test_wrong_event_date_cannot_be_rescued_by_incidental_expected_date(self):
        for family, encoded in (
                ("financial_report", release_html("October 29, 2026", incidental="Comparison event October 29, 2025")),
                ("earnings_call", call_html("October 29, 2026", incidental="Previous call October 29, 2025"))):
            with self.subTest(family=family):
                with self.assertRaises(ValueError):
                    issuer_event.extract_official_body(encoded, family)

    def test_missing_end_marker_is_failure_without_page_chrome_fallback(self):
        for family, encoded in (("financial_report", release_html(ending=False)),
                                ("earnings_call", call_html(ending=False))):
            with self.subTest(family=family):
                with self.assertRaises(ValueError):
                    issuer_event.extract_official_body(encoded, family)

    def test_navigation_and_captcha_pages_are_not_issuer_documents(self):
        for family in issuer_event.SOURCES:
            for encoded in (b"<nav>Microsoft earnings investor relations</nav>",
                            b"<h1>Verify you are human</h1><p>CAPTCHA challenge</p>"):
                with self.subTest(family=family, encoded=encoded):
                    with self.assertRaises(ValueError):
                        issuer_event.extract_official_body(encoded, family)

    def test_duplicate_document_title_is_ambiguous_and_refused(self):
        for family, encoded, title in (("financial_report", release_html(), RELEASE_TITLE),
                                       ("earnings_call", call_html(), CALL_TITLE)):
            with self.subTest(family=family):
                encoded = encoded.replace(("<h1>" + title).encode(), ("<h1>" + title + "</h1><h1>" + title).encode())
                with self.assertRaises(ValueError):
                    issuer_event.extract_official_body(encoded, family)

    def test_financial_report_requires_declared_body_root_period_and_table(self):
        variants = (release_html().replace(b"pressreleasecontent", b"unrelatedcontent"),
                    release_html().replace(b"September 30, 2025", b"Some other period"),
                    release_html().replace(b"Operating income", b"No financial table"))
        for encoded in variants:
            with self.subTest(encoded=encoded):
                with self.assertRaises(ValueError):
                    issuer_event.extract_official_body(encoded, "financial_report")

    def test_transcript_requires_speaker_and_qa_not_only_event_heading(self):
        for encoded in (call_html().replace(b"JONATHAN NEILSON:", b"No speaker:"),
                        call_html().replace(b"Q&amp;A", b"No questions")):
            with self.subTest(encoded=encoded):
                with self.assertRaises(ValueError):
                    issuer_event.extract_official_body(encoded, "earnings_call")

    def test_unknown_source_family_and_invalid_utf8_are_not_silently_extracted(self):
        with self.assertRaises(ValueError):
            issuer_event.extract_official_body(release_html(), "news")
        with self.assertRaises(UnicodeError):
            issuer_event.extract_official_body(b"\xff\xfe", "earnings_call")

    def test_historical_cutoff_excludes_current_receipts_and_creates_no_labels_or_model_results(self):
        capture, _, _, _ = self.capture()
        before = {str(path.relative_to(capture)): path.read_bytes()
                  for path in capture.rglob("*") if path.is_file()}
        output, report = self.audit(capture)
        self.assertEqual(report["verified_body_documents"], 2)
        audit = report["source_eligibility"]
        self.assertEqual(audit["eligible_source_records"], 0)
        self.assertEqual(audit["reason_counts"]["text_version_received_after_cutoff"], 2)
        self.assertEqual(audit["reason_counts"]["original_publication_time_missing"], 2)
        for row in audit["records"]:
            self.assertEqual(row["status"], "After cutoff")
            self.assertTrue(row["first_actual_receipt_at"].startswith("2026-10-17"))
            self.assertIsNone(row["original_published_at"])
            self.assertEqual(row["event_calendar_date"], "2025-10-29")
            self.assertEqual(row["fiscal_period"], "FY2026 Q1")
        self.assertEqual(report["formal_labels_created"], 0)
        self.assertEqual(report["real_model_evaluation"], "Not evaluated")
        self.assertFalse(report["profit_alpha_established"])
        for name in ("labels.json", "split.json", "comparison.json"):
            self.assertFalse((output / name).exists())
        self.assertEqual(before, {str(path.relative_to(capture)): path.read_bytes()
                                  for path in capture.rglob("*") if path.is_file()})

    def test_issuer_documents_share_one_event_not_two_independent_events(self):
        capture, _, _, _ = self.capture()
        output, report = self.audit(capture)
        records = load_json(output / "source_records.json")
        events = {event for row in records for event in row["event_ids"]}
        self.assertEqual(events, {"MSFT:earnings:FY2026Q1:2025-10-29"})
        mapping = load_json(output / "event_mapping.json")
        self.assertFalse(mapping["independent_human_gold"])
        self.assertEqual(mapping["event"], issuer_event.EVENT)
        self.assertEqual(report["source_eligibility"]["families"]["earnings_call"]["distinct_supplied_event_ids"], 1)

    def test_news_and_comment_coverage_is_unknown_not_a_zero_market_sample_claim(self):
        capture, _, _, _ = self.capture()
        _, report = self.audit(capture)
        self.assertIsNone(report["news_same_event_records"])
        self.assertIsNone(report["investor_comment_same_event_records"])
        self.assertIn("not requested", report["coverage_note"])
        self.assertIn("news", report["source_eligibility"]["missing_source_families"])
        self.assertIn("investor_comment", report["source_eligibility"]["missing_source_families"])

    def test_missing_response_timestamp_remains_unknown_despite_early_request_start(self):
        capture, _, _, _ = self.capture()

        def remove_receipts(value):
            for row in value["requests"]:
                row["request_started_at"] = "2025-10-29T20:00:00Z"
                row["response_received_at"] = None

        self.alter_manifest(capture, remove_receipts)
        _, report = self.audit(capture)
        self.assertEqual(report["source_eligibility"]["eligible_source_records"], 0)
        self.assertEqual(report["source_eligibility"]["reason_counts"]["text_version_receipt_evidence_missing"], 2)
        for row in report["source_eligibility"]["records"]:
            self.assertIsNone(row["first_actual_receipt_at"])
            self.assertIsNone(row["first_response_received_at"])

    def test_publication_evidence_and_saved_body_hashes_remain_reproducible(self):
        capture, _, _, _ = self.capture()
        output, _ = self.audit(capture)
        evidence = load_json(output / "publication_evidence.json")
        run = load_json(output / "run_manifest.json")
        self.assertEqual(len(run["input_files_sha256"]), 3)
        for path, expected in run["input_files_sha256"].items():
            self.assertEqual(digest(Path(path).read_bytes()), expected)
        for name, expected in run["outputs_sha256"].items():
            self.assertEqual(digest((output / name).read_bytes()), expected)
        for family in issuer_event.SOURCES:
            body = (output / "bodies" / (family + ".txt")).read_bytes()
            expected_body, _ = issuer_event.extract_official_body((capture / "raw" / (family + ".html")).read_bytes(), family)
            self.assertEqual(body, expected_body.encode())
            self.assertEqual(evidence[family]["text_version_sha256"], digest(body))
            self.assertIsNone(evidence[family]["original_published_at"])

    def test_raw_tampering_fails_before_audit_output_creation(self):
        capture, _, _, _ = self.capture()
        raw = capture / "raw/financial_report.html"
        raw.write_bytes(raw.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "hash/length"):
            self.audit(capture)
        self.assertFalse((self.root / "audit").exists())

    def test_wrong_url_method_or_event_are_rejected_before_output_creation(self):
        for index, change in enumerate((
                lambda value: value["requests"][0].update(url="https://example.invalid/not-issuer"),
                lambda value: value["requests"][0].update(method="POST"),
                lambda value: value["event"].update(event_calendar_date="2026-10-29"))):
            with self.subTest(index=index):
                capture, _, _, _ = self.capture(name=f"capture-{index}")
                self.alter_manifest(capture, change)
                with self.assertRaises(ValueError):
                    self.audit(capture, name=f"audit-{index}")
                self.assertFalse((self.root / f"audit-{index}").exists())

    def test_manifest_must_retain_both_fixed_attempts_and_cannot_redirect_raw_path(self):
        for index, change in enumerate((
                lambda value: value["requests"].pop(),
                lambda value: value["requests"][1].update(source_family="financial_report"),
                lambda value: value["requests"][0].update(raw_file="../outside.html"))):
            with self.subTest(index=index):
                capture, _, _, _ = self.capture(name=f"capture-{index}")
                self.alter_manifest(capture, change)
                with self.assertRaises(ValueError):
                    self.audit(capture, name=f"audit-{index}")
                self.assertFalse((self.root / f"audit-{index}").exists())

    def test_raw_response_symlink_is_rejected_even_when_bytes_match(self):
        capture, _, _, _ = self.capture()
        raw = capture / "raw/financial_report.html"
        outside = self.root / "outside.html"
        outside.write_bytes(raw.read_bytes())
        raw.unlink()
        raw.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "capture archive"):
            self.audit(capture)
        self.assertFalse((self.root / "audit").exists())

    def test_existing_and_dangling_symlink_outputs_cannot_be_overwritten(self):
        capture, _, _, _ = self.capture()
        for name, kind in (("existing", "directory"), ("alias", "symlink"), ("dangling", "dangling")):
            target = self.root / name
            if kind == "directory":
                target.mkdir()
                (target / "keep").write_text("unchanged")
            else:
                target.symlink_to(capture if kind == "symlink" else self.root / "missing-target", target_is_directory=True)
            with self.subTest(name=name):
                with mock.patch.object(issuer_event, "_fetch") as fetch:
                    with self.assertRaises(FileExistsError):
                        issuer_event.capture_event(target)
                fetch.assert_not_called()
                with self.assertRaises(FileExistsError):
                    issuer_event.audit_capture(capture, target, cutoff=CUTOFF)
        self.assertEqual((self.root / "existing/keep").read_text(), "unchanged")

    def test_audit_cannot_write_inside_input_or_through_an_alias_to_input(self):
        capture, _, _, _ = self.capture()
        alias = self.root / "input-alias"
        alias.symlink_to(capture, target_is_directory=True)
        for output in (capture / "audit", alias / "audit"):
            with self.subTest(output=output):
                with self.assertRaisesRegex(ValueError, "outside"):
                    issuer_event.audit_capture(capture, output, cutoff=CUTOFF)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
