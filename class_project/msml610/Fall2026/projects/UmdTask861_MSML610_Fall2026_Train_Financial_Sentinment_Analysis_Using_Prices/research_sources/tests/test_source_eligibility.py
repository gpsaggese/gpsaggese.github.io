"""Source eligibility contracts use invented text/evidence, never private data."""

import copy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_preflight import audit_source_records


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def capture(version=None, received="2025-10-30T11:30:00Z", **changes):
    request_start = ((datetime.fromisoformat(received.replace("Z", "+00:00")) - timedelta(seconds=1))
                     .isoformat().replace("+00:00", "Z")) if received else "2025-10-30T11:29:59Z"
    result = {
        "response_received_at": received,
        "request_started_at": request_start,
        "request_evidence_id": "fixture-request-1",
        "retrieval_evidence_id": "fixture-response-1",
        "response_sha256": digest("fixture saved HTTP response"),
        "text_version_sha256": version or digest("fixture original text"),
        "status": 200,
        "archive_integrity_verified": True,
    }
    result.update(changes)
    return result


def source(identifier="fixture-1", family="news", **changes):
    version = digest("fixture original text")
    result = {
        "record_id": identifier,
        "source_id": "source-" + identifier,
        "source_family": family,
        "document_kind": "fixture-body",
        "event_ids": ["fixture-event-earnings-1"],
        "event_identity_evidence_id": "fixture-reviewed-event-map-1",
        "event_calendar_date": "2025-10-29",
        "original_published_at": "2025-10-29T21:00:00Z",
        "version_published_at": "2025-10-29T21:00:00Z",
        "original_publication_evidence_id": "fixture-page-timestamp-1",
        "text_version_sha256": version,
        "text_available": True,
        "captures": [capture(version)],
    }
    result.update(changes)
    return result


class SourceEligibilityTests(unittest.TestCase):
    cutoff = "2025-10-30T12:45:00Z"  # 08:45 New York, before the DST change.
    as_of = "2025-11-01T16:00:00Z"

    def audit(self, records, **changes):
        return audit_source_records(records, cutoff=changes.get("cutoff", self.cutoff),
                                    as_of=changes.get("as_of", self.as_of))

    def row(self, value):
        return self.audit([value])["records"][0]

    def test_complete_supplied_evidence_can_pass_each_source_family(self):
        records = [source(f"fixture-{family}", family) for family in
                   ("financial_report", "earnings_call", "news", "investor_comment")]
        report = self.audit(records)
        self.assertEqual(report["eligible_source_records"], 4)
        self.assertEqual(report["missing_source_families"], [])
        for row in report["records"]:
            self.assertEqual(row["status"], "Eligible under supplied evidence")
            self.assertEqual(row["reasons"], [])
            self.assertEqual(row["first_actual_receipt_at"], "2025-10-30T11:30:00Z")
            self.assertTrue(row["source_contract_eligible"])

    def test_after_cutoff_comment_publication_cannot_become_morning_input(self):
        row = self.row(source(family="investor_comment",
                              original_published_at="2025-10-30T12:46:00Z",
                              version_published_at="2025-10-30T12:46:00Z",
                              captures=[capture(received="2025-10-30T12:47:00Z")]))
        self.assertFalse(row["source_contract_eligible"])
        self.assertEqual(row["status"], "After cutoff")
        self.assertIn("original_published_at_after_cutoff", row["reasons"])
        self.assertIn("text_version_received_after_cutoff", row["reasons"])

    def test_republication_before_original_is_inconsistent_even_with_later_actual_receipt(self):
        row = self.row(source(republished_at="2025-10-28T21:00:00Z"))
        self.assertFalse(row["source_contract_eligible"])
        self.assertIn("republication_precedes_original_publication", row["reasons"])

    def test_comment_published_before_cutoff_but_received_later_is_unavailable(self):
        row = self.row(source(family="investor_comment", captures=[
            capture(received="2025-10-30T12:46:00Z")]))
        self.assertNotIn("original_published_at_after_cutoff", row["reasons"])
        self.assertIn("text_version_received_after_cutoff", row["reasons"])
        self.assertFalse(row["source_contract_eligible"])

    def test_old_version_receipt_cannot_be_borrowed_by_revised_hash(self):
        old = source("old", source_id="same-article")
        revised_hash = digest("fixture revised text")
        revised = source("revision", source_id="same-article",
                         text_version_sha256=revised_hash,
                         version_published_at="2025-10-30T12:50:00Z",
                         captures=[old["captures"][0], capture(revised_hash, "2025-10-30T13:00:00Z")])
        report = self.audit([old, revised])
        original, latest = report["records"]
        self.assertTrue(original["source_contract_eligible"])
        self.assertEqual(latest["first_actual_receipt_at"], "2025-10-30T13:00:00Z")
        self.assertEqual(latest["source_first_retained_text_receipt_at"], "2025-10-30T11:30:00Z")
        self.assertFalse(latest["source_contract_eligible"])
        self.assertIn("version_published_at_after_cutoff", latest["reasons"])

    def test_unreceived_revision_stays_unknown_despite_valid_old_capture(self):
        old_hash = digest("fixture original text")
        row = self.row(source(text_version_sha256=digest("fixture revised text"),
                              version_published_at="2025-10-30T12:00:00Z",
                              captures=[capture(old_hash)]))
        self.assertIsNone(row["first_actual_receipt_at"])
        self.assertEqual(row["first_response_received_at"], "2025-10-30T11:30:00Z")
        self.assertIn("text_version_receipt_evidence_missing", row["reasons"])

    def test_revision_with_its_own_pre_cutoff_receipt_can_pass(self):
        original = source("old", source_id="same-article")
        revised_hash = digest("fixture corrected before cutoff")
        revised = source("revision", source_id="same-article", text_version_sha256=revised_hash,
                         version_published_at="2025-10-30T12:00:00Z",
                         captures=[original["captures"][0], capture(revised_hash, "2025-10-30T12:10:00Z")])
        row = self.audit([original, revised])["records"][1]
        self.assertTrue(row["source_contract_eligible"])
        self.assertEqual(row["first_actual_receipt_at"], "2025-10-30T12:10:00Z")
        self.assertEqual(row["source_first_retained_text_receipt_at"], "2025-10-30T11:30:00Z")

    def test_same_hash_cannot_backdate_new_republished_edition(self):
        row = self.row(source(republished_at="2025-10-30T13:00:00Z"))
        self.assertFalse(row["source_contract_eligible"])
        self.assertIn("republished_at_after_cutoff", row["reasons"])
        self.assertIn("text_receipt_precedes_version_publication", row["reasons"])

    def test_republication_before_cutoff_still_rejects_earlier_edition_receipt(self):
        row = self.row(source(republished_at="2025-10-30T12:00:00Z"))
        self.assertFalse(row["source_contract_eligible"])
        self.assertIn("text_receipt_precedes_version_publication", row["reasons"])
        self.assertEqual(row["status"], "Missing or inconsistent evidence")

    def test_syndicated_copies_with_reviewed_shared_event_do_not_inflate_event_count(self):
        copies = [source("wire-a"), source("wire-b", text_version_sha256=digest("syndicated wording"),
                                          captures=[capture(digest("syndicated wording"))])]
        report = self.audit(copies)
        self.assertEqual(report["families"]["news"]["records"], 2)
        self.assertEqual(report["families"]["news"]["distinct_source_ids"], 2)
        self.assertEqual(report["families"]["news"]["distinct_supplied_event_ids"], 1)
        self.assertEqual(report["eligible_source_records"], 2)
        self.assertIn("event-ID counts are not independent samples", " ".join(report["limitations"]))

    def test_unknown_event_identity_is_not_fabricated_from_article_or_url(self):
        row = source(event_ids=[], event_identity_evidence_id=None,
                     url="https://example.invalid/earnings/event-123")
        report = self.audit([row])
        self.assertEqual(report["records"][0]["event_ids"], [])
        self.assertEqual(report["families"]["news"]["distinct_supplied_event_ids"], 0)
        self.assertEqual(report["families"]["news"]["missing_event_identity_records"], 1)
        self.assertFalse(report["records"][0]["source_contract_eligible"])

    def test_fiscal_label_is_preserved_separately_from_explicit_calendar_event(self):
        # Microsoft FY2026 Q1 is a 2025-10-29 event. This is a boundary fixture,
        # not a retrieved transcript or a claim to have received its text then.
        # https://www.microsoft.com/en-us/investor/events/fy-2026/earnings-fy-2026-q1
        row = self.row(source(family="earnings_call", fiscal_period="FY2026 Q1",
                              event_calendar_date="2025-10-29"))
        self.assertEqual(row["event_calendar_date"], "2025-10-29")
        self.assertEqual(row["fiscal_period"], "FY2026 Q1")
        self.assertTrue(row["source_contract_eligible"])

    def test_fiscal_year_url_and_period_end_cannot_supply_missing_event_date(self):
        row = self.row(source(family="financial_report", fiscal_period="FY2026 Q1",
                              event_calendar_date=None, report_period_end="2025-09-30",
                              url="https://example.invalid/fy-2026/earnings-fy-2026-q1"))
        self.assertIsNone(row["event_calendar_date"])
        self.assertIn("financial_event_calendar_date_missing", row["reasons"])
        self.assertFalse(row["source_contract_eligible"])

    def test_event_day_is_not_a_substitute_for_exact_original_publication(self):
        row = self.row(source(family="earnings_call", fiscal_period="FY2026 Q1",
                              original_published_at=None, version_published_at=None))
        self.assertIsNone(row["original_published_at"])
        self.assertIn("original_publication_time_missing", row["reasons"])
        self.assertFalse(row["source_contract_eligible"])

    def test_request_start_first_seen_and_elapsed_cannot_supply_actual_receipt(self):
        row = self.row(source(first_seen_at="2025-10-30T11:29:59Z", captures=[
            capture(received=None, first_seen_at="2025-10-30T11:29:59Z", elapsed_seconds=1.0)]))
        self.assertIsNone(row["first_actual_receipt_at"])
        self.assertIsNone(row["first_response_received_at"])
        self.assertIn("text_version_receipt_evidence_missing", row["reasons"])

    def test_http_failure_does_not_prove_text_receipt_even_with_hash(self):
        row = self.row(source(captures=[capture(status=403)]))
        self.assertEqual(row["first_response_received_at"], "2025-10-30T11:30:00Z")
        self.assertIsNone(row["first_actual_receipt_at"])
        self.assertFalse(row["source_contract_eligible"])

    def test_response_before_its_request_start_is_not_accepted_as_receipt(self):
        row = self.row(source(captures=[capture(request_started_at="2025-10-30T11:31:00Z")]))
        self.assertIsNone(row["first_response_received_at"])
        self.assertIsNone(row["first_actual_receipt_at"])
        self.assertIn("response_precedes_request_start", row["reasons"])
        self.assertIn("text_version_receipt_evidence_missing", row["reasons"])
        self.assertFalse(row["source_contract_eligible"])

    def test_request_start_is_optional_and_does_not_replace_response_evidence(self):
        row = self.row(source(captures=[capture(request_started_at=None)]))
        self.assertEqual(row["first_actual_receipt_at"], "2025-10-30T11:30:00Z")
        self.assertTrue(row["source_contract_eligible"])
        self.assertEqual(row["retrieval_evidence"][0]["request_evidence_id"], "fixture-request-1")

    def test_whitespace_or_structured_values_are_not_evidence_identifiers(self):
        for invalid in ("", "   \n", {"name": "fake"}, ["fake"], 1, True):
            for field, reason in (("original_publication_evidence_id", "original_publication_evidence_missing"),
                                  ("event_identity_evidence_id", "event_identity_unreviewed_or_missing"),
                                  ("retrieval_evidence_id", "text_version_receipt_evidence_missing")):
                with self.subTest(field=field, invalid=invalid):
                    value = source()
                    if field == "retrieval_evidence_id":
                        value["captures"][0][field] = invalid
                    else:
                        value[field] = invalid
                    row = self.row(value)
                    self.assertFalse(row["source_contract_eligible"])
                    self.assertIn(reason, row["reasons"])

    def test_missing_or_unverified_raw_response_evidence_cannot_certify_text(self):
        for change in ({"response_sha256": None}, {"retrieval_evidence_id": None},
                       {"archive_integrity_verified": False}, {"text_version_sha256": None},
                       {"status": True}):
            with self.subTest(change=change):
                row = self.row(source(captures=[capture(**change)]))
                self.assertIsNone(row["first_actual_receipt_at"])
                self.assertIn("text_version_receipt_evidence_missing", row["reasons"])
                self.assertFalse(row["source_contract_eligible"])

    def test_future_capture_is_not_authenticated_at_earlier_as_of(self):
        row = self.row(source(captures=[capture(received="2025-11-02T11:00:00Z")]))
        self.assertIsNone(row["first_actual_receipt_at"])
        self.assertIsNone(row["first_response_received_at"])
        self.assertIn("response_after_as_of", row["reasons"])
        self.assertFalse(row["source_contract_eligible"])

    def test_cutoff_cannot_exceed_audit_as_of(self):
        report = self.audit([source()], cutoff="2025-10-30T12:45:00Z",
                            as_of="2025-10-30T12:45:00Z")
        self.assertTrue(report["records"][0]["source_contract_eligible"])
        with self.assertRaises(ValueError):
            self.audit([source()], cutoff="2025-10-30T12:45:00Z", as_of="2025-10-30T12:44:59Z")

    def test_metadata_only_filings_do_not_count_as_financial_or_call_text(self):
        report = self.audit([source(family="financial_report", document_kind="filing_metadata",
                                    text_available=False, text_version_sha256=None, captures=[])])
        row = report["records"][0]
        self.assertIn("retained_text_missing", row["reasons"])
        self.assertIn("financial_report", report["missing_source_families"])
        self.assertIn("earnings_call", report["missing_source_families"])
        self.assertEqual(report["families"]["financial_report"]["retained_text_records"], 0)
        self.assertEqual(report["families"]["earnings_call"]["records"], 0)
        self.assertEqual(report["eligible_source_records"], 0)

    def test_earliest_verified_same_version_receipt_ignores_input_capture_order(self):
        value = source(captures=[capture(received="2025-10-30T12:30:00Z"),
                                 capture(received="2025-10-30T11:00:00Z"),
                                 capture(received="2025-10-30T10:00:00Z", status=403)])
        row = self.row(value)
        self.assertEqual(row["first_actual_receipt_at"], "2025-10-30T11:00:00Z")
        self.assertEqual(row["first_response_received_at"], "2025-10-30T10:00:00Z")
        self.assertTrue(row["source_contract_eligible"])

    def test_publication_requires_explicit_evidence_and_version_chronology(self):
        for changes, reason in (({"original_publication_evidence_id": None},
                                 "original_publication_evidence_missing"),
                                ({"version_published_at": "2025-10-29T20:00:00Z"},
                                 "version_precedes_original_publication")):
            with self.subTest(reason=reason):
                row = self.row(source(**changes))
                self.assertIn(reason, row["reasons"])
                self.assertFalse(row["source_contract_eligible"])

    def test_audit_does_not_mutate_inputs_or_emit_raw_body_text(self):
        value = source(raw_text="PRIVATE-FIXTURE-BODY-MUST-NOT-APPEAR",
                       text="PRIVATE-FIXTURE-BODY-MUST-NOT-APPEAR",
                       content="PRIVATE-FIXTURE-BODY-MUST-NOT-APPEAR")
        value["captures"][0]["raw_response"] = "PRIVATE-FIXTURE-RAW-MUST-NOT-APPEAR"
        before = copy.deepcopy(value)
        report = self.audit([value])
        self.assertEqual(value, before)
        serialized = json.dumps(report)
        self.assertNotIn("PRIVATE-FIXTURE", serialized)
        self.assertNotIn("labels", report)
        self.assertNotIn("predictions", report)

    def test_invalid_or_ambiguous_identity_contracts_raise_instead_of_guessing(self):
        invalid = ([source(), source()], [source(source_family="comment")],
                   [source(event_ids=["same", "same"])], [source(source_id="")],
                   [source(original_published_at="2025-10-29T21:00:00")],
                   [source(event_calendar_date="2025-10")])
        for records in invalid:
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    self.audit(records)


if __name__ == "__main__":
    unittest.main()
