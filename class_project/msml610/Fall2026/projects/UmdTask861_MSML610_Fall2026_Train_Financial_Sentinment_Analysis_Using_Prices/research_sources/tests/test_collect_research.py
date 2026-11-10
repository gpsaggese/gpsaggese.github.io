"""Mock-only capture tests; no credentials, application runner, or network access."""

from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from research.financial_sentiment_using_prices.collect_research import (
    BASE_URL, BARS_ROUTE, NEWS_ROUTE, CollectionError, collect_research,
)


def news(identity="one", content="<p>Palantir &amp; partners</p>", updated="2026-10-07T15:00:00Z"):
    return {"id": identity, "headline": "Palantir  &amp; partners", "content": content,
            "summary": "<p>A  summary</p>", "source": "Test Publisher",
            "created_at": "2026-10-07T14:00:00Z", "updated_at": updated,
            "url": "https://example.org/article", "symbols": ["PLTR"]}


def bars(token=None):
    return {"bars": {"PLTR": [{"t": "2026-10-07T04:00:00Z", "o": 100, "c": 101}]},
            "next_page_token": token}


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.headers, self.calls = {}, []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class Clock:
    def __init__(self):
        self.value = datetime(2026, 11, 10, 12, tzinfo=timezone.utc)

    def __call__(self):
        self.value += timedelta(seconds=1)
        return self.value


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "new-run"

    def collect(self, session, **kwargs):
        return collect_research(self.output, ["PLTR"], "2026-10-01", "2026-10-09",
                                session=session, credential_loader=lambda: ("fixture-key", "fixture-secret"),
                                clock=Clock(), **kwargs)

    def read(self, filename):
        return json.loads((self.output / filename).read_text())

    def compatible_inputs(self):
        module_dir = str(Path(__file__).resolve().parents[1])
        with patch.object(sys, "path", [module_dir, *sys.path]):
            runner = importlib.import_module("run_research")
            return runner.read_inputs(self.output)

    def test_success_is_read_inputs_compatible_with_receipts_and_hashes(self):
        session = Session([Response(bars()), Response({"news": [news()], "next_page_token": None})])
        manifest = self.collect(session)
        self.assertEqual(manifest["status"], "complete")
        articles, prices, _ = self.compatible_inputs()
        self.assertEqual(articles[0]["headline"], "Palantir & partners")
        self.assertEqual(articles[0]["text"], "Palantir & partners")
        self.assertEqual(articles[0]["summary"], "A summary")
        self.assertEqual(prices, bars())
        self.assertFalse(articles[0]["historical_availability_verified"])
        self.assertEqual(articles[0]["retrieved_at"], manifest["requests"][1]["response_received_at"])
        for record in manifest["requests"]:
            self.assertLess(record["started_at"], record["response_received_at"])
            self.assertEqual(record["sha256"], hashlib.sha256((self.output / record["file"]).read_bytes()).hexdigest())
        saved = "".join(path.read_text() for path in self.output.glob("*.json"))
        self.assertNotIn("fixture-key", saved)
        self.assertNotIn("fixture-secret", saved)
        self.assertNotIn("APCA-API", saved)
        self.assertEqual([url for url, _ in session.calls], [BASE_URL + BARS_ROUTE, BASE_URL + NEWS_ROUTE])
        for _, options in session.calls:
            self.assertFalse(options["allow_redirects"])
            self.assertEqual(options["timeout"], 30)
        params = session.calls[0][1]["params"]
        self.assertEqual(params["symbols"], "PLTR,SPY,QQQ,IGV,SOXX,XLE")
        self.assertEqual((params["feed"], params["timeframe"], params["adjustment"], params["limit"]), ("sip", "1Day", "raw", 10000))
        self.assertEqual(params["start"], "2026-10-01T04:00:00+00:00")
        self.assertEqual(params["end"], "2026-10-09T20:00:00+00:00")
        self.assertEqual(session.calls[1][1]["params"]["limit"], 50)

    def test_duplicate_id_keeps_last_version_and_last_actual_receipt(self):
        session = Session([Response(bars()),
                           Response({"news": [news()], "next_page_token": "page-2"}),
                           Response({"news": [news(content="<p>Updated body</p>", updated="2026-10-08T15:00:00Z")], "next_page_token": None})])
        manifest = self.collect(session)
        articles, _, _ = self.compatible_inputs()
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["text"], "Updated body")
        self.assertEqual(articles[0]["retrieved_at"], manifest["requests"][2]["response_received_at"])
        self.assertEqual(session.calls[2][1]["params"]["page_token"], "page-2")
        summary = manifest["news_summary"]
        self.assertEqual((summary["raw_article_rows"], summary["duplicate_rows"], summary["duplicate_ids"], summary["changed_duplicate_rows"]), (2, 1, 1, 1))

    def test_old_article_recent_revision_is_audited_and_pagination_continues(self):
        older = news("older", updated="2026-10-08T15:00:00Z")
        older["created_at"] = "2026-09-30T14:00:00Z"
        session = Session([Response(bars()), Response({"news": [older], "next_page_token": "page-2"}),
                           Response({"news": [news()], "next_page_token": None})])
        manifest = self.collect(session)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(manifest["news_summary"]["raw_article_rows"], 2)
        self.assertEqual(manifest["news_summary"]["accepted_article_rows"], 1)
        self.assertEqual(manifest["news_summary"]["rejected_article_rows"], 1)
        rejection = self.read("news_rejected_records.json")[0]
        self.assertEqual(rejection["id"], "older")
        self.assertTrue(rejection["updated_within_requested_window"])
        self.assertEqual(rejection["response_received_at"], manifest["requests"][1]["response_received_at"])
        articles, _, provenance = self.compatible_inputs()
        self.assertEqual([a["id"] for a in articles], ["one"])
        self.assertEqual(provenance["verified_rejected_records"], 1)
        self.assertEqual(self.read("news_raw_000.json")["news"][0], older)

    def test_publication_after_window_is_rejected_without_changing_endpoint(self):
        later = news("future", updated="2026-10-10T15:00:00Z")
        later["created_at"] = "2026-10-10T14:00:00Z"
        session = Session([Response(bars()), Response({"news": [later, news()], "next_page_token": None})])
        manifest = self.collect(session)
        self.assertEqual(manifest["status"], "complete")
        self.assertFalse(self.read("news_rejected_records.json")[0]["updated_within_requested_window"])
        self.assertEqual([a["id"] for a in self.compatible_inputs()[0]], ["one"])
        self.assertEqual(session.calls[1][1]["params"]["end"], "2026-10-09T20:00:00+00:00")

    def test_invalid_update_order_still_fails(self):
        session = Session([Response(bars()), Response({"news": [news(updated="2026-10-06T15:00:00Z")], "next_page_token": None})])
        with self.assertRaisesRegex(CollectionError, "precedes"):
            self.collect(session)
        self.assertEqual(self.read("manifest.json")["status"], "failed")
        self.assertFalse((self.output / "news_normalized.json").exists())

    def test_rejection_audit_cannot_hide_an_in_window_article(self):
        self.collect(Session([Response(bars()), Response({"news": [news()], "next_page_token": None})]))
        manifest = self.read("manifest.json")
        rejection = [{"file": "news_raw_000.json", "row_index": 0, "id": "one",
                      "created_at": "2026-10-07T14:00:00Z", "updated_at": "2026-10-07T15:00:00Z",
                      "response_received_at": manifest["requests"][1]["response_received_at"],
                      "reason": "publication_outside_requested_window", "updated_within_requested_window": True}]
        path = self.output / "news_rejected_records.json"
        path.write_text(json.dumps(rejection))
        manifest["rejected_records_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["news_summary"]["rejected_article_rows"] = 1
        (self.output / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "reason"):
            self.compatible_inputs()

    def test_tampered_rejection_audit_hash_is_rejected(self):
        self.collect(Session([Response(bars()), Response({"news": [news()], "next_page_token": None})]))
        (self.output / "news_rejected_records.json").write_text('[{}]')
        with self.assertRaisesRegex(ValueError, "hash"):
            self.compatible_inputs()

    def test_page_cap_is_explicitly_truncated_and_still_auditable(self):
        session = Session([Response(bars()), Response({"news": [news()], "next_page_token": "more"})])
        manifest = self.collect(session, max_news_pages=1)
        self.assertEqual(manifest["status"], "truncated")
        self.assertTrue(manifest["news_summary"]["truncated"])
        self.assertFalse(self.read("news_summary.json")["pagination_complete"])
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(len(self.compatible_inputs()[0]), 1)

    def test_http_failure_keeps_failed_manifest_without_normalized_success(self):
        session = Session([Response(bars()), Response({"message": "do not persist response bodies on errors"}, 403)])
        with self.assertRaisesRegex(CollectionError, "HTTP 403"):
            self.collect(session)
        manifest = self.read("manifest.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["requests"][-1]["status"], 403)
        self.assertIsNotNone(manifest["requests"][-1]["response_received_at"])
        self.assertFalse((self.output / "news_normalized.json").exists())
        self.assertNotIn("message", self.read("news_raw_000.json"))

    def test_transport_failure_has_no_invented_response_receipt(self):
        session = Session([TimeoutError("secret text must not be logged")])
        with self.assertRaises(CollectionError):
            self.collect(session)
        manifest = self.read("manifest.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertIsNone(manifest["requests"][0]["status"])
        self.assertIsNone(manifest["requests"][0]["response_received_at"])
        self.assertNotIn("secret text", (self.output / "manifest.json").read_text())

    def test_bars_pagination_is_rejected_instead_of_silent_truncation(self):
        session = Session([Response(bars("more-bars"))])
        with self.assertRaisesRegex(CollectionError, "one-page budget"):
            self.collect(session)
        manifest = self.read("manifest.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["bars_summary"]["status"], "truncated")
        self.assertFalse(manifest["bars_summary"]["pagination_complete"])
        self.assertEqual(len(session.calls), 1)

    def test_repeated_news_token_fails_and_retains_received_pages(self):
        session = Session([Response(bars()), Response({"news": [news()], "next_page_token": "same"}),
                           Response({"news": [], "next_page_token": "same"})])
        with self.assertRaisesRegex(CollectionError, "repeated"):
            self.collect(session)
        self.assertEqual(self.read("manifest.json")["status"], "failed")
        self.assertTrue((self.output / "news_raw_001.json").exists())

    def test_malformed_json_and_shape_fail(self):
        for response in (Response(ValueError("invalid JSON")), Response({"news": []}), Response([])):
            with self.subTest(payload=response.payload):
                self.output = Path(self.temp.name) / f"failure-{len(list(Path(self.temp.name).iterdir()))}"
                with self.assertRaises(CollectionError):
                    self.collect(Session([response]))
                self.assertEqual(self.read("manifest.json")["status"], "failed")

    def test_existing_output_and_oversized_window_are_rejected_before_requests(self):
        self.output.mkdir()
        session = Session([])
        with self.assertRaises(FileExistsError):
            self.collect(session)
        self.assertEqual(session.calls, [])
        with self.assertRaisesRegex(ValueError, "93"):
            collect_research(Path(self.temp.name) / "too-long", ["PLTR"], "2026-01-01", "2026-10-09", session=session)

    def test_new_york_winter_endpoint_and_future_date_rejection(self):
        session = Session([Response(bars()), Response({"news": [], "next_page_token": None})])
        collect_research(self.output, ["PLTR"], "2026-11-02", "2026-11-03", session=session,
                         credential_loader=lambda: ("fixture-key", "fixture-secret"), clock=Clock())
        params = session.calls[0][1]["params"]
        self.assertEqual(params["start"], "2026-11-02T05:00:00+00:00")
        self.assertEqual(params["end"], "2026-11-03T21:00:00+00:00")
        with self.assertRaisesRegex(ValueError, "not occurred"):
            collect_research(Path(self.temp.name) / "future", ["PLTR"], "2026-11-10", "2026-11-11",
                             session=Session([]), clock=Clock())


if __name__ == "__main__":
    unittest.main()
