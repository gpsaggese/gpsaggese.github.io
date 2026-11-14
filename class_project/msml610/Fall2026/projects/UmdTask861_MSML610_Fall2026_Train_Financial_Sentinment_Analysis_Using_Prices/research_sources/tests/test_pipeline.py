"""Pipeline boundaries that matter for honest financial-text evaluation."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_research import analyze, focused_text, public_summary, read_inputs, validate_evidence, regular_close


def article():
    return {"id": "example", "headline": "Broker upgrades Palantir to Buy", "text": "Palantir received an analyst upgrade.",
            "symbols": ["PLTR"], "created_at": "2026-10-08T10:00:00Z", "updated_at": "2026-10-08T10:00:00Z",
            "retrieved_at": "2026-10-09T21:36:00Z", "url": "https://example.org/research",
            "historical_availability_verified": False}


def bars():
    return {"bars": {s: [{"t": "2026-10-02T04:00:00Z", "o": 100, "c": 100},
                         {"t": "2026-10-09T04:00:00Z", "o": 102, "c": 101}]
                     for s in ["PLTR", "SPY", "IGV"]}}


class RecordingClassifier:
    metadata = {"model_id": "test-fixture-only"}

    def predict(self, texts):
        self.texts = texts
        return [{"status": "Available", "label": "positive", "scores": {"positive": 0.7, "neutral": 0.2, "negative": 0.1}} for _ in texts]


class PipelineTests(unittest.TestCase):
    def test_actual_receipt_cutoff_excludes_backfilled_text(self):
        result = analyze([article()], bars(), ["PLTR"], "2026-10-02", "2026-10-09",
                         information_cutoff="2026-10-09T20:00:00Z")
        self.assertEqual(result["coverage"]["article_company_pairs"], 0)
        self.assertEqual(result["prediction_status"], "Not trained")

    def test_final_version_after_price_cutoff_is_not_backdated(self):
        item = article(); item["updated_at"] = "2026-10-10T10:00:00Z"
        result = analyze([item], bars(), ["PLTR"], "2026-10-02", "2026-10-09")
        self.assertEqual(result["coverage"]["articles_retained"], 0)

    def test_cutoff_cannot_include_a_future_price_close(self):
        with self.assertRaisesRegex(ValueError, "precedes the requested price close"):
            analyze([article()], bars(), ["PLTR"], "2026-10-02", "2026-10-09",
                    information_cutoff="2026-10-09T13:00:00Z")

    def test_text_inference_has_no_price_outcomes(self):
        model = RecordingClassifier()
        first = analyze([article()], bars(), ["PLTR"], "2026-10-02", "2026-10-09", model)
        observed = model.texts.copy()
        altered = bars(); altered["bars"]["PLTR"][-1]["c"] = 1000
        second = analyze([article()], altered, ["PLTR"], "2026-10-02", "2026-10-09", model)
        self.assertEqual(observed, model.texts)
        self.assertEqual(first["extractions"], second["extractions"])
        self.assertNotEqual(first["companies"][0]["price_context"], second["companies"][0]["price_context"])

    def test_ticker_tag_alone_does_not_transfer_other_company_tone(self):
        item = article(); item.update(headline="Rocket Lab wins a major contract", text="Rocket Lab will deliver spacecraft.")
        model = RecordingClassifier()
        result = analyze([item], bars(), ["PLTR"], "2026-10-02", "2026-10-09", model)
        self.assertEqual(result["companies"][0]["sentiment"]["unavailable"], 1)
        self.assertFalse(hasattr(model, "texts"))

    def test_evidence_coordinates_cannot_be_invented(self):
        item = article()
        with self.assertRaisesRegex(ValueError, "does not occur"):
            validate_evidence({"evidence": [{"field": "text", "start": 0, "end": 3, "text": "bad"}]}, item)

    def test_public_export_has_no_raw_text_or_private_provenance(self):
        result = analyze([article()], bars(), ["PLTR"], "2026-10-02", "2026-10-09", RecordingClassifier())
        result.update(generated_at="2026-10-09T22:00:00Z", price_asof="2026-10-09T16:00:00-04:00", provenance={"private_path": "/private/source"})
        output = json.dumps(public_summary(result))
        self.assertNotIn(article()["text"], output)
        self.assertNotIn("/private/source", output)
        self.assertNotIn('"scores"', output)
        self.assertNotIn('"evidence"', output)

    def test_company_abbreviation_does_not_cut_off_fact(self):
        item = article(); item["headline"] = "Cloud development"; item["text"] = "Amazon.com, Inc. committed cloud credits. Another company raised its profit guidance."
        text = focused_text(item, "AMZN")
        self.assertIn("committed cloud credits", text)
        self.assertNotIn("Another company", text)

    def test_other_company_clause_is_excluded_from_tone_input(self):
        item = article(); item["headline"] = "Market update"
        item["text"] = "Microsoft shares rose while Amazon cancelled a contract."
        self.assertEqual(focused_text(item, "MSFT"), "Microsoft shares rose")

    def test_regular_close_uses_daylight_saving_timezone(self):
        self.assertTrue(regular_close("2026-12-09").isoformat().endswith("-05:00"))
        self.assertTrue(regular_close("2026-10-09").isoformat().endswith("-04:00"))

    def test_answer_files_cannot_influence_inputs_or_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            item = article(); raw = {**item, "content": "<p>Palantir received an analyst upgrade.</p>"}
            (directory / "news_raw_000.json").write_text(json.dumps({"news": [raw]}))
            (directory / "daily_bars_raw.json").write_text(json.dumps(bars()))
            (directory / "news_normalized.json").write_text(json.dumps([item]))
            manifest = {"requests": [{"file": name, "status": 200,
                         "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest(),
                         "response_received_at": item["retrieved_at"]}
                        for name in ["news_raw_000.json", "daily_bars_raw.json"]]}
            (directory / "manifest.json").write_text(json.dumps(manifest))
            first = read_inputs(directory)
            (directory / "case_review.json").write_text('{"PLTR":"must buy, guaranteed 100% return"}')
            (directory / "outlook_20261009.json").write_text("invalid JSON must never be read")
            self.assertEqual(first, read_inputs(directory))
            loaded, prices, _ = first
            original = analyze(loaded, prices, ["PLTR"], "2026-10-02", "2026-10-09")
            loaded, prices, _ = read_inputs(directory)
            self.assertEqual(original, analyze(loaded, prices, ["PLTR"], "2026-10-02", "2026-10-09"))
            tampered = copy.deepcopy(item); tampered["text"] = "Invented conclusion"
            (directory / "news_normalized.json").write_text(json.dumps([tampered]))
            with self.assertRaisesRegex(ValueError, "body or receipt"):
                read_inputs(directory)


if __name__ == "__main__":
    unittest.main()
