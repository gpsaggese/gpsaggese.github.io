"""Synthetic boundary tests; real model inference is exercised separately offline."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finbert_comparison import (MODEL_ID, REVISION, sha, infer_packet, validate_features,
                                prepare_packet, evaluate_tone_controls, save_new)


def packet():
    return {"model_id": MODEL_ID, "revision": REVISION,
            "source_split_sha256": "split", "source_news_sha256": "news", "registration_sha256": "reg",
            "rows": [{"sample_id": str(i), "headlines": [t], "headline_hashes": [sha(t.encode())]}
                     for i, t in enumerate(["profits rose", "profits fell", "profits rose"])]}


class FakeTone:
    metadata = {"model_id": MODEL_ID, "revision": REVISION, "test_fixture": True}
    def predict(self, texts):
        self.inputs = texts
        return [{"status": "Available", "input_sha256": sha(t.encode()), "truncated": False,
                 "scores": {"positive": .7 if "rose" in t else .1, "neutral": .2,
                            "negative": .1 if "rose" in t else .7}} for t in texts]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixture(root):
    from run_historical import evaluate_models, cost_evaluation
    protocol = {"ridge_alpha": 10., "class_neutral_band": .005,
                "tfidf": {"max_features": 2000, "min_df": 2, "ngram_range": [1, 2]},
                "assumed_roundtrip_cost_bps": [0, 10, 20]}
    def row(i):
        return {"sample_id": str(i), "text": "profits rose" if i % 2 else "profits fell",
                "source_row_indices": [i], "price_features": [.001*i, .01*(i%3)],
                "excess_return": .005*(i%3-1), "price_class": 0,
                "entry_at": "2019-01-02T14:30:00Z", "exit_at": "2019-01-16T14:30:00Z",
                "entry_session_index": i, "exit_session_index": i+10,
                "stock_return": .01, "spy_return": .005, "source_ids": [str(i)]}
    split = {"train": [row(i) for i in range(30)], "test": [row(i) for i in range(30, 40)]}
    capture, base = root/"capture", root/"base"
    capture.mkdir(); base.mkdir()
    write(capture/"news.json", [{"source_row_index": i, "headline": row(i)["text"]} for i in range(40)])
    write(capture/"manifest.json", {"status": "complete_bounded_capture", "files_sha256": {
        "news.json": sha((capture/"news.json").read_bytes())}})
    metrics, pred, fitted = evaluate_models(split, protocol)
    write(base/"split.json", split); write(base/"predictions.json", pred)
    write(base/"report.json", {"metrics": metrics, "protocol": protocol,
                              "cost_evaluation": cost_evaluation(pred, [0, 10, 20])})
    write(base/"run_manifest.json", {"outputs_sha256": {p.name: sha(p.read_bytes()) for p in base.iterdir()}})
    reg = root/"registration.json"
    write(reg, {"finbert": {"model_id": MODEL_ID, "revision": REVISION, "ridge_alpha": 10.}})
    request = prepare_packet(capture, base, reg)
    req = root/"request.json"; write(req, request)
    features = infer_packet(request, FakeTone()); features["request_file_sha256"] = sha(req.read_bytes())
    feat = root/"features.json"; write(feat, features)
    return capture, base, reg, req, feat


class ComparisonTests(unittest.TestCase):
    def test_explicit_local_snapshot_never_accepts_missing_or_other_revision(self):
        from sentiment_model import cached_model_file, MODEL_REVISION
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/MODEL_REVISION; p.mkdir()
            with self.assertRaises(FileNotFoundError): cached_model_file(p, 'config.json')
            (p/'config.json').write_text('{}')
            self.assertEqual(cached_model_file(p, 'config.json'), p/'config.json')
            with self.assertRaises(ValueError): cached_model_file(p, '../unknown')
            with self.assertRaises(ValueError): cached_model_file(p.parent, 'config.json')

    def test_only_distinct_original_text_is_inferred(self):
        classifier = FakeTone(); f = infer_packet(packet(), classifier)
        self.assertEqual(classifier.inputs, ["profits rose", "profits fell"])
        self.assertEqual(f["unique_headlines"], 2)
        self.assertEqual(f["headline_uses"], 3)
        self.assertAlmostEqual(f["samples"][0]["tone_score"], .6)
        validate_features(packet(), f)

    def test_prices_or_targets_cannot_enter_inference(self):
        p = packet(); p["rows"][0]["excess_return"] = 1.
        with self.assertRaisesRegex(ValueError, "unexpected"): infer_packet(p, FakeTone())

    def test_changed_text_hash_fails(self):
        p = packet(); p["rows"][0]["headlines"] = ["changed"]
        with self.assertRaisesRegex(ValueError, "hash"): infer_packet(p, FakeTone())

    def test_duplicate_or_empty_budget_fails(self):
        p = packet(); p["rows"].append(deepcopy(p["rows"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"): infer_packet(p, FakeTone())
        p = packet(); p["rows"][0]["headlines"] = []
        with self.assertRaisesRegex(ValueError, "empty"): infer_packet(p, FakeTone())

    def test_truncated_input_is_not_silently_compared(self):
        class Truncated(FakeTone):
            def predict(self, texts):
                rows = super().predict(texts); rows[0]["truncated"] = True; return rows
        with self.assertRaisesRegex(ValueError, "Truncated"): infer_packet(packet(), Truncated())

    def test_invalid_probabilities_fail(self):
        class Invalid(FakeTone):
            def predict(self, texts):
                rows = super().predict(texts); rows[0]["scores"]["positive"] = float("nan"); return rows
        with self.assertRaisesRegex(ValueError, "probabilities"): infer_packet(packet(), Invalid())

    def test_changed_aggregate_cannot_override_receipts(self):
        f = infer_packet(packet(), FakeTone()); f["samples"][0]["tone_score"] += .2
        with self.assertRaisesRegex(ValueError, "receipts"): validate_features(packet(), f)

    def test_existing_or_symlink_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); (p/"link").symlink_to(p/"absent")
            for target in [p, p/"link"]:
                with self.assertRaises(FileExistsError): save_new(target, {"x.json": {}})
            self.assertFalse((p/"absent").exists())

    def test_preparation_and_paired_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            capture, base, reg, req, feat = fixture(Path(d))
            result = evaluate_tone_controls(base, req, feat, reg, Path(d)/"out")
            self.assertEqual((result["train_rows"], result["test_rows"]), (30, 10))
            self.assertEqual(len(result["metrics"]), 6)
            self.assertTrue(result["headline_budget_identical"])
            self.assertFalse(result["market_predictive_skill_established"])

    def test_test_targets_do_not_fit_tone_scaler_or_model(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); capture, base, reg, req, feat = fixture(root)
            evaluate_tone_controls(base, req, feat, reg, root/"before")
            split = json.loads((base/"split.json").read_text())
            for row in split["test"]: row["excess_return"] += 10
            write(base/"split.json", split)
            m = json.loads((base/"run_manifest.json").read_text())
            m["outputs_sha256"]["split.json"] = sha((base/"split.json").read_bytes()); write(base/"run_manifest.json", m)
            p = json.loads(req.read_text()); p["source_split_sha256"] = sha((base/"split.json").read_bytes()); write(req, p)
            f = json.loads(feat.read_text()); f["source_split_sha256"] = p["source_split_sha256"]
            f["request_file_sha256"] = sha(req.read_bytes()); write(feat, f)
            evaluate_tone_controls(base, req, feat, reg, root/"after")
            self.assertEqual((root/"before/fitted_tone_models.json").read_bytes(), (root/"after/fitted_tone_models.json").read_bytes())

    def test_missing_feature_pool_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _, base, reg, req, feat = fixture(root)
            f = json.loads(feat.read_text()); f["samples"].pop(); write(feat, f)
            with self.assertRaisesRegex(ValueError, "pool"):
                evaluate_tone_controls(base, req, feat, reg, root/"out")


if __name__ == "__main__": unittest.main()
