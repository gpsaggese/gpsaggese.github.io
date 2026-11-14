"""Meaningful timing, provenance, missingness and calibration-isolation checks."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_signal_adapter import (
    FEATURE_NAMES, RETROSPECTIVE_MODE, STRICT_MODE, SentimentResidualAdapter,
    audit_coverage, build_asof_features, fit_residual_calibration, load_archives,
    make_observation,
)


def observation(**updates):
    fields = dict(source="news", source_id="n1", symbols=["SNDK"], title="SNDK upgraded",
                  published_at="2026-10-01T13:00:00Z", updated_at="2026-10-01T13:00:00Z",
                  received_at="2026-10-01T14:00:00Z", receipt_verified=True)
    fields.update(updates)
    return make_observation(**fields)


def snapshot(day=1, mode=STRICT_MODE):
    timestamp = f"2026-10-{day:02d}T14:05:00+00:00"
    row = observation(source_id=f"n{day}", published_at=timestamp, updated_at=timestamp, received_at=timestamp)
    return build_asof_features([row], "SNDK", f"2026-10-{day:02d}T14:10:00+00:00", mode=mode)


def examples(mode=STRICT_MODE):
    return [{"snapshot": snapshot(day, mode), "base_prediction": 0.001,
             "realized_return": 0.002 + day * 0.00001,
             "label_end_at": f"2026-10-{day:02d}T14:40:00+00:00",
             "target_horizon_minutes": 30} for day in range(1, 6)]


class TimingTests(unittest.TestCase):
    def test_receipt_and_processing_delay_are_respected(self):
        row = observation()
        before = build_asof_features([row], "SNDK", "2026-10-01T14:00:59Z")
        at = build_asof_features([row], "SNDK", "2026-10-01T14:01:00Z")
        self.assertFalse(before["has_text"])
        self.assertTrue(at["primary_asof_eligible"])
        self.assertEqual(at["evidence"][0]["available_at"], "2026-10-01T14:01:00+00:00")

    def test_missing_receipt_does_not_become_neutral_observation(self):
        row = observation(received_at=None, receipt_verified=False)
        strict = build_asof_features([row], "SNDK", "2026-10-01T15:00:00Z")
        exploratory = build_asof_features([row], "SNDK", "2026-10-01T15:00:00Z", mode=RETROSPECTIVE_MODE)
        self.assertEqual(strict["features"]["any_text_missing"], 1.0)
        self.assertEqual(strict["source_counts"]["news"], 0)
        self.assertTrue(exploratory["has_text"])
        self.assertFalse(exploratory["primary_asof_eligible"])

    def test_future_revised_text_cannot_replace_earlier_version(self):
        old = observation(title="SNDK upgraded")
        new = observation(title="SNDK downgraded", updated_at="2026-10-01T15:00:00Z", received_at="2026-10-01T15:05:00Z")
        before = build_asof_features([new, old], "SNDK", "2026-10-01T14:30:00Z")
        after = build_asof_features([new, old], "SNDK", "2026-10-01T15:10:00Z")
        self.assertEqual(before["features"]["news_stance_balance"], 1.0)
        self.assertEqual(after["features"]["news_stance_balance"], -1.0)
        self.assertEqual(after["source_counts"]["news"], 1)

    def test_published_old_news_received_today_is_not_new(self):
        row = observation(received_at="2026-10-05T14:00:00Z")
        self.assertFalse(build_asof_features([row], "SNDK", "2026-10-05T14:05:00Z")["has_text"])

    def test_wrong_company_has_no_feature(self):
        self.assertFalse(build_asof_features([observation()], "NVDA", "2026-10-01T14:05:00Z")["has_text"])

    def test_questions_and_negation_are_unknown(self):
        for title in ["Should SNDK be upgraded?", "SNDK not upgraded", "如果看涨就买入"]:
            snap = build_asof_features([observation(title=title)], "SNDK", "2026-10-01T14:05:00Z")
            self.assertEqual(snap["features"]["news_scored_fraction"], 0)
            self.assertEqual(snap["features"]["any_text_missing"], 0)

    def test_stance_is_a_feature_not_direct_trade(self):
        snap = build_asof_features([observation()], "SNDK", "2026-10-01T14:05:00Z")
        result = SentimentResidualAdapter().predict(-0.001, snap)
        self.assertEqual(result["prediction"], -0.001)
        self.assertFalse(result["text_changed_prediction"])

    def test_multicompany_stance_is_not_assigned_to_every_company(self):
        snap = build_asof_features([observation(symbols=["SNDK", "NVDA"])], "SNDK", "2026-10-01T14:05:00Z")
        self.assertEqual(snap["source_counts"]["news"], 1)
        self.assertEqual(snap["features"]["news_scored_fraction"], 0)

    def test_naive_or_negative_delay_fails(self):
        with self.assertRaises(ValueError):
            build_asof_features([], "SNDK", "2026-10-01T14:05:00")
        with self.assertRaises(ValueError):
            build_asof_features([], "SNDK", "2026-10-01T14:05:00Z", processing_delay_seconds=-1)

    def test_old_comparison_interval_has_zero_available_text(self):
        decisions = [{"symbol": "SNDK", "decision_at": "2026-09-30T14:05:00Z"}]
        result = audit_coverage([observation()], decisions)
        self.assertEqual(result["primary_asof_eligible_decisions"], 0)


class CalibrationTests(unittest.TestCase):
    def fit(self, rows=None, mode=STRICT_MODE):
        return fit_residual_calibration(rows or examples(mode), "2026-10-06T13:00:00Z",
                                        minimum_text_rows=5, minimum_sessions=5,
                                        minimum_nonoverlapping_intervals=5, mode=mode)

    def test_insufficient_sessions_falls_back(self):
        fitted = fit_residual_calibration(examples()[:1], "2026-10-06T13:00:00Z")
        self.assertEqual(fitted["status"], "unavailable")
        out = SentimentResidualAdapter(fitted).predict(0.001, snapshot(7))
        self.assertEqual(out["prediction"], 0.001)

    def test_only_mature_past_labels_fit(self):
        rows = examples()
        original = self.fit(rows)
        future = copy.deepcopy(rows[0])
        future["snapshot"]["decision_at"] = "2026-10-07T13:00:00Z"
        future["label_end_at"] = "2026-10-07T13:30:00Z"
        future["realized_return"] = 1000.0
        modified = self.fit([*rows, future])
        self.assertEqual(original["weights"], modified["weights"])
        self.assertEqual(original["intercept"], modified["intercept"])
        self.assertEqual(modified["excluded_counts"]["outcome_not_mature_before_training_cutoff"], 1)

    def test_capped_same_horizon_residual(self):
        out = SentimentResidualAdapter(self.fit(), max_abs_adjustment=0.0003).predict(0.001, snapshot(7))
        self.assertTrue(out["applied"])
        self.assertAlmostEqual(out["adjustment"], 0.0003)
        self.assertAlmostEqual(out["prediction"], 0.0013)
        self.assertEqual(out["target_horizon_minutes"], 30)

    def test_future_calibration_and_cross_horizon_rejected(self):
        fitted = self.fit()
        self.assertFalse(SentimentResidualAdapter(fitted).predict(0.0, snapshot(2))["applied"])
        wrong = SentimentResidualAdapter(fitted, target_horizon_minutes=5).predict(0.0, snapshot(7))
        self.assertEqual(wrong["fallback_reason"], "horizon_mismatch")
        rows = examples()
        rows[0]["target_horizon_minutes"] = 10 * 24 * 60
        with self.assertRaises(ValueError):
            self.fit(rows)

    def test_declared_thirty_minutes_cannot_hide_ten_day_target(self):
        rows = examples()
        rows[0]["label_end_at"] = "2026-10-04T14:40:00Z"
        with self.assertRaises(ValueError):
            self.fit(rows)

    def test_explicit_next_open_label_start(self):
        rows = examples()
        for row in rows:
            day = row["snapshot"]["decision_at"][:10]
            row["label_start_at"] = day + "T14:15:00Z"
            row["label_end_at"] = day + "T14:45:00Z"
        self.assertEqual(self.fit(rows)["status"], "fitted_development_only")

    def test_publication_assumption_cannot_mix_with_primary(self):
        fitted = self.fit(mode=RETROSPECTIVE_MODE)
        out = SentimentResidualAdapter(fitted).predict(0.0, snapshot(7))
        self.assertEqual(out["fallback_reason"], "availability_mode_mismatch")
        exploratory = SentimentResidualAdapter(fitted).predict(0.0, snapshot(7, RETROSPECTIVE_MODE))
        self.assertTrue(exploratory["applied"])
        self.assertFalse(exploratory["primary_asof_eligible"])

    def test_duplicate_decision_rows_fail(self):
        rows = examples()
        with self.assertRaises(ValueError):
            self.fit([*rows, rows[0]])

    def test_no_text_prediction_exactly_matches_C(self):
        snap = build_asof_features([], "SNDK", "2026-10-07T14:10:00Z")
        out = SentimentResidualAdapter(self.fit()).predict(-0.00017, snap)
        self.assertEqual(out["prediction"], -0.00017)
        self.assertEqual(out["adjustment"], 0.0)


class ArchiveTests(unittest.TestCase):
    def make_archive(self, directory, receipt="2026-10-01T14:00:00Z"):
        article = {"id": "1", "symbols": ["SNDK"], "headline": "SNDK upgraded",
                   "content": "A retained body.", "created_at": "2026-10-01T13:00:00Z",
                   "updated_at": "2026-10-01T13:00:00Z"}
        raw = json.dumps({"news": [article]}).encode()
        (directory / "news_raw_000.json").write_bytes(raw)
        (directory / "manifest.json").write_text(json.dumps({"requests": [{"file": "news_raw_000.json", "status": 200,
            "sha256": hashlib.sha256(raw).hexdigest(), "response_received_at": receipt}]}))
        normalized = dict(article, text=article["content"], retrieved_at=receipt)
        (directory / "news_normalized.json").write_text(json.dumps([normalized]))

    def test_manifest_verified_and_repeated_captures_keep_first_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "one", Path(temp) / "two"
            first.mkdir(); second.mkdir()
            self.make_archive(first)
            self.make_archive(second, "2026-10-02T14:00:00Z")
            rows, audit = load_archives([second, first])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["received_at"], "2026-10-01T14:00:00+00:00")
            self.assertEqual(audit["versions_with_verified_receipt"], 1)

    def test_tampered_response_fails_before_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_archive(root)
            (root / "news_raw_000.json").write_text('{}')
            with self.assertRaises(ValueError):
                load_archives([root])

    def test_hand_edited_normalized_text_is_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_archive(root)
            p = root / "news_normalized.json"
            rows = json.loads(p.read_text())
            rows[0]["text"] = "Known later price outcome."
            p.write_text(json.dumps(rows))
            observations, audit = load_archives([root])
            self.assertEqual(observations, [])
            self.assertEqual(len(audit["exclusions"]), 1)


if __name__ == "__main__":
    unittest.main()
