"""Synthetic contracts only; these fixtures are not human or market evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import hashlib
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
from move_attribution import capture_inputs, detect_moves, evaluate_attributions, prepare_review


def bars(returns, benchmark_returns=None):
    benchmark_returns = benchmark_returns or [0] * len(returns)
    start = datetime(2023, 1, 1, 14, 30, tzinfo=timezone.utc)
    stock = benchmark = 100.0
    output = [{"symbol": "TEST", "session_index": 0, "at": start.isoformat(),
               "open": stock, "benchmark_open": benchmark}]
    for i, (raw, market) in enumerate(zip(returns, benchmark_returns), 1):
        stock *= 1 + raw
        benchmark *= 1 + market
        output.append({"symbol": "TEST", "session_index": i,
            "at": (start + timedelta(days=i)).isoformat(), "open": stock,
            "benchmark_open": benchmark})
    return output


def detection(n=1):
    return {"parameters": {}, "counts": {}, "moves": [
        {"move_id": f"move-{i}", "symbol": "TEST",
         "interval_start": "2023-02-06T14:30:00Z",
         "interval_end": "2023-02-07T14:30:00Z"} for i in range(n)]}


def news(key="news-1", **updates):
    return {"news_id": key, "symbols": ["TEST"], "entity_verified": True,
            "historical_availability_verified": True, "source_ref": "fixture://news/1",
            "published_at": "2023-02-06T13:00:00Z", "version_at": "2023-02-06T13:00:00Z",
            "received_at": "2023-02-06T13:01:00Z", **updates}


def prediction(i=0, *, decision="attributed", key="news-1"):
    return {"move_id": f"move-{i}", "decision": decision,
            "news_id": key if decision == "attributed" else None,
            "method": "synthetic_contract_fixture", "frozen_at": "2026-11-01T00:00:00Z"}


def label(i=0, *, verdict="attributed", ids=None):
    return {"move_id": f"move-{i}", "verdict": verdict,
            "acceptable_news_ids": (["news-1"] if ids is None else ids) if verdict == "attributed" else [],
            "evidence_refs": ["fixture://synthetic-coverage-audit"],
            "rationale": "Synthetic label used solely to test arithmetic."}


def human_review(labels):
    return {"reviewer": {"kind": "human", "independent": True,
            "blinded_to_predictions": True, "reviewer_id": "synthetic-test-identity",
            "reviewed_at": "2026-11-02T00:00:00Z"}, "cases": labels}


class TestDetection(unittest.TestCase):
    def test_shock_is_not_in_its_own_volatility_and_future_is_irrelevant(self):
        history = [.01, -.01, .01, -.01]
        original = bars(history + [.2])
        result = detect_moves(original, lookback=4)
        self.assertEqual(len(result["moves"]), 1)
        move = result["moves"][0]
        self.assertAlmostEqual(move["prior_sample_std"], statistics.stdev(history))
        self.assertAlmostEqual(move["standardized_excess_move"], .2 / statistics.stdev(history))
        extended = detect_moves(bars(history + [.2, -.8, .9]), lookback=4)
        self.assertEqual(move, extended["moves"][0])
        self.assertEqual(original, bars(history + [.2]))

    def test_missing_session_breaks_window_instead_of_bridging(self):
        rows = bars([.01, -.01, .01, -.01, .2])
        del rows[3]
        result = detect_moves(rows, lookback=3)
        self.assertEqual(result["moves"], [])
        self.assertEqual(result["counts"]["missing_session_intervals"], 1)

    def test_market_move_is_not_stock_specific_excess(self):
        result = detect_moves(bars([.01, -.01, .01, .2], [.01, -.01, .01, .2]), lookback=3)
        self.assertEqual(result["moves"], [])
        self.assertEqual(result["counts"]["zero_volatility_intervals"], 1)

    def test_invalid_prices_dates_and_calendar_fail(self):
        for field, value in [("open", float("nan")), ("open", 0),
                             ("benchmark_open", True), ("at", "2023-01-01"),
                             ("session_index", -1), ("session_index", True)]:
            with self.subTest(field=field, value=value):
                rows = bars([.01, -.01, .2])
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    detect_moves(rows, lookback=2)
        rows = bars([.01, -.01, .2])
        with self.assertRaises(ValueError):
            detect_moves(rows + [rows[0]], lookback=2)
        rows[1]["at"] = rows[0]["at"]
        with self.assertRaises(ValueError):
            detect_moves(rows, lookback=2)


class TestPrecedingCandidates(unittest.TestCase):
    def test_preceding_version_only_and_unassessed_is_not_unexplained(self):
        items = [news(), news("at-start", received_at="2023-02-06T14:30:00Z"),
                 news("late-version", version_at="2023-02-06T15:00:00Z", received_at="2023-02-06T15:01:00Z"),
                 news("old", published_at="2023-02-01T13:00:00Z"),
                 news("retrospective", historical_availability_verified=False)]
        packet = prepare_review(detection(), items)
        case = packet["cases"][0]
        self.assertEqual([n["news_id"] for n in case["candidates"]], ["news-1"])
        self.assertEqual(case["decision_status"], "unassessed")
        self.assertIsNone(case["selected_news_id"])
        self.assertEqual(case["excluded_news_counts"]["not_received_before_move_start"], 2)
        self.assertFalse(packet["causal_explanation_model_implemented"])

    def test_exact_timezones_and_window_boundary(self):
        items = [news("boundary", published_at="2023-02-03T14:30:00Z"),
                 news("latest", received_at="2023-02-06T09:29:59-05:00")]
        packet = prepare_review(detection(), items)
        self.assertEqual([n["news_id"] for n in packet["cases"][0]["candidates"]], ["latest", "boundary"])

    def test_date_only_entity_and_clock_order_are_not_silently_repaired(self):
        items = [news("date-only", published_at="2023-02-06"),
                 news("entity", entity_verified=False),
                 news("clock", version_at="2023-02-06T12:00:00Z")]
        case = prepare_review(detection(), items)["cases"][0]
        self.assertEqual(case["candidates"], [])
        self.assertEqual(case["excluded_news_counts"], {
            "missing_or_invalid_exact_clock": 1, "entity_unverified": 1,
            "inconsistent_version_receipt_order": 1})

    def test_news_ids_and_move_ids_cannot_duplicate(self):
        with self.assertRaises(ValueError):
            prepare_review(detection(), [news(), news()])
        duplicated = detection()
        duplicated["moves"] *= 2
        with self.assertRaises(ValueError):
            prepare_review(duplicated, [])


class TestHumanEvaluation(unittest.TestCase):
    def setUp(self):
        self.packet = prepare_review(detection(3), [news(), news("news-2")])

    def test_no_predictions_or_no_gold_never_means_zero_false_attribution(self):
        result = evaluate_attributions(self.packet)
        self.assertEqual(result["status"], "awaiting_predictions")
        self.assertIsNone(result["metrics"])
        predictions = [prediction(i, decision="unexplained") for i in range(3)]
        result = evaluate_attributions(self.packet, predictions)
        self.assertEqual(result["status"], "awaiting_independent_human_review")
        self.assertEqual(result["model_unexplained_fraction"], 1)
        self.assertIsNone(result["metrics"])
        self.assertEqual(result["human_reviewed_moves"], 0)

    def test_false_claim_rate_uses_claims_and_gold_uncertainty_is_excluded(self):
        predictions = [prediction(0), prediction(1, key="news-2"), prediction(2, decision="unexplained")]
        gold = human_review([label(0), label(1, verdict="unexplained"), label(2, verdict="uncertain")])
        result = evaluate_attributions(self.packet, predictions, gold)
        self.assertEqual(result["status"], "partially_reviewed")
        self.assertEqual(result["human_reviewed_moves"], 3)
        self.assertEqual(result["conclusive_reviewed_moves"], 2)
        self.assertEqual(result["uncertain_reviewed_moves"], 1)
        self.assertEqual(result["metrics"]["false_attribution_rate_among_claims"], .5)
        self.assertEqual(result["metrics"]["decision_accuracy_on_reviewed"], .5)

    def test_partial_review_does_not_use_unreviewed_moves_as_negative_truth(self):
        predictions = [prediction(i) for i in range(3)]
        result = evaluate_attributions(self.packet, predictions, human_review([label(0, verdict="unexplained")]))
        self.assertEqual(result["review_coverage"], 1 / 3)
        self.assertEqual(result["metrics"]["denominator_reviewed_conclusive"], 1)
        self.assertEqual(result["metrics"]["false_attribution_rate_among_claims"], 1)

    def test_abstaining_on_every_move_has_no_claim_precision(self):
        predictions = [prediction(i, decision="unexplained") for i in range(3)]
        gold = human_review([label(i, verdict="unexplained") for i in range(3)])
        result = evaluate_attributions(self.packet, predictions, gold)
        self.assertEqual(result["status"], "evaluated")
        self.assertIsNone(result["metrics"]["false_attribution_rate_among_claims"])
        self.assertEqual(result["metrics"]["decision_accuracy_on_reviewed"], 1)

    def test_gold_can_name_a_missed_source_and_retrieval_miss_remains_visible(self):
        predictions = [prediction(i, decision="unexplained") for i in range(3)]
        gold = human_review([label(0, ids=["outside-retrieval-news"])])
        result = evaluate_attributions(self.packet, predictions, gold)
        self.assertEqual(result["metrics"]["candidate_recall_on_gold_explained"], 0)
        self.assertEqual(result["metrics"]["decision_accuracy_on_reviewed"], 0)

    def test_assistant_nonindependent_or_unblinded_review_is_rejected(self):
        predictions = [prediction(i) for i in range(3)]
        for field, value in [("kind", "assistant"), ("kind", "model"),
                             ("independent", False), ("blinded_to_predictions", False)]:
            with self.subTest(field=field, value=value):
                gold = human_review([label()])
                gold["reviewer"][field] = value
                with self.assertRaises(ValueError):
                    evaluate_attributions(self.packet, predictions, gold)

    def test_predictions_require_complete_unique_cohort_and_eligible_explanations(self):
        valid = [prediction(i) for i in range(3)]
        invalid = [valid[:2], valid + [prediction()], [prediction(0, key="post-move-news"), *valid[1:]]]
        for rows in invalid:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    evaluate_attributions(self.packet, rows)

    def test_gold_requires_evidence_and_predictions_frozen_first(self):
        predictions = [prediction(i) for i in range(3)]
        for mutation in ["no-evidence", "early-review", "duplicate", "invalid-verdict"]:
            gold = human_review([label()])
            if mutation == "no-evidence": gold["cases"][0]["evidence_refs"] = []
            if mutation == "early-review": gold["reviewer"]["reviewed_at"] = "2026-10-31T00:00:00Z"
            if mutation == "duplicate": gold["cases"] *= 2
            if mutation == "invalid-verdict": gold["cases"][0]["verdict"] = "unexplained"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                evaluate_attributions(self.packet, predictions, gold)

    def test_inputs_are_not_mutated(self):
        predictions, gold = [prediction(i) for i in range(3)], human_review([label()])
        before = deepcopy((self.packet, predictions, gold))
        evaluate_attributions(self.packet, predictions, gold)
        self.assertEqual((self.packet, predictions, gold), before)


class TestOfflineCommand(unittest.TestCase):
    def test_capture_checks_raw_source_hashes_and_keeps_date_only_news_unverified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents = {
                "protocol.json": {"symbol": "TEST"},
                "calendar.json": {"complete": True, "sessions": [
                    {"date": "2023-01-03", "open_at": "2023-01-03T14:30:00Z"}]},
                "news.json": [{"source_row_index": 0, "headline": "Synthetic fixture title",
                               "source_date": "2023-01-02"}],
                "prices_000.json": {"bars": {symbol: [{"t": "2023-01-03T05:00:00Z", "o": 100}]
                                               for symbol in ["TEST", "SPY"]}, "next_page_token": None}}
            for name, value in documents.items():
                (root / name).write_text(json.dumps(value))
            (root / "source_prefix.csv").write_text("headline,date\nSynthetic fixture title,2023-01-02\n")
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            manifest = {"status": "complete_bounded_capture", "files_sha256": hashes,
                        "protocol_sha256": hashes["protocol.json"],
                        "bars_summary": {"pagination_complete": True, "records": {"TEST": 1, "SPY": 1}},
                        "requests": [{"file": "prices_000.json", "status": 200,
                                      "params": {"adjustment": "split", "timeframe": "1Day"},
                                      "next_page_token": None}]}
            (root / "manifest.json").write_text(json.dumps(manifest))
            normalized_bars, normalized_news, actual_hashes = capture_inputs(root)
            self.assertEqual(len(normalized_bars), 1)
            self.assertFalse(normalized_news[0]["historical_availability_verified"])
            self.assertFalse(normalized_news[0]["entity_verified"])
            self.assertIsNone(normalized_news[0]["published_at"])
            self.assertEqual(actual_hashes["source_prefix.csv"], hashes["source_prefix.csv"])
            (root / "source_prefix.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                capture_inputs(root)

    def test_cli_produces_private_hashed_pending_receipt_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bars.json").write_text(json.dumps(bars([.01, -.01, .01, -.01, .2])))
            (root / "news.json").write_text("[]")
            command = [sys.executable, str(MODULE / "move_attribution.py"),
                       "--bars", str(root / "bars.json"), "--news", str(root / "news.json"),
                       "--output", str(root / "out"), "--lookback", "4"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((root / "out/evaluation_receipt.json").read_text())
            self.assertEqual(receipt["evaluation"]["total_moves"], 1)
            self.assertIsNone(receipt["evaluation"]["metrics"])
            self.assertFalse(receipt["is_causal_or_trading_alpha_evidence"])
            self.assertEqual(len(receipt["input_sha256"]["bars"]), 64)
            self.assertEqual((root / "out/.gitignore").read_text(), "*\n")
            again = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(again.returncode, 0)


if __name__ == "__main__":
    unittest.main()
