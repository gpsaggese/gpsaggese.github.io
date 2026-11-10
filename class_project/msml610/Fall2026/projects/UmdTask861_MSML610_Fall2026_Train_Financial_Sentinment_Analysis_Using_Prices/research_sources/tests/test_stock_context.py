"""Offline regression tests for timing, price arithmetic and public-data boundaries."""

import copy
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.financial_sentiment_using_prices.stock_context import (
    _atomic_write, build_stock_context, classify_stance, load_stock_context, main,
)


CUTOFF = "2026-10-12T14:45:00Z"
GENERATED = "2026-10-12T15:00:00Z"
SYMBOLS = ("SNDK", "SPY", "QQQ", "SOXX", "MU", "WDC", "STX")


def bar(timestamp, close, *, opening=None, volume=10, vwap=None):
    opening = close if opening is None else opening
    return {"timestamp": timestamp, "open": opening, "high": max(close, opening) + 1,
            "low": min(close, opening) - 1, "close": close, "volume": volume, "vwap": vwap}


def fixture():
    day, days = date(2026, 10, 9), []
    while len(days) < 60:
        if day.weekday() < 5:
            days.append(day)
        day -= timedelta(days=1)
    days.reverse()
    daily = {"request": {"timeframe": "1Day"}, "bars": {
        s: [bar(f"{d.isoformat()}T04:00:00Z", 100 + i) for i, d in enumerate(days)] for s in SYMBOLS}}
    intra = {"request": {"timeframe": "5Min"}, "bars": {
        s: [bar("2026-10-12T13:25:00Z", 149, volume=100000, vwap=149),
            *[bar((datetime(2026, 10, 12, 13, 30, tzinfo=timezone.utc) + timedelta(minutes=5 * i)).isoformat(),
                  150 + i, opening=150 + i, volume=10 * (i + 1), vwap=150 + i - .25) for i in range(16)]]
        for s in SYMBOLS}}
    return daily, intra


def context(daily=None, intra=None, **kwargs):
    defaults = fixture()
    return build_stock_context(daily or defaults[0], intra or defaults[1], cutoff=CUTOFF,
                               generated_at=GENERATED, **kwargs)


def post(id_, text, *, publication="2026-10-12T14:00:00Z", receipt="2026-10-12T14:30:00Z", **kwargs):
    return {"post_id": id_, "query_symbol": "SNDK", "title": text, "excerpt": text,
            "published_at": publication, "response_received_at": receipt,
            "first_seen_in_this_audit_at": "2026-10-12T13:00:00Z", **kwargs}


class StockContextTests(unittest.TestCase):
    def test_complete_bars_end_timestamps_and_opening_decomposition(self):
        result = context()
        own = [p for p in result["intraday"] if p["symbol"] == "SNDK"]
        self.assertEqual(own[0]["timestamp"], "2026-10-12T13:30:00Z")
        self.assertEqual(own[0]["session"], "premarket")
        self.assertEqual(own[1]["session"], "regular")
        self.assertEqual(own[-1]["timestamp"], CUTOFF)
        self.assertEqual(result["summary"]["last_price"], 164)
        self.assertEqual(result["summary"]["previous_close"], 159)
        summary = result["summary"]
        self.assertAlmostEqual(summary["change_pct"], (164 / 159 - 1) * 100)
        self.assertAlmostEqual(1 + summary["change_pct"] / 100,
                               (1 + summary["gap_pct"] / 100) * (1 + summary["open_to_cutoff_pct"] / 100))
        self.assertEqual(summary["excess_vs_spy_pp"], 0)
        self.assertEqual(summary["excess_vs_sector_pp"], 0)

    def test_future_daily_and_unfinished_bars_cannot_change_past_report(self):
        daily, intra = fixture()
        expected = context(daily, intra)
        for s in SYMBOLS:
            daily["bars"][s].extend([{"timestamp": "2026-10-12T04:00:00Z", "close": float("nan")}] * 2)
            intra["bars"][s][-1]["close"] = float("nan")
            intra["bars"][s].append({"timestamp": "2026-10-12T15:00:00Z", "close": -1})
        self.assertEqual(context(daily, intra), expected)

    def test_provider_vwap_weights_volume_and_excludes_premarket(self):
        result = context()
        self.assertEqual(result["summary"]["session_volume"], 1200)
        expected = sum((150 + i - .25) * 10 * (i + 1) for i in range(15)) / 1200
        self.assertAlmostEqual(result["summary"]["session_vwap"], expected)
        self.assertIn("Session VWAP method in this snapshot: volume-weighted provider bar VWAP.", result["limitations"])

    def test_one_missing_provider_vwap_uses_consistent_proxy_for_whole_session(self):
        daily, intra = fixture()
        intra["bars"]["SNDK"][2]["vwap"] = None
        result = context(daily, intra)
        expected = sum((150 + i) * 10 * (i + 1) for i in range(15)) / 1200
        self.assertAlmostEqual(result["summary"]["session_vwap"], expected)
        self.assertTrue(any("method in this snapshot" in x and "proxy" in x for x in result["limitations"]))

    def test_levels_tr_and_moving_averages_only_use_prior_sessions(self):
        daily, intra = fixture()
        result = context(daily, intra, manifest={"price_received_at": GENERATED})
        self.assertEqual(result["summary"]["sma20"], 149.5)
        self.assertEqual(result["summary"]["sma50"], 134.5)
        self.assertEqual(result["summary"]["mean_true_range_14"], 2)
        levels = {r["id"]: r for r in result["levels"]}
        self.assertEqual(levels["previous_low"]["price"], 158)
        self.assertEqual(levels["rolling_low_20"]["price"], 139)
        self.assertEqual(levels["rolling_high_20"]["price"], 160)
        self.assertEqual({r["known_at"] for r in result["levels"]}, {"2026-10-09T20:00:00Z"})
        self.assertTrue(all("reconstructed after the event" in r["basis"] for r in result["levels"]))
        intra["bars"]["SNDK"][1]["low"] = 1
        self.assertEqual(context(daily, intra)["levels"], result["levels"])
        self.assertTrue(any("post_cutoff" in x for x in result["limitations"]))

    def test_daily_common_baseline_and_matching_five_twenty_session_endpoints(self):
        daily, intra = fixture()
        del daily["bars"]["STX"][0]
        result = context(daily, intra)
        first = result["daily"][:7]
        self.assertEqual(len({r["date"] for r in first}), 1)
        self.assertEqual({r["return_pct"] for r in first}, {0})
        own = next(r for r in result["benchmarks"] if r["symbol"] == "SNDK")
        self.assertAlmostEqual(own["return_5d_pct"], (159 / 154 - 1) * 100)
        self.assertAlmostEqual(own["return_20d_pct"], (159 / 139 - 1) * 100)
        del daily["bars"]["SNDK"][-6]
        missing = context(daily, intra)
        own = next(r for r in missing["benchmarks"] if r["symbol"] == "SNDK")
        self.assertIsNone(own["return_5d_pct"])
        self.assertIsNone(missing["summary"]["sma20"])
        self.assertIsNone(missing["summary"]["mean_true_range_14"])

    def test_missing_benchmark_at_same_timestamp_never_uses_stale_price(self):
        daily, intra = fixture()
        del intra["bars"]["SPY"][-2]
        result = context(daily, intra)
        self.assertIsNone(result["summary"]["excess_vs_spy_pp"])
        self.assertIsNotNone(result["summary"]["change_pct"])
        self.assertTrue(any("SPY lacks a bar at" in x for x in result["limitations"]))

    def test_missing_previous_day_is_not_replaced_with_older_close(self):
        daily, intra = fixture()
        del daily["bars"]["SNDK"][-1]
        result = context(daily, intra)
        self.assertIsNone(result["summary"]["change_pct"])
        self.assertIsNone(result["summary"]["previous_close"])
        self.assertEqual(result["levels"], [])
        self.assertFalse(any(r["symbol"] == "SNDK" for r in result["intraday"]))

    def test_duplicate_invalid_and_wrong_timeframe_inputs_fail_explicitly(self):
        for corruption in ("duplicate", "nonfinite", "range", "timeframe", "naive"):
            with self.subTest(corruption=corruption):
                daily, intra = fixture()
                if corruption == "duplicate":
                    intra["bars"]["SNDK"].append(intra["bars"]["SNDK"][0])
                elif corruption == "nonfinite":
                    intra["bars"]["SNDK"][0]["close"] = float("inf")
                elif corruption == "range":
                    intra["bars"]["SNDK"][0]["high"] = 1
                elif corruption == "timeframe":
                    intra["request"]["timeframe"] = "1Min"
                else:
                    intra["bars"]["SNDK"][0]["timestamp"] = "2026-10-12T13:25:00"
                with self.assertRaises(ValueError):
                    context(daily, intra)

    def test_stance_lexicon_negation_mixed_and_unknown_are_not_predictions(self):
        for text, expected in (("看多，准备买入", "bullish"), ("做空并减仓", "bearish"),
                               ("有人看多，有人看空", "mixed"), ("不要抄底", "unknown"),
                               ("不看涨也不看跌", "unknown"), ("今天价格150", "unknown"), ("", "unknown")):
            self.assertEqual(classify_stance(text), expected)

    def test_instrument_name_is_not_a_personal_bullish_stance(self):
        text = "$三倍做多半导体ETF-Direxion (SOXL.US)$ $闪迪 (SNDK.US)$ 今日列表"
        self.assertEqual(classify_stance(text), "unknown")
        self.assertEqual(classify_stance(text + "。本人看多。"), "bullish")

    def test_community_versions_counts_blank_text_and_late_receipts(self):
        rows = [post("a", "看多"), post("b", "看空"), post("c", "看多，也看空"), post("d", ""),
                post("e", "今天行情"), post(None, "买入"),
                post("a", "卖出", receipt="2026-10-12T15:00:00Z")]
        original = copy.deepcopy(rows)
        result = context(community_rows=rows)["community"]
        self.assertEqual(result["records"], 7)
        self.assertEqual(result["unique_posts"], 5)
        self.assertEqual(result["available_at_cutoff"], 4)
        self.assertEqual([result[k] for k in ("bullish", "bearish", "mixed", "unknown")], [0, 2, 1, 2])
        self.assertEqual(result["classified"], 3)
        self.assertEqual(result["stance_balance"], -100)
        self.assertEqual(result["received_at"], "2026-10-12T15:00:00Z")
        self.assertNotRegex(result["method"], r"[\u3400-\u9fff]")
        self.assertIn("not confidence", result["method"])
        self.assertEqual(rows, original)
        self.assertNotIn("post_id", json.dumps(result))

    def test_early_publication_late_receipt_cannot_backdate_text(self):
        rows = [post("a", "看多", receipt="2026-10-12T15:00:00Z"), post("b", "买入", receipt=None)]
        result = context(community_rows=rows)["community"]
        self.assertEqual(result["available_at_cutoff"], 0)
        self.assertEqual(result["bullish"], 2)
        self.assertIsNone(context(community_rows=[post("x", "")])["community"]["stance_balance"])

    def test_public_news_timing_and_raw_field_allowlist(self):
        news = {"status": "Observed", "received_at": GENERATED, "records_total": 50,
                "precutoff_publications": 49, "usable_at_cutoff": 49,
                "raw": "SECRET_BODY", "items": [{"id": "n", "source": "Company IR",
                    "published_at": "2026-10-12T13:00:00Z", "received_at": GENERATED,
                    "timing": "available_at_cutoff", "association": "direct",
                    "summary": "The company announced a new product.", "url": "https://example.com/news",
                    "evidence_type": "company_statement", "body": "SECRET_BODY", "account": "SECRET_ACCOUNT"}]}
        sources = [{"id": "price", "label": "Market prices", "status": "Observed", "records": 70,
                    "received_at": GENERATED, "detail": "Saved daily bars", "local_path": "/Users/private/input.json"}]
        result = context(news_context=news, context_sources=sources)
        self.assertEqual(result["news"]["records_total"], 50)
        self.assertEqual(len(result["news"]["items"]), 1)
        self.assertEqual(result["news"]["usable_at_cutoff"], 0)
        self.assertEqual(result["news"]["items"][0]["timing"], "received_after_cutoff")
        serialized = json.dumps(result, allow_nan=False)
        self.assertNotIn("SECRET", serialized)
        self.assertNotIn("/Users/", serialized)
        news["items"][0]["url"] = "https://user:password@example.com/news"
        with self.assertRaises(ValueError):
            context(news_context=news)

    def test_unknown_collection_receipt_cannot_certify_aggregate_news_availability(self):
        news = {"status": "Observed", "received_at": None, "records_total": 10,
                "precutoff_publications": 10, "usable_at_cutoff": 10, "items": []}
        self.assertEqual(context(news_context=news)["news"]["usable_at_cutoff"], 0)

    def test_loader_missing_optional_inputs_ignores_human_case_and_cli(self):
        daily, intra = fixture()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "input"
            root.mkdir()
            (root / "daily.json").write_text(json.dumps(daily))
            (root / "intraday.json").write_text(json.dumps(intra))
            (root / "levels.json").write_text("NOT VALID JSON: human-selected prices must never be read")
            (root / "case_review.json").write_text("SECRET_BODY")
            (root / "generation_audit.json").write_text("original archive audit")
            original = {path.name: path.read_bytes() for path in root.iterdir()}
            result = load_stock_context(root, cutoff=CUTOFF, generated_at=GENERATED)
            self.assertEqual(result["news"]["status"], "Not collected")
            self.assertEqual(result["community"]["unique_posts"], 0)
            self.assertIsNone(result["price_received_at"])
            output = Path(folder) / "public" / "public.json"
            audit_output = Path(folder) / "private" / "new_generation_audit.json"
            main(["--input-dir", str(root), "--output", str(output),
                  "--audit-output", str(audit_output), "--cutoff", CUTOFF, "--symbol", "SNDK"])
            written = json.loads(output.read_text())
            self.assertEqual(written["summary"], result["summary"])
            self.assertNotIn(folder, output.read_text())
            audit = json.loads(audit_output.read_text())
            self.assertEqual(audit["public_output_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(audit["inputs"]["daily.json"]["sha256"], hashlib.sha256((root / "daily.json").read_bytes()).hexdigest())
            self.assertFalse(audit["inputs"]["news_context.json"]["present"])
            self.assertNotIn("levels.json", audit["inputs"])
            self.assertEqual({path.name: path.read_bytes() for path in root.iterdir()}, original)

    def test_failed_atomic_replacement_preserves_last_report_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "public.json"
            path.write_text("original report")
            with patch.object(Path, "replace", side_effect=OSError("simulated replacement failure")):
                with self.assertRaises(OSError):
                    _atomic_write(path, "replacement report")
            self.assertEqual(path.read_text(), "original report")
            self.assertEqual(list(Path(folder).iterdir()), [path])


class StockContextPublicationTests(unittest.TestCase):
    """Use disposable inputs only; a failed rerun must preserve previous evidence."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root / "archive"
        self.archive.mkdir()
        daily, intra = fixture()
        (self.archive / "daily.json").write_text(json.dumps(daily))
        (self.archive / "intraday.json").write_text(json.dumps(intra))
        (self.archive / "generation_audit.json").write_text("original archive audit")
        self.output = self.root / "public" / "snapshot.json"
        self.output.parent.mkdir()
        self.output.write_text("previous public report")
        self.audit = self.root / "private" / "new_generation_audit.json"
        self.original_report = self.output.read_bytes()
        self.original_inputs = self._input_bytes()

    def _input_bytes(self):
        return {str(path.relative_to(self.archive)): path.read_bytes()
                for path in self.archive.rglob("*") if path.is_file()}

    def _run(self, *, output=None, audit=None, input_dir=None):
        with patch("builtins.print"):
            main(["--input-dir", str(input_dir or self.archive),
                  "--output", str(output or self.output), "--audit-output", str(audit or self.audit),
                  "--cutoff", CUTOFF])

    def _assert_preserved(self):
        self.assertEqual(self.output.read_bytes(), self.original_report)
        self.assertEqual(self._input_bytes(), self.original_inputs)
        self.assertEqual(list(self.root.rglob(".*.tmp")), [])

    def test_successful_update_uses_fresh_private_audit_and_preserves_archive(self):
        self._run()
        audit = json.loads(self.audit.read_text())
        report = json.loads(self.output.read_text())
        self.assertEqual(report["symbol"], "SNDK")
        self.assertEqual(audit["public_output_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(self._input_bytes(), self.original_inputs)
        self.assertNotIn(str(self.root), self.output.read_text())

    def test_existing_audit_refuses_rerun_before_replacing_previous_report(self):
        self.audit.parent.mkdir()
        self.audit.write_bytes(b"existing private audit")
        with self.assertRaises(FileExistsError):
            self._run()
        self.assertEqual(self.audit.read_bytes(), b"existing private audit")
        self._assert_preserved()

    def test_existing_audit_symlinks_are_never_replaced(self):
        self.audit.parent.mkdir()
        for target_exists in (False, True):
            with self.subTest(target_exists=target_exists):
                target = self.root / f"older_audit_{target_exists}.json"
                if target_exists:
                    target.write_bytes(b"older audit")
                self.audit.symlink_to(target)
                with self.assertRaises(FileExistsError):
                    self._run()
                self.assertTrue(self.audit.is_symlink())
                if target_exists:
                    self.assertEqual(target.read_bytes(), b"older audit")
                else:
                    self.assertFalse(target.exists())
                self.audit.unlink()
                self._assert_preserved()

    def test_report_and_audit_cannot_share_a_destination(self):
        with self.assertRaises(ValueError):
            self._run(audit=self.output)
        nested = self.output.parent / "nested"
        nested.mkdir()
        route = self.root / "report_route"
        route.symlink_to(nested, target_is_directory=True)
        with self.assertRaises(ValueError):
            self._run(audit=route / ".." / self.output.name)
        self._assert_preserved()

    def test_all_outputs_inside_archive_are_refused_including_symlink_routes(self):
        alias = self.root / "archive_alias"
        alias.symlink_to(self.archive, target_is_directory=True)
        outside = self.root / "outside"
        outside.mkdir()
        (self.archive / "escape").symlink_to(outside, target_is_directory=True)
        for destination in (self.archive / "new.json", alias / "new.json",
                            self.archive / "escape" / "new.json", outside / "new.json"):
            for role in ("output", "audit"):
                with self.subTest(destination=destination, role=role):
                    with self.assertRaises(ValueError):
                        self._run(**{role: destination})
                    self.assertFalse(destination.exists())
                    self.assertFalse(self.audit.exists())
                    self._assert_preserved()

    def test_external_targets_of_archived_input_symlinks_are_protected(self):
        for name in ("daily.json", "generation_audit.json"):
            source = self.archive / name
            external = self.root / f"external_{name}"
            source.rename(external)
            source.symlink_to(external)
            original = external.read_bytes()
            for role in ("output", "audit"):
                with self.subTest(name=name, role=role):
                    with self.assertRaises(ValueError):
                        self._run(**{role: external})
                    self.assertEqual(external.read_bytes(), original)
                    self._assert_preserved()

    def test_symlink_parent_segments_cannot_disguise_a_write_inside_archive(self):
        nested = self.archive / "nested"
        nested.mkdir()
        route = self.root / "routing"
        route.mkdir()
        (route / "hop").symlink_to(nested, target_is_directory=True)
        disguised = route / "hop" / ".." / "new.json"
        for role in ("output", "audit"):
            with self.subTest(role=role):
                with self.assertRaises(ValueError):
                    self._run(**{role: disguised})
                self.assertFalse((self.archive / "new.json").exists())
                self._assert_preserved()

    def test_input_symlink_parent_segments_resolve_to_the_real_archive(self):
        nested = self.root / "nested"
        nested.mkdir()
        route = self.root / "routing"
        route.mkdir()
        (route / "hop").symlink_to(nested, target_is_directory=True)
        disguised_input = route / "hop" / ".." / "archive"
        with self.assertRaises(ValueError):
            self._run(input_dir=disguised_input, output=self.archive / "new.json")
        self.assertFalse((self.archive / "new.json").exists())
        self._assert_preserved()

    def test_audit_cannot_be_mistaken_for_report_output(self):
        original = json.dumps({"schema_version": "stock-context-generation-audit-1.0"}).encode()
        self.output.write_bytes(original)
        with self.assertRaises(FileExistsError):
            self._run()
        self.assertEqual(self.output.read_bytes(), original)
        self.assertFalse(self.audit.exists())
        self.assertEqual(self._input_bytes(), self.original_inputs)

    def test_invalid_input_keeps_previous_report_and_creates_no_audit(self):
        daily = json.loads((self.archive / "daily.json").read_text())
        daily["bars"]["SNDK"][0]["close"] = -1
        (self.archive / "daily.json").write_text(json.dumps(daily))
        self.original_inputs = self._input_bytes()
        with self.assertRaises(ValueError):
            self._run()
        self.assertFalse(self.audit.exists())
        self._assert_preserved()

    def test_audit_publication_failure_preserves_previous_report(self):
        with patch("research.financial_sentiment_using_prices.stock_context.os.link",
                   side_effect=OSError("simulated audit publication failure")):
            with self.assertRaises(OSError):
                self._run()
        self.assertFalse(self.audit.exists())
        self._assert_preserved()

    def test_concurrent_audit_creation_is_never_overwritten_or_deleted(self):
        def conflict(_temporary, destination):
            Path(destination).write_bytes(b"concurrent audit")
            raise FileExistsError("simulated concurrent audit creation")

        with patch("research.financial_sentiment_using_prices.stock_context.os.link", side_effect=conflict):
            with self.assertRaises(FileExistsError):
                self._run()
        self.assertEqual(self.audit.read_bytes(), b"concurrent audit")
        self._assert_preserved()

    def test_report_publication_failure_rolls_back_only_new_audit(self):
        with patch.object(Path, "replace", side_effect=OSError("simulated report publication failure")):
            with self.assertRaises(OSError):
                self._run()
        self.assertFalse(self.audit.exists())
        self._assert_preserved()

    def test_report_failure_preserves_audit_replaced_by_another_writer(self):
        other_audit = self.root / "other_audit.json"
        other_audit.write_bytes(b"another writer's audit")

        def fail_report(_temporary, _output):
            os.replace(other_audit, self.audit)
            raise OSError("simulated report failure after another writer replaced the audit")

        with patch.object(Path, "replace", fail_report):
            with self.assertRaises(OSError):
                self._run()
        self.assertEqual(self.audit.read_bytes(), b"another writer's audit")
        self._assert_preserved()


if __name__ == "__main__":
    unittest.main()
