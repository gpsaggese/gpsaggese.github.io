"""Offline, time-bounded move review and independently labeled evaluation.

Candidate retrieval is a recency baseline, not an LLM or a causal explanation.
No candidate is automatically promoted to an explanation. A missing human review
produces unavailable evaluation metrics, never a zero false-attribution rate.
All inputs are local JSON; this module performs no collection or trading.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo


def aware(value):
    """Reject date-only/naive clocks rather than inventing a publication time."""
    if not isinstance(value, str):
        raise ValueError("Require an explicit timezone-aware timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Require an explicit timezone-aware timestamp")
    return result.astimezone(timezone.utc)


def finite(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (positive and value <= 0)):
        raise ValueError("Invalid finite " + name)
    return float(value)


def identity(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing " + name)
    return value


def detect_moves(bars, *, lookback=20, sigma_threshold=3.0):
    """Detect open-to-open excess moves using only earlier, contiguous returns.

    Rows require symbol, session_index, at, open and benchmark_open. A session
    index comes from an independently supplied calendar, not enumerate(missing
    bars). A missing session breaks the volatility window; it is not bridged.
    News retrieval subsequently ends BEFORE the start of each move interval.
    """
    if type(lookback) is not int or lookback < 2:
        raise ValueError("Require at least two prior returns")
    sigma_threshold = finite(sigma_threshold, "sigma threshold", positive=True)
    companies = defaultdict(list)
    for bar in bars:
        symbol = identity(bar.get("symbol"), "symbol")
        index = bar.get("session_index")
        if type(index) is not int or index < 0:
            raise ValueError("Require a nonnegative calendar session index")
        companies[symbol].append({"index": index, "at": aware(bar.get("at")),
            "open": finite(bar.get("open"), "open", positive=True),
            "benchmark": finite(bar.get("benchmark_open"), "benchmark open", positive=True)})
    moves, counts = [], Counter()
    for symbol, rows in sorted(companies.items()):
        rows.sort(key=lambda row: row["index"])
        if len({row["index"] for row in rows}) != len(rows):
            raise ValueError("Duplicate company/session price")
        if any(a["at"] >= b["at"] for a, b in zip(rows, rows[1:])):
            raise ValueError("Calendar indices and timestamps disagree")
        history = []
        for previous, current in zip(rows, rows[1:]):
            if current["index"] != previous["index"] + 1:
                history = []
                counts["missing_session_intervals"] += 1
                continue
            raw = current["open"] / previous["open"] - 1
            benchmark = current["benchmark"] / previous["benchmark"] - 1
            excess = raw - benchmark
            for name, value in [("stock return", raw), ("benchmark return", benchmark), ("excess return", excess)]:
                finite(value, name)
            counts["observed_intervals"] += 1
            # Compute the baseline before adding the current interval.
            if len(history) < lookback:
                counts["warmup_intervals"] += 1
            else:
                baseline = history[-lookback:]
                mean, std = statistics.mean(baseline), statistics.stdev(baseline)
                if std <= 1e-12:
                    counts["zero_volatility_intervals"] += 1
                else:
                    counts["tested_intervals"] += 1
                    z = (excess - mean) / std
                    if abs(z) >= sigma_threshold:
                        start, end = previous["at"].isoformat(), current["at"].isoformat()
                        moves.append({"move_id": f"{symbol}:{start}:{end}", "symbol": symbol,
                            "interval_start": start, "interval_end": end,
                            "raw_return": raw, "benchmark_return": benchmark,
                            "excess_return": excess, "prior_mean": mean,
                            "prior_sample_std": std, "prior_return_count": lookback,
                            "standardized_excess_move": z})
            history.append(excess)
    return {"moves": moves, "counts": dict(counts), "parameters": {
        "lookback": lookback, "sigma_threshold": sigma_threshold,
        "price_kind": "open_to_open", "return_kind": "stock_minus_benchmark",
        "volatility_kind": "prior_sample_standard_deviation"}}


def prepare_review(detection, news, *, preceding_hours=72):
    """Keep exact, received pre-move versions; unverified timing stays excluded.

    Input news requires news_id, symbols, entity_verified, published_at,
    version_at, received_at, historical_availability_verified and source_ref.
    Date-only 2023 headlines cannot satisfy this strict attribution contract.
    The packet is private: it can contain supplied headlines and source links.
    """
    preceding_hours = finite(preceding_hours, "news window", positive=True)
    lookup = {}
    for item in news:
        key = identity(item.get("news_id"), "news identity")
        if key in lookup:
            raise ValueError("Duplicate news identity; resolve versions explicitly")
        symbols = item.get("symbols")
        if not isinstance(symbols, list) or any(not isinstance(s, str) for s in symbols):
            raise ValueError("Require explicit news symbols")
        lookup[key] = item
    cases, move_ids = [], set()
    for move in detection["moves"]:
        move_id = identity(move.get("move_id"), "move identity")
        if move_id in move_ids:
            raise ValueError("Duplicate move identity")
        move_ids.add(move_id)
        start, end = aware(move["interval_start"]), aware(move["interval_end"])
        if end <= start:
            raise ValueError("Move interval must have positive duration")
        lower = start - timedelta(hours=preceding_hours)
        candidates, rejected = [], Counter()
        for key, item in sorted(lookup.items()):
            if move["symbol"] not in item["symbols"]:
                continue
            reason = None
            if item.get("entity_verified") is not True:
                reason = "entity_unverified"
            elif item.get("historical_availability_verified") is not True:
                reason = "historical_availability_unverified"
            else:
                try:
                    published = aware(item.get("published_at"))
                    version = aware(item.get("version_at"))
                    received = aware(item.get("received_at"))
                except ValueError:
                    reason = "missing_or_invalid_exact_clock"
                else:
                    if not published <= version <= received:
                        reason = "inconsistent_version_receipt_order"
                    elif received >= start:
                        reason = "not_received_before_move_start"
                    elif published < lower:
                        reason = "published_outside_preceding_window"
                    elif not isinstance(item.get("source_ref"), str) or not item["source_ref"].strip():
                        reason = "missing_source_reference"
            if reason:
                rejected[reason] += 1
                continue
            candidate = {k: item[k] for k in ["news_id", "source_ref", "published_at", "version_at", "received_at"]}
            candidate["headline"] = item.get("headline", "")
            candidates.append(candidate)
        candidates.sort(key=lambda row: (-aware(row["received_at"]).timestamp(), row["news_id"]))
        cases.append({**move, "candidates": candidates, "excluded_news_counts": dict(rejected),
            "decision_status": "unassessed", "selected_news_id": None})
    return {"schema_version": 1, "detection": {k: detection[k] for k in ["counts", "parameters"]},
        "preceding_hours": preceding_hours, "candidate_ranking": "receipt_recency_only",
        "causal_explanation_model_implemented": False, "cases": cases,
        "limitations": [
            "Recency ranks retrieval candidates; it does not establish that news caused a move.",
            "News received during or after the daily move interval is excluded; intraday onset is unknown.",
            "Missing historical clocks are not replaced with a date-only proxy or a later receipt.",
            "Unassessed cases are not unexplained ground truth."]}


def evaluate_attributions(packet, predictions=None, human_review=None):
    """Evaluate frozen predictions only against declared independent human labels.

    This validates an explicit provenance contract, not the reviewer's identity.
    Human review must declare independence and blindness to model predictions.
    External news IDs can appear in gold: a retrieval miss must remain a miss.
    All metrics state their actual reviewed denominator; absent gold stays null.
    """
    cases = {c["move_id"]: c for c in packet["cases"]}
    if len(cases) != len(packet["cases"]):
        raise ValueError("Duplicate review case")
    result = {"status": "awaiting_predictions", "total_moves": len(cases),
        "predictions": 0, "model_unexplained_fraction": None,
        "human_reviewed_moves": 0, "conclusive_reviewed_moves": 0,
        "review_coverage": 0.0 if cases else None, "metrics": None,
        "provenance_authentication": "declared_by_input_not_independently_authenticated"}
    if predictions is None:
        if human_review is not None:
            raise ValueError("Freeze predictions before evaluating human labels")
        return result
    lookup = {}
    for prediction in predictions:
        key = prediction.get("move_id")
        if key not in cases or key in lookup:
            raise ValueError("Unknown or duplicate prediction identity")
        decision, selected = prediction.get("decision"), prediction.get("news_id")
        if decision not in {"attributed", "unexplained"}:
            raise ValueError("Predictions must explicitly attribute or abstain")
        candidates = {n["news_id"] for n in cases[key]["candidates"]}
        if (decision == "attributed" and selected not in candidates) or (decision == "unexplained" and selected is not None):
            raise ValueError("Selected explanation is not an eligible preceding candidate")
        identity(prediction.get("method"), "prediction method")
        aware(prediction.get("frozen_at"))
        lookup[key] = prediction
    if set(lookup) != set(cases):
        raise ValueError("Predictions must cover the entire frozen move cohort")
    result.update(status="awaiting_independent_human_review", predictions=len(lookup),
        model_unexplained_fraction=sum(p["decision"] == "unexplained" for p in predictions) / len(cases) if cases else None)
    if human_review is None:
        return result
    reviewer = human_review.get("reviewer", {})
    if (reviewer.get("kind") != "human" or reviewer.get("independent") is not True
            or reviewer.get("blinded_to_predictions") is not True):
        raise ValueError("Require explicitly independent, blinded human review; assistant/model labels are not gold")
    identity(reviewer.get("reviewer_id"), "reviewer identity")
    reviewed_at = aware(reviewer.get("reviewed_at"))
    labels, conclusive = {}, []
    for label in human_review.get("cases", []):
        key = label.get("move_id")
        if key not in cases or key in labels:
            raise ValueError("Unknown or duplicate reviewed move")
        if aware(lookup[key]["frozen_at"]) > reviewed_at:
            raise ValueError("Predictions were not frozen before review")
        verdict, ids = label.get("verdict"), label.get("acceptable_news_ids")
        if verdict not in {"attributed", "unexplained", "uncertain"}:
            raise ValueError("Unknown human verdict")
        if not isinstance(ids, list) or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError("Invalid human explanation identities")
        if (verdict == "attributed" and not ids) or (verdict != "attributed" and ids):
            raise ValueError("Human verdict and explanation identities disagree")
        refs = label.get("evidence_refs")
        if not isinstance(refs, list) or not refs or any(not isinstance(v, str) or not v.strip() for v in refs):
            raise ValueError("Every review requires source references or a coverage audit")
        identity(label.get("rationale"), "review rationale")
        labels[key] = label
        if verdict != "uncertain":
            conclusive.append(label)
    claimed, false_claimed, correct, gold_unexplained, retrieval_hits, gold_explained = 0, 0, 0, 0, 0, 0
    for label in conclusive:
        key, verdict = label["move_id"], label["verdict"]
        prediction = lookup[key]
        gold_unexplained += verdict == "unexplained"
        if verdict == "attributed":
            gold_explained += 1
            retrieval_hits += bool(set(label["acceptable_news_ids"]) & {n["news_id"] for n in cases[key]["candidates"]})
        match = (prediction["decision"] == "unexplained" and verdict == "unexplained") or (
            prediction["decision"] == "attributed" and prediction["news_id"] in label["acceptable_news_ids"])
        correct += match
        if prediction["decision"] == "attributed":
            claimed += 1
            false_claimed += not match
    n = len(conclusive)
    result.update(status="evaluated" if len(labels) == len(cases) and n == len(cases) and n else "partially_reviewed",
        human_reviewed_moves=len(labels), conclusive_reviewed_moves=n,
        uncertain_reviewed_moves=len(labels) - n, review_coverage=len(labels) / len(cases) if cases else None)
    if n:
        result["metrics"] = {"denominator_reviewed_conclusive": n,
            "attribution_claims_on_reviewed": claimed, "false_attributions": false_claimed,
            "false_attribution_rate_among_claims": false_claimed / claimed if claimed else None,
            "decision_accuracy_on_reviewed": correct / n,
            "human_unexplained_fraction_on_reviewed": gold_unexplained / n,
            "gold_explained_moves": gold_explained,
            "candidate_recall_on_gold_explained": retrieval_hits / gold_explained if gold_explained else None}
    return result


def read_local(path, *, parse_json=True):
    path = Path(path)
    stat = path.stat()
    if not path.is_file() or getattr(stat, "st_flags", 0) & 0x40000000:
        raise ValueError("Input must be a materialized local file")
    if stat.st_size > 64 * 1024 * 1024:
        raise ValueError("Input exceeds the bounded 64 MiB JSON limit")
    raw = path.read_bytes()
    return (json.loads(raw) if parse_json else raw), hashlib.sha256(raw).hexdigest()


def capture_inputs(directory):
    """Normalize a hash-bound run_historical capture without rerunning a model.

    Dataset company tags and reported dates do not become verified entities or
    historical publication/receipt evidence. All such news remains ineligible.
    """
    directory = Path(directory).resolve()
    manifest, manifest_hash = read_local(directory / "manifest.json")
    if (manifest.get("status") != "complete_bounded_capture"
            or manifest.get("bars_summary", {}).get("pagination_complete") is not True):
        raise ValueError("Require a complete captured price query")
    loaded, hashes = {}, {"manifest.json": manifest_hash}
    for name, expected in manifest["files_sha256"].items():
        if Path(name).name != name or (directory / name).resolve().parent != directory:
            raise ValueError("Capture files must remain inside their archive")
        value, actual = read_local(directory / name, parse_json=name.endswith(".json"))
        if actual != expected:
            raise ValueError("Capture input hash changed: " + name)
        loaded[name], hashes[name] = value, actual
    protocol, protocol_hash = read_local(directory / "protocol.json")
    if protocol_hash != manifest["protocol_sha256"]:
        raise ValueError("Capture protocol hash changed")
    hashes["protocol.json"] = protocol_hash
    calendar = loaded["calendar.json"]
    sessions = calendar["sessions"]
    dates = [row["date"] for row in sessions]
    if calendar.get("complete") is not True or dates != sorted(set(dates)):
        raise ValueError("Require an ordered, complete independent session calendar")
    symbol = protocol["symbol"]
    prices, counts, prior_token = {}, Counter(), None
    requests = [r for r in manifest["requests"] if r["file"].startswith("prices_")]
    if not requests:
        raise ValueError("Capture has no price responses")
    for i, request in enumerate(requests):
        params = request["params"]
        if (request["file"] not in loaded or request.get("status") != 200
                or params.get("page_token") != prior_token
                or params.get("adjustment") != "split" or params.get("timeframe") != "1Day"):
            raise ValueError("Invalid split-adjusted daily price response or pagination chain")
        page = loaded[request["file"]]
        prior_token = page.get("next_page_token")
        if prior_token != request.get("next_page_token") or bool(prior_token) != (i < len(requests) - 1):
            raise ValueError("Incomplete or conflicting captured pagination")
        for ticker, rows in page["bars"].items():
            for row in rows:
                day = aware(row["t"]).astimezone(ZoneInfo("America/New_York")).date().isoformat()
                if day not in dates or (ticker, day) in prices:
                    raise ValueError("Duplicate or out-of-calendar price")
                prices[ticker, day] = finite(row["o"], "captured open", positive=True)
                counts[ticker] += 1
    if dict(counts) != manifest["bars_summary"]["records"] or not all(counts[s] for s in [symbol, "SPY"]):
        raise ValueError("Captured price count mismatch")
    bars = [{"symbol": symbol, "session_index": i, "at": row["open_at"],
             "open": prices[symbol, row["date"]], "benchmark_open": prices["SPY", row["date"]]}
            for i, row in enumerate(sessions)
            if (symbol, row["date"]) in prices and ("SPY", row["date"]) in prices]
    news = [{"news_id": f"{symbol}:{row['source_row_index']}", "symbols": [symbol],
             "headline": row["headline"], "source_ref": row.get("url", ""),
             "entity_verified": False, "historical_availability_verified": False,
             "published_at": row.get("reported_timestamp"), "version_at": None,
             "received_at": row.get("response_received_at")}
            for row in loaded["news.json"]]
    return bars, news, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bars", type=Path)
    inputs.add_argument("--capture", type=Path)
    parser.add_argument("--news", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--human-review", type=Path)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--sigma-threshold", type=float, default=3.0)
    parser.add_argument("--preceding-hours", type=float, default=72)
    args = parser.parse_args()
    if args.capture:
        if args.news:
            parser.error("--capture uses its bound news; do not override with --news")
        bars, news, input_hashes = capture_inputs(args.capture)
    else:
        if not args.news:
            parser.error("--bars requires --news")
        bars, bar_hash = read_local(args.bars)
        news, news_hash = read_local(args.news)
        input_hashes = {"bars": bar_hash, "news": news_hash}
    packet = prepare_review(detect_moves(bars, lookback=args.lookback, sigma_threshold=args.sigma_threshold),
                            news, preceding_hours=args.preceding_hours)
    predictions = read_local(args.predictions) if args.predictions else (None, None)
    review = read_local(args.human_review) if args.human_review else (None, None)
    evaluation = evaluate_attributions(packet, predictions[0], review[0])
    # New private directory only; never overwrite prior evidence or publish text.
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (args.output / ".gitignore").write_text("*\n")
    receipt = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {**input_hashes, "predictions": predictions[1], "human_review": review[1]},
        "parameters": packet["detection"]["parameters"], "preceding_hours": args.preceding_hours,
        "observations": {"bars": len(bars), "news": len(news)},
        "evaluation": evaluation, "is_causal_or_trading_alpha_evidence": False}
    for name, content in [("move_review_packet.json", packet), ("evaluation_receipt.json", receipt)]:
        (args.output / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    print(json.dumps(evaluation, allow_nan=False))


if __name__ == "__main__":
    main()
