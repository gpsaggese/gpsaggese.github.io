"""Offline, as-of text features and a same-horizon Quant.ai residual adapter.

This module never imports the trading backend, fetches data, or places orders.
The primary mode requires retained response-receipt evidence. A fixed processing
delay is an explicit replay assumption, not a historical execution receipt.
The optional publication-assumption mode is counterfactual development only.
Lexical stance is a deliberately limited feature, not a return probability.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo


STRICT_MODE = "strict_received"
RETROSPECTIVE_MODE = "retrospective_publication_assumption"
FEATURE_NAMES = (
    "news_log_count", "futu_log_count", "news_stance_balance",
    "futu_stance_balance", "news_scored_fraction", "futu_scored_fraction",
    "unique_text_fraction", "mean_publication_age_fraction",
    "mean_receipt_age_fraction", "any_text_missing", "news_missing", "futu_missing",
)
EXTRACTOR_VERSION = "quant-text-features-1.0"
NY = ZoneInfo("America/New_York")
_POSITIVE_EN = {"upgrade", "upgrades", "upgraded", "bullish", "outperform", "beats"}
_NEGATIVE_EN = {"downgrade", "downgrades", "downgraded", "bearish", "underperform", "misses"}
_POSITIVE_ZH = ("看多", "做多", "看涨", "看漲", "买入", "買入")
_NEGATIVE_ZH = ("看空", "做空", "看跌", "卖出", "賣出")


def _time(value):
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("An aware ISO timestamp is required")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Naive timestamps are not allowed")
    return parsed.astimezone(timezone.utc)


def _stamp(value):
    return _time(value).isoformat()


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _clean(value):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))).strip()


def _read_json(path, receipts):
    data = Path(path).read_bytes()
    receipts[str(Path(path).resolve())] = _hash_bytes(data)
    return json.loads(data)


def _symbol(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", value):
        raise ValueError("Invalid symbol")
    return value


def make_observation(*, source, source_id, symbols, title, text="", published_at,
                     updated_at=None, received_at=None, receipt_verified=False,
                     provenance=None):
    """Normalize explicit inputs; callers must not invent receipt_verified evidence."""
    if source not in {"news", "futu"} or not str(source_id):
        raise ValueError("Unsupported source or empty source ID")
    published = _time(published_at)
    updated = _time(updated_at or published_at)
    if updated < published:
        raise ValueError("Update precedes publication")
    received = _time(received_at) if received_at else None
    if receipt_verified and received is None:
        raise ValueError("Verified receipt requires an actual timestamp")
    clean_title, clean_text = _clean(title), _clean(text)
    if not clean_title and not clean_text:
        raise ValueError("Text is empty")
    if not symbols:
        raise ValueError("At least one source-associated company is required")
    return {
        "source": source, "source_id": str(source_id),
        "symbols": sorted({_symbol(s) for s in symbols}),
        "title": clean_title, "text": clean_text,
        "content_hash": _hash_bytes((clean_title + "\n" + clean_text).encode()),
        "published_at": published.isoformat(), "updated_at": updated.isoformat(),
        "received_at": received.isoformat() if received else None,
        "receipt_verified": bool(receipt_verified),
        "provenance": provenance or {},
    }


def _news_archive(directory, files, exclusions):
    normalized_path = directory / "news_normalized.json"
    rows = _read_json(normalized_path, files)
    if not isinstance(rows, list):
        raise ValueError("News normalization must be a list")
    manifest_path = directory / "manifest.json"
    raw_versions = []
    if manifest_path.is_file():
        manifest = _read_json(manifest_path, files)
        for request in manifest.get("requests", []):
            filename = request.get("file", "")
            if not filename.startswith("news_raw_"):
                continue
            if Path(filename).name != filename or request.get("status") != 200:
                raise ValueError("Invalid news response path or failed response")
            raw_path = directory / filename
            raw = _read_json(raw_path, files)
            if files[str(raw_path.resolve())] != request.get("sha256"):
                raise ValueError("News response hash does not match its manifest")
            receipt = _stamp(request["response_received_at"])
            for article in raw.get("news", []):
                raw_versions.append((article, receipt, raw_path))
    by_id = {}
    for article, receipt, path in raw_versions:
        by_id.setdefault(str(article.get("id")), []).append((article, receipt, path))
    observations = []
    for index, row in enumerate(rows):
        try:
            matching = []
            for raw, receipt, raw_path in by_id.get(str(row.get("id")), []):
                same = (str(row["id"]) == str(raw["id"])
                        and row.get("symbols") == raw.get("symbols")
                        and _clean(row.get("headline")) == _clean(raw.get("headline"))
                        and _clean(row.get("text")) == _clean(raw.get("content"))
                        and _time(row["created_at"]) == _time(raw["created_at"])
                        and _time(row["updated_at"]) == _time(raw["updated_at"]))
                if same:
                    matching.append((receipt, raw_path))
            if raw_versions and not matching:
                raise ValueError("Normalized article differs from retained responses")
            receipt, response_path = min(matching) if matching else (None, None)
            observations.append(make_observation(
                source="news", source_id=row["id"], symbols=row["symbols"],
                title=row.get("headline", ""), text=row.get("text", ""),
                published_at=row["created_at"], updated_at=row.get("updated_at"),
                received_at=receipt, receipt_verified=bool(matching),
                provenance={"normalized_file": str(normalized_path.resolve()),
                            "normalized_row": index, "response_file": str(response_path.resolve()) if response_path else None,
                            "receipt_reason": "retained_response_manifest" if matching else "missing_actual_receipt"},
            ))
        except (ValueError, KeyError, TypeError) as exc:
            exclusions.append({"path": str(normalized_path), "row": index, "reason": str(exc)})
    return observations


def _dict_rows(value):
    if isinstance(value, dict):
        if "id" in value and "title" in value:
            yield value
        for child in value.values():
            yield from _dict_rows(child)
    elif isinstance(value, list):
        for child in value:
            yield from _dict_rows(child)


def _futu_archive(directory, files, exclusions):
    normalized_path = directory / "all_normalized.json"
    rows = _read_json(normalized_path, files)
    if not isinstance(rows, list):
        raise ValueError("Futu normalization must be a list")
    verified = {}
    for symbol in sorted({_symbol(row["query_symbol"]) for row in rows}):
        request_path, raw_path = directory / f"{symbol}_request.json", directory / f"{symbol}_raw.json"
        if not request_path.is_file() or not raw_path.is_file():
            continue
        request = _read_json(request_path, files)
        raw = _read_json(raw_path, files)
        if request.get("http_status") != 200 or files[str(raw_path.resolve())] != request.get("sha256"):
            raise ValueError("Futu response failed or its hash does not match")
        verified[symbol] = (request.get("response_received_at"), list(_dict_rows(raw)), raw_path)
    observations = []
    for index, row in enumerate(rows):
        try:
            receipt, raw_rows, raw_path = verified.get(row["query_symbol"], (None, [], None))
            candidates = [raw for raw in raw_rows if str(raw.get("id")) == str(row["post_id"])]
            same = any(_clean(raw.get("title")) == _clean(row.get("title"))
                       and _clean(raw.get("desc")) == _clean(row.get("excerpt"))
                       and _time(row["published_at"]) == datetime.fromtimestamp(float(raw["publish_time"]), timezone.utc)
                       for raw in candidates)
            if raw_rows and not same:
                raise ValueError("Normalized Futu text differs from retained response")
            observations.append(make_observation(
                source="futu", source_id=row["post_id"], symbols=[row["query_symbol"]],
                title=row.get("title", ""), text=row.get("excerpt", ""),
                published_at=row["published_at"], received_at=receipt if same else None,
                receipt_verified=bool(receipt and same),
                provenance={"normalized_file": str(normalized_path.resolve()),
                            "normalized_row": index, "response_file": str(raw_path.resolve()) if raw_path else None,
                            "receipt_reason": "retained_response_manifest" if receipt and same else "missing_actual_receipt"},
            ))
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            exclusions.append({"path": str(normalized_path), "row": index, "reason": str(exc)})
    return observations


def load_archives(paths):
    """Load an explicit archive allowlist; no answer files or price labels are read."""
    files, exclusions, observations = {}, [], []
    for path in paths:
        directory = Path(path)
        if (directory / "news_normalized.json").is_file():
            observations.extend(_news_archive(directory, files, exclusions))
        elif (directory / "all_normalized.json").is_file():
            observations.extend(_futu_archive(directory, files, exclusions))
        else:
            raise ValueError(f"No supported normalized text archive: {directory}")
    # Merge repeated identical captures without moving a version's first receipt.
    # Query-symbol attribution is retained; one post returned for two companies
    # does not become two votes for the same company.
    merged = {}
    for row in observations:
        key = (row["source"], row["source_id"], row["content_hash"], row["published_at"], row["updated_at"], tuple(row["symbols"]))
        old = merged.get(key)
        if old is None:
            merged[key] = row
            continue
        prefer = (row["receipt_verified"] and
                  (not old["receipt_verified"] or _time(row["received_at"]) < _time(old["received_at"])))
        if prefer:
            merged[key] = row
    result = sorted(merged.values(), key=lambda r: (r["published_at"], r["source"], r["source_id"], r["content_hash"]))
    return result, {
        "extractor_version": EXTRACTOR_VERSION, "input_sha256": files,
        "normalized_rows_before_deduplication": len(observations),
        "retained_versions": len(result),
        "retained_versions_by_source": dict(Counter(r["source"] for r in result)),
        "versions_with_verified_receipt": sum(r["receipt_verified"] for r in result),
        "earliest_verified_receipt": min((r["received_at"] for r in result if r["receipt_verified"]), default=None),
        "exclusions": exclusions,
        "notes": ["Publication timestamps and receipt estimates are not actual receipt evidence.",
                  "Source tags/query association are not independent semantic relevance labels.",
                  "Identical title hashes are duplicate-text proxies, not verified event identities."],
    }


def load_default_archives(repo_root):
    root = Path(repo_root)
    candidates = [
        root / "reports/market_impact_data_audit_20261007/alpaca",
        root / "reports/market_impact_data_audit_20261007/futu",
        root / "reports/financial_sentiment_runs/event_review_20261009",
        root / "reports/financial_sentiment_runs/collector_trial_20261009",
        root / "reports/financial_sentiment_runs/stock_context_20261012_1200/futu_current",
    ]
    daily = root / "reports/financial_sentiment_runs/daily_610"
    if daily.is_dir():
        candidates.extend(sorted(daily.glob("capture_*")))
    return load_archives(p for p in candidates if p.is_dir())


def _stance(row):
    # Exclude question/conditional/negated language instead of guessing intent.
    text = row["title"] if row["source"] == "news" else row["title"] + " " + row["text"]
    # Vendor multi-company tags and multiple Futu ticker mentions do not provide
    # target-specific stance attribution. Retain the record, leave stance unknown.
    if len(row["symbols"]) != 1 or len(set(re.findall(r"\(([A-Z][A-Z0-9.\-]*)\.US\)", text))) > 1:
        return 0.0, False
    words = set(re.findall(r"[a-z]+", text.lower()))
    if any(x in text for x in ("?", "？", "如果", "不是", "不要", "不看", "不做")) or words & {"not", "never", "if", "could", "might"}:
        return 0.0, False
    positive = bool(words & _POSITIVE_EN) or any(x in text for x in _POSITIVE_ZH)
    negative = bool(words & _NEGATIVE_EN) or any(x in text for x in _NEGATIVE_ZH)
    if positive == negative:
        return 0.0, False
    return (1.0 if positive else -1.0), True


def build_asof_features(observations, symbol, decision_at, *, lookback_hours=48,
                        processing_delay_seconds=60, mode=STRICT_MODE):
    """Return version-aware features at one decision; missing text stays missing."""
    symbol, decision = _symbol(symbol), _time(decision_at)
    if mode not in {STRICT_MODE, RETROSPECTIVE_MODE}:
        raise ValueError("Unknown availability mode")
    if not math.isfinite(lookback_hours) or lookback_hours <= 0:
        raise ValueError("Lookback must be positive")
    if not math.isfinite(processing_delay_seconds) or processing_delay_seconds < 0:
        raise ValueError("Processing delay cannot be negative")
    window = timedelta(hours=lookback_hours)
    delay = timedelta(seconds=processing_delay_seconds)
    versions, reasons = {}, Counter()
    for row in observations:
        if symbol not in row["symbols"]:
            continue
        published, updated = _time(row["published_at"]), _time(row["updated_at"])
        if mode == STRICT_MODE:
            if not row["receipt_verified"] or not row["received_at"]:
                reasons["missing_verified_receipt"] += 1
                continue
            received = _time(row["received_at"])
            available = max(published, updated, received) + delay
        else:
            received = _time(row["received_at"]) if row["received_at"] else None
            available = max(published, updated) + delay
        if available > decision:
            reasons["not_available_yet"] += 1
            continue
        # Old content newly collected does not become a fresh information event.
        if decision - updated > window or decision - available > window:
            reasons["outside_lookback"] += 1
            continue
        key = (row["source"], row["source_id"])
        candidate = (updated, available, row["content_hash"], row, received)
        if key not in versions or candidate[:3] > versions[key][:3]:
            versions[key] = candidate
    selected = sorted(versions.values(), key=lambda x: (x[0], x[3]["source"], x[3]["source_id"]))
    by_source = {s: [x for x in selected if x[3]["source"] == s] for s in ("news", "futu")}
    features = {}
    for source, rows in by_source.items():
        scores = [_stance(x[3]) for x in rows]
        scored = [value for value, known in scores if known]
        features[source + "_log_count"] = math.log1p(len(rows))
        features[source + "_stance_balance"] = sum(scored) / len(scored) if scored else 0.0
        features[source + "_scored_fraction"] = len(scored) / len(rows) if rows else 0.0
        features[source + "_missing"] = float(not rows)
    # Lowercase normalized titles are a duplicate-text proxy, not semantic events.
    hashes = {_hash_bytes(x[3]["title"].lower().encode()) for x in selected}
    features["unique_text_fraction"] = len(hashes) / len(selected) if selected else 0.0
    features["mean_publication_age_fraction"] = (sum(min((decision - _time(x[3]["published_at"])).total_seconds() / window.total_seconds(), 1.0) for x in selected) / len(selected) if selected else 0.0)
    # Assumption mode uses availability age, never a negative future receipt age.
    features["mean_receipt_age_fraction"] = (sum((decision - x[1]).total_seconds() / window.total_seconds() for x in selected) / len(selected) if selected else 0.0)
    features["any_text_missing"] = float(not selected)
    return {
        "schema_version": 1, "extractor_version": EXTRACTOR_VERSION,
        "symbol": symbol, "decision_at": decision.isoformat(), "mode": mode,
        "primary_asof_eligible": mode == STRICT_MODE and bool(selected),
        "has_text": bool(selected), "features": {k: features[k] for k in FEATURE_NAMES},
        "feature_names": list(FEATURE_NAMES), "feature_vector": [features[k] for k in FEATURE_NAMES],
        "source_counts": {s: len(v) for s, v in by_source.items()},
        "excluded_version_counts": dict(reasons),
        "availability_policy": {"lookback_hours": lookback_hours, "processing_delay_seconds": processing_delay_seconds,
                                "processing_clock": "simulated_fixed_delay", "actual_historical_model_run_verified": False},
        "evidence": [{"source": x[3]["source"], "source_id": x[3]["source_id"],
                      "content_hash": x[3]["content_hash"], "published_at": x[3]["published_at"],
                      "updated_at": x[3]["updated_at"], "received_at": x[3]["received_at"],
                      "available_at": x[1].isoformat(), "receipt_verified": x[3]["receipt_verified"],
                      "primary_asof_eligible": mode == STRICT_MODE,
                      "provenance": x[3]["provenance"]} for x in selected],
        "limitations": ["Keyword stance is not calibrated financial sentiment or a stock-return probability.",
                        "A verified receipt plus simulated delay is an offline replay, not a historical execution claim.",
                        "No broad retail-population representation or event novelty is certified."],
    }


def _vector(snapshot):
    if snapshot.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Feature schema mismatch")
    vector = snapshot.get("feature_vector")
    if not isinstance(vector, list) or len(vector) != len(FEATURE_NAMES) or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in vector):
        raise ValueError("Invalid feature vector")
    return vector


def _solve(matrix, rhs):
    """Small positive-definite ridge system; no external ML/runtime dependency."""
    n = len(rhs)
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(rows[row][col]))
        if abs(rows[pivot][col]) < 1e-14:
            raise ValueError("Singular calibration system")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        rows[col] = [v / scale for v in rows[col]]
        for row in range(n):
            if row != col:
                factor = rows[row][col]
                rows[row] = [a - factor * b for a, b in zip(rows[row], rows[col])]
    return [row[-1] for row in rows]


def fit_residual_calibration(examples, training_cutoff, *, target_horizon_minutes=30,
                             minimum_text_rows=100, minimum_sessions=5,
                             minimum_nonoverlapping_intervals=30, ridge_alpha=10.0,
                             mode=STRICT_MODE):
    """Fit training-only C residuals; each row includes explicit label maturity.

    Example keys: snapshot, base_prediction, realized_return, label_end_at;
    label_start_at is optional and defaults to the decision instant. It must be
    supplied when next-open execution starts later than the decision instant.
    The target is realized_return - base_prediction at the SAME minute horizon.
    Count/session gates are engineering minimums, not evidence of statistical power.
    """
    cutoff = _time(training_cutoff)
    if mode not in {STRICT_MODE, RETROSPECTIVE_MODE}:
        raise ValueError("Unknown mode")
    if target_horizon_minutes <= 0 or ridge_alpha <= 0 or not math.isfinite(ridge_alpha):
        raise ValueError("Invalid horizon or ridge penalty")
    if min(minimum_text_rows, minimum_sessions, minimum_nonoverlapping_intervals) < 1:
        raise ValueError("Training gates must be positive")
    retained, excluded = [], Counter()
    for example in examples:
        snapshot = example["snapshot"]
        decision = _time(snapshot["decision_at"])
        label_start = _time(example.get("label_start_at", snapshot["decision_at"]))
        label_end = _time(example["label_end_at"])
        if decision >= cutoff or label_end >= cutoff:
            excluded["outcome_not_mature_before_training_cutoff"] += 1
            continue
        if example.get("target_horizon_minutes", target_horizon_minutes) != target_horizon_minutes:
            raise ValueError("Cross-horizon calibration is forbidden")
        if label_start < decision or label_end - label_start != timedelta(minutes=target_horizon_minutes):
            raise ValueError("Label interval must match the declared minute target after the decision")
        if snapshot.get("mode") != mode:
            excluded["availability_mode_mismatch"] += 1
            continue
        if not snapshot["has_text"] or (mode == STRICT_MODE and not snapshot["primary_asof_eligible"]):
            excluded["no_eligible_text"] += 1
            continue
        values = (example["base_prediction"], example["realized_return"])
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in values):
            raise ValueError("Nonfinite prediction or outcome")
        retained.append((decision, label_end, snapshot["symbol"], _vector(snapshot), values[1] - values[0]))
    ids = [(r[2], r[0]) for r in retained]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate company-decision training rows")
    sessions = sorted({r[0].astimezone(NY).date().isoformat() for r in retained})
    # Count non-overlapping intervals per company, without claiming cross-company independence.
    ends, nonoverlap = {}, 0
    for decision, label_end, symbol, _, _ in sorted(retained):
        if symbol not in ends or decision >= ends[symbol]:
            nonoverlap += 1
            ends[symbol] = label_end
    artifact = {
        "schema_version": 1, "status": "unavailable", "mode": mode,
        "primary_asof_eligible": mode == STRICT_MODE, "training_cutoff": cutoff.isoformat(),
        "target_horizon_minutes": target_horizon_minutes, "feature_names": list(FEATURE_NAMES),
        "training_text_rows": len(retained), "training_sessions": len(sessions),
        "training_session_dates": sessions, "nonoverlapping_intervals_per_company_total": nonoverlap,
        "excluded_counts": dict(excluded), "ridge_alpha": ridge_alpha,
        "gates": {"minimum_text_rows": minimum_text_rows, "minimum_sessions": minimum_sessions,
                  "minimum_nonoverlapping_intervals": minimum_nonoverlapping_intervals},
        "alpha_established": False, "target": "same_horizon_realized_return_minus_C_prediction",
        "limitations": ["Session and interval gates do not certify independent events or adequate statistical power.",
                        "Calibration must be assessed on strictly later data after realistic trading costs."],
    }
    if len(retained) < minimum_text_rows or len(sessions) < minimum_sessions or nonoverlap < minimum_nonoverlapping_intervals:
        artifact["reason"] = "insufficient_eligible_training_coverage"
        return artifact
    columns = list(zip(*(r[3] for r in retained)))
    means = [sum(col) / len(col) for col in columns]
    scales = [math.sqrt(sum((v - mean) ** 2 for v in col) / len(col)) or 1.0 for col, mean in zip(columns, means)]
    x = [[(v - mean) / scale for v, mean, scale in zip(r[3], means, scales)] for r in retained]
    targets = [r[4] for r in retained]
    intercept = sum(targets) / len(targets)
    n = len(FEATURE_NAMES)
    gram = [[sum(row[i] * row[j] for row in x) + (ridge_alpha if i == j else 0.0) for j in range(n)] for i in range(n)]
    rhs = [sum(row[i] * (y - intercept) for row, y in zip(x, targets)) for i in range(n)]
    artifact.update(status="fitted_development_only", reason=None, means=means, scales=scales,
                    weights=_solve(gram, rhs), intercept=intercept,
                    latest_training_label_end=max(r[1] for r in retained).isoformat())
    return artifact


class SentimentResidualAdapter:
    """Apply only a matching, mature calibration; unavailable inputs give B == C."""

    def __init__(self, calibration=None, *, target_horizon_minutes=30, max_abs_adjustment=0.0005):
        if target_horizon_minutes <= 0 or not math.isfinite(max_abs_adjustment) or max_abs_adjustment <= 0:
            raise ValueError("Invalid prediction horizon or adjustment cap")
        self.calibration = calibration
        self.target_horizon_minutes = target_horizon_minutes
        self.max_abs_adjustment = max_abs_adjustment

    def predict(self, base_prediction, snapshot):
        if not isinstance(base_prediction, (int, float)) or isinstance(base_prediction, bool) or not math.isfinite(base_prediction):
            raise ValueError("Base prediction must be finite")
        vector = _vector(snapshot)
        calibration = self.calibration
        reason = None
        if not snapshot["has_text"]:
            reason = "no_asof_text"
        elif calibration is None or calibration.get("status") != "fitted_development_only":
            reason = "no_eligible_calibration"
        elif calibration["target_horizon_minutes"] != self.target_horizon_minutes:
            reason = "horizon_mismatch"
        elif calibration["mode"] != snapshot["mode"]:
            reason = "availability_mode_mismatch"
        elif _time(calibration["training_cutoff"]) >= _time(snapshot["decision_at"]):
            reason = "calibration_not_available_before_decision"
        elif calibration["feature_names"] != list(FEATURE_NAMES):
            reason = "feature_schema_mismatch"
        elif snapshot["mode"] == STRICT_MODE and not snapshot["primary_asof_eligible"]:
            reason = "not_primary_asof_eligible"
        delta = 0.0
        raw_delta = None
        if reason is None:
            coefficients = (calibration["means"], calibration["scales"], calibration["weights"])
            if any(len(c) != len(FEATURE_NAMES) for c in coefficients) or any(not math.isfinite(v) for c in coefficients for v in c) or any(v <= 0 for v in calibration["scales"]):
                raise ValueError("Invalid calibration coefficients")
            raw_delta = calibration["intercept"] + sum(w * (v - m) / s for v, m, s, w in zip(vector, *coefficients))
            if not math.isfinite(raw_delta):
                raise ValueError("Invalid residual prediction")
            delta = max(-self.max_abs_adjustment, min(self.max_abs_adjustment, raw_delta))
        return {"base_prediction": base_prediction, "prediction": base_prediction + delta,
                "adjustment": delta, "raw_adjustment": raw_delta, "applied": reason is None,
                "text_changed_prediction": abs(delta) > 1e-15,
                "fallback_reason": reason, "mode": snapshot["mode"],
                "primary_asof_eligible": snapshot["primary_asof_eligible"],
                "decision_at": snapshot["decision_at"], "target_horizon_minutes": self.target_horizon_minutes,
                "adjustment_cap": self.max_abs_adjustment}


def audit_coverage(observations, decisions, **feature_options):
    """Summarize explicit decisions; no invented trading-calendar denominator."""
    snapshots = [build_asof_features(observations, row["symbol"], row["decision_at"], **feature_options) for row in decisions]
    days = sorted({s["decision_at"][:10] for s in snapshots if s["has_text"]})
    return {"decisions": len(snapshots), "decisions_with_text": sum(s["has_text"] for s in snapshots),
            "primary_asof_eligible_decisions": sum(s["primary_asof_eligible"] for s in snapshots),
            "dates_with_text": days, "by_symbol": {symbol: {"decisions": sum(s["symbol"] == symbol for s in snapshots),
            "with_text": sum(s["symbol"] == symbol and s["has_text"] for s in snapshots)} for symbol in sorted({s["symbol"] for s in snapshots})},
            "mode": feature_options.get("mode", STRICT_MODE),
            "actual_historical_model_run_verified": False}
