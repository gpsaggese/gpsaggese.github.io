"""Inspect saved financial-text samples and verify exploratory price labels.

This module is an offline Quant.ai adaptation of the UMD project template.
It never fetches data, trains a model, places an order, or changes the archive.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_TITLE = "Train_Financial_Sentinment_Analysis_Using_Prices"
AUDIT_DIRECTORY = "market_impact_data_audit_20261007"
RESEARCH_SYMBOLS = frozenset({"NVDA", "SNDK", "TSLA", "PLTR"})
MODEL_STATUS = "Not trained"
EVALUATION_STATUS = "Not evaluated"


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Resolve an explicit path, an environment override, or the Quant archive.

    :param data_dir: Optional path to the existing audit directory.
    :return: Existing absolute directory; never create or download data.
    """
    configured = data_dir or os.environ.get("FINANCIAL_SENTIMENT_DATA_DIR")
    path = Path(configured).expanduser() if configured else (
        Path(__file__).resolve().parents[2] / "reports" / AUDIT_DIRECTORY
    )
    if not path.is_dir():
        raise FileNotFoundError(
            f"Audit data directory is unavailable: {path}. Restore the existing "
            "local audit or set FINANCIAL_SENTIMENT_DATA_DIR. No data was downloaded."
        )
    return path.resolve()


def _read_json(path: Path) -> Any:
    """Read a required archive file with an actionable error on missing data."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required audit file is missing: {path}") from exc


def load_audit(data_dir: str | Path | None = None) -> dict[str, Any]:
    """Load the retained audit without using any online service.

    :param data_dir: Optional directory override.
    :return: Source samples and their original summary/validation records.
    """
    base = resolve_data_dir(data_dir)
    files = {
        "news": "alpaca/news_normalized.json",
        "labels": "alpaca/exploratory_labels.json",
        "news_summary": "alpaca/summary.json",
        "futu": "futu/all_normalized.json",
        "futu_summary": "futu/summary.json",
        "sec_summary": "sec/summary.json",
        "validation": "validation.json",
    }
    return {"data_dir": base, **{name: _read_json(base / path) for name, path in files.items()}}


def field_dictionary() -> list[dict[str, str]]:
    """Describe fields used in the starter without suggesting live availability."""
    return [
        {"dataset": "News", "field": "id / symbols", "meaning": "Article identifier and vendor ticker tags; tags are not verified company relevance."},
        {"dataset": "News", "field": "headline / text", "meaning": "Archived headline and available body; final historical text may include revisions."},
        {"dataset": "News", "field": "created_at / updated_at", "meaning": "Vendor publication/revision times, not a recorded historical receipt time."},
        {"dataset": "News", "field": "historical_availability_verified", "meaning": "False in this snapshot; the historical text version was not proven available in real time."},
        {"dataset": "Futu", "field": "post_id / query_symbol", "meaning": "Post identity and query association; the same post may occur in multiple stock queries."},
        {"dataset": "Futu", "field": "title / excerpt", "meaning": "Original text; these fields often repeat each other. No validated sentiment label exists."},
        {"dataset": "Futu", "field": "response_received_at", "meaning": "Missing in the first snapshot; request start must not be substituted for receipt."},
        {"dataset": "Labels", "field": "event_id / symbol / status", "meaning": "One candidate article-stock pair and its eligibility/exclusion reason."},
        {"dataset": "Labels", "field": "entry_at / exit_at", "meaning": "Hypothetical five-minute bar opens, 60 minutes apart, using an explicitly assumed delay."},
        {"dataset": "Labels", "field": "excess_return", "meaning": "Stock open-to-open return minus matching SPY return; an outcome label, not a prediction or net trading return."},
    ]


def coverage_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize observed data coverage, keeping record types separate."""
    news, futu, labels = audit["news"], audit["futu"], audit["labels"]
    valid = [row for row in labels if row["status"] == "exploratory_label_available"]
    return [
        {"measure": "Unique news article IDs", "count": len({row["id"] for row in news}), "scope": "Historical final text; event clusters not deduplicated."},
        {"measure": "Articles with body text", "count": sum(bool(row["text"]) for row in news), "scope": "Text presence does not prove company-specific information."},
        {"measure": "Futu query rows", "count": len(futu), "scope": "Four bounded queries; not the full community."},
        {"measure": "Unique Futu post IDs", "count": len({row["post_id"] for row in futu}), "scope": "No author IDs, interaction history, or validated sentiment labels."},
        {"measure": "Regular-session five-minute bars", "count": sum(row["rth_bars"] for row in audit["news_summary"]["bars"].values()), "scope": "Seven instruments over five sessions; read from the saved source summary."},
        {"measure": "SEC filings in the 90-day window", "count": audit["sec_summary"]["totals"]["window_filings"], "scope": "Metadata only; document retrieval did not succeed."},
        {"measure": "Candidate article-stock pairs", "count": len(labels), "scope": "Ticker tags generate candidates, not independent events."},
        {"measure": "Available exploratory return labels", "count": len(valid), "scope": "Outcomes under historical timing assumptions; not training or evaluation results."},
    ]


def sample_records(audit: dict[str, Any], source: str, limit: int = 3) -> list[dict[str, Any]]:
    """Return brief real source records without copying full article bodies."""
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    if source == "news":
        return [{"id": row["id"], "symbols": row["symbols"], "title": row["headline"],
                 "body_chars": len(row["text"]), "published_at": row["created_at"],
                 "updated_at": row["updated_at"], "historical_availability_verified": row["historical_availability_verified"]}
                for row in audit["news"][:limit]]
    if source == "futu":
        return [{"id": row["post_id"], "symbol": row["query_symbol"], "title": row["title"],
                 "excerpt_chars": len(row["excerpt"]), "published_at": row["published_at"],
                 "response_received_at": row.get("response_received_at")}
                for row in audit["futu"][:limit]]
    raise ValueError("source must be 'news' or 'futu'")


def _timestamp(value: str) -> datetime:
    """Parse an aware timestamp and reject timezone-free input."""
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError(f"Timezone is missing from {value!r}")
    return result.astimezone(timezone.utc)


def _require(condition: bool, message: str) -> None:
    """Keep verification enabled even when Python runs with optimization."""
    if not condition:
        raise ValueError(message)


def verify_exploratory_labels(audit: dict[str, Any]) -> dict[str, Any]:
    """Recompute archived outcomes and check their explicit timing assumptions.

    :param audit: Data returned by load_audit(), including its archive path.
    :return: Counts and check results, never a prediction or trading score.
    """
    news, labels = audit["news"], audit["labels"]
    articles = {row["id"]: row for row in news}
    _require(len(articles) == len(news), "Duplicate news article identifiers")
    _require(all(row["historical_availability_verified"] is False for row in news),
             "This starter expects historical availability to remain unverified")
    pairs = [(row["event_id"], row["symbol"]) for row in labels]
    expected_pairs = {(row["id"], symbol) for row in news for symbol in set(row["symbols"]) & RESEARCH_SYMBOLS}
    _require(len(pairs) == len(set(pairs)), "Duplicate article-stock candidate")
    _require(set(pairs) == expected_pairs, "Article-stock candidates do not match source tags")
    _require(all(row["historical_availability_verified"] is False for row in labels),
             "Exploratory labels must not claim verified historical availability")
    statuses = dict(Counter(row["status"] for row in labels))
    _require(statuses == audit["news_summary"]["label_status_counts"], "Label status counts differ from the saved audit")
    prices: dict[str, dict[datetime, dict[str, Any]]] = {}
    base = Path(audit["data_dir"]) / "alpaca"
    for symbol, expected in audit["news_summary"]["bars"].items():
        rows = [row for file in sorted(base.glob(f"bars_{symbol}_[0-9]*.json"))
                for row in _read_json(file)["bars"][symbol]]
        prices[symbol] = {_timestamp(row["t"]): row for row in rows}
        _require(len(rows) == len(prices[symbol]) == expected["all_bars"], f"Missing or duplicate bars for {symbol}")
    available = [row for row in labels if row["status"] == "exploratory_label_available"]
    maximum_error = 0.0
    dates: Counter[str] = Counter()
    for row in available:
        article = articles[row["event_id"]]
        start, end = _timestamp(row["entry_at"]), _timestamp(row["exit_at"])
        delayed = max(_timestamp(article["created_at"]), _timestamp(article["updated_at"])) + timedelta(seconds=60)
        next_open = delayed.replace(minute=(delayed.minute // 5) * 5, second=0, microsecond=0)
        if next_open < delayed:
            next_open += timedelta(minutes=5)
        _require(start == next_open, "Entry is not the first five-minute boundary after the assumed delay")
        _require(end - start == timedelta(minutes=60), "Label horizon is not 60 minutes")
        local_start, local_end = start.astimezone(ZoneInfo("America/New_York")), end.astimezone(ZoneInfo("America/New_York"))
        _require(local_start.date() == local_end.date(), "Label crosses a session boundary")
        _require(local_start.strftime("%H:%M") >= "09:30" and local_end.strftime("%H:%M") <= "15:55", "Label lies outside the audited regular-session window")
        stock_prices, market_prices = prices[row["symbol"]], prices["SPY"]
        stock_return = stock_prices[end]["o"] / stock_prices[start]["o"] - 1
        spy_return = market_prices[end]["o"] / market_prices[start]["o"] - 1
        recalculated = {"stock_return": stock_return, "spy_return": spy_return, "excess_return": stock_return - spy_return}
        for field, value in recalculated.items():
            error = abs(value - row[field])
            _require(math.isfinite(error) and error < 1e-12, f"Return mismatch: {row['event_id']} / {row['symbol']} / {field}")
            maximum_error = max(maximum_error, error)
        dates[local_start.date().isoformat()] += 1
    reference = audit["validation"]
    _require(len(labels) == reference["article_stock_rows"], "Candidate count differs from the saved validation")
    _require(len(available) == reference["valid_exploratory_label_rows"], "Available-label count differs from the saved validation")
    _require(dict(dates) == reference["label_rows_by_date"], "Session counts differ from the saved validation")
    hashes = _read_json(base / "hashes.json")
    for filename, expected_hash in hashes.items():
        _require(hashlib.sha256((base / filename).read_bytes()).hexdigest() == expected_hash,
                 f"Archived file hash differs: {filename}")
    return {
        "verification_status": "Passed",
        "candidate_article_stock_pairs": len(labels),
        "available_exploratory_labels": len(available),
        "unique_articles_with_labels": len({row["event_id"] for row in available}),
        "excluded_candidates": len(labels) - len(available),
        "status_counts": statuses,
        "label_rows_by_date": dict(dates),
        "maximum_absolute_return_error": maximum_error,
        "verified_file_hashes": len(hashes),
        "model_status": MODEL_STATUS,
        "evaluation_status": EVALUATION_STATUS,
        "scope": "Internal archive consistency and outcome recomputation only. Historical availability, causal impact, prediction skill, and net trading returns remain unverified.",
    }


if __name__ == "__main__":
    saved_audit = load_audit()
    print(json.dumps({"project": PROJECT_TITLE, "coverage": coverage_rows(saved_audit),
                      "label_verification": verify_exploratory_labels(saved_audit)}, indent=2))
