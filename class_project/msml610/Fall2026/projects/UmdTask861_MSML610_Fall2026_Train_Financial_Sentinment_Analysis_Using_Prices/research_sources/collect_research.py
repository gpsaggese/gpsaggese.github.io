"""Capture bounded, read-only Alpaca news and daily bars into a new local archive.

Only two fixed market-data GET routes are used. Importing this module performs
no I/O; the CLI loads the existing local credentials only when collecting.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import sys
from typing import Callable
from zoneinfo import ZoneInfo


BASE_URL = "https://data.alpaca.markets"
BARS_ROUTE = "/v2/stocks/bars"
NEWS_ROUTE = "/v1beta1/news"
MARKET_SYMBOLS = ("SPY", "QQQ", "IGV", "SOXX", "XLE")
MAX_NEWS_PAGES = 20
MAX_WINDOW_DAYS = 93
MAX_SYMBOLS = 50
NEW_YORK = ZoneInfo("America/New_York")


class CollectionError(RuntimeError):
    """A capture failed; its manifest and any received pages remain on disk."""


class OutsidePublicationWindow(CollectionError):
    """A valid vendor record is outside the registered publication window."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("The collection clock must include a timezone")
    return value.astimezone(timezone.utc).isoformat()


def _parse_timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise CollectionError("Article timestamps must be strings")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("Timezone missing")
        return parsed.astimezone(timezone.utc)
    except ValueError as exc:
        raise CollectionError("Article timestamp is invalid or has no timezone") from exc


def _window(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    try:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dates must use YYYY-MM-DD") from exc
    if start.isoformat() != start_date or end.isoformat() != end_date:
        raise ValueError("Dates must use YYYY-MM-DD")
    if not 1 <= (end - start).days + 1 <= MAX_WINDOW_DAYS:
        raise ValueError(f"The inclusive date window must contain 1 to {MAX_WINDOW_DAYS} calendar days")
    return (datetime.combine(start, time.min, NEW_YORK),
            datetime.combine(end, time(16), NEW_YORK))


def _symbols(values: list[str]) -> list[str]:
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_SYMBOLS:
        raise ValueError(f"Provide 1 to {MAX_SYMBOLS} distinct stock symbols")
    symbols = []
    for value in values:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", value):
            raise ValueError("Symbols must be uppercase stock tickers")
        symbols.append(value)
    if len(set(symbols)) != len(symbols):
        raise ValueError("Duplicate stock symbols")
    return symbols


def _save(path: Path, payload: dict | list) -> str:
    data = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _clean_body(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def _normalize(article: dict, received_at: str, start: datetime, end: datetime) -> dict:
    if not isinstance(article, dict) or not isinstance(article.get("id"), (int, str)) or isinstance(article["id"], bool) or not str(article["id"]):
        raise CollectionError("News response contains an invalid article ID")
    for field in ("headline", "created_at", "updated_at", "url"):
        if not isinstance(article.get(field), str):
            raise CollectionError(f"News response contains an invalid {field}")
    for field in ("content", "summary", "source"):
        if not isinstance(article.get(field, ""), str):
            raise CollectionError(f"News response contains an invalid {field}")
    if not isinstance(article.get("symbols"), list) or any(not isinstance(s, str) for s in article["symbols"]):
        raise CollectionError("News response contains invalid symbol tags")
    created, updated = _parse_timestamp(article["created_at"]), _parse_timestamp(article["updated_at"])
    if updated < created:
        raise CollectionError("Article update precedes its publication timestamp")
    if not start <= created <= end:
        raise OutsidePublicationWindow("Article publication is outside the requested window")
    return {
        "id": str(article["id"]),
        "headline": re.sub(r"\s+", " ", html.unescape(article["headline"])).strip(),
        "text": _clean_body(article.get("content", "")),
        "summary": _clean_body(article.get("summary", "")),
        "symbols": article["symbols"],
        "source": article.get("source", ""),
        "created_at": article["created_at"],
        "updated_at": article["updated_at"],
        "url": article["url"],
        "retrieved_at": received_at,
        "historical_availability_verified": False,
    }


def _local_credentials() -> tuple[str, str]:
    """Reuse the read-only credential helper, without importing the app runner."""
    backend = str(Path(__file__).resolve().parents[2] / "backend")
    if backend not in sys.path:
        sys.path.insert(0, backend)
    from app.market_data.history import credentials
    return credentials()


def collect_research(
    output_dir: str | Path,
    symbols: list[str],
    start_date: str,
    end_date: str,
    *,
    max_news_pages: int = MAX_NEWS_PAGES,
    start_at: str | None = None,
    end_at: str | None = None,
    session=None,
    credential_loader: Callable[[], tuple[str, str]] | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> dict:
    """Save an auditable bounded capture; injectable dependencies permit offline tests.

    Successful or news-page-capped captures produce run_research-compatible
    inputs. HTTP, transport, malformed-data, or bars-pagination failures raise
    CollectionError and retain a failed manifest, without a normalized success
    file. A news page cap returns status 'truncated', never 'complete'. Duplicate
    news IDs retain the last archived occurrence and that occurrence's receipt
    time, matching read_inputs(); duplicate counts remain visible in summaries.
    Valid records created outside the requested window are retained in the raw
    response and an explicit rejected-record audit. A vendor may return a recent
    revision of an older article. That does not silently change our publication
    window into an update-time window. Malformed records still fail the capture.
    """
    symbols = _symbols(symbols)
    start, end = _window(start_date, end_date)
    if (start_at is None) != (end_at is None):
        raise ValueError("Provide both start_at and end_at for an exact window")
    if start_at is not None:
        start, end = _parse_timestamp(start_at), _parse_timestamp(end_at)
        if start >= end:
            raise ValueError("Exact window start must precede end")
        if start.astimezone(NEW_YORK).date().isoformat() != start_date or end.astimezone(NEW_YORK).date().isoformat() != end_date:
            raise ValueError("Exact timestamps must match their New York start/end dates")
    if type(max_news_pages) is not int or not 1 <= max_news_pages <= MAX_NEWS_PAGES:
        raise ValueError(f"max_news_pages must be between 1 and {MAX_NEWS_PAGES}")
    started = _stamp(clock())
    if end > datetime.fromisoformat(started):
        raise ValueError("The requested endpoint has not occurred yet")
    price_start = datetime.combine(start.astimezone(NEW_YORK).date(), time.min, NEW_YORK)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    news_summary = {
        "status": "not_started", "pagination_complete": False, "truncated": False,
        "pages_received": 0, "raw_article_rows": 0, "unique_article_ids": 0,
        "duplicate_rows": 0, "duplicate_ids": 0, "changed_duplicate_rows": 0,
        "accepted_article_rows": 0, "rejected_article_rows": 0,
        "max_pages": max_news_pages, "page_size": 50,
        "duplicate_resolution": "Last archived occurrence and its response receipt time.",
        "coverage_scope": "The bounded vendor query, not a guarantee of all market news.",
        "historical_availability_verified": False,
    }
    rejected_records = []
    manifest = {
        "schema_version": 1, "status": "running", "created_at": started,
        "analysis_asof": _stamp(end), "source": "alpaca_read_only_market_data",
        "symbols": symbols, "bar_symbols": list(dict.fromkeys([*symbols, *MARKET_SYMBOLS])),
        "universe_fixed_before_price_scan": True,
        "window": {"start_date": start_date, "end_date": end_date,
                   "start": _stamp(start), "end": _stamp(end), "timezone": "America/New_York",
                   "close_policy": ("Exact timestamp window including after-hours news; unfinished daily bars remain raw inputs." if start_at is not None else "16:00 reference close; holidays and early-close sessions are not inferred.")},
        "limits": {"news_pages": max_news_pages, "bars_pages": 1,
                   "window_calendar_days": MAX_WINDOW_DAYS, "symbols": MAX_SYMBOLS},
        "bar_window": {"start": _stamp(price_start), "end": _stamp(end),
                       "policy": "Daily bars begin at the New York date boundary so intraday cursor updates retain session bars."},
        "requests": [], "news_summary": news_summary,
        "bars_summary": {"status": "not_started", "pagination_complete": False},
        "historical_availability_verified": False,
        "normalization_policy": "created_at within the inclusive requested window; valid outside-window records remain in raw responses and the rejected-record audit; malformed records fail the capture",
    }

    def save_manifest():
        manifest["rejected_records_sha256"] = _save(output / "news_rejected_records.json", rejected_records)
        _save(output / "manifest.json", manifest)
        _save(output / "news_summary.json", news_summary)

    def request(route, params, filename):
        if route not in {BARS_ROUTE, NEWS_ROUTE}:
            raise CollectionError("Unexpected market-data route")
        record = {"route": route, "method": "GET", "params": dict(params),
                  "started_at": _stamp(clock()), "response_received_at": None,
                  "status": None, "file": filename, "sha256": None}
        manifest["requests"].append(record)
        save_manifest()
        try:
            response = session.get(BASE_URL + route, params=params, timeout=30, allow_redirects=False)
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            record["request_finished_at"] = _stamp(clock())
            record["sha256"] = _save(output / filename, {"capture_error": "Transport request failed; no response received."})
            raise CollectionError("Market-data transport failed; inspect the retained manifest") from exc
        record["response_received_at"] = _stamp(clock())
        record["status"] = response.status_code
        if response.status_code != 200:
            record["response_body_retained"] = False
            record["sha256"] = _save(output / filename, {"capture_error": "HTTP request failed", "http_status": response.status_code})
            raise CollectionError(f"Market-data endpoint returned HTTP {response.status_code}; no feed fallback")
        try:
            payload = response.json()
        except (ValueError, TypeError) as exc:
            record["sha256"] = _save(output / filename, {"capture_error": "Response was not valid JSON"})
            raise CollectionError("Market-data response was not valid JSON") from exc
        record["sha256"] = _save(output / filename, payload)
        save_manifest()
        if not isinstance(payload, dict):
            raise CollectionError("Market-data response must be a JSON object")
        token = payload.get("next_page_token")
        if token is not None and not isinstance(token, str):
            raise CollectionError("Market-data response has an invalid pagination token")
        return payload, record["response_received_at"]

    owned_session = session is None
    save_manifest()
    try:
        key, secret = (credential_loader or _local_credentials)()
        if not key or not secret:
            raise CollectionError("Alpaca market-data credentials are unavailable")
        if owned_session:
            import requests
            session = requests.Session()
        session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        bars_params = {"symbols": ",".join(manifest["bar_symbols"]), "start": _stamp(price_start),
                       "end": _stamp(end), "timeframe": "1Day", "adjustment": "raw",
                       "feed": "sip", "limit": 10000, "sort": "asc"}
        bars, _ = request(BARS_ROUTE, bars_params, "daily_bars_raw.json")
        if not isinstance(bars.get("bars"), dict) or any(not isinstance(rows, list) for rows in bars["bars"].values()):
            raise CollectionError("Daily-bar response must contain a bars dictionary of lists")
        manifest["bars_summary"] = {
            "status": "truncated" if bars.get("next_page_token") else "complete",
            "pagination_complete": not bool(bars.get("next_page_token")),
            "records": {symbol: len(rows) for symbol, rows in bars["bars"].items()},
            "missing_symbols": [symbol for symbol in manifest["bar_symbols"] if not bars["bars"].get(symbol)],
        }
        if bars.get("next_page_token"):
            raise CollectionError("Daily bars exceed the one-page budget; capture is incomplete")
        token, seen_tokens, articles, duplicate_ids = None, set(), {}, set()
        news_summary["status"] = "running"
        for page in range(max_news_pages):
            params = {"symbols": ",".join(symbols), "start": _stamp(start), "end": _stamp(end),
                      "limit": 50, "include_content": "true", "exclude_contentless": "false", "sort": "asc"}
            if token:
                params["page_token"] = token
            payload, received = request(NEWS_ROUTE, params, f"news_raw_{page:03d}.json")
            if not isinstance(payload.get("news"), list):
                raise CollectionError("News response must contain a news list")
            news_summary["pages_received"] += 1
            news_summary["raw_article_rows"] += len(payload["news"])
            for row_index, raw in enumerate(payload["news"]):
                try:
                    article = _normalize(raw, received, start, end)
                except OutsidePublicationWindow:
                    rejected_records.append({
                        "reason": "publication_outside_requested_window",
                        "file": f"news_raw_{page:03d}.json", "row_index": row_index,
                        "id": str(raw["id"]), "created_at": raw["created_at"],
                        "updated_at": raw["updated_at"], "response_received_at": received,
                        "updated_within_requested_window": start <= _parse_timestamp(raw["updated_at"]) <= end,
                    })
                    news_summary["rejected_article_rows"] += 1
                    continue
                news_summary["accepted_article_rows"] += 1
                identity = article["id"]
                if identity in articles:
                    news_summary["duplicate_rows"] += 1
                    duplicate_ids.add(identity)
                    comparable = lambda row: {k: v for k, v in row.items() if k != "retrieved_at"}
                    if comparable(articles[identity]) != comparable(article):
                        news_summary["changed_duplicate_rows"] += 1
                articles[identity] = article
                news_summary["unique_article_ids"] = len(articles)
                news_summary["duplicate_ids"] = len(duplicate_ids)
            token = payload.get("next_page_token")
            if not token:
                news_summary.update(status="complete", pagination_complete=True)
                break
            if token in seen_tokens:
                raise CollectionError("News endpoint repeated a pagination token")
            seen_tokens.add(token)
            if page + 1 == max_news_pages:
                news_summary.update(status="truncated", truncated=True, next_page_token=token,
                                    reason="News page budget reached; this query is not complete.")
            save_manifest()
        normalized = sorted(articles.values(), key=lambda row: (_parse_timestamp(row["created_at"]), row["id"]))
        manifest["normalized_sha256"] = _save(output / "news_normalized.json", normalized)
        manifest["status"] = "complete" if news_summary["pagination_complete"] else "truncated"
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        if news_summary["status"] == "running":
            news_summary["status"] = "failed"
        if isinstance(exc, CollectionError):
            manifest["error"] = str(exc)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, CollectionError):
            raise
        raise CollectionError("Capture failed; inspect the retained manifest") from exc
    finally:
        manifest["finished_at"] = _stamp(clock())
        save_manifest()
        if owned_session and session is not None:
            session.close()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new private archive directory; existing directories are rejected.")
    parser.add_argument("--symbols", nargs="+", default=["PLTR", "AMZN", "MSFT"])
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD, beginning at midnight in New York.")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD, ending at 16:00 in New York; must be in the past.")
    parser.add_argument("--start-at", help="Optional exact timezone-aware start; provide with --end-at")
    parser.add_argument("--end-at", help="Optional exact timezone-aware end, including after-hours; must have occurred")
    parser.add_argument("--max-news-pages", type=int, default=MAX_NEWS_PAGES, help="Bounded news page budget, from 1 to 20.")
    args = parser.parse_args()
    try:
        manifest = collect_research(args.output_dir, args.symbols, args.start_date,
                                    args.end_date, max_news_pages=args.max_news_pages,
                                    start_at=args.start_at, end_at=args.end_at)
    except (CollectionError, ValueError, FileExistsError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps({"status": manifest["status"], "output_dir": str(args.output_dir),
                      "news_summary": manifest["news_summary"], "bars_summary": manifest["bars_summary"]}, indent=2))
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
