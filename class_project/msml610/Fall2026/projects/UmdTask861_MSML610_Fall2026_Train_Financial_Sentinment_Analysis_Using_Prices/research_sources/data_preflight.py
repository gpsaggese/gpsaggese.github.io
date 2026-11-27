"""Audit a completed month's local corpus before constructing price labels.

Only saved, explicitly selected archives are read. Publication-date coverage is
not a decision-session dataset. No sessions, receipts, total returns, event
identities, labels or model performance are inferred from missing evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, time, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from run_research import read_inputs, timestamp

NEW_YORK = ZoneInfo("America/New_York")
SOURCE_FAMILIES = ("financial_report", "earnings_call", "news", "investor_comment")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def _legacy_news(directory: Path) -> tuple[list[dict], dict, list[dict]]:
    """Adapt only the original four news pages; request start is not receipt."""
    hashes = json.loads((directory / "hashes.json").read_text())
    names = ["requests.json", "news_normalized.json", *[
        f"news_recent_{i:03d}.json" for i in range(4)]]
    fingerprints = {str(directory / "hashes.json"): _hash(directory / "hashes.json")}
    for name in names:
        path = directory / name
        if _hash(path) != hashes.get(name):
            raise ValueError(f"Legacy archive hash mismatch: {name}")
        fingerprints[str(path)] = hashes[name]
    requests = json.loads((directory / "requests.json").read_text())
    news_requests = [r for r in requests if r.get("file") in names[2:]]
    if (len(news_requests) != 4 or len({r["file"] for r in news_requests}) != 4
            or any(r.get("http_status") != 200 or r.get("route") != "/v1beta1/news"
                   for r in news_requests)):
        raise ValueError("Legacy news request evidence is incomplete")
    raw = {}
    for name in names[2:]:
        for row in json.loads((directory / name).read_text())["news"]:
            raw[str(row["id"])] = row
    rows = json.loads((directory / "news_normalized.json").read_text())
    if len({str(r["id"]) for r in rows}) != len(rows) or set(raw) != {str(r["id"]) for r in rows}:
        raise ValueError("Legacy normalized IDs differ from raw pages")
    for row in rows:
        original = raw[str(row["id"])]
        for field in ("created_at", "updated_at", "url", "symbols", "source"):
            if row[field] != original[field]:
                raise ValueError(f"Legacy normalized {field} differs from raw page")
        for field, source in (("headline", "headline"), ("text", "content"), ("summary", "summary")):
            if row[field] != _clean(original.get(source, "")):
                raise ValueError("Legacy normalized text differs from raw page")
    return rows, fingerprints, news_requests


def _version(row: dict) -> str:
    content = {key: row[key] for key in (
        "id", "headline", "text", "summary", "symbols", "created_at", "updated_at")}
    content["id"] = str(content["id"])
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def audit_source_records(records: list[dict], *, cutoff: str, as_of: str) -> dict:
    """Audit supplied source/version/retrieval contracts without inferring events.

    Adapters must bind actual response evidence to the retained text version.
    First receipt means the earliest such receipt retained for this version,
    never a global first publication, request start, or an older text's receipt.
    Event identities require explicit evidence. This source audit does not make
    price labels, authenticate caller declarations, or establish trading alpha.
    """
    boundary, clock = timestamp(cutoff), timestamp(as_of)
    if boundary > clock or not isinstance(records, list):
        raise ValueError("Require a record list and cutoff <= as_of")

    def optional_time(value):
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Source times must be aware strings or null")
        return timestamp(value)

    def iso(value):
        return value.isoformat().replace("+00:00", "Z") if value else None

    def digest(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    def evidence_id(value):
        return isinstance(value, str) and bool(value.strip())

    results, seen = [], set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each source record must be an object")
        identity, source, family = (record.get(key) for key in ("record_id", "source_id", "source_family"))
        if (not isinstance(identity, str) or not identity.strip() or identity in seen
                or not isinstance(source, str) or not source.strip() or family not in SOURCE_FAMILIES):
            raise ValueError("Require unique record_id, source_id and explicit source_family")
        seen.add(identity)
        events = record.get("event_ids", [])
        if (not isinstance(events, list) or any(not isinstance(e, str) or not e.strip() for e in events)
                or len(events) != len(set(events))):
            raise ValueError("event_ids must contain distinct nonempty strings")
        event_day = record.get("event_calendar_date")
        if event_day is not None:
            if not isinstance(event_day, str) or date.fromisoformat(event_day).isoformat() != event_day:
                raise ValueError("event_calendar_date must be an explicit ISO date")
        # Fiscal periods, URL path years and report-period ends are never parsed
        # into event dates or exact publication timestamps.
        fiscal_period = record.get("fiscal_period")
        if fiscal_period is not None and not isinstance(fiscal_period, str):
            raise ValueError("fiscal_period must be a label or null")
        times = {field: optional_time(record.get(field)) for field in (
            "original_published_at", "version_published_at", "republished_at")}
        reasons, responses, receipts, evidence = [], [], [], []
        version = record.get("text_version_sha256")
        if times["original_published_at"] is None:
            reasons.append("original_publication_time_missing")
        if not evidence_id(record.get("original_publication_evidence_id")):
            reasons.append("original_publication_evidence_missing")
        if record.get("text_available") is not True:
            reasons.append("retained_text_missing")
        if not digest(version):
            reasons.append("text_version_missing")
        if not events or not evidence_id(record.get("event_identity_evidence_id")):
            reasons.append("event_identity_unreviewed_or_missing")
        for field, value in times.items():
            if value is not None and value > boundary:
                reasons.append(field + "_after_cutoff")
        if (times["original_published_at"] is not None and times["version_published_at"] is not None
                and times["version_published_at"] < times["original_published_at"]):
            reasons.append("version_precedes_original_publication")
        if (times["original_published_at"] is not None and times["republished_at"] is not None
                and times["republished_at"] < times["original_published_at"]):
            reasons.append("republication_precedes_original_publication")
        if family in {"financial_report", "earnings_call"}:
            if event_day is None:
                reasons.append("financial_event_calendar_date_missing")
            elif event_day > boundary.astimezone(NEW_YORK).date().isoformat():
                reasons.append("financial_event_after_cutoff")
        captures = record.get("captures", [])
        if not isinstance(captures, list):
            raise ValueError("captures must be a list")
        for capture in captures:
            if not isinstance(capture, dict):
                raise ValueError("Each capture must be an object")
            received = optional_time(capture.get("response_received_at"))
            started = optional_time(capture.get("request_started_at"))
            # Request start is only a consistency check, never receipt evidence;
            # first_seen metadata is deliberately ignored.
            bound_response = (capture.get("archive_integrity_verified") is True
                              and evidence_id(capture.get("retrieval_evidence_id"))
                              and digest(capture.get("response_sha256")))
            if received is not None and started is not None and received < started:
                reasons.append("response_precedes_request_start")
            elif received is not None and received > clock:
                reasons.append("response_after_as_of")
            elif received is not None and bound_response:
                responses.append(received)
                if (type(capture.get("status")) is int and capture["status"] == 200
                        and digest(version) and capture.get("text_version_sha256") == version
                        and record.get("text_available") is True):
                    receipts.append(received)
            evidence.append({key: capture.get(key) for key in (
                "retrieval_evidence_id", "request_evidence_id", "response_sha256", "text_version_sha256",
                "response_received_at", "request_started_at", "status", "archive_integrity_verified")})
        first = min(receipts, default=None)
        if first is None:
            reasons.append("text_version_receipt_evidence_missing")
        else:
            if first > boundary:
                reasons.append("text_version_received_after_cutoff")
            if any(value is not None and first < value for value in times.values()):
                reasons.append("text_receipt_precedes_version_publication")
        reasons = list(dict.fromkeys(reasons))
        late = any(reason.endswith("after_cutoff") for reason in reasons)
        results.append({
            "record_id": identity, "source_id": source, "source_family": family,
            "document_kind": record.get("document_kind"), "event_ids": events.copy(),
            "event_identity_evidence_id": record.get("event_identity_evidence_id"),
            "event_calendar_date": event_day, "fiscal_period": fiscal_period,
            **{key: iso(value) for key, value in times.items()},
            "original_publication_evidence_id": record.get("original_publication_evidence_id"),
            "text_version_sha256": version, "text_available": record.get("text_available") is True,
            "first_response_received_at": iso(min(responses, default=None)),
            "first_actual_receipt_at": iso(first),
            "retrieval_evidence": evidence, "source_contract_eligible": not reasons,
            "status": "After cutoff" if late else "Missing or inconsistent evidence" if reasons else "Eligible under supplied evidence",
            "reasons": reasons,
        })
    # Keep an older version's receipt visible, but never substitute it for the
    # current version's first_actual_receipt_at when auditing revisions/reposts.
    source_first = {}
    for row in results:
        if row["first_actual_receipt_at"]:
            key = row["source_family"], row["source_id"]
            value = timestamp(row["first_actual_receipt_at"])
            source_first[key] = min(source_first.get(key, value), value)
    for row in results:
        row["source_first_retained_text_receipt_at"] = iso(source_first.get((row["source_family"], row["source_id"])))
    families = {}
    for family in SOURCE_FAMILIES:
        rows = [row for row in results if row["source_family"] == family]
        families[family] = {
            "records": len(rows), "retained_text_records": sum(row["text_available"] for row in rows),
            "source_contract_eligible": sum(row["source_contract_eligible"] for row in rows),
            "distinct_source_ids": len({row["source_id"] for row in rows}),
            "distinct_supplied_event_ids": len({event for row in rows for event in row["event_ids"]}),
            "missing_event_identity_records": sum("event_identity_unreviewed_or_missing" in row["reasons"] for row in rows),
        }
    return {
        "schema_version": "financial-source-eligibility-1.0", "cutoff": iso(boundary), "as_of": iso(clock),
        "families": families, "records": results,
        "eligible_source_records": sum(row["source_contract_eligible"] for row in results),
        "reason_counts": dict(Counter(reason for row in results for reason in row["reasons"])),
        "missing_source_families": [family for family in SOURCE_FAMILIES if not families[family]["retained_text_records"]],
        "limitations": [
            "Eligibility checks supplied evidence; it is not independent authentication or a price label.",
            "First receipt is the earliest retained response bound to this exact text version, not global first availability.",
            "Shared event IDs keep syndicated copies associated; event-ID counts are not independent samples.",
            "Unknown event identities remain unknown; no event is invented per article, filing or comment.",
            "Fiscal-year labels and report-period ends do not supply event calendar dates or exact publication times.",
            "No text scoring, return labels, model evaluation or profit evidence is produced.",
        ],
    }


def audit_archives(primary_dir: Path, legacy_alpaca_dir: Path, *, symbols: list[str],
                   month: str, as_of: str, source_snapshot_dir: Path | None = None) -> dict:
    """Return an offline readiness audit, including an explicit negative result.

    ``primary_dir`` is a collector/run_research archive; ``legacy_alpaca_dir``
    is the four-page original audit. The scope does not imply a repository-wide
    search or full vendor/company coverage. No price-label input is generated.
    """
    if not symbols or len(set(symbols)) != len(symbols) or any(
            not isinstance(s, str) or not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", s)
            or s == "SPY" for s in symbols):
        raise ValueError("Provide distinct company tickers, excluding benchmark SPY")
    if not isinstance(month, str) or not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError("month must use YYYY-MM")
    first = date.fromisoformat(month + "-01")
    following = date(first.year + (first.month == 12), first.month % 12 + 1, 1)
    clock = timestamp(as_of)
    if clock < datetime.combine(following, time.min, NEW_YORK):
        raise ValueError("The requested calendar month is not yet complete")
    primary_dir, legacy_alpaca_dir = Path(primary_dir).resolve(), Path(legacy_alpaca_dir).resolve()
    articles, bars, provenance = read_inputs(primary_dir)
    manifest = json.loads((primary_dir / "manifest.json").read_text())
    legacy, fingerprints, old_requests = _legacy_news(legacy_alpaca_dir)
    for name, expected in provenance["input_files"].items():
        fingerprints[str(primary_dir / name)] = expected
    if _hash(primary_dir / "manifest.json") != provenance["input_files"]["manifest.json"]:
        raise ValueError("Primary manifest changed during preflight")
    primary_raw_news, primary_requests_by_id = {}, {}
    for request in manifest["requests"]:
        name = request["file"]
        fingerprints[str(primary_dir / name)] = request["sha256"]
        if name.startswith("news_raw_"):
            for row in json.loads((primary_dir / name).read_text())["news"]:
                primary_raw_news[str(row["id"])] = row
                primary_requests_by_id[str(row["id"])] = request
    # read_inputs already binds headline, body, tags, version times and receipt.
    # Bind the remaining version fields too, rather than fingerprinting an
    # unverified normalized summary as though it were a captured text version.
    for row in articles:
        raw = primary_raw_news[str(row["id"])]
        if row["summary"] != _clean(raw.get("summary", "")) or row["source"] != raw.get("source", ""):
            raise ValueError("Primary normalized summary or source differs from raw page")
    bar_requests = [r for r in manifest["requests"] if r.get("file") == "daily_bars_raw.json"]
    if (len(bar_requests) != 1 or bar_requests[0].get("route") != "/v2/stocks/bars"
            or bar_requests[0].get("params", {}).get("timeframe") != "1Day"
            or bar_requests[0]["params"].get("adjustment") != "raw"):
        raise ValueError("Expected one explicitly raw daily-bar request")
    bar_receipt = timestamp(bar_requests[0]["response_received_at"])
    if bar_receipt > clock or any(timestamp(r["observed_at"]) > clock for r in old_requests):
        raise ValueError("as_of precedes archive capture evidence")
    legacy_requests_by_id = {}
    for request in old_requests:
        for row in json.loads((legacy_alpaca_dir / request["file"]).read_text())["news"]:
            legacy_requests_by_id[str(row["id"])] = request
    captures, source_contracts = [], {}
    for origin, records in (("legacy", legacy), ("primary", articles)):
        for row in records:
            published, updated = timestamp(row["created_at"]), timestamp(row["updated_at"])
            if updated < published or updated > clock:
                raise ValueError("Article version has invalid or future timestamps")
            # The legacy observed_at is a request START; even elapsed_seconds
            # cannot supply an exact response receipt or historical text version.
            received = timestamp(row["retrieved_at"]) if origin == "primary" else None
            if received is not None and (received < updated or received > clock):
                raise ValueError("Article receipt does not follow its version or exceeds as_of")
            captures.append({
                "archive": origin, "article_id": str(row["id"]),
                "source_group_id": "alpaca-news:" + str(row["id"]),
                "version_sha256": _version(row), "symbols": sorted(set(row["symbols"])),
                "publication_date_ny": published.astimezone(NEW_YORK).date().isoformat(),
                "published_at": row["created_at"], "updated_at": row["updated_at"],
                "response_received_at": row["retrieved_at"] if received else None,
                "capture_available_at": received.isoformat() if received else None,
                "body_present": bool(row["text"].strip()),
                "historical_availability_verified": False,
                "availability_reason": "Observed later capture; historical version not proven"
                if received else "Exact receipt missing; request start is not receipt",
            })
            request = (primary_requests_by_id if origin == "primary" else legacy_requests_by_id)[str(row["id"])]
            directory = primary_dir if origin == "primary" else legacy_alpaca_dir
            raw_path = str(directory / request["file"])
            version = _version(row)
            source_contracts[(origin, str(row["id"]))] = {
                "record_id": origin + ":alpaca-news:" + str(row["id"]),
                "source_family": "news", "source_id": "alpaca-news:" + str(row["id"]),
                "event_ids": [], "event_identity_evidence_id": None,
                "original_published_at": row["created_at"], "version_published_at": row["updated_at"],
                "original_publication_evidence_id": raw_path, "republished_at": None,
                "event_calendar_date": None, "fiscal_period": None,
                "text_version_sha256": version,
                "text_available": bool(row["headline"].strip() or row["text"].strip()),
                "document_kind": "news_article" if row["text"].strip() else "news_headline_only",
                "captures": [{
                    "response_received_at": row["retrieved_at"] if received else None,
                    "request_started_at": request.get("started_at") if received else request.get("observed_at"),
                    "status": request["status"] if received else request["http_status"],
                    "retrieval_evidence_id": raw_path,
                    "request_evidence_id": str(directory / ("manifest.json" if received else "requests.json")),
                    "response_sha256": fingerprints[raw_path], "text_version_sha256": version,
                    "archive_integrity_verified": True,
                }],
            }
    selected = [r for r in captures if r["publication_date_ny"].startswith(month)
                and set(r["symbols"]) & set(symbols)]
    unique_ids = {r["article_id"] for r in selected}
    versions = {(r["article_id"], r["version_sha256"]) for r in selected}
    companies = {}
    for symbol in [*symbols, "SPY"]:
        daily = bars.get("bars", {}).get(symbol, [])
        dates = []
        for row in daily:
            day = timestamp(row["t"]).astimezone(NEW_YORK).date().isoformat()
            opening = row.get("o")
            if (isinstance(opening, bool) or not isinstance(opening, (int, float))
                    or not math.isfinite(opening) or opening <= 0):
                raise ValueError("Invalid daily opening value")
            if timestamp(row["t"]) > bar_receipt or day in dates:
                raise ValueError("Future or duplicate daily bar")
            dates.append(day)
        dates.sort()
        company_news = [r for r in selected if symbol in r["symbols"]] if symbol != "SPY" else []
        companies[symbol] = {
            "selected_article_ids": len({r["article_id"] for r in company_news}),
            "selected_body_article_ids": len({r["article_id"] for r in company_news if r["body_present"]}),
            "publication_dates_ny": sorted({r["publication_date_ny"] for r in company_news}),
            "observed_daily_dates_in_month": [d for d in dates if d.startswith(month)],
            "observed_daily_dates_after_month": [d for d in dates if d >= following.isoformat()],
            "daily_records": len(dates), "first_observed_daily_date": min(dates, default=None),
            "last_observed_daily_date": max(dates, default=None),
            "return_basis": "raw_price", "response_received_at": bar_receipt.isoformat(),
            "bar_timestamp_policy": "Vendor daily date marker, not actual regular open or receipt",
        }
    missing = [
        "complete_month_company_news_query_and_text_versions",
        "historical_response_receipts_for_selected_text",
        "independent_versioned_complete_exchange_calendar",
        "corporate_actions_and_consistent_open_total_return_indices",
        "full_price_history_and_ten_session_label_tail",
        "decision_day_text_and_feature_adapter_with_actual_completion_times",
        "event_grouping_across_articles_and_companies_and_independent_review",
    ]
    summary = {
        "selected_unique_article_ids": len(unique_ids), "selected_distinct_versions": len(versions),
        "selected_article_company_pairs": sum(companies[s]["selected_article_ids"] for s in symbols),
        "selected_body_article_company_pairs": sum(companies[s]["selected_body_article_ids"] for s in symbols),
        "selected_capture_rows": len(selected),
        "selected_captures_missing_exact_receipt": sum(r["response_received_at"] is None for r in selected),
        "decision_session_denominator": None, "ten_session_matchable_fraction": None,
        "whole_month_horizon_mature": None, "strict_eligible_labels_created": 0,
        "real_model_evaluation_status": "Not evaluated",
        "readiness": "Insufficient evidence for formal ten-session dataset",
    }
    source_audit = audit_source_records(
        [source_contracts[(row["archive"], row["article_id"])] for row in selected],
        cutoff=as_of, as_of=as_of)
    source_audit["cutoff_basis"] = "Audit-time inventory only; not a company-day 08:45 replay"
    result = {
        "schema_version": "financial-sentiment-month-preflight-1.0", "as_of": clock.isoformat(),
        "scope": {"symbols": symbols, "publication_month_ny": month,
                  "primary_archive": str(primary_dir), "legacy_alpaca_archive": str(legacy_alpaca_dir),
                  "selection_policy": "Caller-selected scope; inherited development corpus, not blind evaluation",
                  "prior_price_inspection": "Primary archive includes previously price-inspected exploratory cases"},
        "summary": summary, "companies_and_benchmark": companies,
        "news_query_evidence": [
            {"archive": origin, "query_symbols": r["params"].get("symbols"),
             "start": r["params"].get("start"), "end": r["params"].get("end"),
             "target_month_complete_coverage_verified": False}
            for origin, requests in (("legacy", old_requests), ("primary", manifest["requests"]))
            for r in requests if r.get("route") == "/v1beta1/news"],
        "selected_capture_audit": selected, "input_files_sha256": fingerprints,
        "source_family_audit": source_audit,
        "missing_evidence": missing,
        "interpretation": [
            "No news returned by an incomplete query is missing coverage, not confirmed no-news.",
            "Ticker tags are candidate relevance; source IDs are not independent events.",
            "Identical versions across captures are not independent historical versions.",
            "Observed bar dates do not establish a trading calendar, maturity or missing-session count.",
            "Fractions remain null without a verified decision-session denominator.",
            "No raw text or price values are copied; no labels, fits or predictive metrics are produced.",
        ],
    }
    if source_snapshot_dir is not None:
        records, snapshot_inputs, coverage = _snapshot_source_records(source_snapshot_dir)
        snapshot_audit = audit_source_records(records, cutoff=coverage["snapshot_analysis_cutoff"], as_of=as_of)
        snapshot_audit["coverage"] = coverage
        snapshot_audit["cutoff_basis"] = "Original snapshot cutoff; actual later captures remain after cutoff"
        result["source_snapshot_audit"] = snapshot_audit
        result["input_files_sha256"].update(snapshot_inputs)
    return result


def _snapshot_source_records(directory):
    """Bind a saved stock-context snapshot to source records, without inference.

    The old report's six inputs are preserved and verified, but its manually
    selected summaries are never substituted for captured source text. Capture
    receipts establish only these retained observations, not historical first
    availability or independently identified business events.
    """
    directory = Path(directory).resolve()
    fingerprints = {}

    def read(name, expected=None):
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise ValueError("Snapshot input must remain inside its archive")
        encoded = path.read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        if expected is not None and (
                type(expected.get("bytes")) is not int
                or len(encoded) != expected["bytes"]
                or digest != expected.get("sha256")):
            raise ValueError(f"Snapshot input integrity check failed: {name}")
        fingerprints[str(path)] = digest
        return encoded

    def load(name, expected=None):
        return json.loads(read(name, expected))

    audit = load("generation_audit.json")
    required_inputs = {"daily.json", "intraday.json", "manifest.json",
                       "futu_current/all_normalized.json", "news_context.json",
                       "context_sources.json"}
    if set(audit.get("inputs", {})) != required_inputs:
        raise ValueError("Snapshot generation audit must bind all six documented inputs")
    bound = {}
    for name, evidence in audit["inputs"].items():
        if evidence.get("present") is not True:
            raise ValueError("Snapshot generation input is not present")
        bound[name] = read(name, evidence)
    manifest = json.loads(bound["manifest.json"])
    cutoff = timestamp(manifest["analysis_cutoff"])
    if cutoff != timestamp(audit["analysis_cutoff"]):
        raise ValueError("Snapshot cutoff differs from its generation audit")

    requests = load("source_capture/requests.json")
    names = {"news_raw_000.json", "sec_submissions.json", "sec_selected_document.html",
             "sandisk_ir.html", "cxmt_news.html", "nanya_news.html"}
    if (not isinstance(requests, list) or len(requests) != len(names)
            or {r.get("file") for r in requests} != names):
        raise ValueError("Snapshot source requests must bind the six documented responses")
    responses = {}
    request_by_file = {}
    for request in requests:
        name = request["file"]
        if type(request.get("http_status")) is not int:
            raise ValueError("Snapshot request status must be an integer")
        if timestamp(request["request_started_at"]) > timestamp(request["received_at"]):
            raise ValueError("Snapshot request receipt precedes its request start")
        responses[name] = read("source_capture/" + name, request)
        request_by_file[name] = request

    def capture(request, evidence_name, version_hash=None, *, futu=False):
        return {
            "response_received_at": request["response_received_at" if futu else "received_at"],
            "request_started_at": request["request_started_at"],
            "status": request["http_status"],
            "retrieval_evidence_id": str(directory / (
                "futu_current/SNDK_raw.json" if futu else "source_capture/" + request["file"])),
            "request_evidence_id": str(directory / evidence_name),
            "response_sha256": request["sha256"],
            "text_version_sha256": version_hash,
            "archive_integrity_verified": True,
        }

    def record(family, source_id, published, publication_evidence, version_time,
               version_hash, text_available, kind, observation, *, fiscal_period=None,
               event_calendar_date=None):
        return {
            "record_id": "snapshot:" + source_id,
            "source_family": family, "source_id": source_id,
            "event_ids": [], "event_identity_evidence_id": None,
            "original_published_at": published,
            "original_publication_evidence_id": publication_evidence,
            "version_published_at": version_time, "republished_at": None,
            "event_calendar_date": event_calendar_date, "fiscal_period": fiscal_period,
            "text_version_sha256": version_hash, "text_available": text_available,
            "document_kind": kind, "captures": [observation],
        }

    records = []
    news_request = request_by_file["news_raw_000.json"]
    sec_request = request_by_file["sec_submissions.json"]
    if news_request["http_status"] != 200 or sec_request["http_status"] != 200:
        raise ValueError("Snapshot news and SEC metadata require successful raw responses")
    raw_news = json.loads(responses["news_raw_000.json"])["news"]
    normalized_news = load("source_capture/news_normalized.json")
    raw_by_id = {str(r["id"]): r for r in raw_news}
    if (len(raw_by_id) != len(raw_news)
            or len({str(r["id"]) for r in normalized_news}) != len(normalized_news)
            or set(raw_by_id) != {str(r["id"]) for r in normalized_news}):
        raise ValueError("Snapshot normalized news IDs differ from captured news")
    news_path = str(directory / "source_capture/news_raw_000.json")
    for row in normalized_news:
        raw = raw_by_id[str(row["id"])]
        if (any(row[k] != raw[k] for k in (
                "id", "headline", "symbols", "created_at", "updated_at", "url", "source"))
                or row["text"] != _clean(raw.get("content", ""))
                or row["received_at"] != news_request["received_at"]):
            raise ValueError("Snapshot normalized news differs from captured text or receipt")
        published, updated = timestamp(raw["created_at"]), timestamp(raw["updated_at"])
        if not published <= updated <= timestamp(news_request["received_at"]):
            raise ValueError("Snapshot news version is later than its response")
        version = _version({
            "id": raw["id"], "headline": raw["headline"], "text": row["text"],
            "summary": _clean(raw.get("summary", "")), "symbols": raw["symbols"],
            "created_at": raw["created_at"], "updated_at": raw["updated_at"],
        })
        records.append(record(
            "news", "alpaca-news:" + str(raw["id"]), raw["created_at"], news_path,
            raw["updated_at"], version, bool(raw["headline"].strip() or row["text"].strip()),
            "news_article" if row["text"].strip() else "news_headline_only",
            capture(news_request, "source_capture/requests.json", version)))

    futu_request = load("futu_current/SNDK_request.json")
    if (futu_request.get("http_status") != 200
            or timestamp(futu_request["request_started_at"])
            > timestamp(futu_request["response_received_at"])):
        raise ValueError("Snapshot Futu request evidence is invalid")
    futu_raw = load("futu_current/SNDK_raw.json", {
        "bytes": futu_request["response_bytes"], "sha256": futu_request["sha256"]})
    if futu_raw.get("code") != 0 or not isinstance(futu_raw.get("data"), list):
        raise ValueError("Snapshot Futu response does not contain successful post data")
    futu_rows = json.loads(bound["futu_current/all_normalized.json"])
    raw_posts = {str(r["id"]): r for r in futu_raw["data"]}
    if (len(raw_posts) != len(futu_raw["data"])
            or len({str(r["post_id"]) for r in futu_rows}) != len(futu_rows)
            or set(raw_posts) != {str(r["post_id"]) for r in futu_rows}):
        raise ValueError("Snapshot normalized Futu IDs differ from captured posts")
    for row in futu_rows:
        raw = raw_posts[str(row["post_id"])]
        title, excerpt = _clean(raw["title"]), _clean(raw["desc"])
        version = hashlib.sha256((title + " " + excerpt).encode()).hexdigest()
        raw_epoch = raw["publish_time"]
        if not isinstance(raw_epoch, str) or not re.fullmatch(r"[0-9]+", raw_epoch):
            raise ValueError("Snapshot Futu publication requires explicit Unix seconds")
        published = datetime.fromtimestamp(int(raw_epoch), timezone.utc)
        if (row["title"] != title or row["excerpt"] != excerpt
                or row["text_sha256"] != version or row["raw_publish_time"] != raw_epoch
                or timestamp(row["published_at"]) != published
                or row["request_started_at"] != futu_request["request_started_at"]
                or row["response_received_at"] != futu_request["response_received_at"]
                or row.get("first_seen_in_this_audit_at") != futu_request["response_received_at"]
                or published > timestamp(futu_request["response_received_at"])):
            raise ValueError("Snapshot normalized Futu text or timing differs from raw evidence")
        item = record(
            "investor_comment", "futu-post:" + str(row["post_id"]), published.isoformat(),
            str(directory / "futu_current/SNDK_raw.json"), None, version,
            bool(title or excerpt), "investor_comment",
            capture(futu_request, "futu_current/SNDK_request.json", version, futu=True))
        item["first_seen_in_this_audit_at"] = row["first_seen_in_this_audit_at"]
        records.append(item)

    sec = json.loads(responses["sec_submissions.json"])
    recent = sec["filings"]["recent"]
    sec_rows = load("source_capture/sec_rows.json")
    fields = ("filingDate", "acceptanceDateTime", "form", "accessionNumber", "primaryDocument")
    size = len(recent["accessionNumber"])
    if any(len(recent[key]) != size for key in (*fields, "reportDate")):
        raise ValueError("Snapshot SEC metadata columns have inconsistent lengths")
    recent_by_id = {recent["accessionNumber"][i]: i for i in range(size)}
    if (len(recent_by_id) != size
            or len({r["accessionNumber"] for r in sec_rows}) != len(sec_rows)):
        raise ValueError("Snapshot SEC accession IDs must be unique")
    annual_count = 0
    for row in sec_rows:
        index = recent_by_id.get(row["accessionNumber"])
        if index is None or any(row[key] != recent[key][index] for key in fields):
            raise ValueError("Snapshot selected SEC metadata differs from its raw response")
        if row["form"] != "10-K":
            continue
        annual_count += 1
        accepted = timestamp(row["acceptanceDateTime"])
        if accepted > timestamp(sec_request["received_at"]):
            raise ValueError("Snapshot SEC acceptance is later than its metadata receipt")
        records.append(record(
            "financial_report", "sec-accession:" + row["accessionNumber"],
            accepted.isoformat(), str(directory / "source_capture/sec_submissions.json"),
            None, None, False, "sec_10k_metadata",
            capture(sec_request, "source_capture/requests.json"),
            fiscal_period=recent["reportDate"][index] or None,
            event_calendar_date=accepted.astimezone(NEW_YORK).date().isoformat()))

    def retrieval(name):
        request = request_by_file[name]
        return {"file": name, **capture(request, "source_capture/requests.json"),
                "text_available": False, "event_calendar_date": None}

    coverage = {
        "snapshot_analysis_cutoff": cutoff.isoformat(),
        "news_records": len(normalized_news), "investor_comment_records": len(futu_rows),
        "sec_metadata_records": len(sec_rows),
        "financial_report_metadata_records": annual_count,
        "other_sec_metadata_records": len(sec_rows) - annual_count,
        "financial_report_text_records": 0, "earnings_call_records": 0,
        "failed_body_retrievals": [retrieval(name) for name in (
            "sec_selected_document.html", "sandisk_ir.html")
            if request_by_file[name]["http_status"] != 200],
        "unparsed_html_retrievals": [retrieval(name) for name in (
            "cxmt_news.html", "nanya_news.html")],
        "historical_first_receipt_verified": False,
        "independent_event_identity_verified": False,
        "notes": [
            "SEC acceptance and report period are distinct; filing metadata is not report text.",
            "Manual IR/news summaries are not substituted for successful original document capture.",
            "Industry HTML retrieval is bound but publication, text version and event identity remain unparsed.",
            "first_seen_in_this_audit_at is only this retained capture, not historical first availability.",
        ],
    }
    if any(_hash(Path(path)) != digest for path, digest in fingerprints.items()):
        raise ValueError("Snapshot inputs changed during source audit")
    return records, fingerprints, coverage


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--legacy-alpaca-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--symbols", nargs="+", default=["PLTR", "AMZN", "MSFT"])
    parser.add_argument("--month", required=True)
    parser.add_argument("--as-of", required=True, help="Timezone-aware audit time; does not prove maturity")
    parser.add_argument("--source-snapshot-dir", type=Path, help="Optional existing multi-source snapshot; keep its original cutoff")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    protected = [args.input_dir, args.legacy_alpaca_dir]
    if args.source_snapshot_dir is not None:
        protected.append(args.source_snapshot_dir)
    if args.output_dir.is_symlink() or any(output.is_relative_to(p.resolve()) for p in protected):
        raise ValueError("Output must be new and outside all input archives")
    report = audit_archives(args.input_dir, args.legacy_alpaca_dir,
                            symbols=args.symbols, month=args.month, as_of=args.as_of,
                            source_snapshot_dir=args.source_snapshot_dir)
    # Recheck all input fingerprints before publishing to a fresh private run.
    if any(_hash(Path(p)) != h for p, h in report["input_files_sha256"].items()):
        raise ValueError("Input archive changed during preflight")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    (output / ".gitignore").write_text("*\n", encoding="utf-8")
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (output / "preflight.json").write_text(payload, encoding="utf-8")
    manifest = {
        "status": "Completed evidence audit; dataset not ready",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_files_sha256": report["input_files_sha256"],
        "generation_sources_sha256": {str(Path(__file__).resolve()): _hash(Path(__file__))},
        "preflight_sha256": _hash(output / "preflight.json"),
    }
    for name in ("run_research.py", "event_extraction.py", "price_context.py"):
        source = Path(__file__).with_name(name)
        manifest["generation_sources_sha256"][str(source)] = _hash(source)
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
