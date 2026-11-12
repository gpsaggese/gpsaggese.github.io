"""Bounded SEC Company Facts capture and conservative financial fact normalization.

Imports are inert. Only four declared companies and one GET per company are
permitted. Current captures do not establish historical pipeline availability.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener


SYMBOLS = ("NVDA", "SNDK", "TSLA", "PLTR")
API_DOCUMENTATION = "https://www.sec.gov/search-filings/edgar-application-programming-interfaces"
MAX_BYTES = 20 * 1024 * 1024
METRICS = {
    "revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss",),
    "cash_and_cash_equivalents": ("CashAndCashEquivalentsAtCarryingValue",),
    "total_assets": ("Assets",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
}
INSTANT_METRICS = {"cash_and_cash_equivalents", "total_assets"}
PERIODIC_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}


def now():
    return datetime.now(timezone.utc)


def stamp(value):
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("A timezone-aware capture clock is required")
    return value.astimezone(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def save_json(path, payload):
    data = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    with path.open("wb") as stream:
        stream.write(data)
    path.chmod(0o600)
    return digest(data)


def verified_issuers(submissions_dir, symbols=SYMBOLS):
    """Use existing SEC submissions identity, never a guessed ticker/CIK map."""
    result = {}
    for symbol in symbols:
        path = Path(submissions_dir) / f"{symbol}_submissions.response"
        raw = path.read_bytes()
        payload = json.loads(raw)
        cik = str(int(payload["cik"]))
        if not cik.isdigit() or len(cik) > 10 or int(cik) <= 0 or symbol not in payload.get("tickers", []):
            raise ValueError(f"Submissions identity does not verify {symbol}")
        result[symbol] = dict(cik=cik.zfill(10), name=payload["name"],
                              submissions_source=str(path), submissions_sha256=digest(raw))
    return result


def period_type(metric, start, end):
    """Duration buckets do not invent fiscal quarters or derive Q4 from YTD."""
    if start is None:
        return ("instant", None) if metric in INSTANT_METRICS else ("missing_duration", None)
    days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
    if days <= 0:
        raise ValueError("Financial period ends before it begins")
    if metric in INSTANT_METRICS:
        return "unexpected_duration_for_balance", days
    if 75 <= days <= 105:
        return "quarter_length_duration", days
    if 150 <= days <= 210:
        return "six_month_cumulative_duration", days
    if 240 <= days <= 300:
        return "nine_month_cumulative_duration", days
    if 330 <= days <= 400:
        return "annual_duration", days
    return "other_duration", days


def normalize_company(payload, symbol, identity, received_at, as_of_date):
    """Retain every candidate fact/version; cutoff only gates descriptive views.

    Quarter-length means reported start/end spans 75–105 days. It is a candidate
    company quarter, not a claim about fiscal numbering or eight consecutive
    quarters. Six/nine-month durations remain cumulative, never divided by 2/3.
    """
    cutoff = date.fromisoformat(as_of_date)
    receipt = datetime.fromisoformat(received_at.replace("Z", "+00:00"))
    if receipt.tzinfo is None or cutoff > receipt.date():
        raise ValueError("Cutoff must not be later than the actual capture date")
    if int(payload["cik"]) != int(identity["cik"]):
        raise ValueError("Company Facts CIK differs from verified submissions")
    taxonomy = payload.get("facts", {}).get("us-gaap", {})
    records, rejected = [], []
    for metric, candidates in METRICS.items():
        for tag in candidates:
            concept = taxonomy.get(tag)
            if concept is None:
                continue
            for unit, facts in concept.get("units", {}).items():
                for index, original in enumerate(facts):
                    try:
                        start = original.get("start")
                        end, filed = original["end"], original["filed"]
                        end_date, filed_date = date.fromisoformat(end), date.fromisoformat(filed)
                        if start is not None:
                            date.fromisoformat(start)
                        value = original["val"]
                        if type(value) not in (int, float) or not math.isfinite(value):
                            raise ValueError("Nonfinite or nonnumeric financial fact")
                        if not isinstance(original.get("accn"), str) or not original["accn"]:
                            raise ValueError("Missing accession")
                        kind, days = period_type(metric, start, end)
                        reasons = []
                        if filed_date > cutoff:
                            reasons.append("filed_after_cutoff")
                        if end_date > cutoff:
                            reasons.append("period_ends_after_cutoff")
                        if original.get("form") not in PERIODIC_FORMS:
                            reasons.append("not_periodic_10k_10q")
                        if kind in {"missing_duration", "unexpected_duration_for_balance"}:
                            reasons.append(kind)
                        body = dict(symbol=symbol, cik=identity["cik"], metric=metric, taxonomy="us-gaap", tag=tag,
                                    unit=unit, value=value, start=start, end=end, filed=filed,
                                    fy=original.get("fy"), fp=original.get("fp"), form=original.get("form"),
                                    accession=original["accn"], frame=original.get("frame"),
                                    period_type=kind, duration_days=days)
                        fact_id = digest(json.dumps(body, sort_keys=True, allow_nan=False).encode())
                        records.append({**body, "fact_id": fact_id, "source_row_index": index,
                                        "response_received_at": received_at,
                                        "historical_pipeline_received_at": None,
                                        "historical_availability_verified": False,
                                        "descriptive_as_of_eligible": not reasons, "exclusion_reasons": reasons})
                    except (KeyError, ValueError, TypeError) as exc:
                        rejected.append(dict(metric=metric, tag=tag, unit=unit, source_row_index=index,
                                             error_type=type(exc).__name__))
    return dict(schema_version="sec-companyfacts-1", symbol=symbol, cik=identity["cik"],
                entity_name=payload.get("entityName"), as_of_date=as_of_date,
                response_received_at=received_at, historical_availability_verified=False,
                records=records, rejected_rows=rejected,
                limitations=["Company Facts does not replace filing narrative or risk factors.",
                             "No automated investment rating or trading authorization.",
                             "Filed is a date, not a verified historical public or pipeline receipt timestamp.",
                             "Custom tags and segment/dimensional facts are outside this normalized coverage.",
                             "Duration buckets are descriptive; quarterly values are not derived from YTD."])


def summarize_company(normalized):
    """Select an explicit tag and latest eligible version; never sum aliases."""
    rows = normalized["records"]
    metrics = {}
    for metric, candidates in METRICS.items():
        all_rows = [r for r in rows if r["metric"] == metric]
        eligible = [r for r in all_rows if r["descriptive_as_of_eligible"]]
        groups = defaultdict(list)
        for row in eligible:
            groups[(row["unit"], row["start"], row["end"], row["period_type"])].append(row)
        periods = []
        for (unit, start, end, kind), versions in groups.items():
            tag = min((r["tag"] for r in versions), key=candidates.index)
            tag_versions = [r for r in versions if r["tag"] == tag]
            latest_filed = max(r["filed"] for r in tag_versions)
            latest = [r for r in tag_versions if r["filed"] == latest_filed]
            values = {r["value"] for r in latest}
            conflict = len(values) != 1
            periods.append(dict(unit=unit, start=start, end=end, period_type=kind, selected_tag=tag,
                                latest_filed=latest_filed, value=None if conflict else latest[0]["value"],
                                conflicting_latest_values=conflict, selected_fact_ids=sorted({r["fact_id"] for r in latest}),
                                version_count=len(versions), available_tags=sorted({r["tag"] for r in versions}),
                                derived=False))
        periods.sort(key=lambda row: (row["end"], row["start"] or "", row["unit"]))
        units = {}
        for unit in sorted({row["unit"] for row in periods}):
            selected = [r for r in periods if r["unit"] == unit]
            quarters = [r for r in selected if r["period_type"] == "quarter_length_duration" and not r["conflicting_latest_values"]]
            by_quarter_end = defaultdict(list)
            for row in quarters:
                by_quarter_end[row["end"]].append(row)
            ambiguous_ends = sorted(end for end, contexts in by_quarter_end.items() if len(contexts) != 1)
            # Multiple start dates ending on one day are not multiple quarters.
            quarters = [contexts[0] for end, contexts in sorted(by_quarter_end.items()) if len(contexts) == 1]
            recent = quarters[-8:]
            gaps = [(date.fromisoformat(b["end"]) - date.fromisoformat(a["end"])).days
                    for a, b in zip(recent, recent[1:])]
            units[unit] = dict(period_type_counts=dict(Counter(r["period_type"] for r in selected)),
                               latest_reported_end=max((r["end"] for r in selected), default=None),
                               latest_filed=max((r["latest_filed"] for r in selected), default=None),
                               latest_eight_quarter_length_periods=recent,
                               eight_reported_quarter_length_periods_available=len(recent) == 8,
                               eight_consecutive_fiscal_quarters_verified=False,
                               ambiguous_quarter_end_dates=ambiguous_ends,
                               recent_quarter_end_gaps_days=gaps,
                               missing_or_irregular_quarter_gaps=any(not 75 <= gap <= 105 for gap in gaps),
                               latest_by_period_type={kind: [r for r in selected if r["period_type"] == kind][-1]
                                                      for kind in sorted({r["period_type"] for r in selected})})
        metrics[metric] = dict(candidate_tags=list(candidates), normalized_rows=len(all_rows), eligible_rows=len(eligible),
                               missing=len(eligible) == 0, units=units)
    eligible = [r for r in rows if r["descriptive_as_of_eligible"]]
    return dict(symbol=normalized["symbol"], cik=normalized["cik"], entity_name=normalized["entity_name"],
                as_of_date=normalized["as_of_date"], response_received_at=normalized["response_received_at"],
                historical_availability_verified=False, normalized_fact_versions=len(rows), rejected_rows=len(normalized["rejected_rows"]),
                latest_reported_period_end=max((r["end"] for r in eligible), default=None),
                latest_eligible_filing_date=max((r["filed"] for r in eligible), default=None),
                missing_metrics=[m for m, v in metrics.items() if v["missing"]], metrics=metrics,
                limitations=normalized["limitations"])


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_once(url, user_agent):
    request = Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    try:
        return build_opener(NoRedirect()).open(request, timeout=40)
    except HTTPError as response:
        return response


def collect_company_facts(output_dir, submissions_dir, *, symbols=SYMBOLS, as_of_date=None, fetch=fetch_once, clock=now, pause=time.sleep):
    symbols = tuple(symbols)
    if not symbols or len(symbols) > 4 or len(set(symbols)) != len(symbols) or not set(symbols) <= set(SYMBOLS):
        raise ValueError("Choose 1–4 distinct declared company symbols")
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a new capture directory; captures are not overwritten")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / ".gitignore").write_text("*\n", encoding="utf-8")
    manifest = dict(schema_version="sec-companyfacts-capture-1", status="running", method="GET", requests=[],
                    started_at=stamp(clock()), source_documentation=API_DOCUMENTATION,
                    collector_sha256=digest(Path(__file__).read_bytes()), historical_availability_verified=False)
    save_json(output / "manifest.json", manifest)
    try:
        issuers = verified_issuers(submissions_dir, symbols)
    except Exception as exc:
        manifest.update(status="identity_verification_failed", error_type=type(exc).__name__)
        save_json(output / "manifest.json", manifest)
        raise
    manifest["verified_issuers"] = issuers
    cutoff = as_of_date or clock().date().isoformat()
    if date.fromisoformat(cutoff) > clock().date():
        manifest.update(status="invalid_future_cutoff")
        save_json(output / "manifest.json", manifest)
        raise ValueError("Future cutoff is forbidden")
    user_agent = os.environ.get("SEC_USER_AGENT") or "QuantAI research/1.0 https://github.com/ypeng12/Quant.ai"
    manifest["user_agent_source"] = "SEC_USER_AGENT" if os.environ.get("SEC_USER_AGENT") else "honest_project_url"
    summaries = []
    for symbol, identity in issuers.items():
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{identity['cik']}.json"
        request = dict(symbol=symbol, url=url, method="GET", started_at=stamp(clock()), status="pending")
        manifest["requests"].append(request)
        save_json(output / "manifest.json", manifest)
        try:
            response = fetch(url, user_agent)
            try:
                raw = response.read(MAX_BYTES + 1)
                request.update(response_received_at=stamp(clock()), http_status=response.getcode(),
                               response_bytes=len(raw), truncated=len(raw) > MAX_BYTES)
            finally:
                response.close()
            raw = raw[:MAX_BYTES]
            raw_path = output / f"{symbol}_companyfacts.response"
            raw_path.write_bytes(raw)
            raw_path.chmod(0o600)
            request.update(raw_file=raw_path.name, raw_sha256=digest(raw))
            if request["http_status"] != 200 or request["truncated"]:
                request["status"] = "http_failed" if request["http_status"] != 200 else "oversize_failed"
            else:
                normalized = normalize_company(json.loads(raw), symbol, identity, request["response_received_at"], cutoff)
                summary = summarize_company(normalized)
                request.update(status="complete", normalized_sha256=save_json(output / f"{symbol}_normalized.json", normalized),
                               summary_sha256=save_json(output / f"{symbol}_summary.json", summary))
                summaries.append(summary)
        except Exception as exc:
            request.update(status="failed", error_type=type(exc).__name__)
        save_json(output / "manifest.json", manifest)
        pause(0.25)  # Four serial requests; no retries, redirect bypass or browser impersonation.
    manifest.update(status="complete" if all(r["status"] == "complete" for r in manifest["requests"]) else "partial_or_failed",
                    completed_at=stamp(clock()), company_count=len(summaries))
    save_json(output / "summary.json", dict(companies=summaries, status=manifest["status"]))
    save_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--submissions-dir", type=Path, required=True)
    parser.add_argument("--as-of-date")
    parser.add_argument("--symbols", nargs="+", choices=SYMBOLS, default=SYMBOLS)
    args = parser.parse_args()
    manifest = collect_company_facts(args.output_dir, args.submissions_dir, symbols=args.symbols, as_of_date=args.as_of_date)
    print(json.dumps({"status": manifest["status"], "companies": manifest["company_count"],
                      "requests": [{k: r.get(k) for k in ("symbol", "http_status", "status")} for r in manifest["requests"]]}))


if __name__ == "__main__":
    main()
