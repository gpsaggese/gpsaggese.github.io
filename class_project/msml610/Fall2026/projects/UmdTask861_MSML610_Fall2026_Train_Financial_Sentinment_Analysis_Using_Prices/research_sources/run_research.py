"""Generate an evidence review from saved news and daily prices, without trading.

Text extraction never receives price outcomes or hand-written case conclusions.
The report is descriptive: no price-trained predictor or trading signal exists.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from event_extraction import extract_article, cluster_events, company_clauses
from price_context import build_price_context

ROOT = Path(__file__).resolve().parents[2]
ALIASES = {"PLTR": ["Palantir", "PLTR"], "AMZN": ["Amazon", "AWS", "AMZN"],
           "MSFT": ["Microsoft", "MSFT"]}
BENCHMARKS = {"PLTR": "IGV", "AMZN": "QQQ", "MSFT": "IGV"}
SCHEMA_VERSION = "1.0"


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("An explicit timezone is required")
    return result.astimezone(timezone.utc)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regular_close(day: str) -> datetime:
    """Reference close for a regular NY session; no half-day calendar inference."""
    return datetime.fromisoformat(day + "T16:00:00").replace(tzinfo=ZoneInfo("America/New_York"))


def read_inputs(directory: Path) -> tuple[list[dict], dict, dict]:
    """Read only this allowlist, not saved human analyses or return labels."""
    files = ["manifest.json", "news_normalized.json", "daily_bars_raw.json"]
    inputs = {name: json.loads((directory / name).read_text()) for name in files}
    manifest = inputs["manifest.json"]
    verified = []
    for request in manifest["requests"]:
        filename = request["file"]
        if Path(filename).name != filename or not filename.endswith(".json"):
            raise ValueError("Invalid archive response path")
        if request.get("status") != 200 or digest(directory / filename) != request["sha256"]:
            raise ValueError(f"Archive integrity check failed: {filename}")
        verified.append(filename)
    articles = inputs["news_normalized.json"]
    if len({str(x["id"]) for x in articles}) != len(articles):
        raise ValueError("Duplicate article IDs in the input archive")
    for article in articles:
        for field in ["created_at", "updated_at", "retrieved_at"]:
            timestamp(article[field])
        if timestamp(article["updated_at"]) < timestamp(article["created_at"]):
            raise ValueError("Article update precedes publication")
    # Confirm the normalized texts were retained from these actual responses.
    raw_articles = {}
    receipts = {}
    rejected = {}
    rejection_hash = manifest.get("rejected_records_sha256")
    if rejection_hash is not None:
        rejection_path = directory / "news_rejected_records.json"
        if digest(rejection_path) != rejection_hash:
            raise ValueError("Rejected-record audit hash differs from manifest")
        rejection_rows = json.loads(rejection_path.read_text())
        if not isinstance(rejection_rows, list):
            raise ValueError("Rejected-record audit must be a list")
        for row in rejection_rows:
            key = (row.get("file"), row.get("row_index"))
            if (key[0] not in verified or not key[0].startswith("news_raw_")
                    or type(key[1]) is not int or key[1] < 0 or key in rejected):
                raise ValueError("Invalid or duplicate rejected-record location")
            rejected[key] = row
        if manifest["news_summary"]["rejected_article_rows"] != len(rejected):
            raise ValueError("Rejected-record count differs from manifest")
    verified_rejections = set()
    for filename in verified:
        if filename.startswith("news_raw_"):
            receipt = next(r["response_received_at"] for r in manifest["requests"] if r["file"] == filename)
            for row_index, article in enumerate(json.loads((directory / filename).read_text())["news"]):
                key = (filename, row_index)
                if key in rejected:
                    row = rejected[key]
                    start, end = timestamp(manifest["window"]["start"]), timestamp(manifest["window"]["end"])
                    created, updated = timestamp(article["created_at"]), timestamp(article["updated_at"])
                    if (row.get("reason") != "publication_outside_requested_window"
                            or row.get("id") != str(article["id"])
                            or row.get("created_at") != article["created_at"]
                            or row.get("updated_at") != article["updated_at"]
                            or row.get("response_received_at") != receipt
                            or updated < created or start <= created <= end
                            or row.get("updated_within_requested_window") is not (start <= updated <= end)):
                        raise ValueError("Rejected-record reason does not match the retained response")
                    verified_rejections.add(key)
                    continue
                raw_articles[str(article["id"])] = article
                receipts[str(article["id"])] = receipt
    if verified_rejections != set(rejected):
        raise ValueError("Rejected-record audit refers to a missing raw occurrence")
    if set(raw_articles) != {str(x["id"]) for x in articles}:
        raise ValueError("Normalized article IDs differ from archived responses")
    for article in articles:
        raw = raw_articles[str(article["id"])]
        for field in ["created_at", "updated_at", "url", "symbols"]:
            if article[field] != raw[field]:
                raise ValueError(f"Normalized {field} differs from raw article")
        if article["headline"] != re.sub(r"\s+", " ", html.unescape(raw["headline"])).strip():
            raise ValueError("Normalized headline differs from raw article")
        normalized_body = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw.get("content", "")))).strip()
        if article["text"] != normalized_body or article["retrieved_at"] != receipts[str(article["id"])]:
            raise ValueError("Normalized body or receipt time differs from raw response")
    provenance = {"input_files": {name: digest(directory / name) for name in files},
                  "verified_raw_responses": verified,
                  "ignored_answer_files": ["case_review.json", "outlook_20261009.json", "price_scan.json"],
                  "historical_availability_verified": False}
    if rejection_hash is not None:
        provenance["input_files"]["news_rejected_records.json"] = rejection_hash
        provenance["verified_rejected_records"] = len(verified_rejections)
    news_files = sorted(name for name in verified if name.startswith("news_raw_"))
    provenance["news_pagination_complete"] = bool(news_files) and not json.loads((directory / news_files[-1]).read_text()).get("next_page_token")
    if inputs["daily_bars_raw.json"].get("next_page_token"):
        raise ValueError("Daily bar pagination is incomplete")
    return articles, inputs["daily_bars_raw.json"], provenance


def focused_text(article: dict, symbol: str) -> str:
    """Reuse the extractor's literal company-clause boundaries for text tone."""
    return "\n".join(company_clauses(article, symbol)[:4])


def validate_evidence(extraction: dict, article: dict) -> None:
    for evidence in extraction["evidence"]:
        field, start, end = evidence["field"], evidence["start"], evidence["end"]
        if field not in {"headline", "text"} or type(start) is not int or type(end) is not int:
            raise ValueError("Invalid evidence coordinates")
        if not 0 <= start < end <= len(article[field]) or article[field][start:end] != evidence["text"]:
            raise ValueError("Evidence does not occur in its source field")


def analyze(articles: list[dict], raw_bars: dict, symbols: list[str], start: str, end: str,
            classifier=None, *, information_cutoff: str | None = None) -> dict:
    """Pure orchestration with injectable tone inference for offline tests."""
    cutoff = timestamp(information_cutoff) if information_cutoff else None
    content_cutoff = regular_close(end)
    if cutoff is not None and cutoff < content_cutoff:
        raise ValueError("Information cutoff precedes the requested price close; shorten the price window.")
    retained = [a for a in articles if max(timestamp(a["created_at"]), timestamp(a["updated_at"])) <= content_cutoff
                and (cutoff is None or max(timestamp(a["created_at"]), timestamp(a["updated_at"]),
                                           timestamp(a["retrieved_at"])) <= cutoff)]
    extracted, texts, indices = [], [], []
    by_id = {str(a["id"]): a for a in retained}
    for article in sorted(retained, key=lambda x: (timestamp(x["created_at"]), str(x["id"]))):
        for symbol in symbols:
            if symbol not in article["symbols"]:
                continue
            row = extract_article(article, symbol)
            validate_evidence(row, article)
            row["source_url"] = article["url"]
            row["headline_only"] = not bool(article.get("text", "").strip())
            row["sentiment"] = {"status": "Unavailable", "label": "unavailable",
                                "reason": "No verified company-bearing text or model inference disabled."}
            text = focused_text(article, symbol)
            if text and row["relevance"] in {"headline", "body"} and classifier:
                indices.append(len(extracted)); texts.append(text)
                row["sentiment_input"] = text
            extracted.append(row)
    if classifier and texts:
        predictions = classifier.predict(texts)
        if len(predictions) != len(indices):
            raise ValueError("Tone inference output count mismatch")
        for index, prediction in zip(indices, predictions):
            extracted[index]["sentiment"] = prediction
    # These modules see neither the saved hand-written review nor its outlook.
    clusters = cluster_events(extracted)
    prices = build_price_context(raw_bars, symbols, start, end,
                                 {s: BENCHMARKS.get(s, "SPY") for s in symbols})
    companies = []
    for symbol in symbols:
        records = [x for x in extracted if x["symbol"] == symbol]
        groups = [x for x in clusters if x["symbol"] == symbol]
        price = prices["companies"][symbol]
        tone_counts = Counter(x["sentiment"]["label"] for x in records)
        types = Counter(x["event_type"] for x in records)
        headline_only = sum(x["headline_only"] for x in records)
        findings = [f"{len(records)} tagged articles produced {len(groups)} candidate event groups; grouping uses conservative rules.",
                    f"{types['analyst_revision']} articles contain analyst-revision candidates; these are opinions, not company revenue."]
        if price.get("return_pct") is not None:
            findings.append(f"Observed price return from {start} close to {end} close: {price['return_pct']:+.2f}%.")
        if price.get("excess_vs_spy_pp") is not None and price.get("excess_vs_sector_pp") is not None:
            findings.append(f"Observed return difference: {price['excess_vs_spy_pp']:+.2f} percentage points versus SPY and {price['excess_vs_sector_pp']:+.2f} versus {BENCHMARKS.get(symbol, 'SPY')}; no causal attribution.")
        for session in price.get("daily_sessions", []):
            if session.get("return_pct") is not None and session["return_pct"] > 0 and session.get("gap_pct") is not None and session["gap_pct"] > 0 and session.get("intraday_pct") is not None and session["intraday_pct"] <= 0:
                findings.append(f"On {session['date']}, the stock gapped {session['gap_pct']:+.2f}% before the regular open, then returned {session['intraday_pct']:+.2f}% from open to close. The daily gain was not an after-open gain.")
        events = []
        for group in groups:
            members = group["extractions"]
            representative = max(members, key=lambda x: (bool(x["evidence"]), not x["headline_only"], x.get("updated_at") or ""))
            available = [x["sentiment"]["label"] for x in members if x["sentiment"]["label"] != "unavailable"]
            tone = available[0] if available and len(set(available)) == 1 else "mixed" if available else "unavailable"
            urls = sorted({by_id[str(x["article_id"])]["url"] for x in members})
            urls = [u for u in urls if urlparse(u).scheme == "https"]
            events.append({"id": group["event_key"], "event_type": representative["event_type"],
                           "claim_kind": representative["claim_kind"], "summary": representative["summary"],
                           "article_count": len(members), "earliest_reported_at": group["earliest_reported_at"],
                           "latest_updated_at": group["latest_updated_at"], "source_urls": urls,
                           "tone": tone, "headline_only": all(x["headline_only"] for x in members)})
        limitations = ["Text tone is not an expected return or a calibrated probability of a stock rising.",
                      "Historical text was collected after the price window; this is a retrospective review.",
                      "Event grouping and fact/opinion labels use rules and require human review.",
                      "Only the retained news feed is searched. Company IR pages and earlier reports elsewhere are not retrieved automatically.",
                      "The earliest report in a candidate group is only the earliest retained article; later details must not be backdated to that time.",
                      "The report combines rule templates with price arithmetic, not a generative analyst or a causal explanation.",
                      "Vendor ticker tags and company-bearing sentences do not prove direct economic relevance.",
                      "No current Futu crowding measurement or trained ten-day return forecast is available."]
        if headline_only:
            limitations.append(f"{headline_only} articles have no body; their claims are headline-only.")
        companies.append({"symbol": symbol, "article_count": len(records), "headline_only_count": headline_only,
                          "cluster_count": len(groups), "price_context": price,
                          "sentiment": {label: tone_counts[label] for label in ["positive", "negative", "neutral", "unavailable"]},
                          "findings": findings, "limitations": limitations, "events": events})
    return {"schema_version": SCHEMA_VERSION, "analysis_mode": "Retrospective evidence review",
            "extraction_method": "Rules v1 + FinBERT tone" if classifier else "Rules v1; tone inference disabled",
            "model_status": "Pretrained inference only" if classifier else "Not run",
            "prediction_status": "Not trained", "ab_status": "Not evaluated",
            "coverage": {"articles_loaded": len(articles), "articles_retained": len(retained),
                         "article_company_pairs": len(extracted), "classified_pairs": len(indices), "clusters": len(clusters)},
            "companies": companies, "extractions": extracted, "market_context": prices,
            "tone_model": classifier.metadata if classifier else None,
            "information_cutoff": information_cutoff,
            "publication_version_cutoff": content_cutoff.isoformat(),
            "tone_input_policy": "Up to four company-bearing clauses, sharing extractor boundaries; max 512 tokens. English-only baseline; unresolved multi-company clauses need review.",
            "earliest_actual_receipt": min((a["retrieved_at"] for a in retained), default=None),
            "latest_actual_receipt": max((a["retrieved_at"] for a in retained), default=None)}


def public_summary(report: dict) -> dict:
    """Explicit aggregate/summary allowlist; private sentences never exported."""
    result = {key: report[key] for key in ["schema_version", "generated_at", "price_asof", "analysis_mode",
              "extraction_method", "model_status", "prediction_status", "ab_status"]}
    result["coverage"] = {key: report["coverage"][key] for key in ["articles_loaded", "article_company_pairs", "classified_pairs", "clusters"]}
    result["companies"] = []
    for company in report["companies"]:
        row = {key: company[key] for key in ["symbol", "article_count", "headline_only_count", "cluster_count", "sentiment", "findings", "limitations"]}
        row["price_context"] = {key: company["price_context"].get(key) for key in ["status", "return_pct", "excess_vs_spy_pp", "excess_vs_sector_pp"]}
        row["price_context"]["sector_benchmark"] = BENCHMARKS.get(company["symbol"], "SPY")
        row["events"] = [{key: event[key] for key in ["id", "event_type", "claim_kind", "summary", "article_count", "earliest_reported_at", "source_urls", "tone", "headline_only"]} for event in company["events"]]
        result["companies"].append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+", default=["PLTR", "AMZN", "MSFT"])
    parser.add_argument("--start-date", default="2026-10-02")
    parser.add_argument("--end-date", default="2026-10-09")
    parser.add_argument("--information-cutoff", help="Aware timestamp; filters by actual receipt AND version times.")
    parser.add_argument("--offline", action="store_true", help="Require model files already present in the local HF cache.")
    parser.add_argument("--rules-only", action="store_true", help="Skip pretrained tone inference; never invent model results.")
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Output directory already exists; choose a new run directory")
    if len(set(args.symbols)) != len(args.symbols) or any(s not in ALIASES for s in args.symbols):
        raise ValueError("This prototype supports distinct PLTR, AMZN and MSFT symbols")
    started = datetime.now(timezone.utc).isoformat()
    articles, bars, provenance = read_inputs(args.input_dir)
    code_files = ["run_research.py", "sentiment_model.py", "event_extraction.py", "price_context.py"]
    provenance["code_files"] = {name: digest(Path(__file__).parent / name) for name in code_files}
    classifier = None
    if not args.rules_only:
        from sentiment_model import FinBertTone
        print("Loading pinned local FinBERT tone model...", flush=True)
        classifier = FinBertTone(offline=args.offline)
    print(f"Analyzing {len(articles)} saved articles for {', '.join(args.symbols)}...", flush=True)
    report = analyze(articles, bars, args.symbols, args.start_date, args.end_date,
                     classifier, information_cutoff=args.information_cutoff)
    report.update({"generated_at": datetime.now(timezone.utc).isoformat(), "run_started_at": started,
                   "price_asof": regular_close(args.end_date).isoformat(),
                   "provenance": provenance})
    if provenance["code_files"] != {name: digest(Path(__file__).parent / name) for name in code_files}:
        raise ValueError("Research code changed during inference; repeat with a stable version")
    if not provenance["news_pagination_complete"]:
        for company in report["companies"]:
            company["limitations"].append("News pagination was truncated; this report covers only the retained pages.")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / ".gitignore").write_text("*\n!.gitignore\n")
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    summary = public_summary(report)
    (args.output_dir / "public_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.public_output:
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.public_output.with_suffix(args.public_output.suffix + ".tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(args.public_output)
    print(json.dumps({"output_dir": str(args.output_dir), "coverage": report["coverage"],
                      "companies": [{"symbol": c["symbol"], "sentiment": c["sentiment"], "groups": c["cluster_count"]} for c in report["companies"]]}, indent=2))


if __name__ == "__main__":
    main()
