"""Capture and audit two official documents for one declared earnings event.

Only the MSFT FY2026 Q1 issuer pages are requested. Captures are current
observations, not historical point-in-time proof. No credentials or trading
runtime are loaded; source eligibility uses the existing preflight contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
from http.client import HTTPException
import json
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from data_preflight import audit_source_records
from run_research import timestamp


EVENT = {"issuer": "MSFT", "fiscal_period": "FY2026 Q1", "event_calendar_date": "2025-10-29"}
SOURCES = {
    "financial_report": "https://www.microsoft.com/en-us/investor/earnings/fy-2026-q1/press-release-webcast",
    "earnings_call": "https://www.microsoft.com/en-us/investor/events/fy-2026/earnings-fy-2026-q1",
}
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
HEADER_ALLOWLIST = ("Content-Type", "Date", "Last-Modified", "ETag")


def _now():
    return datetime.now(timezone.utc)


def _stamp(value):
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Capture clock must include a timezone")
    return value.astimezone(timezone.utc).isoformat()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _fetch(url):
    request = Request(url, headers={"User-Agent": "Quant-course-source-audit/1.0", "Accept": "text/html"})
    try:
        return build_opener(_NoRedirect()).open(request, timeout=30)
    except HTTPError as response:
        return response


def capture_event(output_dir: Path, *, fetch=None, clock=_now) -> dict:
    """Retain at most two fixed public GET responses in a new private directory.

    response_received_at is recorded after reading the retained body. HTTP Date
    and Last-Modified are retrieval metadata, never original publication proof.
    Failed/truncated responses remain visible and are not retried or replaced.
    """
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a new capture directory")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    (output / ".gitignore").write_text("*\n", encoding="utf-8")
    (output / "raw").mkdir(mode=0o700)
    manifest = {
        "schema_version": "issuer-event-capture-1.0", "event": dict(EVENT),
        "created_at": _stamp(clock()), "status": "running", "requests": [],
        "historical_pipeline_receipt_verified": False,
        "capture_source_sha256": _digest(Path(__file__).read_bytes()),
    }

    def save():
        (output / "manifest.json").write_text(_json(manifest), encoding="utf-8")

    save()
    for family, url in SOURCES.items():
        record = {"source_family": family, "url": url, "method": "GET",
                  "request_started_at": _stamp(clock()), "response_received_at": None,
                  "http_status": None, "raw_file": None, "sha256": None,
                  "bytes": 0, "truncated": False, "headers": {}}
        manifest["requests"].append(record)
        save()
        try:
            response = (fetch or _fetch)(url)
            try:
                data = response.read(MAX_RESPONSE_BYTES + 1)
                record["response_received_at"] = _stamp(clock())
                record["http_status"] = response.getcode()
                record["headers"] = {name: response.headers.get(name) for name in HEADER_ALLOWLIST
                                     if response.headers.get(name) is not None}
            finally:
                response.close()
            record["truncated"] = len(data) > MAX_RESPONSE_BYTES
            data = data[:MAX_RESPONSE_BYTES]
            name = "raw/" + family + ".html"
            with (output / name).open("xb") as stream:
                stream.write(data)
            record.update(raw_file=name, sha256=_digest(data), bytes=len(data))
        except (OSError, ValueError, HTTPException) as exc:
            record["error_type"] = type(exc).__name__
        save()
    manifest["status"] = "complete" if all(
        row["http_status"] == 200 and not row["truncated"] and row["raw_file"]
        for row in manifest["requests"]) else "partial"
    manifest["completed_at"] = _stamp(clock())
    save()
    return manifest


class _BodyText(HTMLParser):
    """Retain issuer text blocks, excluding executable and page-chrome content."""
    void = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
            "param", "source", "track", "wbr"}
    blocks = {"p", "h1", "h2", "h3", "h4", "li", "tr", "br", "div"}

    def __init__(self, target_id=None):
        super().__init__(convert_charrefs=True)
        self.target_id, self.stack, self.parts, self.targets = target_id, [], [], 0

    def _enabled(self):
        return (not any(tag in {"script", "style", "noscript", "template", "head"}
                        for tag, _ in self.stack)
                and (self.target_id is None or any(identity == self.target_id for _, identity in self.stack)))

    def handle_starttag(self, tag, attrs):
        identity = dict(attrs).get("id")
        if identity == self.target_id and self.target_id is not None:
            self.targets += 1
        if tag not in self.void:
            self.stack.append((tag, identity))
        if self._enabled() and tag in self.blocks:
            self.parts.append("\n")
        elif self._enabled() and tag in {"td", "th"}:
            self.parts.append("\t")

    def handle_endtag(self, tag):
        if self._enabled() and tag in self.blocks:
            self.parts.append("\n")
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                self.stack = self.stack[:index]
                break

    def handle_data(self, data):
        if self._enabled():
            self.parts.append(data)

    def lines(self):
        return [value for line in "".join(self.parts).splitlines() if (value := " ".join(line.split()))]


def extract_official_body(encoded: bytes, family: str) -> tuple[str, dict]:
    """Extract one declared document, retaining date precision and text offsets.

    The release dateline supports a publication DATE. The transcript heading's
    date supports only the EVENT date, not when that transcript was posted.
    No HTTP header, webcast start time or fiscal label supplies publication time.
    """
    if family not in SOURCES:
        raise ValueError("Unknown fixed issuer source family")
    parser = _BodyText("pressreleasecontent" if family == "financial_report" else None)
    parser.feed(encoded.decode("utf-8-sig"))
    lines = parser.lines()
    title = ("Microsoft Cloud and AI Strength Drives First Quarter Results" if family == "financial_report"
             else "Microsoft FY26 First Quarter Earnings Conference Call")
    starts = [(index, width) for index in range(len(lines)) for width in (1, 2, 3)
              if " ".join(lines[index:index + width]) == title]
    if len(starts) != 1 or (family == "financial_report" and parser.targets != 1):
        raise ValueError("Expected one issuer document title/body boundary")
    start, title_lines = starts[0]
    marker = "IMPORTANT NOTICE TO USERS" if family == "financial_report" else "END"
    ends = [index for index in range(start + 1, len(lines))
            if (marker in lines[index] if family == "financial_report" else lines[index] == marker)]
    if not ends:
        raise ValueError("Issuer document end boundary missing; no page-chrome fallback")
    text = "\n".join(lines[start:ends[0] + (family == "earnings_call")]) + "\n"
    if family == "financial_report":
        date_index = start + title_lines
        expression = r"^REDMOND,\s*Wash\.\s*[—-]\s*(?P<date>[A-Za-z]+\s+\d{1,2},\s+\d{4})(?:\s*[—-]|\s)"
        match = re.search(expression, lines[date_index]) if date_index < len(lines) else None
    else:
        expression = (r"^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+"
                      r"(?P<date>[A-Za-z]+\s+\d{1,2},\s+\d{4})$")
        header_dates = [(index, re.search(expression, lines[index]))
                        for index in range(start + title_lines, min(start + title_lines + 3, len(lines)))]
        header_dates = [(index, value) for index, value in header_dates if value is not None]
        date_index, match = header_dates[0] if len(header_dates) == 1 else (None, None)
    if match is None or datetime.strptime(match.group("date"), "%B %d, %Y").date().isoformat() != EVENT["event_calendar_date"]:
        raise ValueError("Explicit issuer dateline/event-header date does not match the declared 2025 event")
    offset = sum(len(line) + 1 for line in lines[start:date_index]) + match.start("date")
    if family == "earnings_call" and ("JONATHAN NEILSON:" not in text or "Q&A" not in text):
        raise ValueError("Transcript body must include speaker remarks and Q&A")
    if family == "financial_report" and ("September 30, 2025" not in text or "Operating income" not in text):
        raise ValueError("Release body must include the reported period and financial tables")
    evidence = {
        "event_calendar_date": EVENT["event_calendar_date"], "fiscal_period": EVENT["fiscal_period"],
        "original_publication_date": EVENT["event_calendar_date"] if family == "financial_report" else None,
        "date_role": "release_dateline" if family == "financial_report" else "call_event_date_only",
        "original_publication_precision": "date" if family == "financial_report" else "unknown",
        "original_published_at": None, "version_published_at": None,
        "historical_pipeline_response_received_at": None,
        "date_evidence": {"text": match.group("date"), "start": offset,
                          "end": offset + len(match.group("date"))},
        "body_boundaries": {"start_marker": title, "end_marker": marker},
        "webcast_start_and_http_headers_are_not_publication_times": True,
        "table_policy": "Original HTML retains table structure; plain text is not parsed financial values",
    }
    return text, evidence


def audit_capture(input_dir: Path, output_dir: Path, *, cutoff: str, clock=_now) -> dict:
    """Verify saved bytes, reproduce bodies, then reuse the existing source audit.

    This is a single historical-event source check. No calendar, prices, labels,
    training, text scoring, same-sample model comparison or profit is generated.
    """
    source, output = Path(input_dir).resolve(), Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a new audit directory")
    if output.resolve().is_relative_to(source):
        raise ValueError("Audit output must be outside the capture archive")
    as_of = _stamp(clock())
    if timestamp(cutoff) > timestamp(as_of):
        raise ValueError("Historical reference cutoff must not follow audit as_of")
    sources = [Path(__file__), *[Path(__file__).with_name(name) for name in (
        "data_preflight.py", "run_research.py", "event_extraction.py", "price_context.py")]]
    generation_sources = {str(path.resolve()): _digest(path.read_bytes()) for path in sources}
    manifest_path = source / "manifest.json"
    encoded_manifest = manifest_path.read_bytes()
    manifest = json.loads(encoded_manifest)
    if manifest.get("schema_version") != "issuer-event-capture-1.0" or manifest.get("event") != EVENT:
        raise ValueError("Capture event differs from the explicitly declared issuer event")
    requests = manifest.get("requests")
    if (not isinstance(requests, list) or len(requests) != len(SOURCES)
            or {row.get("source_family") for row in requests} != set(SOURCES)):
        raise ValueError("Capture must retain exactly the two fixed document attempts")
    inputs = {str(manifest_path): _digest(encoded_manifest)}
    prepared, failures = {}, {}
    for row in requests:
        family = row["source_family"]
        if row.get("url") != SOURCES[family] or row.get("method") != "GET":
            raise ValueError("Capture URL/method differs from the fixed public source")
        name = row.get("raw_file")
        if name is None:
            failures[family] = "No response body retained"
            continue
        if name != "raw/" + family + ".html":
            raise ValueError("Unexpected raw-response path")
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError("Raw response must stay inside its capture archive")
        encoded = path.read_bytes()
        if len(encoded) != row["bytes"] or _digest(encoded) != row["sha256"]:
            raise ValueError("Captured raw response hash/length mismatch")
        inputs[str(path)] = _digest(encoded)
        if row.get("http_status") != 200 or row.get("truncated") is not False:
            failures[family] = "HTTP failure or truncated response; not retained original text"
            continue
        try:
            text, publication = extract_official_body(encoded, family)
            prepared[family] = (row, text, publication)
        except (ValueError, UnicodeError) as exc:
            failures[family] = str(exc)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    (output / ".gitignore").write_text("*\n", encoding="utf-8")
    (output / "bodies").mkdir(mode=0o700)
    publication_evidence, mapping, records = {}, {}, []
    event_id = "MSFT:earnings:FY2026Q1:2025-10-29"
    for family, (row, text, publication) in prepared.items():
        body = text.encode("utf-8")
        text_path = output / "bodies" / (family + ".txt")
        with text_path.open("xb") as stream:
            stream.write(body)
        if _digest(text_path.read_bytes()) != _digest(body):
            raise ValueError("Saved body differs from extracted text version")
        publication_evidence[family] = {**publication, "source_url": row["url"],
                                       "raw_response_sha256": row["sha256"], "text_version_sha256": _digest(body)}
        mapping[family] = {"source_url": row["url"], "event_id": event_id,
                           "event_calendar_date": publication["event_calendar_date"],
                           "fiscal_period": publication["fiscal_period"], "text_version_sha256": _digest(body)}
        records.append({
            "record_id": "issuer-capture:" + family, "source_id": row["url"], "source_family": family,
            "document_kind": "earnings_release" if family == "financial_report" else "earnings_transcript",
            "event_ids": [event_id], "event_identity_evidence_id": str(output / "event_mapping.json"),
            "event_calendar_date": publication["event_calendar_date"], "fiscal_period": publication["fiscal_period"],
            "original_published_at": None, "version_published_at": None, "republished_at": None,
            "original_publication_evidence_id": str(output / "publication_evidence.json") if family == "financial_report" else None,
            "text_version_sha256": _digest(body), "text_available": True,
            "captures": [{"request_started_at": row["request_started_at"],
                          "response_received_at": row["response_received_at"], "status": row["http_status"],
                          "retrieval_evidence_id": str(source / row["raw_file"]),
                          "request_evidence_id": str(manifest_path), "response_sha256": row["sha256"],
                          "text_version_sha256": _digest(body), "archive_integrity_verified": True}],
        })
    documents = {
        "publication_evidence.json": publication_evidence,
        "event_mapping.json": {"event": dict(EVENT), "documents": mapping,
                               "method": "Issuer headings and explicit calendar-date evidence; conservative shared group for split isolation",
                               "independent_human_gold": False,
                               "limitation": "A shared earnings group is not a claim that every discussed business fact originated at this event"},
        "source_records.json": records,
    }
    for name, value in documents.items():
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(_json(value))
    audit = audit_source_records(records, cutoff=cutoff, as_of=as_of)
    report = {
        "schema_version": "issuer-event-audit-1.0", "event": dict(EVENT), "audit_as_of": as_of,
        "capture_attempts": len(requests), "verified_body_documents": len(prepared),
        "body_characters": {family: len(text) for family, (_, text, _) in prepared.items()},
        "body_failures": failures, "source_eligibility": audit,
        "original_historical_receipts_verified": False,
        "historical_text_version_verified": False,
        "news_same_event_records": None, "investor_comment_same_event_records": None,
        "coverage_note": "News/comments were not requested here; the 2026 local corpus is not paired to this 2025 event",
        "ten_session_target": "Stock opening total return minus SPY, explicit exchange sessions; no labels constructed here",
        "formal_labels_created": 0, "real_model_evaluation": "Not evaluated", "profit_alpha_established": False,
        "pending": ["Precise original body publication and historical version availability",
                    "Same-event permitted news/comment corpus and independent event review",
                    "Independent session calendar, total-return/corporate-action evidence and prepared adapter",
                    "Time/event-isolated ten-session labels and same-sample price/tone/text controls"],
    }
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        stream.write(_json(report))
    if any(_digest(Path(path).read_bytes()) != expected for path, expected in inputs.items()):
        raise ValueError("Capture inputs changed during audit")
    if any(_digest(Path(path).read_bytes()) != expected for path, expected in generation_sources.items()):
        raise ValueError("Generation source changed during audit")
    run_manifest = {"schema_version": "issuer-event-audit-manifest-1.0", "input_files_sha256": inputs,
                    "generation_sources_sha256": generation_sources,
                    "outputs_sha256": {str(path.relative_to(output)): _digest(path.read_bytes())
                                       for path in output.rglob("*") if path.is_file()}}
    with (output / "run_manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(_json(run_manifest))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture", action="store_true")
    source.add_argument("--input-dir", type=Path, help="Offline audit of a saved two-document capture")
    parser.add_argument("--cutoff", help="Aware historical reference cutoff; required for offline audit")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.capture:
        result = capture_event(args.output_dir)
        print(json.dumps({"status": result["status"], "responses": [
            {k: row[k] for k in ("source_family", "http_status", "bytes", "truncated", "response_received_at")}
            for row in result["requests"]]}))
        return 0 if result["status"] == "complete" else 2
    if not args.cutoff:
        parser.error("--cutoff is required for the offline audit")
    result = audit_capture(args.input_dir, args.output_dir, cutoff=args.cutoff)
    print(json.dumps({"verified_body_documents": result["verified_body_documents"],
                      "eligible_sources_at_cutoff": result["source_eligibility"]["eligible_source_records"],
                      "formal_labels_created": 0, "model_evaluation": "Not evaluated"}))
    return 0 if not result["body_failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
