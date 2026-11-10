"""Bounded public FNSPID headlines and existing Alpaca split-adjusted bars.

This is a retrospective development archive, never a point-in-time dataset.
Only selected original headlines are used; provided summaries/sentiment are not
model inputs. Capture timestamps are actual, not historical publication times.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import re

from collect_research import _local_credentials, _symbols
from calendar_audit import export_calendar

REPOSITORY = "Zihan1004/FNSPID"
REVISION = "bf9189c41527198897d1af3e17b1a0095279fc45"
NEWS_FILE = "Stock_news/All_external.csv"
PRICE_URL = "https://data.alpaca.markets/v2/stocks/bars"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def parse_prefix(data):
    """Drop an incomplete trailing CSV record; preserve source row positions."""
    reader = csv.reader(io.StringIO(data.decode("utf-8", errors="strict")), strict=True)
    header = next(reader)
    required = {"Date", "Article_title", "Stock_symbol", "Url"}
    if not required.issubset(header):
        raise ValueError("Unexpected FNSPID schema")
    records, incomplete, parser_error, last_appended = [], False, False, False
    try:
        for index, row in enumerate(reader):
            last_appended = False
            if len(row) != len(header):
                incomplete = True
                continue
            # A final record without a line terminator may have been cut in
            # an unquoted last field. It is conservatively discarded below.
            records.append({**dict(zip(header, row)), "source_row_index": index})
            last_appended = True
    except csv.Error:
        incomplete, parser_error = True, True
    if data and not data.endswith((b"\n", b"\r")) and records and last_appended and not parser_error:
        records.pop()
        incomplete = True
    return records, incomplete


def capture(output_dir, protocol, *, prefix_bytes=2 * 1024 * 1024,
            session=None, credential_loader=None, max_price_pages=5):
    """Capture a fixed byte budget and finite pages; failure remains auditable."""
    import requests
    symbols = _symbols([protocol["symbol"]])
    if symbols == ["SPY"]:
        raise ValueError("Select a company, not the benchmark")
    if type(prefix_bytes) is not int or not 1024 <= prefix_bytes <= 32 * 1024 * 1024:
        raise ValueError("prefix_bytes must be between 1 KiB and 32 MiB")
    if type(max_price_pages) is not int or not 1 <= max_price_pages <= 20:
        raise ValueError("Require a finite price page budget from 1 to 20")
    revision = protocol.get("revision", REVISION)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pin an immutable dataset revision")
    first = date.fromisoformat(protocol["source_date_start"])
    last = date.fromisoformat(protocol["source_date_end_exclusive"])
    if not first < last or last >= datetime.now(timezone.utc).date():
        raise ValueError("Require a finished historical source-date window")
    output = Path(output_dir)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / ".gitignore").write_text("*\n")
    write_json(output / "protocol.json", protocol)  # before any outcomes
    manifest = {"schema_version": "historical-backfill-1.0", "status": "running",
                "started_at": stamp(), "data_kind": "observed_retrospective",
                "historical_availability_verified": False, "requests": [],
                "protocol_sha256": digest((output / "protocol.json").read_bytes()),
                "generation_sources_sha256": {Path(__file__).name: digest(Path(__file__).read_bytes())},
                "source": REPOSITORY, "revision": revision,
                "coverage_scope": "Bounded published CSV prefix; not complete company or market coverage",
                "return_basis": "split_adjusted_open_price_excluding_cash_dividends"}
    owned = session is None
    session = session or requests.Session()
    try:
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{NEWS_FILE}"
        record = {"url": url, "started_at": stamp(), "file": "news_prefix.csv",
                  "requested_range": f"bytes=0-{prefix_bytes - 1}"}
        manifest["requests"].append(record)
        # Credentials are supplied per request ONLY to the fixed Alpaca host;
        # never put them on this shared session or any public-data request.
        with session.get(url, headers={"Range": record["requested_range"]},
                         stream=True, timeout=30) as response:
            record.update(status=response.status_code,
                          content_range=response.headers.get("Content-Range"))
            if response.status_code not in (200, 206):
                raise ValueError("Public dataset request failed")
            parts, retained = [], 0
            for part in response.iter_content(chunk_size=65536):
                remaining = prefix_bytes - retained
                if not remaining:
                    break
                part = part[:remaining]
                parts.append(part)
                retained += len(part)
                if retained == prefix_bytes:
                    break
            raw = b"".join(parts)
        (output / record["file"]).write_bytes(raw)
        record.update(response_received_at=stamp(), retained_bytes=len(raw), sha256=digest(raw),
                      entire_dataset_downloaded=False)
        rows, incomplete = parse_prefix(raw)
        selected = []
        for row in rows:
            if row["Stock_symbol"] not in symbols:
                continue
            day = row["Date"][:10]
            date.fromisoformat(day)
            if not first.isoformat() <= day < last.isoformat() or not row["Article_title"].strip():
                continue
            selected.append({"headline": row["Article_title"], "source_date": day,
                             "reported_timestamp": row["Date"], "symbol": row["Stock_symbol"],
                             "url": row["Url"], "publisher": row.get("Publisher"),
                             "source_row_index": row["source_row_index"],
                             "response_received_at": record["response_received_at"],
                             "historical_availability_verified": False})
        write_json(output / "news.json", selected)
        manifest.update(prefix_rows=len(rows), partial_trailing_record_discarded=incomplete,
                        selected_headlines=len(selected), retained_bodies=0)
        if not selected:
            raise ValueError("Selected company/date window is absent from the bounded prefix; increase budget or choose a justified source")
        key, secret = (credential_loader or _local_credentials)()
        price_first, price_last = first - timedelta(days=61), last + timedelta(days=45)
        params = {"symbols": ",".join([*symbols, "SPY"]),
                  "start": price_first.isoformat() + "T00:00:00Z",
                  "end": price_last.isoformat() + "T00:00:00Z", "timeframe": "1Day",
                  "adjustment": "split", "feed": "sip", "limit": 10000, "sort": "asc"}
        seen_tokens, price_counts = set(), {}
        for page in range(max_price_pages):
            record = {"url": PRICE_URL, "params": dict(params), "started_at": stamp(),
                      "file": f"prices_{page:03}.json"}
            manifest["requests"].append(record)
            response = session.get(PRICE_URL, params=params,
                                   headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
                                   timeout=30, allow_redirects=False)
            raw = response.content
            record.update(status=response.status_code, response_received_at=stamp(),
                          bytes=len(raw), sha256=digest(raw))
            (output / record["file"]).write_bytes(raw)
            if response.status_code != 200:
                raise ValueError("Existing Alpaca access rejected price backfill; no feed fallback attempted")
            payload = json.loads(raw)
            if not isinstance(payload.get("bars"), dict):
                raise ValueError("Invalid price payload")
            if any(not isinstance(v, list) for v in payload["bars"].values()):
                raise ValueError("Invalid bar collection")
            for ticker, bars in payload["bars"].items():
                price_counts[ticker] = price_counts.get(ticker, 0) + len(bars)
            token = payload.get("next_page_token")
            record["next_page_token"] = token
            if token is not None and not isinstance(token, str):
                raise ValueError("Invalid price pagination token")
            if not token:
                break
            if token in seen_tokens:
                raise ValueError("Repeated price pagination token")
            seen_tokens.add(token)
            params["page_token"] = token
        else:
            raise ValueError("Price page budget exhausted; this archive is incomplete")
        manifest["bars_summary"] = {"pagination_complete": True, "records": price_counts,
                                    "missing_symbols": [s for s in [*symbols, "SPY"] if not price_counts.get(s)]}
        if manifest["bars_summary"]["missing_symbols"]:
            raise ValueError("No observed bars for a required symbol")
        write_json(output / "calendar.json", export_calendar(price_first.isoformat(), price_last.isoformat()))
        manifest["status"] = "complete_bounded_capture"
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        if isinstance(exc, ValueError):
            manifest["error"] = str(exc)
        raise
    finally:
        manifest["finished_at"] = stamp()
        manifest["files_sha256"] = {p.name: digest(p.read_bytes()) for p in output.iterdir()
                                    if p.is_file() and p.name not in {"manifest.json", ".gitignore"}}
        write_json(output / "manifest.json", manifest)
        if owned:
            session.close()
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True, help="Frozen JSON protocol, registered before reading outcomes")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--news-prefix-bytes", type=int, default=2097152)
    p.add_argument("--max-price-pages", type=int, default=5)
    a = p.parse_args(argv)
    m = capture(a.output_dir, json.loads(a.protocol.read_text()), prefix_bytes=a.news_prefix_bytes,
                max_price_pages=a.max_price_pages)
    print(json.dumps({k: m[k] for k in ["status", "prefix_rows", "selected_headlines", "retained_bodies"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
