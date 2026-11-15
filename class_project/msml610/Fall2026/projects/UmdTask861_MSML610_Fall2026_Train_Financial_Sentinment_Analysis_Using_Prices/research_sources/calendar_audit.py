"""Export an independent rule calendar and audit raw opening-price coverage.

This offline diagnostic does not create labels, reconstruct historical receipt,
adjust corporate actions, fit models, or estimate trading returns.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
REFERENCES = [
    "https://www.nyse.com/markets/hours-calendars",
    "https://www.nasdaq.com/market-activity/stock-market-holiday-schedule",
    "https://github.com/gerrymanoim/exchange_calendars",
]


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("Expected an ISO date")
    return date.fromisoformat(value)


def _time(value):
    if not isinstance(value, str):
        raise ValueError("Expected an aware timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Expected an aware timestamp")
    return result.astimezone(timezone.utc)


def _iso(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def export_calendar(start, end):
    """XNYS rule sessions, independent of observed bar dates; no network calls.

    ``complete`` describes this package's inclusive rule range only. The
    export is not exchange certification of extraordinary closures, historic
    announcements, or per-symbol opening auctions. Retain the rule provenance.
    """
    if _day(start) > _day(end):
        raise ValueError("Calendar start must not follow end")
    import exchange_calendars as xc

    calendar = xc.get_calendar("XNYS", start=start, end=end)
    sessions = [{"date": day.date().isoformat(), "open_at": _iso(row["open"]),
                 "close_at": _iso(row["close"])}
                for day, row in calendar.schedule.loc[start:end].iterrows()]
    if not sessions:
        raise ValueError("Calendar range contains no sessions")
    root = Path(xc.__file__).resolve().parent
    source_files = {str(p.relative_to(root)): _hash(p) for p in sorted(root.rglob("*.py"))}
    version = importlib.metadata.version("exchange_calendars")
    return {"calendar_id": "XNYS", "calendar_version": "exchange_calendars:" + version,
            "complete": True, "sessions": sessions,
            "provenance": {"package": "exchange_calendars", "package_version": version,
                           "requested_start": start, "requested_end": end,
                           "package_source_sha256": source_files,
                           "rules_sha256": hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest(),
                           "session_sha256": hashlib.sha256(json.dumps(sessions, sort_keys=True).encode()).hexdigest(),
                           "independent_of_price_dates": True, "references": REFERENCES,
                           "dependencies": {p: importlib.metadata.version(p) for p in ("pandas", "numpy", "tzdata")},
                           "completeness_scope": "Installed XNYS rules in the requested inclusive range",
                           "limitations": [
                               "Not an exchange-certified historical schedule or proof of point-in-time rule availability.",
                               "Extraordinary closures and cross-venue differences require separate source review.",
                               "XNYS is a common US-equity session reference, not proof of each symbol's auction or execution.",
                               "No price, text availability or corporate-action evidence is supplied by this calendar.",
                           ]}}


def audit_archive(archive, *, month, as_of, symbols, calendar):
    """Count paired company/SPY raw open endpoints on every rule decision day.

    A pair is counted only when all four endpoint opens are finite and positive
    and the retained response was received by ``as_of``. This is an endpoint
    coverage denominator, never a denominator of eligible text observations.
    """
    if not isinstance(month, str) or not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError("month must be YYYY-MM")
    first = _day(month + "-01")
    following = date(first.year + (first.month == 12), first.month % 12 + 1, 1)
    clock = _time(as_of)
    if clock.astimezone(NY).date() < following:
        raise ValueError("Audit month is not complete as_of")
    if (not isinstance(symbols, list) or not symbols or len(set(symbols)) != len(symbols)
            or any(not isinstance(s, str) or not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", s)
                   or s == "SPY" for s in symbols)):
        raise ValueError("Require distinct company symbols, excluding SPY")
    # Bind sessions to their export rather than accepting a shortened or edited
    # supplied calendar that could silently move all horizon endpoints.
    provenance = calendar["provenance"]
    if (calendar.get("calendar_id") != "XNYS" or calendar.get("complete") is not True
            or _day(provenance["requested_start"]) > first
            or _day(provenance["requested_end"]) < following
            or hashlib.sha256(json.dumps(calendar["sessions"], sort_keys=True).encode()).hexdigest()
            != provenance["session_sha256"]):
        raise ValueError("Calendar export does not cover/bind the audit month")
    sessions = calendar["sessions"]
    dates = [s["date"] for s in sessions]
    if dates != sorted(set(dates)):
        raise ValueError("Calendar sessions must be ordered and unique")
    decisions = [(i, s) for i, s in enumerate(sessions) if s["date"].startswith(month)]
    if not decisions or decisions[-1][0] + 20 >= len(sessions):
        raise ValueError("Calendar must include twenty exit sessions beyond the last decision")
    archive = Path(archive).resolve()
    manifest_path, bars_path = archive / "manifest.json", archive / "daily_bars_raw.json"
    fingerprints = {str(p): _hash(p) for p in (manifest_path, bars_path)}
    manifest = json.loads(manifest_path.read_text())
    requests = [r for r in manifest["requests"] if r.get("file") == bars_path.name]
    if (len(requests) != 1 or requests[0].get("status") != 200
            or requests[0].get("route") != "/v2/stocks/bars"
            or requests[0].get("params", {}).get("timeframe") != "1Day"
            or requests[0]["params"].get("adjustment") != "raw"
            or requests[0].get("sha256") != fingerprints[str(bars_path)]):
        raise ValueError("Require one hash-bound successful raw daily-bar response")
    request = requests[0]
    receipt = _time(request["response_received_at"])
    if receipt > clock or _time(request["started_at"]) > receipt:
        raise ValueError("Invalid response receipt relative to request or as_of")
    raw = json.loads(bars_path.read_text())
    if raw.get("next_page_token"):
        raise ValueError("Daily-bar pagination incomplete")
    prices, invalid_dates = {}, []
    for symbol in [*symbols, "SPY"]:
        prices[symbol] = {}
        for bar in raw["bars"].get(symbol, []):
            # Alpaca daily timestamps are date markers, not opening/receipt
            # timestamps. Convert that marker to its exchange-local date.
            day = _time(bar["t"]).astimezone(NY).date().isoformat()
            if day in prices[symbol]:
                raise ValueError(f"Duplicate daily bar: {symbol} {day}")
            value = bar.get("o")
            try:
                valid = type(value) in (int, float) and math.isfinite(value) and value > 0
            except OverflowError:
                valid = False
            prices[symbol][day] = valid
            if provenance["requested_start"] <= day <= provenance["requested_end"] and day not in dates:
                invalid_dates.append({"symbol": symbol, "date": day, "reason": "bar_on_non_session"})
    rows = []
    for symbol in symbols:
        for position, entry in decisions:
            for horizon in (5, 10, 20):
                exit_ = sessions[position + horizon]
                missing = [{"symbol": ticker, "date": day} for ticker in (symbol, "SPY")
                           for day in (entry["date"], exit_["date"])
                           if prices[ticker].get(day) is not True]
                mature = _time(exit_["open_at"]) <= clock
                # A daily bar captured before its session closes can still have
                # a valid open; never count an open received before the auction.
                early = [day for day in (entry, exit_) if _time(day["open_at"]) > receipt]
                paired = not missing and not early
                rows.append({"symbol": symbol, "decision_session": entry["date"],
                             "horizon_sessions": horizon, "entry_at": entry["open_at"],
                             "exit_session": exit_["date"], "exit_at": exit_["open_at"],
                             "calendar_horizon_mature_as_of": mature,
                             "paired_raw_open_endpoints_present": paired,
                             "missing_or_invalid_open_endpoints": missing,
                             "endpoint_open_after_response_receipt": [s["date"] for s in early],
                             "formal_label_eligible": False})
    coverage = {}
    for symbol in symbols:
        coverage[symbol] = {str(h): {
            "decision_sessions": len(decisions),
            "calendar_mature_sessions": sum(r["calendar_horizon_mature_as_of"] for r in rows
                                           if r["symbol"] == symbol and r["horizon_sessions"] == h),
            "paired_raw_open_endpoints": sum(r["paired_raw_open_endpoints_present"] for r in rows
                                           if r["symbol"] == symbol and r["horizon_sessions"] == h)}
            for h in (5, 10, 20)}
    if any(_hash(path) != expected for path, expected in fingerprints.items()):
        raise ValueError("Input changed during audit")
    return {"schema_version": "financial-calendar-endpoint-audit-1.0", "as_of": _iso(clock),
            "scope": {"archive": str(archive), "decision_month_ny": month, "symbols": symbols,
                      "benchmark": "SPY", "horizons": [5, 10, 20], "return_basis": "raw_price"},
            "input_sha256": fingerprints,
            "price_response_received_at": _iso(receipt), "price_request": request,
            "calendar": {k: v for k, v in calendar.items() if k != "sessions"},
            "summary": {"decision_sessions_in_month": len(decisions),
                        "coverage": coverage, "strict_eligible_labels_created": 0,
                        "model_evaluation_status": "Not evaluated", "trading_evaluation_status": "Not evaluated"},
            "missing_month_open_dates": {s: [d["date"] for _, d in decisions if not prices[s].get(d["date"])]
                                         for s in [*symbols, "SPY"]},
            "unexpected_bar_dates": invalid_dates, "rows": rows,
            "limitations": [
                "Counts measure raw open endpoint coverage only; no text observations or labels are constructed.",
                "Raw opening prices do not include dividends/splits and are not total-return indices.",
                "Calendar maturity is distinct from endpoint receipt, text availability and label eligibility.",
                "Existing development archives are not an unseen evaluation set.",
                "Text versions, historical receipt, events and feature completion still need independent audit.",
            ]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--calendar-end", required=True, help="Inclusive range; allow twenty tail sessions")
    parser.add_argument("--symbols", nargs="+", default=["PLTR", "AMZN", "MSFT"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output, archive = args.output_dir.resolve(), args.archive.resolve()
    if output == archive or output.is_relative_to(archive) or archive.is_relative_to(output):
        raise ValueError("Output must be separate from the input archive")
    if output.exists():
        raise FileExistsError("Output directory must be new")
    generator = Path(__file__).resolve().read_bytes()
    calendar = export_calendar(args.month + "-01", args.calendar_end)
    report = audit_archive(archive, month=args.month, as_of=args.as_of,
                           symbols=args.symbols, calendar=calendar)
    documents = {"calendar.json": calendar, "report.json": report}
    encoded = {name: (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
               for name, value in documents.items()}
    encoded["generation_source.py"] = generator
    manifest = {"schema_version": "calendar-endpoint-run-1.0", "generated_at": _iso(datetime.now(timezone.utc)),
                "python_version": sys.version, "input_sha256": report["input_sha256"],
                "generation_source_sha256": hashlib.sha256(generator).hexdigest(),
                "outputs": {name: {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
                            for name, value in encoded.items()}}
    if (Path(__file__).resolve().read_bytes() != generator
            or any(_hash(path) != expected for path, expected in report["input_sha256"].items())):
        raise ValueError("Input or generator changed before publication")
    encoded["run_manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    # Validate before creating any output, and never overwrite a prior run.
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    (output / ".gitignore").write_text("*\n")
    for name, value in encoded.items():
        with (output / name).open("xb") as stream:
            stream.write(value)
    print(json.dumps(report["summary"]))
    return report


if __name__ == "__main__":
    main()
