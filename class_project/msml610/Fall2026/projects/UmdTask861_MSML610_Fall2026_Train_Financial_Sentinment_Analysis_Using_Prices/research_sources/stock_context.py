"""Offline, descriptive stock context from saved observations (stdlib only).

This module neither collects data nor imports the application/trading runtime.
Input text stays local: only counts and explicitly supplied public news summaries
are exported. All price calculations are reproducible from the two bar archives.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, time, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo


UTC = timezone.utc
NY = ZoneInfo("America/New_York")
PEERS = ("SPY", "QQQ", "SOXX", "MU", "WDC", "STX")
BULLISH = ("看多", "看涨", "做多", "买入", "加仓", "抄底")
BEARISH = ("看空", "看跌", "做空", "卖出", "减仓", "清仓")
NEGATORS = ("不", "别", "未", "没有", "避免", "拒绝")


def _dt(value):
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("A timezone-aware ISO timestamp is required")
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Naive timestamps are not allowed")
    return result.astimezone(UTC)


def _iso(value):
    return _dt(value).isoformat().replace("+00:00", "Z") if value is not None else None


def _optional_dt(value):
    return _dt(value) if value else None


def _number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite numeric value")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError("Invalid price or volume")
    return result


def _pct(value, baseline):
    return (value / baseline - 1) * 100 if value is not None and baseline else None


def _difference(left, right):
    return left - right if left is not None and right is not None else None


def _session_time(day, hour, minute=0):
    return datetime.combine(day, time(hour, minute), NY).astimezone(UTC)


def _bars(raw, symbols, cutoff, *, daily):
    """Filter future observations before validating their price fields."""
    timeframe = raw.get("request", {}).get("timeframe")
    if timeframe is not None and timeframe != ("1Day" if daily else "5Min"):
        raise ValueError("Unexpected bar timeframe")
    source = raw.get("bars", {})
    if not isinstance(source, dict):
        raise ValueError("bars must be a symbol-to-list object")
    day = cutoff.astimezone(NY).date()
    result = {}
    for symbol in symbols:
        output = {}
        for row in source.get(symbol, []):
            start = _dt(row.get("timestamp", row.get("t")))
            local = start.astimezone(NY)
            if daily:
                if local.date() >= day:
                    continue
                key = local.date()
            else:
                end = start + timedelta(minutes=5)
                if local.date() != day or end > cutoff:
                    continue
                # This report covers premarket and regular trading only.
                if not (time(4) <= local.time().replace(tzinfo=None) < time(16)):
                    continue
                if start.second or start.microsecond or start.minute % 5:
                    raise ValueError("Five-minute bars must begin on a five-minute boundary")
                key = end
            if key in output:
                raise ValueError(f"Duplicate {'daily' if daily else 'intraday'} bar for {symbol}")
            if row.get("symbol", symbol) != symbol:
                raise ValueError("Bar symbol does not match its enclosing series")
            bar = {field: _number(row.get(field, row.get(short)), positive=True)
                   for field, short in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c"))}
            if bar["low"] > min(bar["open"], bar["close"]) or bar["high"] < max(bar["open"], bar["close"]):
                raise ValueError("Inconsistent OHLC range")
            bar["volume"] = _number(row.get("volume", row.get("v", 0)))
            vwap = row.get("vwap", row.get("vw"))
            bar["vwap"] = _number(vwap, positive=True) if vwap is not None else None
            output[key] = bar
        result[symbol] = dict(sorted(output.items()))
    return result


def _count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Counts must be nonnegative integers")
    return value


def _public_text(value, maximum=1000):
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError("Expected short public text")
    # Never allow copied local file paths or secret assignments in public fields.
    if re.search(r"(?:/Users/|/home/|file://|Bearer\s|(?:api[_-]?key|token|password)\s*[=:])", value, re.I):
        raise ValueError("Private material is not allowed in public text")
    return value


def _public_news(raw, cutoff):
    if not raw:
        return {"status": "Not collected", "received_at": None, "records_total": 0,
                "precutoff_publications": 0, "usable_at_cutoff": 0, "items": []}
    received = _optional_dt(raw.get("received_at"))
    total = _count(raw.get("records_total", 0))
    published = _count(raw.get("precutoff_publications", 0))
    usable = _count(raw.get("usable_at_cutoff", 0))
    if not 0 <= usable <= published <= total:
        raise ValueError("Inconsistent news coverage counts")
    # Coverage counts describe the whole collection, not just selected items.
    # Without a collection receipt we cannot certify its claimed usable count.
    if received is None or received > cutoff:
        usable = 0
    items = []
    for row in raw.get("items", []):
        publication = _optional_dt(row.get("published_at"))
        receipt = _optional_dt(row.get("received_at"))
        if receipt is not None and publication is not None and publication > receipt:
            raise ValueError("News publication cannot follow its observed receipt")
        if receipt is not None and receipt > cutoff:
            timing = "received_after_cutoff"
        elif receipt is not None and publication is not None and publication <= cutoff:
            timing = "available_at_cutoff"
        else:
            timing = "publication_time_unknown"
        association = row.get("association")
        evidence = row.get("evidence_type")
        if association not in {"direct", "sector", "macro"} or evidence not in {
            "reported_fact", "company_statement", "opinion", "unverified"
        }:
            raise ValueError("Invalid news association or evidence type")
        url = _public_text(row.get("url"), 2000)
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Public source URLs must use HTTPS without credentials")
        items.append({"id": _public_text(str(row.get("id", "")), 200),
                      "source": _public_text(row.get("source", ""), 200),
                      "published_at": _iso(publication), "received_at": _iso(receipt),
                      "timing": timing, "association": association,
                      "summary": _public_text(row.get("summary", "")), "url": url,
                      "evidence_type": evidence})
    return {"status": "post_cutoff" if received and received > cutoff else _public_text(raw.get("status", "Observed"), 200),
            "received_at": _iso(received), "records_total": total,
            "precutoff_publications": published, "usable_at_cutoff": usable, "items": items}


def _public_sources(raw):
    output = []
    for row in raw or []:
        output.append({"id": _public_text(row.get("id", ""), 100),
                       "label": _public_text(row.get("label", ""), 200),
                       "status": _public_text(row.get("status", ""), 200),
                       "records": _count(row["records"]) if row.get("records") is not None else None,
                       "received_at": _iso(_optional_dt(row.get("received_at"))),
                       "detail": _public_text(row.get("detail", ""))})
    return output


def classify_stance(text):
    """Count explicit Chinese stance cues; nearby negation suppresses, not inverts."""
    # "Long" embedded in a leveraged ETF's tagged product name is not a stance.
    text = re.sub(r"\$[^$\n]*\([A-Za-z0-9.^-]+\)\s*\$", " ", text)
    def has_cue(cues):
        for clause in re.split(r"[，。！？!?;；,\n]", text):
            for cue in cues:
                for match in re.finditer(re.escape(cue), clause):
                    prefix = clause[max(0, match.start() - 4):match.start()]
                    if not any(word in prefix for word in NEGATORS):
                        return True
        return False
    bullish, bearish = has_cue(BULLISH), has_cue(BEARISH)
    return "mixed" if bullish and bearish else "bullish" if bullish else "bearish" if bearish else "unknown"


def _post_text(row):
    parts = []
    for key in ("title", "excerpt"):
        value = row.get(key)
        if isinstance(value, str):
            normalized = " ".join(value.split())
            if normalized and normalized not in parts:
                parts.append(normalized)
    return "\n".join(parts)


def _community(raw, cutoff, symbol):
    limitations = [
        "Counts describe all observed unique post versions, not posts available for trading at the cutoff or unique users.",
        "Chinese phrase cues are a transparent heuristic, not a trained sentiment model; subject, sarcasm, conditions and retrospective remarks are not resolved.",
        "The fixed simplified-Chinese lexicon does not comprehensively cover traditional Chinese, Cantonese, questions, slang or option-position shorthand; unknown is not neutral.",
        "Query association does not establish that a post is about this stock. Author and engagement coverage is not assumed.",
        "Stance balance is 100 * (bullish - bearish) / (bullish + bearish), on a -100 to +100 scale; mixed and unknown remain visible and are excluded. It is not an up-move probability.",
        "No mechanical contrarian trade or probability of a future price move is inferred.",
    ]
    rows = [r for r in raw or [] if r.get("query_symbol", symbol) == symbol]
    unique = {}
    missing_ids = 0
    for row in rows:
        post_id = row.get("post_id")
        if not isinstance(post_id, (str, int)) or isinstance(post_id, bool) or not str(post_id).strip():
            missing_ids += 1
            continue
        key = str(post_id)
        receipt = _optional_dt(row.get("response_received_at"))
        # An undated version cannot be backdated using first_seen or another version.
        order = receipt or datetime.max.replace(tzinfo=UTC)
        old = unique.get(key)
        if old is None or order >= old[0]:
            unique[key] = (order, row)
    posts = [item[1] for item in unique.values()]
    counts = Counter(classify_stance(_post_text(row)) for row in posts)
    receipts = [_optional_dt(r.get("response_received_at")) for r in posts]
    publications = [_optional_dt(r.get("published_at")) for r in posts]
    available = sum(receipt is not None and publication is not None and
                    publication <= receipt <= cutoff
                    for receipt, publication in zip(receipts, publications))
    known_receipts = [value for value in receipts if value is not None]
    known_publications = [value for value in publications if value is not None]
    post_cutoff = sum(receipt is not None and receipt > cutoff for receipt in receipts)
    if not rows:
        status = "Not collected"
    elif posts and post_cutoff == len(posts):
        status = "post_cutoff"
    elif posts and available == len(posts):
        status = "available_at_cutoff"
    else:
        status = "Partial or unknown cutoff availability"
    if post_cutoff:
        limitations.append(f"{post_cutoff} observed post versions were received after the cutoff and cannot support a contemporaneous replay.")
    if missing_ids:
        limitations.append(f"{missing_ids} records have no valid post ID; they remain in records but are excluded from unique-post counts.")
    empty = sum(not _post_text(row) for row in posts)
    if empty:
        limitations.append(f"{empty} unique posts contain no text and are counted as unknown.")
    if len(rows) > len(posts) + missing_ids:
        limitations.append("Duplicate post IDs retain the latest received version; earlier first_seen timestamps do not backdate its text.")
    denominator = counts["bullish"] + counts["bearish"]
    return {"source": "Futu", "received_at": _iso(max(known_receipts)) if known_receipts else None,
            "status": status, "records": len(rows), "unique_posts": len(posts),
            "available_at_cutoff": available,
            "window_start": _iso(min(known_publications)) if known_publications else None,
            "window_end": _iso(max(known_publications)) if known_publications else None,
            **{key: counts[key] for key in ("bullish", "bearish", "mixed", "unknown")},
            "classified": len(posts) - counts["unknown"],
            "stance_balance": 100 * (counts["bullish"] - counts["bearish"]) / denominator if denominator else None,
            "method": "Chinese phrase lexicon v1 uses six bullish and six bearish phrases. Instrument-name cashtags are removed before matching. Negation within four preceding characters suppresses a cue; title and excerpt are concatenated only when distinct. Balance is 100 * (bullish - bearish) / (bullish + bearish), a descriptive scale from -100 to +100, not confidence or a price-move probability. Mixed and unknown posts are excluded from this balance.",
            "limitations": limitations}


def build_stock_context(daily_raw, intraday_raw, *, cutoff, symbol="SNDK", manifest=None,
                        community_rows=None, news_context=None, context_sources=None, generated_at=None):
    """Build context-1.0. No input objects are mutated and no network calls occur."""
    cutoff = _dt(cutoff)
    generated = _dt(generated_at) if generated_at else datetime.now(UTC)
    symbol = str(symbol).upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", symbol):
        raise ValueError("Invalid symbol")
    symbols = tuple(dict.fromkeys((symbol, *PEERS)))
    receipt = _optional_dt((manifest or {}).get("price_received_at"))
    daily = _bars(daily_raw, symbols, cutoff, daily=True)
    intraday = _bars(intraday_raw, symbols, cutoff, daily=False)
    limits = [
        "Retrospective research only. These descriptive observations are not forecasts, buy/sell instructions, calibrated probabilities or an evaluated trading strategy.",
        "Intraday timestamps are five-minute bar END times; a bar is included only after that entire interval is complete. Missing bars are not filled.",
        "Intraday returns use the preceding observed market session close. Premarket is shown separately; the gap and open-to-cutoff return use the regular-session opening bar.",
        "Daily return_pct is a raw close-price return normalized to the first date shared by the returned available series, not a total return or portfolio return.",
        "Five- and twenty-session returns use matching endpoints on the observed reference calendar. SMA and simple mean true range require complete input windows; no official exchange-calendar completeness claim is made.",
        "Candidate levels use only prior completed daily bars and are not guarantees of support or resistance. known_at is the underlying price-period close, not system receipt. Reference computed from prior completed prices; reconstructed after the event. Actual receipt is recorded separately as price_received_at.",
        "Session VWAP excludes premarket. Provider bar VWAP is volume-weighted when available for every positive-volume regular bar; otherwise all bars use the OHLC typical-price proxy (high + low + close) / 3.",
        "News counts may cover more articles than the selected public summaries. Publication before the cutoff alone does not establish that the system had received the text.",
    ]
    if receipt is None:
        limits.append("Price receipt time was not recorded; historical contemporaneous availability is unknown.")
    elif receipt > cutoff:
        limits.append("Price data was received after the analysis cutoff (post_cutoff); this snapshot cannot establish contemporaneous system availability.")
    day = cutoff.astimezone(NY).date()
    # SPY is the explicit reference calendar. With no SPY, use the union and disclose it.
    calendar = list(daily["SPY"]) or sorted(set().union(*(set(rows) for rows in daily.values())))
    if not daily["SPY"]:
        limits.append("SPY daily calendar is missing; observed dates from the other series define the reference calendar.")
    previous_day = calendar[-1] if calendar else None
    previous = {s: daily[s].get(previous_day) for s in symbols}
    previous_close = {s: previous[s]["close"] if previous[s] else None for s in symbols}
    # Use one common timestamp for all headline comparisons, never stale benchmark prices.
    target_times = list(intraday[symbol])
    last_time = target_times[-1] if target_times else None
    last_bar = intraday[symbol].get(last_time)
    last_price = last_bar["close"] if last_bar else None
    changes = {s: _pct(intraday[s][last_time]["close"], previous_close[s])
               if last_time in intraday[s] else None for s in symbols}
    if last_time and last_time < cutoff - timedelta(minutes=5):
        limits.append("The latest completed target bar is more than five minutes older than the requested cutoff.")
    for s in symbols:
        if previous[s] is None:
            limits.append(f"{s} lacks the common previous-session daily bar; its intraday return is unavailable.")
        if last_time and last_time not in intraday[s]:
            limits.append(f"{s} lacks a bar at the target stock's final timestamp; its headline intraday comparison is unavailable.")
    opening_end = _session_time(day, 9, 30) + timedelta(minutes=5)
    opening = intraday[symbol].get(opening_end)
    open_price = opening["open"] if opening else None
    regular = [(end, bar) for end, bar in intraday[symbol].items() if end > _session_time(day, 9, 30)]
    volume = sum(bar["volume"] for _, bar in regular) if regular else None
    positive_bars = [bar for _, bar in regular if bar["volume"] > 0]
    provider_vwap = bool(positive_bars) and all(bar["vwap"] is not None for bar in positive_bars)
    vwap = None
    if volume:
        vwap = sum((bar["vwap"] if provider_vwap else (bar["high"] + bar["low"] + bar["close"]) / 3) * bar["volume"]
                   for bar in positive_bars) / volume
        limits.append("Session VWAP method in this snapshot: " + ("volume-weighted provider bar VWAP." if provider_vwap else "volume-weighted OHLC typical-price proxy; this is not trade-level VWAP."))
    else:
        limits.append("Session VWAP is unavailable because there is no completed regular-session positive volume.")
    expected_last = min(cutoff, _session_time(day, 16))
    expected_regular_count = max(0, int((expected_last - _session_time(day, 9, 30)).total_seconds() // 300))
    if len(regular) < expected_regular_count:
        limits.append("Regular-session bars are incomplete; session volume and VWAP cover observed bars only.")

    def window(count):
        dates = calendar[-count:]
        return [daily[symbol][d] for d in dates] if len(dates) == count and all(d in daily[symbol] for d in dates) else []

    history20, history50, history15 = window(20), window(50), window(15)
    sma20 = sum(b["close"] for b in history20) / 20 if history20 else None
    sma50 = sum(b["close"] for b in history50) / 50 if history50 else None
    tr14 = sum(max(b["high"] - b["low"], abs(b["high"] - prev["close"]), abs(b["low"] - prev["close"]))
               for prev, b in zip(history15, history15[1:])) / 14 if history15 else None
    levels = []
    level_known = _session_time(previous_day, 16) if previous_day else None

    def level(id_, label, price, basis, kind):
        if price is not None:
            levels.append({"id": id_, "label": label, "price": price, "known_at": _iso(level_known),
                           "basis": basis + " Reference computed from prior completed prices; reconstructed after the event.", "kind": kind})

    if previous[symbol]:
        for field in ("high", "low", "close"):
            level("previous_" + field, "Previous session " + field, previous[symbol][field],
                  f"{previous_day.isoformat()} completed daily bar {field}.", "prior_session")
    if history20:
        level("rolling_high_20", "20-session high", max(b["high"] for b in history20),
              "Maximum daily high over the 20 completed reference sessions before today.", "rolling")
        level("rolling_low_20", "20-session low", min(b["low"] for b in history20),
              "Minimum daily low over the 20 completed reference sessions before today.", "rolling")
    level("sma20", "20-session moving average", sma20, "Simple mean close over 20 completed prior sessions.", "moving_average")
    level("sma50", "50-session moving average", sma50, "Simple mean close over 50 completed prior sessions.", "moving_average")

    def period_return(s, count):
        if len(calendar) < count + 1:
            return None
        start, end = daily[s].get(calendar[-count - 1]), daily[s].get(calendar[-1])
        return _pct(end["close"], start["close"]) if start and end else None

    present = [set(rows) for rows in daily.values() if rows]
    shared = set.intersection(*present) if present else set()
    base_day = min(shared) if shared else None
    daily_points = [{"date": d.isoformat(), "symbol": s, "close": b["close"],
                     "return_pct": _pct(b["close"], daily[s][base_day]["close"])}
                    for s in symbols if base_day in daily[s]
                    for d, b in daily[s].items() if d >= base_day]
    if base_day is not None:
        limits.append(f"Daily normalization baseline in this snapshot: {base_day.isoformat()} close, equal to 0% for each available series.")
    else:
        limits.append("No common daily normalization date exists; the normalized daily series is unavailable.")
    intra_points = [{"timestamp": _iso(end), "symbol": s, "price": b["close"],
                     "return_pct": _pct(b["close"], previous_close[s]), "volume": b["volume"],
                     "session": "premarket" if end <= _session_time(day, 9, 30) else "regular"}
                    for s in symbols if previous_close[s] is not None
                    for end, b in intraday[s].items()]
    return {"schema_version": "context-1.0", "generated_at": _iso(generated),
            "analysis_cutoff": _iso(cutoff), "price_received_at": _iso(receipt), "symbol": symbol,
            "mode": "Retrospective research snapshot", "prediction_status": "Not trained",
            "summary": {"last_price": last_price, "previous_close": previous_close[symbol],
                        "change_pct": changes[symbol], "gap_pct": _pct(open_price, previous_close[symbol]),
                        "open_to_cutoff_pct": _pct(last_price, open_price),
                        "excess_vs_spy_pp": _difference(changes[symbol], changes["SPY"]),
                        "excess_vs_sector_pp": _difference(changes[symbol], changes["SOXX"]),
                        "mean_true_range_14": tr14, "sma20": sma20, "sma50": sma50,
                        "session_vwap": vwap, "session_volume": volume},
            "benchmarks": [{"symbol": s, "change_pct": changes[s], "return_5d_pct": period_return(s, 5),
                            "return_20d_pct": period_return(s, 20)} for s in symbols],
            "intraday": sorted(intra_points, key=lambda p: (p["timestamp"], p["symbol"])),
            "daily": sorted(daily_points, key=lambda p: (p["date"], p["symbol"])), "levels": levels,
            "news": _public_news(news_context, cutoff), "community": _community(community_rows, cutoff, symbol),
            "sources": _public_sources(context_sources), "limitations": limits}


def load_stock_context(input_dir, *, cutoff, symbol="SNDK", generated_at=None, input_audit=None):
    """Read only the documented inputs; human-selected levels/case reviews are ignored."""
    directory = Path(input_dir)

    def read(name, default=None, required=False):
        path = directory / name
        if not path.exists() and not required:
            if input_audit is not None:
                input_audit[name] = {"present": False}
            return default
        encoded = path.read_bytes()
        if input_audit is not None:
            input_audit[name] = {"present": True, "bytes": len(encoded),
                                 "sha256": hashlib.sha256(encoded).hexdigest()}
        return json.loads(encoded.decode("utf-8"))

    return build_stock_context(read("daily.json", required=True), read("intraday.json", required=True),
                               cutoff=cutoff, symbol=symbol, generated_at=generated_at,
                               manifest=read("manifest.json", {}),
                               community_rows=read("futu_current/all_normalized.json", []),
                               news_context=read("news_context.json"), context_sources=read("context_sources.json", []))


def _validate_output_paths(input_dir, output, audit_output):
    """Keep the archive and existing audits intact, including path aliases."""
    source = Path(os.path.abspath(input_dir))
    # Resolve the original spelling: abspath would collapse symlink/.. first
    # and could validate a different location from the actual filesystem write.
    source_real = Path(input_dir).resolve()
    protected_roots = {source, source_real}
    # Inspect links without traversing their targets or reading additional files.
    # A directory link is part of the archive even when addressed from outside it.
    protected_roots.update(path.resolve() for path in source_real.rglob("*") if path.is_symlink())
    destinations = []
    for label, path in (("Report", output), ("Audit", audit_output)):
        absolute = Path(os.path.abspath(path))
        resolved = Path(path).resolve()
        if any(candidate == root or root in candidate.parents
               for candidate in (absolute, resolved) for root in protected_roots):
            raise ValueError(f"{label} output must be outside the read-only input directory")
        destinations.append(resolved)
    if destinations[0] == destinations[1]:
        raise ValueError("Report and audit outputs must use different paths")
    audit_output = Path(audit_output)
    if audit_output.exists() or audit_output.is_symlink():
        raise FileExistsError("Audit output already exists; choose a new private audit path")
    output = Path(output)
    if output.is_file():
        try:
            previous = json.loads(output.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            previous = None
        if isinstance(previous, dict) and str(previous.get("schema_version", "")).startswith(
                "stock-context-generation-audit-"):
            raise FileExistsError("Report output must not replace an existing generation audit")


def _atomic_write(path, text, *, exclusive=False):
    """Publish a complete file; exclusive creation never replaces any existing path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        identity = temporary.stat()
        if exclusive:
            # Hard-linking a complete same-filesystem temporary file is atomic
            # and fails if any destination exists, including a dangling symlink.
            os.link(temporary, path)
        else:
            temporary.replace(path)
        return identity.st_dev, identity.st_ino
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_report_and_audit(output, report_text, audit_output, audit_text):
    """Reserve a fresh audit before replacing the report; roll back only our audit."""
    audit_output = Path(audit_output)
    identity = _atomic_write(audit_output, audit_text, exclusive=True)
    try:
        _atomic_write(output, report_text)
    except BaseException:
        try:
            current = audit_output.lstat()
        except FileNotFoundError:
            pass
        else:
            if (current.st_dev, current.st_ino) == identity:
                audit_output.unlink()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build an offline public stock-context research snapshot.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path,
                        help="New private generation-audit file outside the read-only input directory")
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--symbol", default="SNDK")
    args = parser.parse_args(argv)
    _validate_output_paths(args.input_dir, args.output, args.audit_output)
    input_audit = {}
    report = load_stock_context(args.input_dir, cutoff=args.cutoff, symbol=args.symbol, input_audit=input_audit)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    audit = {"schema_version": "stock-context-generation-audit-1.0",
             "generated_at": report["generated_at"], "analysis_cutoff": report["analysis_cutoff"],
             "symbol": report["symbol"], "inputs": input_audit,
             "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "public_output_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
             "public_output_bytes": len(encoded.encode("utf-8")),
             "notes": ["Input hashes describe the exact bytes parsed for this report, not a later reread.",
                       "This audit is private run metadata; raw text and file paths are absent from the public report."]}
    audit_encoded = json.dumps(audit, indent=2, allow_nan=False) + "\n"
    _publish_report_and_audit(args.output, encoded, args.audit_output, audit_encoded)
    print(json.dumps({"schema_version": report["schema_version"], "symbol": report["symbol"],
                      "intraday_points": len(report["intraday"]), "daily_points": len(report["daily"]),
                      "community_posts": report["community"]["unique_posts"]}))


if __name__ == "__main__":
    main()
