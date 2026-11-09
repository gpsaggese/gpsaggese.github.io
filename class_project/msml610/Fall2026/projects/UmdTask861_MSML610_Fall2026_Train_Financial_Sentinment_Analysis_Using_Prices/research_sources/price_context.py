"""Describe saved daily prices without fetching data or forecasting returns.

Only the Python standard library is required. Input prices are used as supplied:
this module does not adjust for splits, dividends, fees, or portfolio weights.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo


NEW_YORK = ZoneInfo("America/New_York")


def _date(value: str, name: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD)") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD)")
    return parsed


def _session_date(value: Any, symbol: str) -> date:
    try:
        if not isinstance(value, str):
            raise ValueError("Timestamp must be a string")
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("Timezone is required")
        return timestamp.astimezone(NEW_YORK).date()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid timezone-aware bar timestamp for {symbol}: {value!r}") from exc


def _price(value: Any, symbol: str, session: date, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid {field} price for {symbol} on {session}: expected a number")
    try:
        price = float(value)
    except OverflowError as exc:
        raise ValueError(f"Invalid {field} price for {symbol} on {session}") from exc
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"Invalid {field} price for {symbol} on {session}: must be finite and positive")
    return price


def _return_pct(end: float | None, start: float | None) -> float | None:
    if end is None or start is None:
        return None
    result = (end / start - 1.0) * 100.0
    if not math.isfinite(result):
        raise ValueError("Price ratio produces a non-finite return")
    return result


def _excess(stock: float | None, benchmark: float | None) -> float | None:
    if stock is None or benchmark is None:
        return None
    result = stock - benchmark
    if not math.isfinite(result):
        raise ValueError("Return difference produces a non-finite excess return")
    return result


def _symbol(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("Symbols must be nonempty strings without surrounding whitespace")
    return value


def _context_row(
    symbol: str,
    bars: dict[date, dict[str, float | None]],
    sessions: list[date],
    start: date,
    end: date,
) -> dict[str, Any]:
    start_close = bars.get(start, {}).get("c")
    end_close = bars.get(end, {}).get("c")
    reasons: dict[str, str] = {}
    if start_close is None or end_close is None:
        missing = [day.isoformat() for day, price in ((start, start_close), (end, end_close)) if price is None]
        reasons["return_pct"] = "Unavailable: exact endpoint close missing on " + ", ".join(dict.fromkeys(missing))
    daily = []
    for previous, current in zip(sessions, sessions[1:]):
        previous_close = bars.get(previous, {}).get("c")
        current_bar = bars.get(current, {})
        open_price, close = current_bar.get("o"), current_bar.get("c")
        gaps: dict[str, str] = {}
        for name, value in (("previous_close", previous_close), ("open", open_price), ("close", close)):
            if value is None:
                missing_day = previous if name == "previous_close" else current
                gaps[name] = f"Unavailable: {name} missing on {missing_day}"
        daily.append({
            "date": current.isoformat(),
            "previous_session_date": previous.isoformat(),
            "status": "Unavailable" if gaps else "Available",
            "previous_close": previous_close,
            "open": open_price,
            "close": close,
            "gap_pct": _return_pct(open_price, previous_close),
            "intraday_pct": _return_pct(close, open_price),
            "return_pct": _return_pct(close, previous_close),
            "unavailable_reasons": gaps,
        })
    return {
        "symbol": symbol,
        "status": "Unavailable" if reasons else "Available",
        "start_close": start_close,
        "end_close": end_close,
        "return_pct": _return_pct(end_close, start_close),
        "unavailable_reasons": reasons,
        "daily_sessions_status": "Unavailable" if any(row["status"] == "Unavailable" for row in daily) else "Available",
        "daily_sessions": daily,
    }


def build_price_context(
    raw_bars: dict,
    symbols: list[str],
    start_date: str,
    end_date: str,
    benchmarks: dict[str, str],
) -> dict:
    """Build descriptive close-to-close company and benchmark comparisons.

    ``raw_bars`` uses the Alpaca shape ``{"bars": {"PLTR": [{"t": ..., "o":
    ..., "c": ...}]}}``. ``benchmarks`` maps companies to caller-selected sector
    proxies, for example ``{"PLTR": "IGV", "AMZN": "QQQ"}``; SPY is always
    included. Both period endpoints must have their exact dated close.

    ``daily_sessions`` covers (start_date, end_date]. The previous close must
    exist on the preceding date in the union of observed requested instruments'
    session dates, plus the two endpoints. Missing bars are not bridged. This
    observed-date grid is not an independently verified exchange calendar.

    Missing data yields ``None`` with an ``Unavailable`` status/reason, preserving
    other computable metrics. Malformed timestamps, duplicate New York dates,
    and supplied invalid prices raise ValueError. Bars outside the date window
    are ignored after timestamp parsing. No input is mutated and no I/O occurs.
    """
    start, end = _date(start_date, "start_date"), _date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date must not be after end_date")
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("symbols must be a nonempty list")
    requested = [_symbol(symbol) for symbol in symbols]
    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate requested symbols")
    if not isinstance(benchmarks, dict):
        raise ValueError("benchmarks must map company symbols to benchmark symbols")
    sector_symbols = {symbol: _symbol(benchmarks[symbol]) for symbol in requested if symbol in benchmarks}
    if not isinstance(raw_bars, dict) or not isinstance(raw_bars.get("bars"), dict):
        raise ValueError("raw_bars must contain a 'bars' dictionary")
    market_symbols = list(dict.fromkeys(["SPY", *sector_symbols.values()]))
    required = list(dict.fromkeys([*requested, *market_symbols]))
    normalized: dict[str, dict[date, dict[str, float | None]]] = {}
    observed_dates = {start, end}
    for symbol in required:
        rows = raw_bars["bars"].get(symbol, [])
        if not isinstance(rows, list):
            raise ValueError(f"Bars for {symbol} must be a list")
        normalized[symbol] = {}
        for bar in rows:
            if not isinstance(bar, dict):
                raise ValueError(f"Each bar for {symbol} must be a dictionary")
            session = _session_date(bar.get("t"), symbol)
            if not start <= session <= end:
                continue
            if session in normalized[symbol]:
                raise ValueError(f"Duplicate New York session date for {symbol}: {session}")
            normalized[symbol][session] = {
                field: _price(bar.get(field), symbol, session, field) for field in ("o", "h", "l", "c")
            }
            observed_dates.add(session)
    sessions = sorted(observed_dates)
    rows_by_symbol = {
        symbol: _context_row(symbol, normalized[symbol], sessions, start, end) for symbol in required
    }
    market_rows = {symbol: rows_by_symbol[symbol] for symbol in market_symbols}
    companies = {}
    spy_return = market_rows["SPY"]["return_pct"]
    for symbol in requested:
        base = rows_by_symbol[symbol]
        row = {**base, "unavailable_reasons": dict(base["unavailable_reasons"])}
        sector = sector_symbols.get(symbol)
        sector_return = market_rows[sector]["return_pct"] if sector else None
        row.update({
            "sector_symbol": sector,
            "spy_return_pct": spy_return,
            "sector_return_pct": sector_return,
            "excess_vs_spy_pp": _excess(row["return_pct"], spy_return),
            "excess_vs_sector_pp": _excess(row["return_pct"], sector_return),
        })
        for field in ("spy_return_pct", "sector_return_pct", "excess_vs_spy_pp", "excess_vs_sector_pp"):
            if row[field] is None:
                row["unavailable_reasons"][field] = "Unavailable: required company or benchmark exact endpoint close missing, or sector benchmark not configured"
        row["status"] = "Unavailable" if row["unavailable_reasons"] else "Available"
        companies[symbol] = row
    return {
        "schema_version": 1,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "America/New_York",
        "return_type": "raw_price_close_to_close",
        "descriptive_only": True,
        "is_forecast": False,
        "methodology": {
            "period": "Exact start-date close to exact end-date close; no endpoint substitution.",
            "daily_sessions": "Dates in (start_date, end_date]; previous close uses the preceding observed-grid date, without filling missing bars.",
            "session_grid": "Union of observed dates for requested companies and benchmarks, plus endpoints; exchange-calendar completeness is not verified.",
            "units": "Returns are percentages; excess returns are percentage-point differences.",
            "decomposition": "1 + daily_return/100 = (1 + gap/100) * (1 + intraday_return/100).",
            "limitations": "Descriptive raw prices as supplied, not portfolio total returns, alpha estimates, or forecasts. No dividend/split adjustment, trading costs, or exposure adjustment is applied.",
            "availability": "Bar timestamps identify sessions, not historical receipt times; this module does not prove when a daily bar became available.",
        },
        "companies": companies,
        "market_benchmarks": market_rows,
    }
