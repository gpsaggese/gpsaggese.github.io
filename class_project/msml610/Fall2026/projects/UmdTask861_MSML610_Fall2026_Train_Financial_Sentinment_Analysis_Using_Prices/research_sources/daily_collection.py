"""One manual daily capture with exact clocks, overlap and a durable cursor.

Run whenever desired; this command does not install a schedule or place orders.
Every capture/version is retained. Only a completed query advances the cursor.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

from collect_research import collect_research, _parse_timestamp, _symbols, NEW_YORK


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def run_daily(output_root, symbols=None, *, start_at=None, end_at=None,
              lookback_days=3, overlap_minutes=60, lag_minutes=20, max_news_pages=20,
              collector=collect_research, clock=lambda: datetime.now(timezone.utc)):
    symbols = _symbols(symbols or ["PLTR", "AMZN", "MSFT"])
    if type(lookback_days) is not int or not 1 <= lookback_days <= 90:
        raise ValueError("Initial lookback must be 1 to 90 days")
    if type(overlap_minutes) is not int or not 1 <= overlap_minutes <= 1440:
        raise ValueError("Overlap must be 1 to 1440 minutes")
    if type(lag_minutes) is not int or not 0 <= lag_minutes <= 1440:
        raise ValueError("Data lag must be 0 to 1440 minutes")
    now = clock()
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Collection clock must include timezone")
    end = _parse_timestamp(end_at) if end_at else now.astimezone(timezone.utc) - timedelta(minutes=lag_minutes)
    if end > now:
        raise ValueError("Daily endpoint is in the future")
    root = Path(output_root)
    if root.is_symlink():
        raise ValueError("Output root cannot be a symlink")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    ignore = root / ".gitignore"
    if not ignore.exists():
        with ignore.open("x") as stream:
            stream.write("*\n")
    cursor, lock = root / "cursor.json", root / ".daily.lock"
    if cursor.is_symlink():
        raise ValueError("Cursor cannot be a symlink")
    lock_token = uuid.uuid4().hex
    with lock.open("x") as stream:
        stream.write(lock_token)
    try:
        before = _hash(cursor)
        state = json.loads(cursor.read_text()) if before else {}
        if state and (state.get("schema_version") != 1 or state.get("symbols") != symbols):
            raise ValueError("Existing cursor belongs to a different schema/company pool; choose a new root")
        previous = _parse_timestamp(state["last_complete_end_at"]) if state else None
        start = (_parse_timestamp(start_at) if start_at else
                 previous - timedelta(minutes=overlap_minutes) if previous else
                 end - timedelta(days=lookback_days))
        if previous and start > previous:
            raise ValueError("Requested start would skip the uncollected cursor gap")
        if start >= end:
            raise ValueError("Daily start must precede end; an older rerun needs an explicit earlier start")
        output = root / ("capture_" + now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8])
        manifest = collector(output, symbols,
                             start.astimezone(NEW_YORK).date().isoformat(),
                             end.astimezone(NEW_YORK).date().isoformat(),
                             start_at=start.isoformat(), end_at=end.isoformat(),
                             max_news_pages=max_news_pages)
        complete = (manifest["status"] == "complete"
                    and manifest["news_summary"]["pagination_complete"]
                    and manifest["bars_summary"]["pagination_complete"])
        if complete:
            # An explicit historical rerun never rolls the current cursor back.
            next_end = max(previous, end) if previous else end
            new_state = {"schema_version": 1, "symbols": symbols,
                         "last_complete_end_at": next_end.isoformat(),
                         "updated_at": clock().astimezone(timezone.utc).isoformat(),
                         "captures": [*state.get("captures", []), output.name]}
            if cursor.is_symlink() or _hash(cursor) != before:
                raise ValueError("Cursor changed during capture; raw output retained, cursor not overwritten")
            fd, temporary = tempfile.mkstemp(prefix=".cursor-", dir=root)
            try:
                with os.fdopen(fd, "w") as stream:
                    json.dump(new_state, stream, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, cursor)
            finally:
                if Path(temporary).exists():
                    Path(temporary).unlink()
        return {"status": manifest["status"], "capture_dir": str(output),
                "window_start_at": start.isoformat(), "window_end_at": end.isoformat(),
                "end_policy": "explicit_endpoint" if end_at else "actual_invocation_minus_configured_data_lag",
                "configured_data_lag_minutes": lag_minutes,
                "cursor_advanced_or_retained": complete,
                "news_summary": manifest["news_summary"], "bars_summary": manifest["bars_summary"],
                "historical_availability_verified": False,
                "note": "Actual receipts remain in each capture; rerun overlap retains all versions, not independent new events"}
    finally:
        if lock.exists() and lock.read_text() == lock_token:
            lock.unlink()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--symbols", nargs="+", default=["PLTR", "AMZN", "MSFT"])
    p.add_argument("--start-at", help="Explicit aware start for an intentional historical rerun")
    p.add_argument("--end-at", help="Explicit aware past endpoint; default is actual invocation minus configured data lag")
    p.add_argument("--lookback-days", type=int, default=3, help="Used only for the first run")
    p.add_argument("--overlap-minutes", type=int, default=60)
    p.add_argument("--lag-minutes", type=int, default=20, help="Default 20-minute SIP history delay; use 0 only with existing real-time access")
    p.add_argument("--max-news-pages", type=int, default=20)
    a = p.parse_args(argv)
    r = run_daily(a.output_root, a.symbols, start_at=a.start_at, end_at=a.end_at,
                  lookback_days=a.lookback_days, overlap_minutes=a.overlap_minutes, lag_minutes=a.lag_minutes,
                  max_news_pages=a.max_news_pages)
    print(json.dumps(r, indent=2))
    return 0 if r["cursor_advanced_or_retained"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
