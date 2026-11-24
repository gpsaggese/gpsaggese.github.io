"""Crash-safe, process-safe reservations of API list-price spending.

Money is stored in integer nanodollars. Outstanding requests retain their
entire maximum liability on process failure; an ambiguous transport error is
charged at that maximum. No automatic release of unknown requests is allowed.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

NANO = 1_000_000_000
MODEL = "anthropic/claude-haiku-5-5"
MAX_INPUT = 1_000_000
MAX_OUTPUT = 16_000
# Reserve maximum full-window input, all at the expensive 5-minute write rate,
# plus the full configured output allowance, at the >100K pricing tier.
MAX_REQUEST_NANO = MAX_INPUT * 625 + MAX_OUTPUT * 2500


class BudgetExhausted(RuntimeError):
    pass


class BudgetBusy(RuntimeError):
    pass


class AccountingError(RuntimeError):
    pass


def price_usage(usage: dict) -> int:
    """Normalize LiteLLM's inclusive prompt count, including thinking output."""
    if not usage or "prompt_tokens" not in usage or "completion_tokens" not in usage:
        raise AccountingError("Missing token usage; request maximum must be retained")
    prompt = int(usage["prompt_tokens"])
    details = usage.get("prompt_tokens_details") or {}
    read = int(usage.get("cache_read_input_tokens") or details.get("cached_tokens") or 0)
    write = int(usage.get("cache_creation_input_tokens") or 0)
    uncached = prompt - read - write if prompt >= read + write else prompt
    prompt = max(prompt, uncached + read + write)
    output = int(usage["completion_tokens"])
    if min(prompt, read, write, uncached, output) < 0 or prompt > MAX_INPUT or output > MAX_OUTPUT:
        raise AccountingError("Usage outside the configured provider bounds")
    cache_types = details.get("cache_creation_token_details") or {}
    if cache_types.get("ephemeral_1h_input_tokens", 0):
        raise AccountingError("Unexpected 1-hour cache write")
    if usage.get("service_tier") not in (None, "standard") or usage.get("inference_geo") not in (None, "global"):
        raise AccountingError("Unexpected service tier or inference geography")
    pin, pwrite, pread, pout = (500, 625, 50, 2500) if prompt > 100_000 else (100, 125, 10, 500)
    return uncached * pin + write * pwrite + read * pread + output * pout


class Budget:
    def __init__(self, path: Path, cap_nano: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                create table if not exists settings (id integer primary key, cap integer not null);
                create table if not exists calls (
                    id text primary key, job_id text not null, created real not null,
                    request_hash text not null, maximum integer not null,
                    liability integer not null, status text not null,
                    settled real, usage_json text, error_type text);
            """)
            db.execute("insert or ignore into settings values (1,?)", (cap_nano,))
            if db.execute("select cap from settings where id=1").fetchone()[0] != cap_nano:
                raise AccountingError("An existing budget cannot silently change its ceiling")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=120, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("pragma busy_timeout=120000")
        try:
            yield db
        finally:
            db.close()

    def reserve(self, job_id: str, request_hash: str, maximum: int = MAX_REQUEST_NANO) -> str:
        if maximum <= 0:
            raise ValueError("Reservation must be positive")
        with self.connection() as db:
            db.execute("begin immediate")
            cap = db.execute("select cap from settings").fetchone()[0]
            total, active = db.execute("select coalesce(sum(liability),0),sum(status='reserved') from calls").fetchone()
            if total + maximum > cap:
                db.rollback()
                if active:
                    raise BudgetBusy("Waiting for requests with outstanding reservations")
                raise BudgetExhausted("The approved ceiling cannot cover another maximum-cost request")
            token = uuid.uuid4().hex
            db.execute("insert into calls (id,job_id,created,request_hash,maximum,liability,status) values (?,?,?,?,?,?,'reserved')",
                       (token, job_id, time.time(), request_hash, maximum, maximum))
            db.commit()
            return token

    def settle(self, token: str, *, usage: dict | None = None, rejected: bool = False, error_type: str | None = None) -> int:
        with self.connection() as db:
            db.execute("begin immediate")
            row = db.execute("select * from calls where id=?", (token,)).fetchone()
            if not row or row["status"] != "reserved":
                db.rollback()
                raise AccountingError("Unknown or already-settled request")
            error = None
            if rejected:
                cost, status = 0, "rejected_before_inference"
            elif usage is None:
                cost, status = row["maximum"], "uncertain_maximum_retained"
            else:
                try:
                    cost = price_usage(usage)
                    if cost > row["maximum"]:
                        raise AccountingError("Reported cost exceeds its reservation")
                    status = "settled"
                except AccountingError as exc:
                    cost, status, error = row["maximum"], "uncertain_maximum_retained", exc
            db.execute("update calls set liability=?,status=?,settled=?,usage_json=?,error_type=? where id=?",
                       (cost, status, time.time(), json.dumps(usage) if usage else None,
                        error_type or (type(error).__name__ if error else None), token))
            db.commit()
        if error:
            raise error
        return cost

    def summary(self) -> dict:
        with self.connection() as db:
            cap = db.execute("select cap from settings").fetchone()[0]
            rows = [dict(r) for r in db.execute("select status,count(*) n,sum(liability) amount from calls group by status")]
        total = sum(r["amount"] for r in rows)
        return {"cap_usd": cap / NANO, "accounted_plus_reserved_usd": total / NANO,
                "available_usd": (cap-total) / NANO, "requests": sum(r["n"] for r in rows),
                "by_status": rows, "within_cap": total <= cap}
