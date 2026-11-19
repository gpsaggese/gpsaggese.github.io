"""
Track API rate limits and token use across runs.

Groq (free tier, `openai/gpt-oss-120b`) limits requests and tokens per minute
and per day. Its response headers report:
- `x-ratelimit-*-requests`: requests per day (RPD)
- `x-ratelimit-*-tokens`: tokens per minute (TPM)

No header reports tokens per day (TPD), so every call is also appended to a
usage ledger shared by all agents using the same key, and the tokens of the
last 24 hours are summed from it.

A 429 caused by a daily limit cannot be fixed by waiting a few seconds: runners
stop cleanly and resume from their caches later.

Import as:

import rate_limit_lib as ratlib
"""

import datetime
import json
import logging
import os
from typing import Any, Dict, Mapping, Optional

import helpers.hio as hio

_LOG = logging.getLogger(__name__)

# Rate limit headers and the name we store each one under.
RATE_LIMIT_HEADERS = {
    "x-ratelimit-remaining-requests": "remaining_requests_day",
    "x-ratelimit-limit-requests": "limit_requests_day",
    "x-ratelimit-reset-requests": "reset_requests_day",
    "x-ratelimit-remaining-tokens": "remaining_tokens_minute",
    "x-ratelimit-limit-tokens": "limit_tokens_minute",
    "x-ratelimit-reset-tokens": "reset_tokens_minute",
}

# Default ledger, shared by all runners.
DEFAULT_LEDGER_FILE = "results/raw/groq_usage.jsonl"

# Free-tier tokens per day for `openai/gpt-oss-120b`.
DEFAULT_DAILY_TOKEN_BUDGET = 200_000

# Longest wait between retries of a per-minute limit.
_MAX_BACKOFF_SECONDS = 60.0


class DailyLimitError(Exception):
    """
    The provider refused a request because a daily limit is used up.
    """


# #############################################################################
# Headers and errors
# #############################################################################


def extract_rate_limits(headers: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """
    Return the rate limit headers of a response under short names.

    :param headers: response headers, or `None`
    :return: e.g., `{"remaining_requests_day": "998", ...}`; missing headers
        are left out
    """
    limits: Dict[str, str] = {}
    if not headers:
        return limits
    for header, key in RATE_LIMIT_HEADERS.items():
        value = headers.get(header)
        if value is not None:
            limits[key] = str(value)
    return limits


def is_daily_limit(message: str) -> bool:
    """
    Return whether a rate limit error is about a daily limit.

    Groq messages name the limit, e.g., "... on tokens per day (TPD): Limit
    200000, Used 199500, Requested 1200 ...".
    """
    text = message.lower()
    is_daily = "per day" in text or "(tpd)" in text or "(rpd)" in text
    return is_daily


def get_retry_seconds(headers: Optional[Mapping[str, str]], attempt: int) -> float:
    """
    Return how long to wait before retrying a per-minute rate limit.

    Uses `retry-after` when the provider sends it, else exponential backoff.

    :param headers: headers of the 429 response, or `None`
    :param attempt: 0-based number of the attempt that failed
    """
    retry_after = (headers or {}).get("retry-after")
    seconds = None
    if retry_after is not None:
        try:
            seconds = float(retry_after)
        except ValueError:
            seconds = None
    if seconds is None:
        seconds = 2.0 ** (attempt + 1)
    seconds = min(max(seconds, 1.0), _MAX_BACKOFF_SECONDS)
    return seconds


# #############################################################################
# Usage ledger
# #############################################################################


def get_utc_now() -> datetime.datetime:
    """
    Return the current time in UTC.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    return now


def append_usage(ledger_file: str, record: Dict[str, Any]) -> None:
    """
    Append one API call to the usage ledger.

    :param record: must have `tokens_in` and `tokens_out`; a `ts` (ISO, UTC) is
        added if missing
    """
    record = dict(record)
    record.setdefault("ts", get_utc_now().isoformat(timespec="seconds"))
    hio.create_dir(os.path.dirname(ledger_file) or ".", incremental=True)
    with open(ledger_file, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def get_tokens_in_window(
    ledger_file: str,
    *,
    now: Optional[datetime.datetime] = None,
    hours: float = 24.0,
) -> int:
    """
    Sum the tokens of the ledger calls in the last `hours`.

    :return: tokens in + tokens out, 0 if the ledger does not exist
    """
    if not os.path.exists(ledger_file):
        return 0
    now = now or get_utc_now()
    start = now - datetime.timedelta(hours=hours)
    total = 0
    with open(ledger_file, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ts = datetime.datetime.fromisoformat(record["ts"])
            if ts >= start:
                total += int(record.get("tokens_in") or 0)
                total += int(record.get("tokens_out") or 0)
    return total


def format_status(
    rate_limits: Dict[str, str], *, tokens_24h: int, daily_budget: int
) -> str:
    """
    Return a one-line summary of how close the key is to its limits.
    """
    requests = rate_limits.get("remaining_requests_day", "?")
    limit = rate_limits.get("limit_requests_day", "?")
    tokens_minute = rate_limits.get("remaining_tokens_minute", "?")
    status = (
        f"Groq: {requests}/{limit} requests left today, "
        f"{tokens_minute} tokens left this minute; "
        f"ledger: {tokens_24h:,}/{daily_budget:,} tokens in the last 24h"
    )
    return status
