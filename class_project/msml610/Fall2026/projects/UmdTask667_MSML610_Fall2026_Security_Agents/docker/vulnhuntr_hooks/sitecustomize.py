"""
Instrument the OpenAI client inside the Vulnhuntr container.

Python imports `sitecustomize` at startup when this dir is on `PYTHONPATH`
(`run_vulnhuntr.py` mounts it at `/hooks`). Vulnhuntr's own code is unchanged;
only the HTTP client of `openai.OpenAI` is wrapped to:
- Append one JSON line per API response to `$VULNHUNTR_CALL_LOG` with the
  status, rate limit headers, token usage, and the error message of a failure
- Retry per-minute rate limits up to `$OPENAI_MAX_RETRIES` times (the `openai`
  client honors `retry-after`)
- Not retry a daily limit or a request too large for the per-minute limit,
  since waiting cannot fix either (sets `x-should-retry: false`)

The API key is never logged: only response headers and bodies are read.
"""

import json
import os
import time

try:
    import httpx
    import openai
except ImportError:  # pragma: no cover
    httpx = None
    openai = None

_HEADERS = (
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-reset-tokens",
    "retry-after",
)


def _is_daily_limit(message: str) -> bool:
    text = message.lower()
    return "per day" in text or "(tpd)" in text or "(rpd)" in text


if httpx is not None and openai is not None:

    class _LoggingTransport(httpx.HTTPTransport):
        """
        Transport that logs each response before the `openai` client sees it.
        """

        def handle_request(self, request: "httpx.Request") -> "httpx.Response":
            response = super().handle_request(request)
            response.read()
            record = {"ts": time.time(), "status": response.status_code}
            for header in _HEADERS:
                value = response.headers.get(header)
                if value is not None:
                    record[header] = value
            body = {}
            try:
                body = json.loads(response.content or b"{}")
            except ValueError:
                pass
            if not isinstance(body, dict):
                body = {}
            usage = body.get("usage") or {}
            record["tokens_in"] = int(usage.get("prompt_tokens") or 0)
            record["tokens_out"] = int(usage.get("completion_tokens") or 0)
            if response.status_code >= 400:
                error = body.get("error") or {}
                message = str(error.get("message", "")) if isinstance(error, dict) else ""
                record["error"] = message[:500]
                record["daily_limit"] = _is_daily_limit(message)
                record["too_large"] = (
                    response.status_code == 413
                    or "request too large" in message.lower()
                )
                if record["daily_limit"] or record["too_large"]:
                    response.headers["x-should-retry"] = "false"
            log_file = os.environ.get("VULNHUNTR_CALL_LOG", "/out/groq_calls.jsonl")
            with open(log_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            return response

    _original_init = openai.OpenAI.__init__

    def _init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault(
            "max_retries", int(os.environ.get("OPENAI_MAX_RETRIES", "8"))
        )
        kwargs.setdefault(
            "http_client", openai.DefaultHttpxClient(transport=_LoggingTransport())
        )
        _original_init(self, *args, **kwargs)

    openai.OpenAI.__init__ = _init
