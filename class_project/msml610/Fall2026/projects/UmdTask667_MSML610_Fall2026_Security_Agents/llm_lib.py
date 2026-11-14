"""
Ask an LLM for security findings through any OpenAI-compatible API.

Used by `runners/run_plain_llm.py` (one fixed prompt, no agent loop) and by the
ensemble agent later. Every response is cached on disk, so a run that stops on
a rate limit can be restarted without paying for the same request twice.

Import as:

import llm_lib as llmlib
"""

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import openai

import helpers.hdbg as hdbg
import helpers.hio as hio

import findings_lib as finlib
import rate_limit_lib as ratlib

_LOG = logging.getLogger(__name__)

# #############################################################################
# Prompt
# #############################################################################

SYSTEM_PROMPT = """You are a senior application security reviewer.
You review one Python file at a time and report security vulnerabilities.
Report only real, exploitable weaknesses. Do not report style issues.
If the file has no vulnerability, report an empty list."""

USER_PROMPT_TEMPLATE = """Review the Python file below. Each line starts with its
line number followed by `|`.

Answer with one JSON object and nothing else, in this format:
{{"findings": [{{"line": <int>, "cwe": <int>, "severity": "critical|high|medium|low",
"confidence": <float 0-1>, "explanation": "<why it is exploitable>",
"fix": "<how to fix it>"}}]}}

Use the line of the statement that a fix must change.

File:
{numbered_code}"""


def number_lines(code: str) -> str:
    """
    Prefix each line with its 1-based number, e.g., `  3| import os`.
    """
    lines = code.rstrip("\n").split("\n")
    width = len(str(len(lines)))
    numbered = "\n".join(
        f"{idx + 1:>{width}}| {line}" for idx, line in enumerate(lines)
    )
    return numbered


def build_messages(code: str) -> List[Dict[str, str]]:
    """
    Build the chat messages asking for findings on one file.
    """
    user = USER_PROMPT_TEMPLATE.format(numbered_code=number_lines(code))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return messages


def get_prompt_version() -> str:
    """
    Return a short hash of the prompt text, stored with each run.
    """
    txt = SYSTEM_PROMPT + USER_PROMPT_TEMPLATE
    version = hashlib.sha1(txt.encode("utf-8")).hexdigest()[:8]
    return version


# #############################################################################
# Parsing
# #############################################################################


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Return the first JSON object in a model answer, or `None`.

    Models sometimes wrap the object in a code fence or add a sentence.
    """
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    if not candidate.strip():
        return None
    obj = json.loads(candidate) if _is_json(candidate) else None
    return obj


def _is_json(text: str) -> bool:
    """
    Return whether `text` parses as JSON.
    """
    decoder = json.JSONDecoder()
    is_valid = True
    # `raw_decode` reports the end of the first value; anything else is junk.
    end = 0
    try:
        _, end = decoder.raw_decode(text)
    except json.JSONDecodeError:
        is_valid = False
    is_valid = is_valid and text[end:].strip() == ""
    return is_valid


def parse_findings(
    text: str, *, case_file: str, agent: str
) -> Optional[List[Dict[str, Any]]]:
    """
    Convert a model answer into findings.

    :param text: raw model answer
    :param case_file: file the answer is about
    :param agent: name of the agent
    :return: findings, or `None` if the answer is not valid JSON in the
        expected format
    """
    obj = _extract_json(text)
    if obj is None or not isinstance(obj.get("findings"), list):
        return None
    findings = []
    for item in obj["findings"]:
        if not isinstance(item, dict) or not str(item.get("line", "")).isdigit():
            _LOG.warning("Skipping finding without a line in '%s'", case_file)
            continue
        severity = str(item.get("severity", "")).lower()
        if severity not in finlib.SEVERITIES:
            severity = ""
        finding = finlib.make_finding(
            case_file=case_file,
            agent=agent,
            line=int(item["line"]),
            cwe=finlib.extract_cwe([f"CWE-{item.get('cwe')}", str(item.get("cwe"))]),
            severity=severity,
            confidence=str(item.get("confidence", "")),
            explanation=str(item.get("explanation", "")),
            fix=str(item.get("fix", "")),
        )
        findings.append(finding)
    return findings


# #############################################################################
# Calls
# #############################################################################


def get_prompt_sha(model: str, messages: List[Dict[str, str]], temperature: float) -> str:
    """
    Return a hash of everything that determines a model answer.

    Stored in each cached response, so that editing a snippet or the prompt
    never reuses a stale answer.
    """
    txt = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature},
        sort_keys=True,
    )
    sha = hashlib.sha1(txt.encode("utf-8")).hexdigest()
    return sha


def is_cached(
    cache_file: str,
    *,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
) -> bool:
    """
    Return whether `cache_file` holds an answer to exactly this request.

    Cache files written before the hash was stored are trusted.
    """
    if not os.path.exists(cache_file):
        return False
    cached = hio.from_json(cache_file)
    stored_sha = cached.get("prompt_sha")
    if stored_sha is None:
        return True
    is_same = stored_sha == get_prompt_sha(model, messages, temperature)
    if not is_same:
        _LOG.warning("Ignoring stale cache '%s': the request changed", cache_file)
    return is_same


def _create_completion(client: Any, params: Dict[str, Any]) -> Tuple[Any, Dict[str, str]]:
    """
    Call the chat API and return the completion and its response headers.

    Uses `with_raw_response` when the client has it, to read the headers.
    """
    raw_api = getattr(client.chat.completions, "with_raw_response", None)
    if raw_api is None:
        return client.chat.completions.create(**params), {}
    raw = raw_api.create(**params)
    return raw.parse(), dict(raw.headers)


def call_with_cache(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, str]],
    cache_file: str,
    temperature: float = 0.0,
    max_retries: int = 8,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    """
    Call a chat model, or return the cached response if it exists.

    Rate limits:
    - A per-minute limit (HTTP 429) is retried after `retry-after` seconds, up
      to `max_retries` times; server errors are retried with backoff
    - A daily limit raises `rate_limit_lib.DailyLimitError` at once, so that the
      caller can stop and resume from the cache later
    Create the client with `max_retries=0`, so that only this loop retries.

    :param client: `openai.OpenAI` client, or any object with the same API
    :param cache_file: JSON file storing the response
    :param sleep_fn: function used to wait, replaced in tests
    :return: dict with `text`, `tokens_in`, `tokens_out`, `seconds`, `cached`,
        and `rate_limits` (headers of the last response, empty if cached)
    """
    if is_cached(cache_file, model=model, messages=messages, temperature=temperature):
        response = hio.from_json(cache_file)
        response["cached"] = True
        response["rate_limits"] = {}
        return response
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    start = time.time()
    for attempt in range(max_retries + 1):
        try:
            completion, headers = _create_completion(client, params)
            break
        except openai.RateLimitError as e:
            message = str(e)
            if ratlib.is_daily_limit(message):
                raise ratlib.DailyLimitError(message) from e
            error_headers = e.response.headers if e.response is not None else None
            if attempt == max_retries:
                raise
            wait = ratlib.get_retry_seconds(error_headers, attempt)
            _LOG.warning(
                "Per-minute rate limit, retry %s/%s in %.0fs",
                attempt + 1,
                max_retries,
                wait,
            )
            sleep_fn(wait)
        except (openai.InternalServerError, openai.APIConnectionError) as e:
            if attempt == max_retries:
                raise
            wait = ratlib.get_retry_seconds(None, attempt)
            _LOG.warning("%s, retry in %.0fs", type(e).__name__, wait)
            sleep_fn(wait)
    usage = completion.usage
    response = {
        "text": completion.choices[0].message.content or "",
        "tokens_in": int(getattr(usage, "prompt_tokens", 0) or 0),
        "tokens_out": int(getattr(usage, "completion_tokens", 0) or 0),
        "seconds": round(time.time() - start, 2),
        "model": model,
        "prompt_sha": get_prompt_sha(model, messages, temperature),
    }
    hio.create_dir(os.path.dirname(cache_file), incremental=True)
    hio.to_json(cache_file, response)
    response["cached"] = False
    response["rate_limits"] = ratlib.extract_rate_limits(headers)
    hdbg.dassert_isinstance(response["text"], str)
    return response
