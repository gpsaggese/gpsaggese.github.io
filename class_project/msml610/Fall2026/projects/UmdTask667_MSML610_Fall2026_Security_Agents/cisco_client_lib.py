"""Cached OpenAI-compatible transport for Cisco's unmodified scan pipeline."""

import hashlib
import json
import logging
import os
from pathlib import Path
import time

import openai

_LOG = logging.getLogger(__name__)


def strip_json_formatting(text):
    """Remove Markdown bold markers outside JSON strings, preserving values."""
    result = []
    quoted = False
    escaped = False
    position = 0
    while position < len(text):
        char = text[position]
        if not quoted and text[position:position + 2] == "**":
            position += 2
            continue
        result.append(char)
        if char == '"' and not escaped:
            quoted = not quoted
        escaped = quoted and char == "\\" and not escaped
        position += 1
    return "".join(result)


class CachedClient:
    """Keep Cisco's prompts while validating and caching model responses."""

    def __init__(self, model=None, base_url=None, api_key=None):
        from llm_client import TokenUsage

        self.model = model or os.environ["LLM_MODEL"]
        self.base_url = base_url or os.environ["LLM_BASE_URL"]
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.usage = TokenUsage()
        self.cache_dir = Path(os.environ["CISCO_CACHE_DIR"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = None

    def is_available(self):
        return bool(self.api_key)

    def chat(self, system_prompt, user_message):
        """Retry rate limits; never let malformed/truncated answers mean safe."""
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0,
            "max_completion_tokens": 4096,
        }
        if self.model.startswith("openai/gpt-oss"):
            request["reasoning_effort"] = "low"
        digest = hashlib.sha256(json.dumps(
            dict(request, base_url=self.base_url), sort_keys=True
        ).encode()).hexdigest()
        cache_file = self.cache_dir / f"{digest}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            self.usage.record(cached["input_tokens"], cached["output_tokens"],
                              self.model)
            return cached["text"], {
                "input_tokens": cached["input_tokens"],
                "output_tokens": cached["output_tokens"],
            }
        if self.client is None:
            self.client = openai.OpenAI(
                base_url=self.base_url, api_key=self.api_key,
                max_retries=0, timeout=90,
            )
        for attempt in range(20):
            try:
                response = self.client.chat.completions.create(**request)
                break
            except openai.RateLimitError as exc:
                message = str(exc).lower()
                if "tokens per day" in message or "requests per day" in message:
                    raise SystemExit(
                        "Provider daily free-tier quota exhausted"
                    )
                if attempt == 19:
                    raise RuntimeError(
                        "Provider rate limit retry budget exhausted"
                    )
                wait = min(60, 10 + attempt * 5)
                _LOG.info("Rate limited; retrying in %s seconds", wait)
                time.sleep(wait)
        choice = response.choices[0]
        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        self.usage.record(tokens_in, tokens_out, self.model)
        text = choice.message.content or ""
        record = {
            "text": text, "input_tokens": tokens_in,
            "output_tokens": tokens_out, "finish_reason": choice.finish_reason,
            "reasoning_effort": request.get("reasoning_effort"),
        }
        # Keep even invalid responses for diagnosis, but never reuse them.
        if choice.finish_reason != "stop":
            cache_file.with_suffix(".invalid.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
            raise RuntimeError("Model response did not finish normally")
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = "\n".join(
                line for line in candidate.splitlines()
                if not line.strip().startswith("```")
            )
        try:
            parsed = json.loads(strip_json_formatting(candidate))
            if not isinstance(parsed, (dict, list)):
                raise ValueError("Expected an object or array")
        except (ValueError, TypeError) as exc:
            cache_file.with_suffix(".invalid.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
            raise RuntimeError(
                "Model response is not valid findings JSON"
            ) from exc
        record["text"] = json.dumps(parsed)
        cache_file.write_text(json.dumps(record, indent=2) + "\n")
        return record["text"], {
            "input_tokens": tokens_in, "output_tokens": tokens_out,
        }
