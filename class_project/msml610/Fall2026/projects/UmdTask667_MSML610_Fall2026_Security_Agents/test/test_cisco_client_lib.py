"""Validate cached transport with local fake API responses."""

import json
from types import SimpleNamespace

import pytest

import cisco_client_lib as clientlib
from cisco_client_lib import CachedClient, strip_json_formatting


def make_client(tmp_path, text, finish_reason="stop"):
    client = object.__new__(CachedClient)
    client.model = "openai/gpt-oss-120b"
    client.base_url = "https://example.invalid/v1"
    client.cache_dir = tmp_path
    records = []
    requests = []
    client.usage = SimpleNamespace(record=lambda *args: records.append(args))

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=text),
            )],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
        )

    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client, records, requests


def test_valid_response_is_cached_and_reused(tmp_path):
    client, records, requests = make_client(tmp_path, '```json\n[]\n```')
    assert client.chat("system", "code")[0] == "[]"
    assert client.chat("system", "code")[0] == "[]"
    assert len(requests) == 1
    assert len(records) == 2
    assert requests[0]["reasoning_effort"] == "low"
    assert len(list(tmp_path.glob("*.json"))) == 1


@pytest.mark.parametrize("text, reason", [
    ('[{"description":', "length"),
    ("invalid json", "stop"),
    ('"not a finding object"', "stop"),
])
def test_bad_response_cannot_be_a_clean_verdict(tmp_path, text, reason):
    client, _, _ = make_client(tmp_path, text, reason)
    with pytest.raises(RuntimeError):
        client.chat("system", "code")
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert files[0].name.endswith(".invalid.json")
    assert json.loads(files[0].read_text())["finish_reason"] == reason


def test_markdown_cleanup_preserves_quoted_values():
    text = '{**"score": 5.3**, "text": "**bold** and \\"quote\\""}'
    parsed = json.loads(strip_json_formatting(text))
    assert parsed == {"score": 5.3, "text": '**bold** and "quote"'}


def test_daily_quota_stops_instead_of_retrying(tmp_path, monkeypatch):
    class QuotaError(Exception):
        pass

    client, _, requests = make_client(tmp_path, "[]")

    def exhausted(**kwargs):
        requests.append(kwargs)
        raise QuotaError("tokens per day (TPD) exceeded")

    monkeypatch.setattr(clientlib.openai, "RateLimitError", QuotaError)
    client.client.chat.completions.create = exhausted
    with pytest.raises(SystemExit, match="daily free-tier quota exhausted"):
        client.chat("system", "code")
    assert len(requests) == 1
