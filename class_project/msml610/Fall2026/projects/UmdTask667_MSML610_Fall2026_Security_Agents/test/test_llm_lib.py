import json
import logging
import os
import types

import httpx
import openai

import helpers.hunit_test as hunitest

import llm_lib as llmlib
import rate_limit_lib as ratlib

_LOG = logging.getLogger(__name__)


class _FakeClient:
    """
    Stand-in for `openai.OpenAI` that returns a fixed answer and counts calls.
    """

    def __init__(self, text: str) -> None:
        self.n_calls = 0
        self._text = text
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.n_calls += 1
        message = types.SimpleNamespace(content=self._text)
        usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=20)
        completion = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)], usage=usage
        )
        return completion


def _rate_limit_error(message: str) -> openai.RateLimitError:
    """
    Build the error the `openai` client raises on an HTTP 429.
    """
    request = httpx.Request("POST", "https://api.example.com/v1/chat")
    response = httpx.Response(
        429, headers={"retry-after": "3"}, request=request
    )
    error = openai.RateLimitError(message, response=response, body=None)
    return error


class _FlakyClient(_FakeClient):
    """
    Fake client that raises the given errors before answering.
    """

    def __init__(self, text: str, errors: list) -> None:
        super().__init__(text)
        self._errors = list(errors)

    def _create(self, **kwargs):  # type: ignore[no-untyped-def]
        if self._errors:
            self.n_calls += 1
            raise self._errors.pop(0)
        return super()._create(**kwargs)


# #############################################################################
# Test_number_lines
# #############################################################################


class Test_number_lines(hunitest.TestCase):
    """
    Test prefixing code lines with their numbers.
    """

    def test1(self) -> None:
        """
        Test that numbers are right-aligned to the widest number.
        """
        # Prepare inputs.
        code = "\n".join(f"x = {i}" for i in range(10)) + "\n"
        # Run test.
        actual = llmlib.number_lines(code).split("\n")
        # Check outputs.
        self.assertEqual(actual[0], " 1| x = 0")
        self.assertEqual(actual[9], "10| x = 9")


# #############################################################################
# Test_parse_findings
# #############################################################################


class Test_parse_findings(hunitest.TestCase):
    """
    Test converting model answers into findings.
    """

    def test1(self) -> None:
        """
        Test a fenced answer with a CWE given as a string.
        """
        # Prepare inputs.
        text = """```json
        {"findings": [{"line": 7, "cwe": "CWE-89", "severity": "High",
                       "confidence": 0.9, "explanation": "x", "fix": "y"}]}
        ```"""
        # Run test.
        findings = llmlib.parse_findings(text, case_file="case_a.py", agent="llm")
        # Check outputs.
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["line"], 7)
        self.assertEqual(findings[0]["cwe"], 89)
        self.assertEqual(findings[0]["severity"], "high")

    def test2(self) -> None:
        """
        Test that an empty list means no findings, not a parse error.
        """
        findings = llmlib.parse_findings(
            '{"findings": []}', case_file="case_a.py", agent="llm"
        )
        self.assertEqual(findings, [])

    def test3(self) -> None:
        """
        Test that an answer that is not JSON is reported as `None`.
        """
        findings = llmlib.parse_findings(
            "The code looks fine to me.", case_file="case_a.py", agent="llm"
        )
        self.assertIsNone(findings)


# #############################################################################
# Test_call_with_cache
# #############################################################################


class Test_call_with_cache(hunitest.TestCase):
    """
    Test that responses are cached on disk.
    """

    def test1(self) -> None:
        """
        Test that the second call reads the cache instead of the API.
        """
        # Prepare inputs.
        client = _FakeClient('{"findings": []}')
        cache_file = os.path.join(self.get_scratch_space(), "case_a.py.json")
        messages = llmlib.build_messages("x = 1\n")
        # Run test.
        first = llmlib.call_with_cache(
            client, model="m", messages=messages, cache_file=cache_file
        )
        second = llmlib.call_with_cache(
            client, model="m", messages=messages, cache_file=cache_file
        )
        # Check outputs.
        self.assertEqual(client.n_calls, 1)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(second["tokens_in"], 100)

    def test2(self) -> None:
        """
        Test that a per-minute 429 is retried after `retry-after`.
        """
        # Prepare inputs.
        error = _rate_limit_error("Limit reached on tokens per minute (TPM)")
        client = _FlakyClient('{"findings": []}', [error])
        cache_file = os.path.join(self.get_scratch_space(), "case_a.py.json")
        waits = []
        # Run test.
        response = llmlib.call_with_cache(
            client,
            model="m",
            messages=llmlib.build_messages("x = 1\n"),
            cache_file=cache_file,
            sleep_fn=waits.append,
        )
        # Check outputs.
        self.assertEqual(client.n_calls, 2)
        self.assertEqual(waits, [3.0])
        self.assertFalse(response["cached"])

    def test3(self) -> None:
        """
        Test that a daily 429 stops at once and caches nothing.
        """
        # Prepare inputs.
        error = _rate_limit_error("Limit reached on tokens per day (TPD)")
        client = _FlakyClient('{"findings": []}', [error, error])
        cache_file = os.path.join(self.get_scratch_space(), "case_a.py.json")
        # Run test.
        with self.assertRaises(ratlib.DailyLimitError):
            llmlib.call_with_cache(
                client,
                model="m",
                messages=llmlib.build_messages("x = 1\n"),
                cache_file=cache_file,
                sleep_fn=lambda _: None,
            )
        # Check outputs.
        self.assertEqual(client.n_calls, 1)
        self.assertFalse(os.path.exists(cache_file))

    def test4(self) -> None:
        """
        Test that editing the snippet invalidates its cached answer.
        """
        # Prepare inputs.
        client = _FakeClient('{"findings": []}')
        cache_file = os.path.join(self.get_scratch_space(), "case_a.py.json")
        old = llmlib.build_messages("x = 1\n")
        new = llmlib.build_messages("x = 2\n")
        # Run test.
        llmlib.call_with_cache(client, model="m", messages=old, cache_file=cache_file)
        second = llmlib.call_with_cache(
            client, model="m", messages=new, cache_file=cache_file
        )
        # Check outputs.
        self.assertEqual(client.n_calls, 2)
        self.assertFalse(second["cached"])
