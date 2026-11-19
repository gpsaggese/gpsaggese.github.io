import datetime
import json
import logging
import os

import helpers.hunit_test as hunitest

import rate_limit_lib as ratlib

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_extract_rate_limits
# #############################################################################


class Test_extract_rate_limits(hunitest.TestCase):
    """
    Test reading Groq rate limit headers.
    """

    def test1(self) -> None:
        """
        Test that known headers are renamed and others are ignored.
        """
        # Prepare inputs.
        headers = {
            "x-ratelimit-remaining-requests": "998",
            "x-ratelimit-remaining-tokens": "7200",
            "content-type": "application/json",
        }
        # Run test.
        actual = ratlib.extract_rate_limits(headers)
        # Check outputs.
        expected = {
            "remaining_requests_day": "998",
            "remaining_tokens_minute": "7200",
        }
        self.assertEqual(actual, expected)


# #############################################################################
# Test_is_daily_limit
# #############################################################################


class Test_is_daily_limit(hunitest.TestCase):
    """
    Test telling daily limits from per-minute limits.
    """

    def test1(self) -> None:
        """
        Test Groq-style messages for TPD, RPD, and TPM.
        """
        # Prepare inputs.
        messages = [
            "Rate limit reached on tokens per day (TPD): Limit 200000",
            "Rate limit reached on requests per day (RPD): Limit 1000",
            "Rate limit reached on tokens per minute (TPM): Limit 8000",
        ]
        # Run test.
        actual = [ratlib.is_daily_limit(message) for message in messages]
        # Check outputs.
        self.assertEqual(actual, [True, True, False])


# #############################################################################
# Test_get_retry_seconds
# #############################################################################


class Test_get_retry_seconds(hunitest.TestCase):
    """
    Test the wait before retrying a per-minute limit.
    """

    def test1(self) -> None:
        """
        Test `retry-after`, backoff without it, and the cap.
        """
        # Run test.
        actual = [
            ratlib.get_retry_seconds({"retry-after": "7"}, 0),
            ratlib.get_retry_seconds(None, 2),
            ratlib.get_retry_seconds({"retry-after": "900"}, 0),
        ]
        # Check outputs.
        self.assertEqual(actual, [7.0, 8.0, 60.0])


# #############################################################################
# Test_get_tokens_in_window
# #############################################################################


class Test_get_tokens_in_window(hunitest.TestCase):
    """
    Test summing ledger tokens over the last 24 hours.
    """

    def test1(self) -> None:
        """
        Test that calls older than the window are left out.
        """
        # Prepare inputs.
        now = datetime.datetime(2026, 11, 1, 12, tzinfo=datetime.timezone.utc)
        ledger = os.path.join(self.get_scratch_space(), "usage.jsonl")
        for hours_ago, tokens in [(30, 5000), (10, 1000), (1, 200)]:
            ts = (now - datetime.timedelta(hours=hours_ago)).isoformat()
            ratlib.append_usage(
                ledger, {"ts": ts, "tokens_in": tokens, "tokens_out": tokens}
            )
        # Run test.
        actual = ratlib.get_tokens_in_window(ledger, now=now)
        # Check outputs.
        self.assertEqual(actual, 2400)

    def test2(self) -> None:
        """
        Test that a missing ledger counts as no usage.
        """
        # Run test.
        actual = ratlib.get_tokens_in_window("does/not/exist.jsonl")
        # Check outputs.
        self.assertEqual(actual, 0)
