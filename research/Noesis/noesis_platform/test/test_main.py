"""
Import as:

import research.Noesis.noesis_platform.test.test_main as rnttmain
"""

import logging
import os
from unittest import mock

import pytest

# `fastapi` is only needed to actually build `main.py`'s module-level `app`
# (imported transitively via `platform_api.py`); skip cleanly instead of
# failing collection where it is not installed, matching
# `test_platform_api.py`.
pytest.importorskip("fastapi")

import helpers.hunit_test as hunitest  # noqa: E402 # pylint: disable=wrong-import-position
import research.Noesis.noesis_platform.main as rnoemain  # noqa: E402 # pylint: disable=wrong-import-position

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test__parse_api_keys
# #############################################################################


class Test__parse_api_keys(hunitest.TestCase):
    """
    Test `main._parse_api_keys()`.
    """

    def test1(self) -> None:
        """
        Test that a well-formed string parses into the expected map.
        """
        # Prepare inputs.
        raw = "key1:acct1,key2:acct2"
        # Prepare outputs.
        expected = {"key1": "acct1", "key2": "acct2"}
        # Run test.
        actual = rnoemain._parse_api_keys(raw)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test2(self) -> None:
        """
        Test that an empty string parses to an empty map.
        """
        # Prepare inputs.
        raw = ""
        # Prepare outputs.
        expected = {}
        # Run test.
        actual = rnoemain._parse_api_keys(raw)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test3(self) -> None:
        """
        Test that a malformed entry (missing `:`) raises `AssertionError`.
        """
        # Prepare inputs.
        raw = "key1acct1"
        # Run test and check output.
        with self.assertRaises(AssertionError):
            rnoemain._parse_api_keys(raw)


# #############################################################################
# Test__get_db_backend
# #############################################################################


class Test__get_db_backend(hunitest.TestCase):
    """
    Test `main._get_db_backend()`.
    """

    def test1(self) -> None:
        """
        Test that `NOESIS_DB_BACKEND` defaults to `"memory"` when unset.
        """
        # Prepare inputs.
        env = dict(os.environ)
        env.pop("NOESIS_DB_BACKEND", None)
        # Prepare outputs.
        expected = "memory"
        # Run test.
        with mock.patch.dict(os.environ, env, clear=True):
            actual = rnoemain._get_db_backend()
        # Check outputs.
        self.assert_equal(actual, expected)
