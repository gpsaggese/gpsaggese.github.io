"""
Test standalone-monitor parsing and no-spend validation.
"""

import argparse
import asyncio
from unittest import mock

import helpers.hunit_test as hunitest

import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.monitor.monitor_cli as rnmoncli
import research.Noesis.monitor.monitor_models as rnmonmod


def _canary() -> rnochmod.Canary:
    return rnochmod.Canary(
        "math-v1-001", "v1", "math", "One plus one?", "2", "integer"
    )


class Test_parse_target(hunitest.TestCase):
    """
    Test explicit provider/model target syntax.
    """

    def test1(self) -> None:
        """
        Test a valid target preserves provider and model.
        """
        actual = rnmoncli.parse_target("groq=meta-llama/model")
        expected = rnmonmod.MonitoringTarget("groq", "meta-llama/model")
        self.assertEqual(actual, expected)

    def test2(self) -> None:
        """
        Test malformed targets fail before any provider call.
        """
        for value in ("groq", "=model", "groq="):
            with self.assertRaises(argparse.ArgumentTypeError):
                rnmoncli.parse_target(value)


class Test_run(hunitest.TestCase):
    """
    Test the default command is validation-only.
    """

    def test1(self) -> None:
        """
        Test no database or provider is touched without `--execute`.
        """
        args = rnmoncli._parse_args(
            ["--target", "groq=test-model", "--canary-count", "1"]
        )
        with (
            mock.patch.object(
                rnmoncli.rnochcan,
                "load_canary_bank",
                return_value=(_canary(),),
            ),
            mock.patch.object(rnmoncli.rnonodb, "connect") as connect,
            mock.patch.object(
                rnmoncli.rnogaope, "OpenRouterProvider"
            ) as provider,
        ):
            result = asyncio.run(rnmoncli._run(args))
        self.assertEqual(result, 0)
        connect.assert_not_called()
        provider.assert_not_called()
