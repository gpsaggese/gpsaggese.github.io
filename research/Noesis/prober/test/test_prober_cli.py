"""
Test command-line parsing and no-spend calibration validation.
"""

import argparse
import asyncio
from unittest import mock

import helpers.hunit_test as hunitest

import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.prober.prober_cli as rnoprcli


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
        Test a valid target preserves provider, model, and role.
        """
        actual = rnoprcli.parse_target(
            "groq=meta-llama/model", rnochmod.CalibrationRole.HEALTHY
        )
        expected = rnochmod.CalibrationTarget(
            rnochmod.CalibrationRole.HEALTHY,
            "groq",
            "meta-llama/model",
        )
        self.assertEqual(actual, expected)

    def test2(self) -> None:
        """
        Test malformed targets fail before any provider call.
        """
        for value in ("groq", "=model", "groq="):
            with self.assertRaises(argparse.ArgumentTypeError):
                rnoprcli.parse_target(value, rnochmod.CalibrationRole.HEALTHY)


class Test_run_calibrate(hunitest.TestCase):
    """
    Test the default calibration mode is validation-only.
    """

    def test1(self) -> None:
        """
        Test no database or provider is touched without `--execute`.
        """
        args = rnoprcli._parse_args(
            [
                "calibrate",
                "--healthy",
                "groq=healthy-model",
                "--weak",
                "deepinfra=weak-model",
                "--minimum-survivors",
                "1",
            ]
        )
        with (
            mock.patch.object(
                rnoprcli.rnochcan,
                "load_canary_bank",
                return_value=(_canary(),),
            ),
            mock.patch.object(rnoprcli.rnonodb, "connect") as connect,
            mock.patch.object(
                rnoprcli.rnogaope, "OpenRouterProvider"
            ) as provider,
        ):
            result = asyncio.run(rnoprcli._run_calibrate(args))
        self.assertEqual(result, 0)
        connect.assert_not_called()
        provider.assert_not_called()
