"""
Test command-line parsing and no-spend calibration validation.
"""

import argparse
import asyncio
from unittest import mock

import helpers.hunit_test as hunitest
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_runtime as rnochrun
import research.Noesis.checker.checker_sink as rnochsink
import research.Noesis.prober.prober_cli as rnoprcli
import research.Noesis.prober.prober_runtime as rnoprrun
import research.Noesis.prober.prober_worker as rnoprwor


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


class Test_run_probe(hunitest.TestCase):
    """
    Test probe command composition without live provider or database calls.
    """

    def test1(self) -> None:
        """
        Test `--once` runs the combined probe/checker cycle with a mock sink.
        """
        args = rnoprcli._parse_args(["probe", "--once"])
        settings = mock.Mock(
            database_url="postgresql://test",
            model_id="test-model",
            max_tokens_cap=64,
            timeout_multiplier=2.0,
        )
        provider = mock.Mock()
        provider.aclose = mock.AsyncMock()
        pool = mock.Mock()
        pool.close = mock.AsyncMock()
        cycle = rnoprwor.ProbeCheckerCycleResult(
            rnoprrun.ProbeCycleResult((), ()),
            rnochrun.CheckerPassResult((), (), ()),
        )
        run_cycle = mock.AsyncMock(return_value=cycle)
        with (
            mock.patch.object(
                rnoprcli,
                "_combined_environment",
                return_value={
                    "OPENROUTER_API_KEY": "test-key",
                    "NOESIS_CANARY_SCHEDULE_SECRET": "test-secret",
                },
            ),
            mock.patch.object(
                rnoprcli.rnonoset,
                "load_settings",
                return_value=settings,
            ),
            mock.patch.object(
                rnoprcli.rnogaope,
                "OpenRouterProvider",
                return_value=provider,
            ),
            mock.patch.object(
                rnoprcli.rnonodb,
                "connect",
                new=mock.AsyncMock(return_value=pool),
            ),
            mock.patch.object(
                rnoprcli.rnonodb,
                "apply_migrations",
                new=mock.AsyncMock(),
            ),
            mock.patch.object(
                rnoprcli.rnoprwor,
                "run_probe_cycle",
                new=run_cycle,
            ),
        ):
            actual = asyncio.run(rnoprcli._run_probe(args))
        self.assertEqual(actual, 0)
        self.assertEqual(run_cycle.await_count, 1)
        self.assertIsInstance(
            run_cycle.await_args.kwargs["sink"],
            rnochsink.RecordingVerdictSink,
        )
        pool.close.assert_awaited_once()
        provider.aclose.assert_awaited_once()
