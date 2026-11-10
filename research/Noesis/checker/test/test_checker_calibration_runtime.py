"""
Test provider execution and persistence for live canary calibration.
"""

import asyncio
from typing import Tuple

import helpers.hunit_test as hunitest

import research.Noesis.checker.checker_calibration as rnochcal
import research.Noesis.checker.checker_calibration_runtime as rnochcaru
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_store as rnochsto
import research.Noesis.common.test.noesis_test_utils as rnctnteut
import research.Noesis.gateway.gateway_fake_provider as rnogafapr
import research.Noesis.gateway.gateway_providers as rnogapro


def _canary() -> rnochmod.Canary:
    return rnochmod.Canary(
        "math-v1-001",
        "v1",
        "math",
        "What is one plus one? Return only the integer.",
        "2",
        "integer",
        1,
    )


def _targets() -> Tuple[rnochmod.CalibrationTarget, ...]:
    return (
        rnochmod.CalibrationTarget(
            rnochmod.CalibrationRole.HEALTHY, "groq", "healthy-model"
        ),
        rnochmod.CalibrationTarget(
            rnochmod.CalibrationRole.HEALTHY, "together", "healthy-model"
        ),
        rnochmod.CalibrationTarget(
            rnochmod.CalibrationRole.WEAK, "weak", "weak-model"
        ),
    )


def _provider() -> rnogafapr.FakeProvider:
    return rnogafapr.FakeProvider(
        {
            "groq": rnogafapr.FakeBehavior(
                "Groq", latency_s=0.2, answer=lambda messages: "2"
            ),
            "together": rnogafapr.FakeBehavior(
                "Together", latency_s=0.3, answer=lambda messages: "2"
            ),
            "weak": rnogafapr.FakeBehavior(
                "Weak", latency_s=0.4, answer=lambda messages: "3"
            ),
        }
    )


# #############################################################################
# Test_execute_calibration
# #############################################################################


class Test_execute_calibration(hunitest.TestCase):
    """
    Test provider calls, deterministic grading, and reporting.
    """

    def test1(self) -> None:
        """
        Test healthy successes and weak misses accept one canary.
        """
        provider = _provider()
        policy = rnochmod.CalibrationPolicy(repetitions=3, concurrency=2)
        execution = asyncio.run(
            rnochcaru.execute_calibration(
                provider, (_canary(),), _targets(), policy
            )
        )
        self.assertEqual(len(provider.calls), 9)
        self.assertTrue(execution.decisions[0].accepted)
        self.assertEqual(execution.decisions[0].healthy_accuracy, 1.0)
        self.assertEqual(execution.decisions[0].weak_accuracy, 0.0)
        self.assertEqual(execution.summary["accepted_count"], 1)
        self.assertEqual(execution.summary["total_calls"], 9)
        for target_stat in execution.summary["target_stats"]:
            expected_accuracy = 0.0 if target_stat["role"] == "WEAK" else 1.0
            self.assertEqual(target_stat["survivor_accuracy"], expected_accuracy)
        self.assertEqual(
            execution.summary["reason_counts"], {rnochcal.CALIBRATED: 1}
        )
        first_call = provider.calls[0][1]
        self.assertEqual(first_call.temperature, 0.0)
        self.assertEqual(first_call.max_tokens, 64)
        self.assertEqual(
            first_call.messages,
            [{"role": "user", "content": _canary().prompt}],
        )

    def test2(self) -> None:
        """
        Test a provider failure is persisted as ungradable, not incorrect.
        """
        provider = _provider()
        provider.behaviors["weak"].fail_rate = 1.0
        provider.behaviors["weak"].fail_status = rnogapro.GATEWAY_ERROR
        policy = rnochmod.CalibrationPolicy(repetitions=1)
        execution = asyncio.run(
            rnochcaru.execute_calibration(
                provider, (_canary(),), _targets(), policy
            )
        )
        weak = [
            row
            for row in execution.observations
            if row.target.role == rnochmod.CalibrationRole.WEAK
        ]
        self.assertEqual(weak[0].provider_status, rnogapro.GATEWAY_ERROR)
        self.assertIsNone(weak[0].correct)
        self.assertEqual(
            execution.decisions[0].reason,
            rnochcal.INCOMPLETE_OBSERVATIONS,
        )


# #############################################################################
# Test_run_calibration
# #############################################################################


class Test_run_calibration(hunitest.TestCase):
    """
    Test one fake-provider run through the PostgreSQL persistence boundary.
    """

    def test1(self) -> None:
        """
        Test the run, raw attempts, summary, and decision are persisted.
        """

        async def _body(pool) -> None:
            policy = rnochmod.CalibrationPolicy(repetitions=2)
            run_id, execution = await rnochcaru.run_calibration(
                pool, _provider(), (_canary(),), _targets(), policy
            )
            rows = await rnctnteut.fetch_all(
                pool,
                "SELECT state, repetitions, summary->>'accepted_count',"
                " configuration->>'grader_version'"
                " FROM canary_calibration_runs WHERE run_id = %s",
                run_id,
            )
            self.assertEqual(rows[0], ("COMPLETED", 2, "1", "m2-v1"))
            rows = await rnctnteut.fetch_all(
                pool,
                "SELECT count(*) FROM canary_calibration_results"
                " WHERE run_id = %s",
                run_id,
            )
            self.assertEqual(rows[0][0], 6)
            rows = await rnctnteut.fetch_all(
                pool,
                "SELECT status, calibrated, healthy_acc, weak_acc"
                " FROM canaries WHERE canary_id = %s",
                _canary().canary_id,
            )
            self.assertEqual(rows[0], ("CALIBRATED", True, 1.0, 0.0))
            self.assertTrue(execution.decisions[0].accepted)
            activated = await rnochsto.activate_calibrated_bank(
                pool, run_id, "v1", 1
            )
            self.assertEqual(activated, 1)
            active = await rnochsto.load_active_canaries(pool, "v1")
            self.assertEqual(active, (_canary(),))

        rnctnteut.run_with_clean_db(self, _body)
