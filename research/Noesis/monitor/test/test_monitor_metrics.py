"""
Test source-independent monitoring-window metrics.
"""

from typing import Optional

import helpers.hunit_test as hunitest

import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.monitor.monitor_metrics as rnmonmet
import research.Noesis.monitor.monitor_models as rnmonmod


def _observation(
    canary_id: str,
    correct: Optional[bool],
    *,
    status: str = rnogapro.OK,
    latency_ms: float = 100.0,
    cost_usd: float = 0.01,
) -> rnmonmod.MonitoringObservation:
    target = rnmonmod.MonitoringTarget("offline", "model")
    return rnmonmod.MonitoringObservation(
        canary_id=canary_id,
        target=target,
        provider_status=status,
        correct=correct,
        grader_reason="TEST",
        grader_version="v1",
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


# #############################################################################
# Test_aggregate_monitoring_window
# #############################################################################


class Test_aggregate_monitoring_window(hunitest.TestCase):
    """
    Test capability, operations, cost, and family aggregation.
    """

    def test1(self) -> None:
        """
        Test mixed observations preserve separate evidence dimensions.
        """
        observations = (
            _observation("math-1", True, latency_ms=80.0),
            _observation("math-2", False, latency_ms=100.0),
            _observation("code-1", None, latency_ms=120.0),
            _observation(
                "code-2",
                None,
                status=rnogapro.UPSTREAM_ERROR,
                cost_usd=0.0,
            ),
        )
        families = {
            "math-1": "math",
            "math-2": "math",
            "code-1": "code",
            "code-2": "code",
        }
        actual = rnmonmet.aggregate_monitoring_window(observations, families, 1)
        self.assertEqual(actual.attempted, 4)
        self.assertEqual(actual.returned, 3)
        self.assertEqual(actual.gradable, 2)
        self.assertEqual(actual.correct, 1)
        self.assertEqual(actual.incorrect, 1)
        self.assertEqual(actual.ungradable, 2)
        self.assertEqual(actual.accuracy, 0.5)
        self.assertEqual(actual.availability, 0.75)
        self.assertEqual(actual.latency_p50_ms, 100.0)
        self.assertEqual(actual.latency_p95_ms, 120.0)
        self.assertEqual(len(actual.families), 2)

    def test2(self) -> None:
        """
        Test zero correct tasks never divide by zero.
        """
        observations = (_observation("math-1", False),)
        actual = rnmonmet.aggregate_monitoring_window(
            observations, {"math-1": "math"}, 1
        )
        self.assertIsNone(actual.cost_per_correct_task_usd)
