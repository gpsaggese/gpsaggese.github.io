"""
Test capability baselines and exact degradation decisions.
"""

import research.Noesis.monitor.monitor_detection as rnmondet
import research.Noesis.monitor.monitor_metrics as rnmonmet

import helpers.hunit_test as hunitest


def _window(
    index: int,
    correct: int,
    gradable: int,
) -> rnmonmet.MonitoringWindowMetrics:
    return rnmonmet.MonitoringWindowMetrics(
        window_index=index,
        attempted=gradable,
        returned=gradable,
        gradable=gradable,
        correct=correct,
        incorrect=gradable - correct,
        ungradable=0,
        accuracy=correct / gradable,
        availability=1.0,
        latency_p50_ms=100.0,
        latency_p95_ms=100.0,
        total_cost_usd=0.0,
        cost_per_task_usd=0.0,
        cost_per_correct_task_usd=0.0 if correct else None,
        families=(),
    )


def _identity() -> rnmondet.MonitoringSeriesIdentity:
    return rnmondet.MonitoringSeriesIdentity(
        "offline",
        "model",
        "v1",
        "grader-v1",
        "config-hash",
    )


# #############################################################################
# Test_one_sided_fisher_quality_p_value
# #############################################################################


class Test_one_sided_fisher_quality_p_value(hunitest.TestCase):
    """
    Test exact lower-tail probabilities.
    """

    def test1(self) -> None:
        """
        Test a large current-quality drop has a small exact p-value.
        """
        actual = rnmondet.one_sided_fisher_quality_p_value(285, 15, 36, 24)
        self.assertLess(actual, 0.01)

    def test2(self) -> None:
        """
        Test equal rates do not look like one-sided degradation.
        """
        actual = rnmondet.one_sided_fisher_quality_p_value(285, 15, 57, 3)
        self.assertGreater(actual, 0.01)


# #############################################################################
# Test_evaluate_capability_window
# #############################################################################


class Test_evaluate_capability_window(hunitest.TestCase):
    """
    Test evidence, effect-size, and confirmation gates.
    """

    def test1(self) -> None:
        """
        Test a material drop needs an independent confirmation window.
        """
        policy = rnmondet.CapabilityDetectionPolicy()
        baseline = rnmondet.build_capability_baseline(
            _identity(),
            tuple(_window(index, 57, 60) for index in range(1, 6)),
            policy,
        )
        current = _window(1, 36, 60)
        warning = rnmondet.evaluate_capability_window(baseline, current, policy)
        self.assertEqual(warning.state, rnmondet.CapabilityState.WARNING)
        confirmed = rnmondet.evaluate_capability_window(
            baseline,
            _window(2, 37, 60),
            policy,
            previous_consecutive_flags=warning.consecutive_flags,
        )
        self.assertEqual(confirmed.state, rnmondet.CapabilityState.NERFED)

    def test2(self) -> None:
        """
        Test sparse evidence remains insufficient regardless of its score.
        """
        policy = rnmondet.CapabilityDetectionPolicy()
        baseline = rnmondet.build_capability_baseline(
            _identity(),
            tuple(_window(index, 57, 60) for index in range(1, 6)),
            policy,
        )
        actual = rnmondet.evaluate_capability_window(
            baseline, _window(1, 5, 10), policy
        )
        self.assertEqual(actual.state, rnmondet.CapabilityState.INSUFFICIENT)
