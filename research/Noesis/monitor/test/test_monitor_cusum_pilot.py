"""Test the deterministic preliminary Bernoulli-CUSUM pilot."""

import research.Noesis.monitor.monitor_capability_experiment as rnmoncaex
import research.Noesis.monitor.monitor_cusum_pilot as rnmoncupi

import helpers.hunit_test as hunitest


class Test_cusum_alarm_query(hunitest.TestCase):
    """Test the sequential detector directly."""

    def test1(self) -> None:
        """Test an all-correct stream does not trigger a lower-tail alarm."""
        actual = rnmoncupi.cusum_alarm_query(
            [True] * 100,
            healthy_accuracy=0.95,
            alternative_accuracy=0.85,
            threshold=3.0,
        )
        self.assertIsNone(actual)

    def test2(self) -> None:
        """Test repeated failures accumulate evidence and trigger an alarm."""
        actual = rnmoncupi.cusum_alarm_query(
            [False] * 10,
            healthy_accuracy=0.95,
            alternative_accuracy=0.85,
            threshold=3.0,
        )
        self.assertEqual(actual, 3)


class Test_run_cusum_pilot(hunitest.TestCase):
    """Test calibration and the paired end-to-end pilot."""

    def test1(self) -> None:
        """Test a short run is reproducible and separates strong controls."""
        protocol = rnmoncupi.CusumPilotProtocol(
            trials_per_scenario=20,
            calibration_trials=100,
        )
        scenarios = (
            rnmoncaex.DEFAULT_CAPABILITY_SCENARIOS[0],
            rnmoncaex.DEFAULT_CAPABILITY_SCENARIOS[1],
        )
        first = rnmoncupi.run_cusum_pilot(protocol, scenarios)
        second = rnmoncupi.run_cusum_pilot(protocol, scenarios)
        self.assertEqual(first, second)
        self.assertLessEqual(
            first.calibration_false_alarm_rate,
            protocol.target_false_alarm_rate,
        )
        self.assertLessEqual(first.scenarios[0].cusum_detection_rate, 0.10)
        self.assertGreaterEqual(first.scenarios[1].cusum_detection_rate, 0.90)
