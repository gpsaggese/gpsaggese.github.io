"""
Test standalone-monitoring value validation.
"""

import helpers.hunit_test as hunitest

import research.Noesis.monitor.monitor_models as rnmonmod


class Test_monitoring_models(hunitest.TestCase):
    """
    Test invalid monitoring targets, policies, and observations fail early.
    """

    def test1(self) -> None:
        """
        Test targets require both provider and model identities.
        """
        with self.assertRaisesRegex(ValueError, "provider_slug"):
            rnmonmod.MonitoringTarget("", "model")
        with self.assertRaisesRegex(ValueError, "model_id"):
            rnmonmod.MonitoringTarget("groq", "")

    def test2(self) -> None:
        """
        Test a run cannot exceed the one-shot canary safety cap.
        """
        with self.assertRaisesRegex(ValueError, "canary_count"):
            rnmonmod.MonitoringPolicy(canary_count=0)
        with self.assertRaisesRegex(ValueError, "canary_count"):
            rnmonmod.MonitoringPolicy(
                canary_count=rnmonmod.MAX_CANARIES_PER_RUN + 1
            )

    def test3(self) -> None:
        """
        Test negative measurement values are rejected.
        """
        target = rnmonmod.MonitoringTarget("groq", "model")
        with self.assertRaisesRegex(ValueError, "latency_ms"):
            rnmonmod.MonitoringObservation(
                "canary-1",
                target,
                "ok",
                True,
                "MATCH",
                "v1",
                -1.0,
                0.0,
            )
