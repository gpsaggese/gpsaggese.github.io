"""
Test rendering of live calibration findings.
"""

import helpers.hunit_test as hunitest

import research.Noesis.checker.checker_calibration_report as rnochcare
import research.Noesis.checker.checker_models as rnochmod


class Test_render_calibration_report(hunitest.TestCase):
    """
    Test the human-reviewable calibration artifact.
    """

    def test1(self) -> None:
        """
        Test thresholds, measurements, cost, and activation are visible.
        """
        summary = {
            "candidate_count": 300,
            "accepted_count": 175,
            "rejected_count": 125,
            "total_calls": 2700,
            "total_cost_usd": 0.123456,
            "target_stats": [
                {
                    "role": "HEALTHY",
                    "provider_slug": "groq",
                    "model_id": "healthy-model",
                    "calls": 900,
                    "ok_calls": 900,
                    "gradable_calls": 900,
                    "accuracy": 1.0,
                    "survivor_calls": 525,
                    "survivor_accuracy": 1.0,
                    "latency_p95_ms": 420.25,
                    "cost_usd": 0.1,
                },
                {
                    "role": "WEAK",
                    "provider_slug": "deepinfra",
                    "model_id": "weak-model",
                    "calls": 900,
                    "ok_calls": 899,
                    "gradable_calls": 899,
                    "accuracy": 0.4,
                    "survivor_calls": 525,
                    "survivor_accuracy": 0.2,
                    "latency_p95_ms": 700.0,
                    "cost_usd": 0.023456,
                },
            ],
        }
        actual = rnochcare.render_calibration_report(
            7,
            "v1",
            rnochmod.CalibrationPolicy(),
            summary,
            minimum_survivors=150,
            activated=True,
        )
        self.assertIn("Accepted survivors: 175", actual)
        self.assertIn("Minimum-survivor gate: **PASS**", actual)
        self.assertIn("Bank activated: **yes**", actual)
        self.assertIn("| HEALTHY | `groq`", actual)
        self.assertIn("100.0%", actual)
        self.assertIn("20.0%", actual)
        self.assertIn("420.2", actual)
        self.assertIn("$0.123456", actual)

    def test2(self) -> None:
        """
        Test a short bank is visibly rejected and remains inactive.
        """
        summary = {
            "candidate_count": 10,
            "accepted_count": 9,
            "rejected_count": 1,
            "total_calls": 30,
            "total_cost_usd": 0.0,
            "target_stats": [],
        }
        actual = rnochcare.render_calibration_report(
            8,
            "v2",
            rnochmod.CalibrationPolicy(),
            summary,
            minimum_survivors=10,
            activated=False,
        )
        self.assertIn("Minimum-survivor gate: **FAIL**", actual)
        self.assertIn("Bank activated: **no**", actual)
