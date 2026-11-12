"""
Test the pluggable capability-measurement boundary.
"""

import helpers.hunit_test as hunitest

import research.Noesis.checker.checker_capability as rnochcap
import research.Noesis.checker.checker_models as rnochmod


class TestDeterministicBenchmarkMeasurer(hunitest.TestCase):
    """
    Test the MVP implementation delegates to deterministic grading.
    """

    def test1(self) -> None:
        """
        Test a normalized exact match returns a versioned success.
        """
        canary = rnochmod.Canary(
            "math-v1-001",
            "v1",
            "math",
            "Return one thousand.",
            "1000",
            "integer",
        )
        result = rnochcap.DEFAULT_MEASURER.measure(canary, "1,000")
        self.assertTrue(result.correct)
        self.assertEqual(
            result.grader_version, rnochcap.DEFAULT_MEASURER.version
        )
