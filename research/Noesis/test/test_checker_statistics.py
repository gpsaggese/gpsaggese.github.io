"""
Test one-sided checker confidence bounds.
"""

import math

import helpers.hunit_test as hunitest

import research.Noesis.checker_statistics as rnochsta

# #############################################################################
# Test_upper_confidence_bound
# #############################################################################


class Test_upper_confidence_bound(hunitest.TestCase):
    """
    Test exact and Wilson bounds and their edge cases.
    """

    def test1(self) -> None:
        """
        Test a documented Wilson example.
        """
        self.assertAlmostEqual(rnochsta.wilson_upper(40, 50), 0.876526, places=6)

    def test2(self) -> None:
        """
        Test the exact zero-success formula and all-success boundary.
        """
        expected = 1 - 0.05 ** (1 / 8)
        self.assertAlmostEqual(rnochsta.clopper_pearson_upper(0, 8), expected)
        self.assertEqual(rnochsta.clopper_pearson_upper(8, 8), 1.0)

    def test3(self) -> None:
        """
        Test no evidence has no confidence bound.
        """
        self.assertIsNone(rnochsta.upper_confidence_bound(0, 0))

    def test4(self) -> None:
        """
        Test bounds increase monotonically with successes.
        """
        for total in range(1, 40):
            bounds = [
                rnochsta.upper_confidence_bound(successes, total)
                for successes in range(total + 1)
            ]
            self.assertEqual(bounds, sorted(bounds))

    def test5(self) -> None:
        """
        Test invalid counts are rejected.
        """
        for successes, total in [(-1, 2), (3, 2), (1, -1), (True, 2), (1.0, 2)]:
            with self.assertRaises((TypeError, ValueError)):
                rnochsta.upper_confidence_bound(successes, total)


class Test_is_conclusive_failure(hunitest.TestCase):
    """
    Test the strict optimistic-bound failure rule.
    """

    def test1(self) -> None:
        """
        Test equality passes and a lower bound fails.
        """
        self.assertTrue(rnochsta.is_conclusive_failure(0.84, 0.85))
        self.assertFalse(rnochsta.is_conclusive_failure(0.85, 0.85))
        self.assertFalse(rnochsta.is_conclusive_failure(None, 0.85))

    def test2(self) -> None:
        """
        Test the weak-model detection probability in the PRD.
        """
        probability = 0.0
        for successes in range(11):
            upper = rnochsta.upper_confidence_bound(successes, 10)
            if rnochsta.is_conclusive_failure(upper, 0.85):
                probability += math.comb(10, successes) * 0.5**10
        self.assertAlmostEqual(probability, 0.828125)
