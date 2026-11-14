"""
Tests for racket_sports_07_speed_vs_accuracy_utils module.

Import as:

import research.Optimal_strategy_for_racket_sports.test.test_racket_sports_07_speed_vs_accuracy_utils as rosfrsttrs0svau
"""

import logging
from typing import Tuple

import helpers.hunit_test as hunitest

from research.Optimal_strategy_for_racket_sports.notebooks import (
    racket_sports_07_speed_vs_accuracy_utils,
)

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_compute_speed_factor
# #############################################################################


class Test_compute_speed_factor(hunitest.TestCase):
    """
    Test `compute_speed_factor()`.

    The speeds are binary fractions of `v0`, so every expected factor is exact.
    """

    def helper(
        self,
        ball_speed_mps: float,
        v0_mps: float,
        a: float,
        expected: Tuple[float, float, float, float],
    ) -> None:
        """
        Check the factor of every shape.

        :param ball_speed_mps: ball speed `v_b`
        :param v0_mps: reference ball speed `v0`
        :param a: coefficient of the factor
        :param expected: the factor of each shape, in the order of
            `SPREAD_SHAPES`: linear, knee, parabola, one-sided parabola
        """
        # Run test.
        actual = tuple(
            racket_sports_07_speed_vs_accuracy_utils.compute_speed_factor(
                ball_speed_mps, v0_mps, a, shape
            )
            for shape in racket_sports_07_speed_vs_accuracy_utils.SPREAD_SHAPES
        )
        # Check outputs.
        self.assertEqual(actual, expected)

    def test1(self) -> None:
        """
        Test that `a = 0` gives 1 for every shape: the Milestone 2 model.
        """
        # Prepare inputs.
        ball_speed_mps = 50.0
        v0_mps = 25.0
        a = 0.0
        # Prepare outputs.
        expected = (1.0, 1.0, 1.0, 1.0)
        # Run test.
        self.helper(ball_speed_mps, v0_mps, a, expected)

    def test2(self) -> None:
        """
        Test that `v_b = v0` gives 1 for every shape, whatever `a` is.
        """
        # Prepare inputs.
        ball_speed_mps = 25.0
        v0_mps = 25.0
        a = 1.0
        # Prepare outputs.
        expected = (1.0, 1.0, 1.0, 1.0)
        # Run test.
        self.helper(ball_speed_mps, v0_mps, a, expected)

    def test3(self) -> None:
        """
        Test a ball slower than `v0`: more accurate with the linear shape,
        less accurate with the parabola, as accurate with the one-sided ones.
        """
        # Prepare inputs.
        ball_speed_mps = 12.5
        v0_mps = 25.0
        a = 1.0
        # Prepare outputs.
        expected = (0.5, 1.0, 1.25, 1.0)
        # Run test.
        self.helper(ball_speed_mps, v0_mps, a, expected)

    def test4(self) -> None:
        """
        Test a ball faster than `v0`: the one-sided shapes equal the two-sided
        ones, and the parabolas grow more slowly than the linear shapes.
        """
        # Prepare inputs.
        ball_speed_mps = 37.5
        v0_mps = 25.0
        a = 1.0
        # Prepare outputs.
        expected = (1.5, 1.5, 1.25, 1.25)
        # Run test.
        self.helper(ball_speed_mps, v0_mps, a, expected)

    def test5(self) -> None:
        """
        Test that every shape gives `1 + a` at `v_b = 2 v0`.
        """
        # Prepare inputs.
        ball_speed_mps = 50.0
        v0_mps = 25.0
        a = 0.5
        # Prepare outputs.
        expected = (1.5, 1.5, 1.5, 1.5)
        # Run test.
        self.helper(ball_speed_mps, v0_mps, a, expected)

    def test6(self) -> None:
        """
        Test that the factor depends on the speeds only through `v_b / v0`,
        with the pickleball `v0` and the same ratio as `test4()`.
        """
        # Prepare inputs.
        ball_speed_mps = 18.0
        v0_mps = 12.0
        a = 1.0
        # Prepare outputs.
        expected = (1.5, 1.5, 1.25, 1.25)
        # Run test.
        self.helper(ball_speed_mps, v0_mps, a, expected)

    def test7(self) -> None:
        """
        Test that `a > 1` is allowed while the factor stays positive.
        """
        # Prepare inputs.
        ball_speed_mps = 12.5
        v0_mps = 25.0
        a = 1.5
        # Prepare outputs.
        expected = (0.25, 1.0, 1.375, 1.0)
        # Run test.
        self.helper(ball_speed_mps, v0_mps, a, expected)

    def test8(self) -> None:
        """
        Test that a factor of 0 is rejected: the linear shape with `a = 2` at
        `v_b = v0 / 2`.
        """
        # Prepare inputs.
        ball_speed_mps = 12.5
        v0_mps = 25.0
        a = 2.0
        shape = "linear"
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_07_speed_vs_accuracy_utils.compute_speed_factor(
                ball_speed_mps, v0_mps, a, shape
            )

    def test9(self) -> None:
        """
        Test that bad inputs are rejected: a negative or `nan` coefficient, an
        unknown shape, and a ball speed that is not positive.
        """
        # Prepare inputs: `(ball_speed_mps, v0_mps, a, shape)`.
        bad_inputs = (
            (50.0, 25.0, -0.5, "linear"),
            (50.0, 25.0, float("nan"), "linear"),
            (50.0, 25.0, 0.5, "cubic"),
            (0.0, 25.0, 0.5, "knee"),
        )
        # Run test and check outputs.
        for inputs in bad_inputs:
            with self.subTest(inputs=inputs), self.assertRaises(AssertionError):
                racket_sports_07_speed_vs_accuracy_utils.compute_speed_factor(
                    *inputs
                )


# #############################################################################
# Test_build_spread_from_ball_speed
# #############################################################################


class Test_build_spread_from_ball_speed(hunitest.TestCase):
    """
    Test `build_spread_from_ball_speed()`.

    The spread at `v0 = 25` m/s is the Milestone 2 default, `(0.5, 1.0)` m.
    """

    def helper(
        self,
        ball_speed_mps: float,
        a_x: float,
        a_y: float,
        shape: str,
        expected: Tuple[float, float],
    ) -> None:
        """
        Check the spread at a ball speed.

        :param ball_speed_mps: ball speed `v_b`
        :param a_x: coefficient of the width factor
        :param a_y: coefficient of the depth factor
        :param shape: one of `SPREAD_SHAPES`
        :param expected: `(std_x2, std_y2)`
        """
        # Prepare inputs.
        std0_x = 0.5
        std0_y = 1.0
        v0_mps = 25.0
        # Run test.
        actual = racket_sports_07_speed_vs_accuracy_utils.build_spread_from_ball_speed(
            ball_speed_mps, std0_x, std0_y, v0_mps, a_x, a_y, shape
        )
        # Check outputs.
        self.assertEqual(actual, expected)

    def test1(self) -> None:
        """
        Test that `a_x = a_y = 0` returns the spread at `v0` exactly.
        """
        # Prepare inputs.
        ball_speed_mps = 50.0
        a_x = 0.0
        a_y = 0.0
        shape = "linear"
        # Prepare outputs.
        expected = (0.5, 1.0)
        # Run test.
        self.helper(ball_speed_mps, a_x, a_y, shape, expected)

    def test2(self) -> None:
        """
        Test that `a_y` acts only on the depth: with `a_x = 0` the width keeps
        its spread at `v0`.
        """
        # Prepare inputs.
        ball_speed_mps = 50.0
        a_x = 0.0
        a_y = 1.0
        shape = "linear"
        # Prepare outputs.
        expected = (0.5, 2.0)
        # Run test.
        self.helper(ball_speed_mps, a_x, a_y, shape, expected)

    def test3(self) -> None:
        """
        Test `a_x = a / 2` and `a_y = a` with `a = 1`: each direction gets its
        own factor.
        """
        # Prepare inputs.
        ball_speed_mps = 37.5
        a_x = 0.5
        a_y = 1.0
        shape = "linear"
        # Prepare outputs.
        expected = (0.625, 1.5)
        # Run test.
        self.helper(ball_speed_mps, a_x, a_y, shape, expected)

    def test4(self) -> None:
        """
        Test that `a_x = a_y` gives both directions the same factor, here the
        parabola below `v0`.
        """
        # Prepare inputs.
        ball_speed_mps = 12.5
        a_x = 1.0
        a_y = 1.0
        shape = "parabola"
        # Prepare outputs.
        expected = (0.625, 1.25)
        # Run test.
        self.helper(ball_speed_mps, a_x, a_y, shape, expected)

    def test5(self) -> None:
        """
        Test that a spread at `v0` that is not positive is rejected.
        """
        # Prepare inputs.
        ball_speed_mps = 50.0
        std0_y = 1.0
        v0_mps = 25.0
        a_x = 0.5
        a_y = 1.0
        shape = "linear"
        # Run test and check outputs.
        for std0_x in (0.0, float("nan")):
            with self.subTest(std0_x=std0_x), self.assertRaises(AssertionError):
                racket_sports_07_speed_vs_accuracy_utils.build_spread_from_ball_speed(
                    ball_speed_mps, std0_x, std0_y, v0_mps, a_x, a_y, shape
                )
