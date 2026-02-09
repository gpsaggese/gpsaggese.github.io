"""
Unit tests for growth_function module.

Import as:

import msml610.tutorials.test.test_growth_function as mtttgrowf
"""

import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import helpers.hunit_test as hunitest
import msml610.tutorials.L05_01_learning_theory_04_growth_function_utils as mtugrfun

_LOG = logging.getLogger(__name__)

# Use non-interactive backend for testing.
matplotlib.use("Agg")


# #############################################################################
# Test_PointGenerator
# #############################################################################


class Test_PointGenerator(hunitest.TestCase):
    """
    Test the PointGenerator class.
    """

    def test1(self) -> None:
        """
        Test random point generation.
        """
        # Prepare inputs.
        generator = mtugrfun.PointGenerator(seed=42)
        # Run test.
        points = generator.generate_random(n=5, d=2, bounds=(-1.0, 1.0))
        # Check outputs.
        self.assertEqual(points.shape, (5, 2))
        self.assertTrue(np.all(points >= -1.0))
        self.assertTrue(np.all(points <= 1.0))
        # Verify reproducibility.
        generator2 = mtugrfun.PointGenerator(seed=42)
        points2 = generator2.generate_random(n=5, d=2, bounds=(-1.0, 1.0))
        np.testing.assert_array_equal(points, points2)

    def test2(self) -> None:
        """
        Test circle point generation.
        """
        # Prepare inputs.
        generator = mtugrfun.PointGenerator()
        # Run test.
        points = generator.generate_circle(n=8, radius=1.0)
        # Check outputs.
        self.assertEqual(points.shape, (8, 2))
        # Verify all points are at distance 1.0 from origin.
        distances = np.sqrt(np.sum(points**2, axis=1))
        np.testing.assert_array_almost_equal(distances, np.ones(8))

    def test3(self) -> None:
        """
        Test grid point generation.
        """
        # Prepare inputs.
        generator = mtugrfun.PointGenerator()
        # Run test.
        points = generator.generate_grid(n=9, d=2, bounds=(-1.0, 1.0))
        # Check outputs.
        self.assertGreater(len(points), 0)
        self.assertTrue(np.all(points >= -1.0))
        self.assertTrue(np.all(points <= 1.0))

    def test4(self) -> None:
        """
        Test collinear point generation.
        """
        # Prepare inputs.
        generator = mtugrfun.PointGenerator()
        # Run test.
        points = generator.generate_collinear(n=5, d=3, bounds=(-1.0, 1.0))
        # Check outputs.
        self.assertEqual(points.shape, (5, 3))
        # Verify all points on line (other dimensions are zero).
        np.testing.assert_array_equal(points[:, 1], np.zeros(5))
        np.testing.assert_array_equal(points[:, 2], np.zeros(5))

    def test5(self) -> None:
        """
        Test 1D line point generation.
        """
        # Prepare inputs.
        generator = mtugrfun.PointGenerator()
        # Run test.
        points = generator.generate_line_1d(n=6, bounds=(-1.0, 1.0))
        # Check outputs.
        self.assertEqual(points.shape, (6, 1))
        self.assertTrue(np.all(points >= -1.0))
        self.assertTrue(np.all(points <= 1.0))


# #############################################################################
# Test_DichotomyEnumerator
# #############################################################################


class Test_DichotomyEnumerator(hunitest.TestCase):
    """
    Test the DichotomyEnumerator class.
    """

    def test1(self) -> None:
        """
        Test dichotomy counting.
        """
        # Prepare inputs.
        enumerator = mtugrfun.DichotomyEnumerator(n=3)
        # Run test.
        actual = enumerator.count_dichotomies()
        # Check outputs.
        self.assertEqual(actual, 8)
        # Prepare inputs.
        enumerator = mtugrfun.DichotomyEnumerator(n=5)
        # Run test.
        actual = enumerator.count_dichotomies()
        # Check outputs.
        self.assertEqual(actual, 32)

    def test2(self) -> None:
        """
        Test getting specific dichotomies.
        """
        # Prepare inputs.
        enumerator = mtugrfun.DichotomyEnumerator(n=3)
        # Run test.
        labels = enumerator.get_dichotomy(0)
        # Check outputs.
        # Dichotomy 0: all -1.
        np.testing.assert_array_equal(labels, np.array([-1, -1, -1]))
        # Run test.
        labels = enumerator.get_dichotomy(7)
        # Check outputs.
        # Dichotomy 7: all +1.
        np.testing.assert_array_equal(labels, np.array([1, 1, 1]))

    def test3(self) -> None:
        """
        Test enumerating all dichotomies.
        """
        # Prepare inputs.
        enumerator = mtugrfun.DichotomyEnumerator(n=3)
        # Run test.
        dichotomies = list(enumerator.enumerate_all())
        # Check outputs.
        self.assertEqual(len(dichotomies), 8)
        # Verify all values in {-1, +1}.
        for labels in dichotomies:
            self.assertTrue(np.all((labels == -1) | (labels == 1)))
        # Verify uniqueness.
        unique = [tuple(d) for d in dichotomies]
        self.assertEqual(len(unique), len(set(unique)))


# #############################################################################
# Test_PerceptronTester
# #############################################################################


class Test_PerceptronTester(hunitest.TestCase):
    """
    Test the PerceptronTester class.
    """

    def test1(self) -> None:
        """
        Test linearly separable case.
        """
        # Prepare inputs.
        tester = mtugrfun.PerceptronTester(random_state=42)
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        labels = np.array([-1, -1, 1, 1])
        # Run test.
        actual = tester.test_dichotomy(points, labels)
        # Check outputs.
        self.assertTrue(actual)

    def test2(self) -> None:
        """
        Test XOR (not linearly separable) case.
        """
        # Prepare inputs.
        tester = mtugrfun.PerceptronTester(random_state=42)
        points = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
        labels = np.array([1, -1, -1, 1])
        # Run test.
        actual = tester.test_dichotomy(points, labels)
        # Check outputs.
        self.assertFalse(actual)

    def test3(self) -> None:
        """
        Test trivial case with all same labels.
        """
        # Prepare inputs.
        tester = mtugrfun.PerceptronTester(random_state=42)
        points = np.array([[0, 0], [1, 0], [0, 1]])
        labels = np.array([1, 1, 1])
        # Run test.
        actual = tester.test_dichotomy(points, labels)
        # Check outputs.
        self.assertTrue(actual)

    def test4(self) -> None:
        """
        Test finding hypothesis parameters.
        """
        # Prepare inputs.
        tester = mtugrfun.PerceptronTester(random_state=42)
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        labels = np.array([-1, -1, 1, 1])
        # Run test.
        result = tester.find_hypothesis(points, labels)
        # Check outputs.
        self.assertIsNotNone(result)
        self.assertIn("weights", result)
        self.assertIn("intercept", result)

    def test5(self) -> None:
        """
        Test that 3 points can be shattered by perceptron.
        """
        # Prepare inputs.
        tester = mtugrfun.PerceptronTester(random_state=42)
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        points = generator.generate_random(n=3, d=2)
        # Run test.
        result = calculator.compute_growth_function(points)
        # Check outputs.
        self.assertEqual(result["m_h_n"], 8)
        self.assertTrue(result["is_shattered"])

    @pytest.mark.slow
    def test6(self) -> None:
        """
        Test break point at 4 points for perceptron.
        """
        # Prepare inputs.
        tester = mtugrfun.PerceptronTester(random_state=42)
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        # Run test.
        result = calculator.compute_growth_function(points)
        # Check outputs.
        self.assertLess(result["m_h_n"], 16)
        self.assertFalse(result["is_shattered"])


# #############################################################################
# Test_PositiveRaysTester
# #############################################################################


class Test_PositiveRaysTester(hunitest.TestCase):
    """
    Test the PositiveRaysTester class.
    """

    def test1(self) -> None:
        """
        Test valid ray pattern.
        """
        # Prepare inputs.
        tester = mtugrfun.PositiveRaysTester()
        points = np.array([[-1], [-0.5], [0.5], [1]])
        labels = np.array([-1, -1, 1, 1])
        # Run test.
        actual = tester.test_dichotomy(points, labels)
        # Check outputs.
        self.assertTrue(actual)

    def test2(self) -> None:
        """
        Test invalid ray pattern with multiple transitions.
        """
        # Prepare inputs.
        tester = mtugrfun.PositiveRaysTester()
        points = np.array([[-1], [-0.5], [0.5], [1]])
        labels = np.array([-1, 1, -1, 1])
        # Run test.
        actual = tester.test_dichotomy(points, labels)
        # Check outputs.
        self.assertFalse(actual)

    def test3(self) -> None:
        """
        Test finding threshold.
        """
        # Prepare inputs.
        tester = mtugrfun.PositiveRaysTester()
        points = np.array([[-1], [0], [1], [2], [3]])
        labels = np.array([-1, -1, 1, 1, 1])
        # Run test.
        result = tester.find_hypothesis(points, labels)
        # Check outputs.
        self.assertIsNotNone(result)
        self.assertIn("threshold", result)
        threshold = result["threshold"]
        self.assertGreater(threshold, 0)
        self.assertLess(threshold, 1)

    def test4(self) -> None:
        """
        Test that m_H(N) = N + 1 for positive rays.
        """
        # Prepare inputs.
        tester = mtugrfun.PositiveRaysTester()
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        # Run test and check outputs.
        for n in [1, 2, 3, 4, 5]:
            points = generator.generate_line_1d(n)
            result = calculator.compute_growth_function(points)
            self.assertEqual(result["m_h_n"], n + 1)


# #############################################################################
# Test_PositiveIntervalsTester
# #############################################################################


class Test_PositiveIntervalsTester(hunitest.TestCase):
    """
    Test the PositiveIntervalsTester class.
    """

    def test1(self) -> None:
        """
        Test valid interval pattern.
        """
        # Prepare inputs.
        tester = mtugrfun.PositiveIntervalsTester()
        points = np.array([[-1], [-0.5], [0], [0.5], [1]])
        labels = np.array([-1, -1, 1, 1, -1])
        # Run test.
        actual = tester.test_dichotomy(points, labels)
        # Check outputs.
        self.assertTrue(actual)

    def test2(self) -> None:
        """
        Test invalid interval pattern with multiple intervals.
        """
        # Prepare inputs.
        tester = mtugrfun.PositiveIntervalsTester()
        points = np.array([[-1], [-0.5], [0], [0.5], [1]])
        labels = np.array([-1, 1, -1, 1, -1])
        # Run test.
        actual = tester.test_dichotomy(points, labels)
        # Check outputs.
        self.assertFalse(actual)

    def test3(self) -> None:
        """
        Test finding interval boundaries.
        """
        # Prepare inputs.
        tester = mtugrfun.PositiveIntervalsTester()
        points = np.array([[-2], [-1], [0], [1], [2], [3]])
        labels = np.array([-1, -1, 1, 1, 1, -1])
        # Run test.
        result = tester.find_hypothesis(points, labels)
        # Check outputs.
        self.assertIsNotNone(result)
        self.assertIn("left_bound", result)
        self.assertIn("right_bound", result)

    def test4(self) -> None:
        """
        Test that m_H(N) = N(N+1)/2 + 1 for positive intervals.
        """
        # Prepare inputs.
        tester = mtugrfun.PositiveIntervalsTester()
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        # Run test and check outputs.
        for n in [1, 2, 3, 4, 5]:
            points = generator.generate_line_1d(n)
            result = calculator.compute_growth_function(points)
            expected = n * (n + 1) // 2 + 1
            self.assertEqual(result["m_h_n"], expected)


# #############################################################################
# Test_ConvexSetsTester
# #############################################################################


class Test_ConvexSetsTester(hunitest.TestCase):
    """
    Test the ConvexSetsTester class.
    """

    def test1(self) -> None:
        """
        Test that all dichotomies are realizable.
        """
        # Prepare inputs.
        tester = mtugrfun.ConvexSetsTester()
        generator = mtugrfun.PointGenerator(seed=42)
        points = generator.generate_circle(n=5)
        enumerator = mtugrfun.DichotomyEnumerator(n=5)
        # Run test and check outputs.
        for i in range(10):
            labels = enumerator.get_dichotomy(i)
            self.assertTrue(tester.test_dichotomy(points, labels))

    def test2(self) -> None:
        """
        Test finding convex hull.
        """
        # Prepare inputs.
        tester = mtugrfun.ConvexSetsTester()
        generator = mtugrfun.PointGenerator(seed=42)
        points = generator.generate_circle(n=6)
        labels = np.array([1, 1, 1, -1, -1, -1])
        # Run test.
        result = tester.find_hypothesis(points, labels)
        # Check outputs.
        self.assertIsNotNone(result)
        self.assertIn("hull_points", result)

    @pytest.mark.slow
    def test3(self) -> None:
        """
        Test that m_H(N) = 2^N for convex sets.
        """
        # Prepare inputs.
        tester = mtugrfun.ConvexSetsTester()
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        # Run test and check outputs.
        for n in [1, 2, 3, 4, 5]:
            points = generator.generate_circle(n)
            result = calculator.compute_growth_function(points)
            expected = 2**n
            self.assertEqual(result["m_h_n"], expected)
            self.assertTrue(result["is_shattered"])


# #############################################################################
# Test_GrowthFunctionCalculator
# #############################################################################


class Test_GrowthFunctionCalculator(hunitest.TestCase):
    """
    Test the GrowthFunctionCalculator class.
    """

    def test1(self) -> None:
        """
        Test computing growth function.
        """
        # Prepare inputs.
        tester = mtugrfun.PositiveRaysTester()
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        points = generator.generate_line_1d(n=4)
        # Run test.
        result = calculator.compute_growth_function(points)
        # Check outputs.
        self.assertIn("n", result)
        self.assertIn("m_h_n", result)
        self.assertIn("max_dichotomies", result)
        self.assertIn("fraction", result)
        self.assertIn("is_shattered", result)
        self.assertIn("realizable_dichotomies", result)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["m_h_n"], 5)

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test computing growth curve.
        """
        # Prepare inputs.
        tester = mtugrfun.PerceptronTester(random_state=42)
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        # Run test.
        results_df = calculator.compute_growth_curve(
            generator, n_range=[1, 2, 3, 4], num_trials=1
        )
        # Check outputs.
        self.assertIn("n", results_df.columns)
        self.assertIn("m_h_n_mean", results_df.columns)
        self.assertIn("max_dichotomies", results_df.columns)
        row_4 = results_df[results_df["n"] == 4].iloc[0]
        self.assertLess(row_4["m_h_n_mean"], 16)

    def test3(self) -> None:
        """
        Test finding break point.
        """
        # Prepare inputs.
        tester = mtugrfun.PerceptronTester(random_state=42)
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        # Run test.
        break_point = calculator.find_break_point(generator, max_n=10)
        # Check outputs.
        self.assertIsNotNone(break_point)
        self.assertLessEqual(break_point, 4)

    def test4(self) -> None:
        """
        Test finding shattered points.
        """
        # Prepare inputs.
        tester = mtugrfun.PerceptronTester(random_state=42)
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        points_3 = generator.generate_random(n=3, d=2)
        # Run test.
        actual = calculator.find_shattered_points(points_3)
        # Check outputs.
        self.assertTrue(actual)

    @pytest.mark.slow
    def test5(self) -> None:
        """
        Test estimating VC dimension.
        """
        # Prepare inputs.
        tester = mtugrfun.PerceptronTester(random_state=42)
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        # Run test.
        result = calculator.estimate_vc_dimension(
            generator, max_n=6, num_trials=3
        )
        # Check outputs.
        self.assertIn("vc_dimension", result)
        self.assertIn("break_point", result)
        self.assertIn("results_by_n", result)
        self.assertGreaterEqual(result["vc_dimension"], 2)


# #############################################################################
# Test_GrowthFunctionVisualizer
# #############################################################################


class Test_GrowthFunctionVisualizer(hunitest.TestCase):
    """
    Test the GrowthFunctionVisualizer class.
    """

    def test1(self) -> None:
        """
        Test plotting growth curve without errors.
        """
        # Prepare inputs.
        visualizer = mtugrfun.GrowthFunctionVisualizer()
        data = {
            "n": [1, 2, 3, 4],
            "m_h_n_mean": [2, 4, 8, 14],
            "m_h_n_std": [0, 0, 0, 0],
            "max_dichotomies": [2, 4, 8, 16],
            "hypothesis": ["Perceptron"] * 4,
        }
        results_df = pd.DataFrame(data)
        # Run test.
        try:
            visualizer.plot_growth_curve(results_df)
            plt.close()
        except Exception as e:
            # Check outputs.
            self.fail(f"plot_growth_curve raised exception: {e}")

    def test2(self) -> None:
        """
        Test plotting multiple growth curves without errors.
        """
        # Prepare inputs.
        visualizer = mtugrfun.GrowthFunctionVisualizer()
        data1 = {
            "n": [1, 2, 3],
            "m_h_n_mean": [2, 3, 4],
            "max_dichotomies": [2, 4, 8],
        }
        data2 = {
            "n": [1, 2, 3],
            "m_h_n_mean": [2, 4, 8],
            "max_dichotomies": [2, 4, 8],
        }
        results_dict = {
            "Positive Rays": pd.DataFrame(data1),
            "Perceptron": pd.DataFrame(data2),
        }
        # Run test.
        try:
            visualizer.plot_multiple_growth_curves(results_dict)
            plt.close()
        except Exception as e:
            # Check outputs.
            self.fail(f"plot_multiple_growth_curves raised exception: {e}")


# #############################################################################
# Test_HelperFunctions
# #############################################################################


class Test_HelperFunctions(hunitest.TestCase):
    """
    Test helper functions.
    """

    def test1(self) -> None:
        """
        Test theoretical growth for positive rays.
        """
        # Run test and check outputs.
        for n in [1, 2, 3, 4, 5]:
            result = mtugrfun.compute_theoretical_growth("Positive Rays", n)
            self.assertEqual(result, n + 1)

    def test2(self) -> None:
        """
        Test theoretical growth for positive intervals.
        """
        # Run test and check outputs.
        for n in [1, 2, 3, 4, 5]:
            result = mtugrfun.compute_theoretical_growth("Positive Intervals", n)
            expected = n * (n + 1) // 2 + 1
            self.assertEqual(result, expected)

    def test3(self) -> None:
        """
        Test theoretical growth for perceptron.
        """
        # Run test and check outputs.
        self.assertEqual(mtugrfun.compute_theoretical_growth("Perceptron", 1), 2)
        self.assertEqual(mtugrfun.compute_theoretical_growth("Perceptron", 2), 4)
        self.assertEqual(mtugrfun.compute_theoretical_growth("Perceptron", 3), 8)
        self.assertEqual(mtugrfun.compute_theoretical_growth("Perceptron", 4), 14)

    def test4(self) -> None:
        """
        Test theoretical growth for convex sets.
        """
        # Run test and check outputs.
        for n in [1, 2, 3, 4, 5]:
            result = mtugrfun.compute_theoretical_growth("Convex Sets", n)
            self.assertEqual(result, 2**n)

    def test5(self) -> None:
        """
        Test comparing with theory.
        """
        # Prepare inputs.
        data = {
            "n": [1, 2, 3, 4],
            "m_h_n_mean": [2, 3, 4, 5],
        }
        results_df = pd.DataFrame(data)
        # Run test.
        compared = mtugrfun.compare_with_theory(results_df, "Positive Rays")
        # Check outputs.
        self.assertIn("theoretical", compared.columns)
        self.assertIn("error", compared.columns)


# #############################################################################
# Test_IntegrationEndToEnd
# #############################################################################


class Test_IntegrationEndToEnd(hunitest.TestCase):
    """
    Integration tests for end-to-end workflows.
    """

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test complete workflow for perceptron.
        """
        # Prepare inputs.
        generator = mtugrfun.PointGenerator(seed=42)
        tester = mtugrfun.PerceptronTester(random_state=42)
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        # Run test.
        results_df = calculator.compute_growth_curve(
            generator, n_range=[1, 2, 3, 4], num_trials=1
        )
        compared = mtugrfun.compare_with_theory(results_df, "Perceptron")
        # Check outputs.
        for n in [1, 2, 3]:
            row = compared[compared["n"] == n].iloc[0]
            self.assertEqual(row["m_h_n_mean"], row["theoretical"])

    def test2(self) -> None:
        """
        Test complete workflow for positive rays.
        """
        # Prepare inputs.
        generator = mtugrfun.PointGenerator(seed=42)
        tester = mtugrfun.PositiveRaysTester()
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        # Run test.
        results_df = calculator.compute_growth_curve(
            generator, n_range=[1, 2, 3, 4, 5], num_trials=1
        )
        # Check outputs.
        for _, row in results_df.iterrows():
            self.assertEqual(row["m_h_n_mean"], row["n"] + 1)

    def test3(self) -> None:
        """
        Test complete workflow for positive intervals.
        """
        # Prepare inputs.
        generator = mtugrfun.PointGenerator(seed=42)
        tester = mtugrfun.PositiveIntervalsTester()
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        # Run test.
        results_df = calculator.compute_growth_curve(
            generator, n_range=[1, 2, 3, 4, 5], num_trials=1
        )
        compared = mtugrfun.compare_with_theory(results_df, "Positive Intervals")
        # Check outputs.
        for _, row in compared.iterrows():
            self.assertEqual(row["m_h_n_mean"], row["theoretical"])

    @pytest.mark.slow
    def test4(self) -> None:
        """
        Test complete workflow for convex sets.
        """
        # Prepare inputs.
        generator = mtugrfun.PointGenerator(seed=42)
        tester = mtugrfun.ConvexSetsTester()
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        # Run test.
        results_df = calculator.compute_growth_curve(
            generator, n_range=[1, 2, 3, 4, 5], num_trials=1
        )
        # Check outputs.
        for _, row in results_df.iterrows():
            self.assertEqual(row["m_h_n_mean"], 2 ** int(row["n"]))


# #############################################################################
# Test_EdgeCases
# #############################################################################


class Test_EdgeCases(hunitest.TestCase):
    """
    Test edge cases and boundary conditions.
    """

    def test1(self) -> None:
        """
        Test with N=1 point.
        """
        # Prepare inputs.
        generator = mtugrfun.PointGenerator(seed=42)
        # Run test and check outputs.
        for tester in [
            mtugrfun.PerceptronTester(random_state=42),
            mtugrfun.PositiveRaysTester(),
            mtugrfun.PositiveIntervalsTester(),
            mtugrfun.ConvexSetsTester(),
        ]:
            calculator = mtugrfun.GrowthFunctionCalculator(
                tester, verbose=False, show_progress=False
            )
            hypothesis_name = tester.get_name()
            if "Positive" in hypothesis_name:
                points = generator.generate_line_1d(n=1)
            elif "Convex" in hypothesis_name:
                points = generator.generate_circle(n=1)
            else:
                points = generator.generate_random(n=1, d=2)
            result = calculator.compute_growth_function(points)
            self.assertEqual(result["m_h_n"], 2)

    def test2(self) -> None:
        """
        Test with N=0 points.
        """
        # Prepare inputs.
        enumerator = mtugrfun.DichotomyEnumerator(n=0)
        # Run test.
        actual = enumerator.count_dichotomies()
        # Check outputs.
        self.assertEqual(actual, 1)

    def test3(self) -> None:
        """
        Test with all identical points.
        """
        # Prepare inputs.
        tester = mtugrfun.PerceptronTester(random_state=42)
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        points = np.array([[0, 0], [0, 0], [0, 0]])
        # Run test.
        result = calculator.compute_growth_function(points)
        # Check outputs.
        self.assertEqual(result["m_h_n"], 2)
