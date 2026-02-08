"""
Unit tests for growth_function module.

Import as:

import msml610.tutorials.test.test_growth_function as mtttgrowf
"""

import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import helpers.hunit_test as hunitest
import msml610.tutorials.growth_function as mtugrfun

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

    def test_generate_random1(self) -> None:
        """
        Test random point generation.
        """
        generator = mtugrfun.PointGenerator(seed=42)
        points = generator.generate_random(n=5, d=2, bounds=(-1.0, 1.0))
        # Verify shape.
        self.assertEqual(points.shape, (5, 2))
        # Verify bounds.
        self.assertTrue(np.all(points >= -1.0))
        self.assertTrue(np.all(points <= 1.0))
        # Verify reproducibility.
        generator2 = mtugrfun.PointGenerator(seed=42)
        points2 = generator2.generate_random(n=5, d=2, bounds=(-1.0, 1.0))
        np.testing.assert_array_equal(points, points2)

    def test_generate_circle1(self) -> None:
        """
        Test circle point generation.
        """
        generator = mtugrfun.PointGenerator()
        points = generator.generate_circle(n=8, radius=1.0)
        # Verify shape.
        self.assertEqual(points.shape, (8, 2))
        # Verify all points are at distance 1.0 from origin.
        distances = np.sqrt(np.sum(points**2, axis=1))
        np.testing.assert_array_almost_equal(distances, np.ones(8))

    def test_generate_grid1(self) -> None:
        """
        Test grid point generation.
        """
        generator = mtugrfun.PointGenerator()
        points = generator.generate_grid(n=9, d=2, bounds=(-1.0, 1.0))
        # Verify shape (should be close to 9).
        self.assertGreater(len(points), 0)
        # Verify bounds.
        self.assertTrue(np.all(points >= -1.0))
        self.assertTrue(np.all(points <= 1.0))

    def test_generate_collinear1(self) -> None:
        """
        Test collinear point generation.
        """
        generator = mtugrfun.PointGenerator()
        points = generator.generate_collinear(n=5, d=3, bounds=(-1.0, 1.0))
        # Verify shape.
        self.assertEqual(points.shape, (5, 3))
        # Verify all points on line (other dimensions are zero).
        np.testing.assert_array_equal(points[:, 1], np.zeros(5))
        np.testing.assert_array_equal(points[:, 2], np.zeros(5))

    def test_generate_line_1d1(self) -> None:
        """
        Test 1D line point generation.
        """
        generator = mtugrfun.PointGenerator()
        points = generator.generate_line_1d(n=6, bounds=(-1.0, 1.0))
        # Verify shape.
        self.assertEqual(points.shape, (6, 1))
        # Verify bounds.
        self.assertTrue(np.all(points >= -1.0))
        self.assertTrue(np.all(points <= 1.0))


# #############################################################################
# Test_DichotomyEnumerator
# #############################################################################


class Test_DichotomyEnumerator(hunitest.TestCase):
    """
    Test the DichotomyEnumerator class.
    """

    def test_count_dichotomies1(self) -> None:
        """
        Test dichotomy counting.
        """
        enumerator = mtugrfun.DichotomyEnumerator(n=3)
        self.assertEqual(enumerator.count_dichotomies(), 8)
        enumerator = mtugrfun.DichotomyEnumerator(n=5)
        self.assertEqual(enumerator.count_dichotomies(), 32)

    def test_get_dichotomy1(self) -> None:
        """
        Test getting specific dichotomies.
        """
        enumerator = mtugrfun.DichotomyEnumerator(n=3)
        # Dichotomy 0: all -1.
        labels = enumerator.get_dichotomy(0)
        np.testing.assert_array_equal(labels, np.array([-1, -1, -1]))
        # Dichotomy 7: all +1.
        labels = enumerator.get_dichotomy(7)
        np.testing.assert_array_equal(labels, np.array([1, 1, 1]))

    def test_enumerate_all1(self) -> None:
        """
        Test enumerating all dichotomies.
        """
        enumerator = mtugrfun.DichotomyEnumerator(n=3)
        dichotomies = list(enumerator.enumerate_all())
        # Verify count.
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

    def test_test_dichotomy_separable1(self) -> None:
        """
        Test linearly separable case.
        """
        tester = mtugrfun.PerceptronTester(random_state=42)
        # Create linearly separable points.
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        labels = np.array([-1, -1, 1, 1])
        # Should be separable.
        self.assertTrue(tester.test_dichotomy(points, labels))

    def test_test_dichotomy_not_separable1(self) -> None:
        """
        Test XOR (not linearly separable) case.
        """
        tester = mtugrfun.PerceptronTester(random_state=42)
        # Create XOR configuration.
        points = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
        labels = np.array([1, -1, -1, 1])
        # Should not be separable (XOR).
        self.assertFalse(tester.test_dichotomy(points, labels))

    def test_all_same_labels1(self) -> None:
        """
        Test trivial case with all same labels.
        """
        tester = mtugrfun.PerceptronTester(random_state=42)
        points = np.array([[0, 0], [1, 0], [0, 1]])
        labels = np.array([1, 1, 1])
        # Trivially separable.
        self.assertTrue(tester.test_dichotomy(points, labels))

    def test_find_hypothesis1(self) -> None:
        """
        Test finding hypothesis parameters.
        """
        tester = mtugrfun.PerceptronTester(random_state=42)
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        labels = np.array([-1, -1, 1, 1])
        # Find hypothesis.
        result = tester.find_hypothesis(points, labels)
        # Should return parameters.
        self.assertIsNotNone(result)
        self.assertIn("weights", result)
        self.assertIn("intercept", result)

    def test_growth_function_3points1(self) -> None:
        """
        Test that 3 points can be shattered by perceptron.
        """
        tester = mtugrfun.PerceptronTester(random_state=42)
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        points = generator.generate_random(n=3, d=2)
        result = calculator.compute_growth_function(points)
        # Should shatter 3 points.
        self.assertEqual(result["m_h_n"], 8)
        self.assertTrue(result["is_shattered"])

    @pytest.mark.slow
    def test_growth_function_4points1(self) -> None:
        """
        Test break point at 4 points for perceptron.
        """
        tester = mtugrfun.PerceptronTester(random_state=42)
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        # Use square configuration to ensure break point.
        points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        result = calculator.compute_growth_function(points)
        # Should not shatter 4 points.
        self.assertLess(result["m_h_n"], 16)
        self.assertFalse(result["is_shattered"])


# #############################################################################
# Test_PositiveRaysTester
# #############################################################################


class Test_PositiveRaysTester(hunitest.TestCase):
    """
    Test the PositiveRaysTester class.
    """

    def test_test_dichotomy_valid_ray1(self) -> None:
        """
        Test valid ray pattern.
        """
        tester = mtugrfun.PositiveRaysTester()
        points = np.array([[-1], [-0.5], [0.5], [1]])
        labels = np.array([-1, -1, 1, 1])
        # Valid ray pattern.
        self.assertTrue(tester.test_dichotomy(points, labels))

    def test_test_dichotomy_invalid_ray1(self) -> None:
        """
        Test invalid ray pattern with multiple transitions.
        """
        tester = mtugrfun.PositiveRaysTester()
        points = np.array([[-1], [-0.5], [0.5], [1]])
        labels = np.array([-1, 1, -1, 1])
        # Invalid ray pattern.
        self.assertFalse(tester.test_dichotomy(points, labels))

    def test_find_hypothesis1(self) -> None:
        """
        Test finding threshold.
        """
        tester = mtugrfun.PositiveRaysTester()
        points = np.array([[-1], [0], [1], [2], [3]])
        labels = np.array([-1, -1, 1, 1, 1])
        # Find hypothesis.
        result = tester.find_hypothesis(points, labels)
        self.assertIsNotNone(result)
        self.assertIn("threshold", result)
        # Threshold should be between 0 and 1.
        threshold = result["threshold"]
        self.assertGreater(threshold, 0)
        self.assertLess(threshold, 1)

    def test_growth_function_theoretical1(self) -> None:
        """
        Test that m_H(N) = N + 1 for positive rays.
        """
        tester = mtugrfun.PositiveRaysTester()
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        for n in [1, 2, 3, 4, 5]:
            points = generator.generate_line_1d(n)
            result = calculator.compute_growth_function(points)
            # Should match theoretical formula.
            self.assertEqual(result["m_h_n"], n + 1)


# #############################################################################
# Test_PositiveIntervalsTester
# #############################################################################


class Test_PositiveIntervalsTester(hunitest.TestCase):
    """
    Test the PositiveIntervalsTester class.
    """

    def test_test_dichotomy_valid_interval1(self) -> None:
        """
        Test valid interval pattern.
        """
        tester = mtugrfun.PositiveIntervalsTester()
        points = np.array([[-1], [-0.5], [0], [0.5], [1]])
        labels = np.array([-1, -1, 1, 1, -1])
        # Valid interval pattern.
        self.assertTrue(tester.test_dichotomy(points, labels))

    def test_test_dichotomy_invalid_interval1(self) -> None:
        """
        Test invalid interval pattern with multiple intervals.
        """
        tester = mtugrfun.PositiveIntervalsTester()
        points = np.array([[-1], [-0.5], [0], [0.5], [1]])
        labels = np.array([-1, 1, -1, 1, -1])
        # Invalid interval pattern.
        self.assertFalse(tester.test_dichotomy(points, labels))

    def test_find_hypothesis1(self) -> None:
        """
        Test finding interval boundaries.
        """
        tester = mtugrfun.PositiveIntervalsTester()
        points = np.array([[-2], [-1], [0], [1], [2], [3]])
        labels = np.array([-1, -1, 1, 1, 1, -1])
        # Find hypothesis.
        result = tester.find_hypothesis(points, labels)
        self.assertIsNotNone(result)
        self.assertIn("left_bound", result)
        self.assertIn("right_bound", result)

    def test_growth_function_theoretical1(self) -> None:
        """
        Test that m_H(N) = N(N+1)/2 + 1 for positive intervals.
        """
        tester = mtugrfun.PositiveIntervalsTester()
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        for n in [1, 2, 3, 4, 5]:
            points = generator.generate_line_1d(n)
            result = calculator.compute_growth_function(points)
            expected = n * (n + 1) // 2 + 1
            # Should match theoretical formula.
            self.assertEqual(result["m_h_n"], expected)


# #############################################################################
# Test_ConvexSetsTester
# #############################################################################


class Test_ConvexSetsTester(hunitest.TestCase):
    """
    Test the ConvexSetsTester class.
    """

    def test_test_dichotomy_always_true1(self) -> None:
        """
        Test that all dichotomies are realizable.
        """
        tester = mtugrfun.ConvexSetsTester()
        generator = mtugrfun.PointGenerator(seed=42)
        points = generator.generate_circle(n=5)
        # Test random dichotomies.
        enumerator = mtugrfun.DichotomyEnumerator(n=5)
        for i in range(10):
            labels = enumerator.get_dichotomy(i)
            # Should always be realizable.
            self.assertTrue(tester.test_dichotomy(points, labels))

    def test_find_hypothesis1(self) -> None:
        """
        Test finding convex hull.
        """
        tester = mtugrfun.ConvexSetsTester()
        generator = mtugrfun.PointGenerator(seed=42)
        points = generator.generate_circle(n=6)
        labels = np.array([1, 1, 1, -1, -1, -1])
        # Find hypothesis.
        result = tester.find_hypothesis(points, labels)
        self.assertIsNotNone(result)
        self.assertIn("hull_points", result)

    @pytest.mark.slow
    def test_growth_function_theoretical1(self) -> None:
        """
        Test that m_H(N) = 2^N for convex sets.
        """
        tester = mtugrfun.ConvexSetsTester()
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        for n in [1, 2, 3, 4, 5]:
            points = generator.generate_circle(n)
            result = calculator.compute_growth_function(points)
            expected = 2**n
            # Should match theoretical formula.
            self.assertEqual(result["m_h_n"], expected)
            self.assertTrue(result["is_shattered"])


# #############################################################################
# Test_GrowthFunctionCalculator
# #############################################################################


class Test_GrowthFunctionCalculator(hunitest.TestCase):
    """
    Test the GrowthFunctionCalculator class.
    """

    def test_compute_growth_function1(self) -> None:
        """
        Test computing growth function.
        """
        tester = mtugrfun.PositiveRaysTester()
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        points = generator.generate_line_1d(n=4)
        result = calculator.compute_growth_function(points)
        # Verify result structure.
        self.assertIn("n", result)
        self.assertIn("m_h_n", result)
        self.assertIn("max_dichotomies", result)
        self.assertIn("fraction", result)
        self.assertIn("is_shattered", result)
        self.assertIn("realizable_dichotomies", result)
        # Verify values.
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["m_h_n"], 5)

    @pytest.mark.slow
    def test_compute_growth_curve1(self) -> None:
        """
        Test computing growth curve.
        """
        tester = mtugrfun.PerceptronTester(random_state=42)
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        results_df = calculator.compute_growth_curve(
            generator, n_range=[1, 2, 3, 4], num_trials=1
        )
        # Verify DataFrame structure.
        self.assertIn("n", results_df.columns)
        self.assertIn("m_h_n_mean", results_df.columns)
        self.assertIn("max_dichotomies", results_df.columns)
        # Verify break point at N=4.
        row_4 = results_df[results_df["n"] == 4].iloc[0]
        self.assertLess(row_4["m_h_n_mean"], 16)

    def test_find_break_point1(self) -> None:
        """
        Test finding break point.
        """
        tester = mtugrfun.PerceptronTester(random_state=42)
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        break_point = calculator.find_break_point(generator, max_n=10)
        # Break point should be 4 for 2D perceptron.
        self.assertIsNotNone(break_point)
        self.assertLessEqual(break_point, 4)

    def test_find_shattered_points1(self) -> None:
        """
        Test finding shattered points.
        """
        tester = mtugrfun.PerceptronTester(random_state=42)
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        # 3 points should be shattered.
        points_3 = generator.generate_random(n=3, d=2)
        self.assertTrue(calculator.find_shattered_points(points_3))

    @pytest.mark.slow
    def test_estimate_vc_dimension1(self) -> None:
        """
        Test estimating VC dimension.
        """
        tester = mtugrfun.PerceptronTester(random_state=42)
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        generator = mtugrfun.PointGenerator(seed=42)
        result = calculator.estimate_vc_dimension(
            generator, max_n=6, num_trials=3
        )
        # Verify result structure.
        self.assertIn("vc_dimension", result)
        self.assertIn("break_point", result)
        self.assertIn("results_by_n", result)
        # VC dimension should be 3 for 2D perceptron.
        self.assertGreaterEqual(result["vc_dimension"], 2)


# #############################################################################
# Test_GrowthFunctionVisualizer
# #############################################################################


class Test_GrowthFunctionVisualizer(hunitest.TestCase):
    """
    Test the GrowthFunctionVisualizer class.
    """

    def test_plot_growth_curve_no_error1(self) -> None:
        """
        Test plotting growth curve without errors.
        """
        visualizer = mtugrfun.GrowthFunctionVisualizer()
        # Create sample data.
        data = {
            "n": [1, 2, 3, 4],
            "m_h_n_mean": [2, 4, 8, 14],
            "m_h_n_std": [0, 0, 0, 0],
            "max_dichotomies": [2, 4, 8, 16],
            "hypothesis": ["Perceptron"] * 4,
        }
        import pandas as pd

        results_df = pd.DataFrame(data)
        # Should not raise error.
        try:
            visualizer.plot_growth_curve(results_df)
            plt.close()
        except Exception as e:
            self.fail(f"plot_growth_curve raised exception: {e}")

    def test_plot_multiple_growth_curves_no_error1(self) -> None:
        """
        Test plotting multiple growth curves without errors.
        """
        visualizer = mtugrfun.GrowthFunctionVisualizer()
        # Create sample data.
        import pandas as pd

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
        # Should not raise error.
        try:
            visualizer.plot_multiple_growth_curves(results_dict)
            plt.close()
        except Exception as e:
            self.fail(f"plot_multiple_growth_curves raised exception: {e}")


# #############################################################################
# Test_HelperFunctions
# #############################################################################


class Test_HelperFunctions(hunitest.TestCase):
    """
    Test helper functions.
    """

    def test_compute_theoretical_growth_positive_rays1(self) -> None:
        """
        Test theoretical growth for positive rays.
        """
        for n in [1, 2, 3, 4, 5]:
            result = mtugrfun.compute_theoretical_growth("Positive Rays", n)
            self.assertEqual(result, n + 1)

    def test_compute_theoretical_growth_positive_intervals1(self) -> None:
        """
        Test theoretical growth for positive intervals.
        """
        for n in [1, 2, 3, 4, 5]:
            result = mtugrfun.compute_theoretical_growth("Positive Intervals", n)
            expected = n * (n + 1) // 2 + 1
            self.assertEqual(result, expected)

    def test_compute_theoretical_growth_perceptron1(self) -> None:
        """
        Test theoretical growth for perceptron.
        """
        # Verify known values.
        self.assertEqual(mtugrfun.compute_theoretical_growth("Perceptron", 1), 2)
        self.assertEqual(mtugrfun.compute_theoretical_growth("Perceptron", 2), 4)
        self.assertEqual(mtugrfun.compute_theoretical_growth("Perceptron", 3), 8)
        self.assertEqual(
            mtugrfun.compute_theoretical_growth("Perceptron", 4), 14
        )

    def test_compute_theoretical_growth_convex1(self) -> None:
        """
        Test theoretical growth for convex sets.
        """
        for n in [1, 2, 3, 4, 5]:
            result = mtugrfun.compute_theoretical_growth("Convex Sets", n)
            self.assertEqual(result, 2**n)

    def test_compare_with_theory1(self) -> None:
        """
        Test comparing with theory.
        """
        import pandas as pd

        # Create sample results.
        data = {
            "n": [1, 2, 3, 4],
            "m_h_n_mean": [2, 3, 4, 5],
        }
        results_df = pd.DataFrame(data)
        # Compare with theory.
        compared = mtugrfun.compare_with_theory(results_df, "Positive Rays")
        # Verify columns added.
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
    def test_complete_workflow_perceptron1(self) -> None:
        """
        Test complete workflow for perceptron.
        """
        generator = mtugrfun.PointGenerator(seed=42)
        tester = mtugrfun.PerceptronTester(random_state=42)
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        # Compute growth curve.
        results_df = calculator.compute_growth_curve(
            generator, n_range=[1, 2, 3, 4], num_trials=1
        )
        # Verify theoretical match.
        compared = mtugrfun.compare_with_theory(results_df, "Perceptron")
        # Verify N=1,2,3 match theory exactly.
        for n in [1, 2, 3]:
            row = compared[compared["n"] == n].iloc[0]
            self.assertEqual(row["m_h_n_mean"], row["theoretical"])

    def test_complete_workflow_positive_rays1(self) -> None:
        """
        Test complete workflow for positive rays.
        """
        generator = mtugrfun.PointGenerator(seed=42)
        tester = mtugrfun.PositiveRaysTester()
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        # Compute growth curve.
        results_df = calculator.compute_growth_curve(
            generator, n_range=[1, 2, 3, 4, 5], num_trials=1
        )
        # Verify m_H(N) = N + 1 for all N.
        for _, row in results_df.iterrows():
            self.assertEqual(row["m_h_n_mean"], row["n"] + 1)

    def test_complete_workflow_positive_intervals1(self) -> None:
        """
        Test complete workflow for positive intervals.
        """
        generator = mtugrfun.PointGenerator(seed=42)
        tester = mtugrfun.PositiveIntervalsTester()
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        # Compute growth curve.
        results_df = calculator.compute_growth_curve(
            generator, n_range=[1, 2, 3, 4, 5], num_trials=1
        )
        # Verify theoretical match.
        compared = mtugrfun.compare_with_theory(results_df, "Positive Intervals")
        for _, row in compared.iterrows():
            self.assertEqual(row["m_h_n_mean"], row["theoretical"])

    @pytest.mark.slow
    def test_complete_workflow_convex_sets1(self) -> None:
        """
        Test complete workflow for convex sets.
        """
        generator = mtugrfun.PointGenerator(seed=42)
        tester = mtugrfun.ConvexSetsTester()
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        # Compute growth curve.
        results_df = calculator.compute_growth_curve(
            generator, n_range=[1, 2, 3, 4, 5], num_trials=1
        )
        # Verify m_H(N) = 2^N for all N.
        for _, row in results_df.iterrows():
            self.assertEqual(row["m_h_n_mean"], 2 ** int(row["n"]))


# #############################################################################
# Test_EdgeCases
# #############################################################################


class Test_EdgeCases(hunitest.TestCase):
    """
    Test edge cases and boundary conditions.
    """

    def test_n_equals_1(self) -> None:
        """
        Test with N=1 point.
        """
        generator = mtugrfun.PointGenerator(seed=42)
        for tester in [
            mtugrfun.PerceptronTester(random_state=42),
            mtugrfun.PositiveRaysTester(),
            mtugrfun.PositiveIntervalsTester(),
            mtugrfun.ConvexSetsTester(),
        ]:
            calculator = mtugrfun.GrowthFunctionCalculator(
                tester, verbose=False, show_progress=False
            )
            # Generate 1 point.
            hypothesis_name = tester.get_name()
            if "Positive" in hypothesis_name:
                points = generator.generate_line_1d(n=1)
            elif "Convex" in hypothesis_name:
                points = generator.generate_circle(n=1)
            else:
                points = generator.generate_random(n=1, d=2)
            result = calculator.compute_growth_function(points)
            # m_H(1) should be 2 for all hypothesis sets.
            self.assertEqual(result["m_h_n"], 2)

    def test_n_equals_0(self) -> None:
        """
        Test with N=0 points.
        """
        enumerator = mtugrfun.DichotomyEnumerator(n=0)
        self.assertEqual(enumerator.count_dichotomies(), 1)

    def test_all_identical_points(self) -> None:
        """
        Test with all identical points.
        """
        tester = mtugrfun.PerceptronTester(random_state=42)
        calculator = mtugrfun.GrowthFunctionCalculator(
            tester, verbose=False, show_progress=False
        )
        # Create 3 identical points.
        points = np.array([[0, 0], [0, 0], [0, 0]])
        result = calculator.compute_growth_function(points)
        # Only 2 dichotomies realizable (all +1 or all -1).
        self.assertEqual(result["m_h_n"], 2)
