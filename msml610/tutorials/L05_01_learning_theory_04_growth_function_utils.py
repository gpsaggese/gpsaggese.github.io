"""
Growth function calculator for VC dimension analysis.

This module provides tools to compute the growth function m_H(N) for different
hypothesis sets by generating all possible dichotomies and testing which can be
realized by the hypothesis.

Import as:

import msml610.tutorials.L05_01_learning_theory_04_growth_function_utils as mtugrfun
"""

import abc
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial
import sklearn.linear_model
from tqdm import tqdm

_LOG = logging.getLogger(__name__)


# #############################################################################
# PointGenerator
# #############################################################################


class PointGenerator:
    """
    Generate N points in D-dimensional space with various configurations.

    This class provides methods to generate points for testing hypothesis sets.
    Supports random generation, special configurations (circle, grid), and
    reproducible generation via random seeds.
    """

    def __init__(self, *, seed: Optional[int] = None) -> None:
        """
        Initialize the point generator.

        :param seed: Random seed for reproducibility
        """
        self._seed = seed
        self._rng = np.random.RandomState(seed)

    def generate_random(
        self, n: int, d: int, bounds: Tuple[float, float] = (-1.0, 1.0)
    ) -> np.ndarray:
        """
        Generate N random points uniformly in D dimensions.

        :param n: Number of points to generate
        :param d: Dimensionality of the space
        :param bounds: Tuple of (min, max) for uniform distribution
        :return: Array of shape (n, d) containing point coordinates
        """
        min_val, max_val = bounds
        # Generate random points uniformly in the specified bounds.
        points = self._rng.uniform(min_val, max_val, size=(n, d))
        return points

    def generate_circle(self, n: int, radius: float = 1.0) -> np.ndarray:
        """
        Generate N points evenly spaced on a circle.

        Useful for 2D visualization and testing convex sets.

        :param n: Number of points to generate
        :param radius: Radius of the circle
        :return: Array of shape (n, 2) containing point coordinates
        """
        # Generate angles evenly spaced around the circle.
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # Convert to Cartesian coordinates.
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        points = np.column_stack([x, y])
        return points

    def generate_grid(
        self, n: int, d: int, bounds: Tuple[float, float] = (-1.0, 1.0)
    ) -> np.ndarray:
        """
        Generate points on a regular grid.

        Points are evenly spaced in each dimension. Total number of points
        may be slightly different from n to form a regular grid.

        :param n: Approximate number of points to generate
        :param d: Dimensionality of the space
        :param bounds: Tuple of (min, max) for grid bounds
        :return: Array of shape (m, d) where m is close to n
        """
        # Compute points per dimension to approximate n total points.
        points_per_dim = int(np.ceil(n ** (1.0 / d)))
        # Create linearly spaced values for each dimension.
        min_val, max_val = bounds
        axes = [np.linspace(min_val, max_val, points_per_dim) for _ in range(d)]
        # Create meshgrid and reshape to (m, d).
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.column_stack([axis.ravel() for axis in mesh])
        return points

    def generate_collinear(
        self, n: int, d: int, bounds: Tuple[float, float] = (-1.0, 1.0)
    ) -> np.ndarray:
        """
        Generate N collinear points (on a line).

        Useful for testing edge cases with perceptrons.

        :param n: Number of points to generate
        :param d: Dimensionality of the space (line will be along first dimension)
        :param bounds: Tuple of (min, max) for line extent
        :return: Array of shape (n, d) containing collinear point coordinates
        """
        # Create points along the first dimension.
        min_val, max_val = bounds
        first_coord = np.linspace(min_val, max_val, n)
        # Set all other dimensions to zero.
        points = np.zeros((n, d))
        points[:, 0] = first_coord
        return points

    def generate_line_1d(
        self, n: int, bounds: Tuple[float, float] = (-1.0, 1.0)
    ) -> np.ndarray:
        """
        Generate N points evenly spaced on a 1D line.

        Useful for testing positive rays and positive intervals.

        :param n: Number of points to generate
        :param bounds: Tuple of (min, max) for line extent
        :return: Array of shape (n, 1) containing point coordinates
        """
        # Generate evenly spaced points on 1D line.
        min_val, max_val = bounds
        points_1d = np.linspace(min_val, max_val, n)
        # Reshape to (n, 1).
        points = points_1d.reshape(-1, 1)
        return points


# #############################################################################
# DichotomyEnumerator
# #############################################################################


class DichotomyEnumerator:
    """
    Enumerate all possible binary classifications (dichotomies) of N points.

    For N points, there are 2^N possible ways to assign binary labels (+1/-1).
    This class generates all such labelings efficiently.
    """

    def __init__(self, n: int) -> None:
        """
        Initialize the dichotomy enumerator.

        :param n: Number of points to classify
        """
        self._n = n

    def count_dichotomies(self) -> int:
        """
        Return the total number of possible dichotomies.

        :return: 2^N where N is the number of points
        """
        return 2**self._n

    def get_dichotomy(self, index: int) -> np.ndarray:
        """
        Get a specific dichotomy by index.

        Uses bit representation to generate the dichotomy:
        index in [0, 2^N-1] maps to a unique binary labeling.

        :param index: Dichotomy index (0 to 2^N - 1)
        :return: Array of shape (N,) with values in {-1, +1}
        """
        # Convert index to binary representation.
        # Each bit corresponds to one point's label.
        labels = np.zeros(self._n, dtype=int)
        for i in range(self._n):
            # Check if bit i is set.
            if index & (1 << i):
                labels[i] = 1
            else:
                labels[i] = -1
        return labels

    def enumerate_all(self) -> Iterator[np.ndarray]:
        """
        Generate all possible dichotomies as an iterator.

        Yields dichotomies one at a time for memory efficiency.

        :return: Iterator yielding arrays of shape (N,) with values in {-1, +1}
        """
        # Iterate through all 2^N possibilities.
        total = self.count_dichotomies()
        for index in range(total):
            yield self.get_dichotomy(index)


# #############################################################################
# HypothesisTester
# #############################################################################


class HypothesisTester(abc.ABC):
    """
    Abstract base class for testing if a dichotomy is realizable.

    Each hypothesis set (perceptron, positive rays, etc.) implements this
    interface to test whether a given labeling of points can be achieved
    by some hypothesis in the set.
    """

    @abc.abstractmethod
    def test_dichotomy(self, points: np.ndarray, labels: np.ndarray) -> bool:
        """
        Test if the given labeling of points is realizable by this hypothesis set.

        :param points: Array of shape (N, D) containing point coordinates
        :param labels: Array of shape (N,) with desired labels in {-1, +1}
        :return: True if realizable, False otherwise
        """
        pass

    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this hypothesis set.

        :return: Human-readable name (e.g., "2D Perceptron")
        """
        pass

    def find_hypothesis(
        self, points: np.ndarray, labels: np.ndarray
    ) -> Optional[Dict]:
        """
        Find a hypothesis that realizes the given labeling.

        Optional method that returns the parameters of a hypothesis that
        achieves the desired labeling. Returns None if not realizable.

        :param points: Array of shape (N, D) containing point coordinates
        :param labels: Array of shape (N,) with desired labels in {-1, +1}
        :return: Dictionary with hypothesis parameters, or None if not realizable
        """
        # Default implementation: just test without returning parameters
        if self.test_dichotomy(points, labels):
            return {}
        return None


# #############################################################################
# PerceptronTester
# #############################################################################


class PerceptronTester(HypothesisTester):
    """
    Test if a dichotomy is realizable by a perceptron (linear separator).

    Uses sklearn.linear_model.Perceptron to fit the data and verify if
    the resulting hyperplane correctly classifies all points.
    """

    def __init__(
        self,
        *,
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the perceptron tester.

        :param max_iter: Maximum iterations for perceptron training
        :param tol: Tolerance for training convergence
        :param random_state: Random seed for reproducibility
        """
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state

    def test_dichotomy(self, points: np.ndarray, labels: np.ndarray) -> bool:
        """
        Test if labels can be separated by a linear hyperplane.

        Uses LinearSVC for more robust linear separability testing than
        the standard perceptron algorithm.

        :param points: Array of shape (N, D) containing point coordinates
        :param labels: Array of shape (N,) with desired labels in {-1, +1}
        :return: True if linearly separable, False otherwise
        """
        # Handle edge case: all labels the same.
        if np.all(labels == labels[0]):
            # Trivially separable.
            return True
        # Handle edge case: only one unique point.
        if len(np.unique(points, axis=0)) == 1:
            # Can only separate if all labels are the same.
            return np.all(labels == labels[0])
        try:
            # Use LinearSVC for robust linear separability test.
            # Large C value enforces hard margin (finds separator if it exists).
            import sklearn.svm

            clf = sklearn.svm.LinearSVC(
                C=1e10,
                max_iter=self._max_iter * 10,
                random_state=self._random_state,
                dual="auto",
            )
            clf.fit(points, labels)
            # Check if perfect classification achieved.
            predictions = clf.predict(points)
            is_separable = np.array_equal(predictions, labels)
            return is_separable
        except Exception as e:
            # If training fails, assume not separable.
            _LOG.debug(f"LinearSVC training failed: {e}")
            return False

    def get_name(self) -> str:
        """
        Get the name of this hypothesis set.

        :return: "Perceptron"
        """
        return "Perceptron"

    def find_hypothesis(
        self, points: np.ndarray, labels: np.ndarray
    ) -> Optional[Dict]:
        """
        Find perceptron weights that realize the labeling.

        :param points: Array of shape (N, D) containing point coordinates
        :param labels: Array of shape (N,) with desired labels in {-1, +1}
        :return: Dict with 'weights' and 'intercept', or None if not realizable
        """
        # Handle edge case: all labels the same.
        if np.all(labels == labels[0]):
            # Return trivial solution.
            d = points.shape[1]
            return {
                "weights": np.zeros(d),
                "intercept": labels[0],
            }
        try:
            # Use LinearSVC for robust linear separability.
            import sklearn.svm

            clf = sklearn.svm.LinearSVC(
                C=1e10,
                max_iter=self._max_iter * 10,
                random_state=self._random_state,
                dual="auto",
            )
            clf.fit(points, labels)
            # Check if perfect classification achieved.
            predictions = clf.predict(points)
            if np.array_equal(predictions, labels):
                return {
                    "weights": clf.coef_[0],
                    "intercept": clf.intercept_[0],
                }
            return None
        except Exception as e:
            _LOG.debug(f"LinearSVC training failed: {e}")
            return None


# #############################################################################
# PositiveRaysTester
# #############################################################################


class PositiveRaysTester(HypothesisTester):
    """
    Test if a dichotomy is realizable by positive rays.

    Positive rays: h(x) = +1 if x >= a, else -1 for some threshold a.
    Valid labelings have at most one transition from -1 to +1 when points
    are sorted.
    """

    def __init__(self) -> None:
        """Initialize the positive rays tester."""
        pass

    def test_dichotomy(self, points: np.ndarray, labels: np.ndarray) -> bool:
        """
        Test if labels form a valid positive ray pattern.

        Points must be 1D. After sorting by position, labels should have
        at most one transition from -1 to +1.

        :param points: Array of shape (N, 1) containing 1D point coordinates
        :param labels: Array of shape (N,) with desired labels in {-1, +1}
        :return: True if valid ray pattern, False otherwise
        """
        # Extract 1D coordinates.
        points_1d = points.flatten()
        # Sort points and labels together.
        sorted_indices = np.argsort(points_1d)
        sorted_labels = labels[sorted_indices]
        # Count transitions from -1 to +1.
        transitions = 0
        for i in range(len(sorted_labels) - 1):
            if sorted_labels[i] == -1 and sorted_labels[i + 1] == 1:
                transitions += 1
            elif sorted_labels[i] == 1 and sorted_labels[i + 1] == -1:
                # Transition from +1 to -1 is not allowed for rays.
                return False
        # Valid if at most one transition from -1 to +1.
        return transitions <= 1

    def get_name(self) -> str:
        """
        Get the name of this hypothesis set.

        :return: "Positive Rays"
        """
        return "Positive Rays"

    def find_hypothesis(
        self, points: np.ndarray, labels: np.ndarray
    ) -> Optional[Dict]:
        """
        Find threshold that realizes the labeling.

        :param points: Array of shape (N, 1) containing 1D point coordinates
        :param labels: Array of shape (N,) with desired labels in {-1, +1}
        :return: Dict with 'threshold', or None if not realizable
        """
        if not self.test_dichotomy(points, labels):
            return None
        # Extract 1D coordinates.
        points_1d = points.flatten()
        # Sort points and labels together.
        sorted_indices = np.argsort(points_1d)
        sorted_points = points_1d[sorted_indices]
        sorted_labels = labels[sorted_indices]
        # Find transition point.
        # If all -1, threshold is after last point.
        if np.all(sorted_labels == -1):
            threshold = sorted_points[-1] + 1.0
        # If all +1, threshold is before first point.
        elif np.all(sorted_labels == 1):
            threshold = sorted_points[0] - 1.0
        else:
            # Find transition from -1 to +1.
            for i in range(len(sorted_labels) - 1):
                if sorted_labels[i] == -1 and sorted_labels[i + 1] == 1:
                    # Place threshold between these two points.
                    threshold = (sorted_points[i] + sorted_points[i + 1]) / 2.0
                    break
        return {"threshold": threshold}


# #############################################################################
# PositiveIntervalsTester
# #############################################################################


class PositiveIntervalsTester(HypothesisTester):
    """
    Test if a dichotomy is realizable by positive intervals.

    Positive intervals: h(x) = +1 if a <= x <= b, else -1 for some interval [a,b].
    Valid labelings have +1 labels forming a contiguous interval when sorted.
    """

    def __init__(self) -> None:
        """Initialize the positive intervals tester."""
        pass

    def test_dichotomy(self, points: np.ndarray, labels: np.ndarray) -> bool:
        """
        Test if labels form a valid positive interval pattern.

        Points must be 1D. After sorting by position, all +1 labels should
        form a single contiguous interval (at most two transitions: -1 to +1, +1 to -1).

        :param points: Array of shape (N, 1) containing 1D point coordinates
        :param labels: Array of shape (N,) with desired labels in {-1, +1}
        :return: True if valid interval pattern, False otherwise
        """
        # Extract 1D coordinates.
        points_1d = points.flatten()
        # Sort points and labels together.
        sorted_indices = np.argsort(points_1d)
        sorted_labels = labels[sorted_indices]
        # Find all transitions and their positions.
        transition_neg_to_pos_pos = None
        transition_pos_to_neg_pos = None
        for i in range(len(sorted_labels) - 1):
            if sorted_labels[i] == -1 and sorted_labels[i + 1] == 1:
                if transition_neg_to_pos_pos is not None:
                    # More than one -1 to +1 transition.
                    return False
                transition_neg_to_pos_pos = i
            elif sorted_labels[i] == 1 and sorted_labels[i + 1] == -1:
                if transition_pos_to_neg_pos is not None:
                    # More than one +1 to -1 transition.
                    return False
                transition_pos_to_neg_pos = i
        # Valid interval: the -1→+1 transition (if present) must come before
        # the +1→-1 transition (if present). Pattern must be: -1*, +1*, -1*
        if (
            transition_neg_to_pos_pos is not None
            and transition_pos_to_neg_pos is not None
        ):
            return transition_neg_to_pos_pos < transition_pos_to_neg_pos
        # If there's at most one transition of each type, and they're in the
        # right order, it's valid.
        return True

    def get_name(self) -> str:
        """
        Get the name of this hypothesis set.

        :return: "Positive Intervals"
        """
        return "Positive Intervals"

    def find_hypothesis(
        self, points: np.ndarray, labels: np.ndarray
    ) -> Optional[Dict]:
        """
        Find interval boundaries that realize the labeling.

        :param points: Array of shape (N, 1) containing 1D point coordinates
        :param labels: Array of shape (N,) with desired labels in {-1, +1}
        :return: Dict with 'left_bound' and 'right_bound', or None if not realizable
        """
        if not self.test_dichotomy(points, labels):
            return None
        # Extract 1D coordinates.
        points_1d = points.flatten()
        # Sort points and labels together.
        sorted_indices = np.argsort(points_1d)
        sorted_points = points_1d[sorted_indices]
        sorted_labels = labels[sorted_indices]
        # Find +1 region.
        positive_indices = np.where(sorted_labels == 1)[0]
        # If no +1 labels, interval is empty.
        if len(positive_indices) == 0:
            # Empty interval: place it outside the range.
            left_bound = sorted_points[-1] + 1.0
            right_bound = sorted_points[-1] + 2.0
        else:
            # Find leftmost and rightmost +1 points.
            first_positive_idx = positive_indices[0]
            last_positive_idx = positive_indices[-1]
            # Set boundaries.
            if first_positive_idx == 0:
                left_bound = sorted_points[0] - 1.0
            else:
                left_bound = (
                    sorted_points[first_positive_idx - 1]
                    + sorted_points[first_positive_idx]
                ) / 2.0
            if last_positive_idx == len(sorted_points) - 1:
                right_bound = sorted_points[-1] + 1.0
            else:
                right_bound = (
                    sorted_points[last_positive_idx]
                    + sorted_points[last_positive_idx + 1]
                ) / 2.0
        return {"left_bound": left_bound, "right_bound": right_bound}


# #############################################################################
# ConvexSetsTester
# #############################################################################


class ConvexSetsTester(HypothesisTester):
    """
    Test if a dichotomy is realizable by convex sets.

    Convex sets: Select all +1 points and take their convex hull.
    All points inside the hull are +1, outside are -1.

    Theoretical result: ALL dichotomies are realizable (m_H(N) = 2^N).
    """

    def __init__(self) -> None:
        """Initialize the convex sets tester."""
        pass

    def test_dichotomy(self, points: np.ndarray, labels: np.ndarray) -> bool:
        """
        Test if labels can be realized by a convex set.

        For convex sets on points in general position, this should always
        return True as every dichotomy is realizable.

        :param points: Array of shape (N, 2) containing 2D point coordinates
        :param labels: Array of shape (N,) with desired labels in {-1, +1}
        :return: Always True (all dichotomies realizable)
        """
        # For convex sets, every dichotomy is realizable by design.
        # Simply select all +1 points and take their convex hull.
        return True

    def get_name(self) -> str:
        """
        Get the name of this hypothesis set.

        :return: "Convex Sets"
        """
        return "Convex Sets"

    def find_hypothesis(
        self, points: np.ndarray, labels: np.ndarray
    ) -> Optional[Dict]:
        """
        Find convex hull that realizes the labeling.

        :param points: Array of shape (N, 2) containing 2D point coordinates
        :param labels: Array of shape (N,) with desired labels in {-1, +1}
        :return: Dict with 'hull_points' (vertices of convex hull)
        """
        # Select all points labeled +1.
        positive_indices = np.where(labels == 1)[0]
        if len(positive_indices) == 0:
            # No positive points: empty hull.
            return {"hull_points": np.array([])}
        positive_points = points[positive_indices]
        if len(positive_points) < 3:
            # Fewer than 3 points: degenerate hull.
            return {"hull_points": positive_points}
        try:
            # Compute convex hull.
            hull = scipy.spatial.ConvexHull(positive_points)
            hull_points = positive_points[hull.vertices]
            return {"hull_points": hull_points}
        except Exception as e:
            # If hull computation fails, return all positive points.
            _LOG.debug(f"Convex hull computation failed: {e}")
            return {"hull_points": positive_points}


# #############################################################################
# GrowthFunctionCalculator
# #############################################################################


class GrowthFunctionCalculator:
    """
    Main calculator for computing growth functions and analyzing VC dimension.

    This class orchestrates the computation of m_H(N) by:
    1. Generating points
    2. Enumerating all dichotomies
    3. Testing which are realizable
    4. Computing statistics and finding break points
    """

    def __init__(
        self,
        hypothesis_tester: HypothesisTester,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> None:
        """
        Initialize the growth function calculator.

        :param hypothesis_tester: Hypothesis tester to use
        :param verbose: Whether to print detailed logs
        :param show_progress: Whether to show progress bars for long computations
        """
        self._hypothesis_tester = hypothesis_tester
        self._verbose = verbose
        self._show_progress = show_progress

    def compute_growth_function(self, points: np.ndarray) -> Dict[str, Any]:
        """
        Compute growth function m_H(N) for the given points.

        Tests all 2^N possible dichotomies and counts how many are realizable.

        :param points: Array of shape (N, D) containing point coordinates
        :return: Dictionary with:
            - 'n': Number of points
            - 'm_h_n': Growth function value (number of realizable dichotomies)
            - 'max_dichotomies': 2^N (total possible dichotomies)
            - 'fraction': m_H(N) / 2^N
            - 'is_shattered': True if m_H(N) == 2^N
            - 'realizable_dichotomies': List of realizable dichotomy indices
        """
        n = len(points)
        if self._verbose:
            _LOG.info(
                f"Computing growth function for {n} points using "
                f"{self._hypothesis_tester.get_name()}"
            )
        # Create dichotomy enumerator.
        enumerator = DichotomyEnumerator(n)
        total_dichotomies = enumerator.count_dichotomies()
        # Test each dichotomy.
        realizable_count = 0
        realizable_indices = []
        iterator = range(total_dichotomies)
        if self._show_progress and total_dichotomies > 16:
            iterator = tqdm(
                iterator,
                desc=f"Testing dichotomies (N={n})",
                disable=not self._show_progress,
            )
        for idx in iterator:
            labels = enumerator.get_dichotomy(idx)
            if self._hypothesis_tester.test_dichotomy(points, labels):
                realizable_count += 1
                realizable_indices.append(idx)
        # Compute statistics.
        fraction = realizable_count / total_dichotomies
        is_shattered = realizable_count == total_dichotomies
        if self._verbose:
            _LOG.info(
                f"m_H({n}) = {realizable_count} / {total_dichotomies} "
                f"({fraction:.2%})"
            )
            if is_shattered:
                _LOG.info(f"{n} points are shattered!")
        return {
            "n": n,
            "m_h_n": realizable_count,
            "max_dichotomies": total_dichotomies,
            "fraction": fraction,
            "is_shattered": is_shattered,
            "realizable_dichotomies": realizable_indices,
        }

    def compute_growth_curve(
        self,
        point_generator: PointGenerator,
        n_range: List[int],
        num_trials: int = 1,
    ) -> pd.DataFrame:
        """
        Compute growth function for a range of N values.

        For each N, generates points and computes m_H(N). Can run multiple
        trials with different random point configurations.

        :param point_generator: Point generator to use
        :param n_range: List of N values to test
        :param num_trials: Number of random trials per N value
        :return: DataFrame with columns:
            - 'n': Number of points
            - 'm_h_n_mean': Mean growth function value across trials
            - 'm_h_n_std': Standard deviation across trials
            - 'max_dichotomies': 2^N
            - 'is_shattered_mean': Fraction of trials where points were shattered
            - 'hypothesis': Name of hypothesis set
        """
        results = []
        for n in n_range:
            if self._verbose:
                _LOG.info(f"Computing for N={n} ({num_trials} trials)")
            trial_results = []
            for trial in range(num_trials):
                # Generate points based on hypothesis type.
                hypothesis_name = self._hypothesis_tester.get_name()
                if "Positive" in hypothesis_name:
                    # Use 1D points for positive rays/intervals.
                    points = point_generator.generate_line_1d(n)
                elif "Convex" in hypothesis_name:
                    # Use circle configuration for convex sets.
                    points = point_generator.generate_circle(n)
                else:
                    # Use random 2D points for perceptron.
                    points = point_generator.generate_random(n, 2)
                # Compute growth function.
                result = self.compute_growth_function(points)
                trial_results.append(result)
            # Aggregate trial results.
            m_h_n_values = [r["m_h_n"] for r in trial_results]
            is_shattered_values = [r["is_shattered"] for r in trial_results]
            results.append(
                {
                    "n": n,
                    "m_h_n_mean": np.mean(m_h_n_values),
                    "m_h_n_std": np.std(m_h_n_values),
                    "max_dichotomies": trial_results[0]["max_dichotomies"],
                    "is_shattered_mean": np.mean(is_shattered_values),
                    "hypothesis": self._hypothesis_tester.get_name(),
                }
            )
        return pd.DataFrame(results)

    def find_break_point(
        self, point_generator: PointGenerator, max_n: int = 10
    ) -> Optional[int]:
        """
        Find the break point (first N where m_H(N) < 2^N).

        :param point_generator: Point generator to use
        :param max_n: Maximum N to test
        :return: Break point N, or None if no break point found up to max_n
        """
        for n in range(1, max_n + 1):
            # Generate points based on hypothesis type.
            hypothesis_name = self._hypothesis_tester.get_name()
            if "Positive" in hypothesis_name:
                points = point_generator.generate_line_1d(n)
            elif "Convex" in hypothesis_name:
                points = point_generator.generate_circle(n)
            else:
                points = point_generator.generate_random(n, 2)
            # Check if shattered.
            if not self.find_shattered_points(points):
                if self._verbose:
                    _LOG.info(f"Break point found at N={n}")
                return n
        if self._verbose:
            _LOG.info(f"No break point found up to N={max_n}")
        return None

    def find_shattered_points(self, points: np.ndarray) -> bool:
        """
        Test if all dichotomies of the given points can be realized.

        :param points: Array of shape (N, D) containing point coordinates
        :return: True if all 2^N dichotomies are realizable (points are shattered)
        """
        result = self.compute_growth_function(points)
        return result["is_shattered"]

    def estimate_vc_dimension(
        self,
        point_generator: PointGenerator,
        max_n: int = 10,
        num_trials: int = 5,
    ) -> Dict[str, Any]:
        """
        Estimate the VC dimension of the hypothesis set.

        VC dimension is the largest N for which some configuration of N points
        can be shattered. Uses multiple random trials to search for shatterable
        configurations.

        :param point_generator: Point generator to use
        :param max_n: Maximum N to test
        :param num_trials: Number of random trials per N value
        :return: Dictionary with:
            - 'vc_dimension': Estimated VC dimension
            - 'break_point': First N where no shattered points found
            - 'results_by_n': Dict mapping N to whether any trial was shattered
        """
        results_by_n = {}
        vc_dimension = 0
        for n in range(1, max_n + 1):
            if self._verbose:
                _LOG.info(
                    f"Testing VC dimension for N={n} ({num_trials} trials)"
                )
            # Try multiple random configurations.
            any_shattered = False
            for trial in range(num_trials):
                # Generate points based on hypothesis type.
                hypothesis_name = self._hypothesis_tester.get_name()
                if "Positive" in hypothesis_name:
                    points = point_generator.generate_line_1d(n)
                elif "Convex" in hypothesis_name:
                    points = point_generator.generate_circle(n)
                else:
                    points = point_generator.generate_random(n, 2)
                # Check if shattered.
                if self.find_shattered_points(points):
                    any_shattered = True
                    break
            results_by_n[n] = any_shattered
            if any_shattered:
                vc_dimension = n
            else:
                # Break point found.
                break
        return {
            "vc_dimension": vc_dimension,
            "break_point": vc_dimension + 1 if vc_dimension < max_n else None,
            "results_by_n": results_by_n,
        }


# #############################################################################
# GrowthFunctionVisualizer
# #############################################################################


class GrowthFunctionVisualizer:
    """
    Create visualizations for growth function analysis.

    Provides methods to plot growth curves, compare with theoretical bounds,
    and visualize realizable vs unrealizable dichotomies.
    """

    def __init__(self, *, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Initialize the visualizer.

        :param figsize: Default figure size for plots
        """
        self._figsize = figsize

    def plot_growth_curve(
        self,
        results_df: pd.DataFrame,
        title: Optional[str] = None,
        show_exponential: bool = True,
    ) -> None:
        """
        Plot growth function m_H(N) vs N.

        :param results_df: DataFrame from compute_growth_curve()
        :param title: Plot title (default: auto-generate from hypothesis name)
        :param show_exponential: Whether to show 2^N reference curve
        """
        fig, ax = plt.subplots(figsize=self._figsize)
        # Plot growth function.
        ax.plot(
            results_df["n"],
            results_df["m_h_n_mean"],
            marker="o",
            linewidth=2,
            label=f"m_H(N) - {results_df['hypothesis'].iloc[0]}",
        )
        # Add error bars if std is available.
        if "m_h_n_std" in results_df.columns:
            ax.fill_between(
                results_df["n"],
                results_df["m_h_n_mean"] - results_df["m_h_n_std"],
                results_df["m_h_n_mean"] + results_df["m_h_n_std"],
                alpha=0.2,
            )
        # Plot exponential reference.
        if show_exponential:
            ax.plot(
                results_df["n"],
                results_df["max_dichotomies"],
                linestyle="--",
                linewidth=2,
                label="2^N",
                alpha=0.7,
            )
        # Format plot.
        ax.set_xlabel("N (number of points)", fontsize=12)
        ax.set_ylabel("Growth Function m_H(N)", fontsize=12)
        if title is None:
            title = f"Growth Function: {results_df['hypothesis'].iloc[0]}"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        plt.tight_layout()
        plt.show()

    def plot_multiple_growth_curves(
        self,
        results_dict: Dict[str, pd.DataFrame],
        title: str = "Growth Functions Comparison",
    ) -> None:
        """
        Plot multiple growth curves on the same axes.

        :param results_dict: Dictionary mapping hypothesis names to result DataFrames
        :param title: Plot title
        """
        fig, ax = plt.subplots(figsize=self._figsize)
        # Plot each hypothesis.
        for name, results_df in results_dict.items():
            ax.plot(
                results_df["n"],
                results_df["m_h_n_mean"],
                marker="o",
                linewidth=2,
                label=name,
            )
        # Plot exponential reference.
        # Use the first result to get N values.
        first_df = next(iter(results_dict.values()))
        ax.plot(
            first_df["n"],
            first_df["max_dichotomies"],
            linestyle="--",
            linewidth=2,
            label="2^N",
            alpha=0.5,
            color="black",
        )
        # Format plot.
        ax.set_xlabel("N (number of points)", fontsize=12)
        ax.set_ylabel("Growth Function m_H(N)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        plt.tight_layout()
        plt.show()

    def plot_dichotomy_grid(
        self,
        points: np.ndarray,
        hypothesis_tester: HypothesisTester,
        max_dichotomies: int = 16,
    ) -> None:
        """
        Visualize a grid of dichotomies showing which are realizable.

        Creates a grid of subplots, each showing one dichotomy. Realizable
        dichotomies are shown with green border, unrealizable with red.

        :param points: Array of shape (N, 2) containing 2D point coordinates
        :param hypothesis_tester: Hypothesis tester to use
        :param max_dichotomies: Maximum number of dichotomies to display
        """
        n = len(points)
        enumerator = DichotomyEnumerator(n)
        total_dichotomies = min(enumerator.count_dichotomies(), max_dichotomies)
        # Determine grid size.
        grid_size = int(np.ceil(np.sqrt(total_dichotomies)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        # Plot each dichotomy.
        for idx in range(total_dichotomies):
            ax = axes[idx]
            labels = enumerator.get_dichotomy(idx)
            is_realizable = hypothesis_tester.test_dichotomy(points, labels)
            # Plot points.
            for i, (point, label) in enumerate(zip(points, labels)):
                color = "blue" if label == 1 else "red"
                ax.scatter(
                    point[0],
                    point[1],
                    c=color,
                    s=100,
                    edgecolors="black",
                    linewidths=1,
                    zorder=3,
                )
            # Set border color based on realizability.
            border_color = "green" if is_realizable else "red"
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
            ax.set_title(f"#{idx}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        # Hide unused subplots.
        for idx in range(total_dichotomies, len(axes)):
            axes[idx].axis("off")
        plt.suptitle(
            f"Dichotomies for {n} points - {hypothesis_tester.get_name()}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    def plot_shatter_test(
        self,
        points: np.ndarray,
        hypothesis_tester: HypothesisTester,
    ) -> None:
        """
        Visualize whether points can be shattered.

        Shows statistics about realizable vs unrealizable dichotomies.

        :param points: Array of shape (N, D) containing point coordinates
        :param hypothesis_tester: Hypothesis tester to use
        """
        # Compute growth function.
        calculator = GrowthFunctionCalculator(
            hypothesis_tester, verbose=False, show_progress=True
        )
        result = calculator.compute_growth_function(points)
        # Create visualization.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self._figsize)
        # Plot 1: Points configuration.
        if points.shape[1] == 1:
            # 1D points.
            ax1.scatter(
                points[:, 0],
                np.zeros(len(points)),
                c="blue",
                s=200,
                edgecolors="black",
                linewidths=2,
            )
            ax1.set_yticks([])
        else:
            # 2D points.
            ax1.scatter(
                points[:, 0],
                points[:, 1],
                c="blue",
                s=200,
                edgecolors="black",
                linewidths=2,
            )
        ax1.set_title(f"Point Configuration (N={result['n']})", fontsize=12)
        ax1.grid(True, alpha=0.3)
        # Plot 2: Statistics.
        ax2.axis("off")
        text_content = f"Hypothesis Set: {hypothesis_tester.get_name()}\n\n"
        text_content += f"N (points):           {result['n']}\n"
        text_content += f"m_H(N):               {result['m_h_n']}\n"
        text_content += f"2^N:                  {result['max_dichotomies']}\n"
        text_content += f"Fraction:             {result['fraction']:.2%}\n\n"
        if result["is_shattered"]:
            text_content += "STATUS: Points are SHATTERED! ✓\n"
            text_content += "All dichotomies are realizable.\n"
        else:
            text_content += "STATUS: Points are NOT shattered.\n"
            unrealizable = result["max_dichotomies"] - result["m_h_n"]
            text_content += f"{unrealizable} dichotomies are unrealizable.\n"
        ax2.text(
            0.1,
            0.5,
            text_content,
            fontsize=12,
            verticalalignment="center",
            family="monospace",
        )
        plt.suptitle(
            "Shatter Test",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()


# #############################################################################
# Helper Functions
# #############################################################################


def compute_theoretical_growth(hypothesis_name: str, n: int) -> int:
    """
    Compute theoretical growth function value for known hypothesis sets.

    :param hypothesis_name: Name of hypothesis set
    :param n: Number of points
    :return: Theoretical m_H(n) value, or -1 if unknown
    """
    hypothesis_name_lower = hypothesis_name.lower()
    if (
        "positive rays" in hypothesis_name_lower
        or "ray" in hypothesis_name_lower
    ):
        # m_H(N) = N + 1 for positive rays.
        return n + 1
    elif (
        "positive intervals" in hypothesis_name_lower
        or "interval" in hypothesis_name_lower
    ):
        # m_H(N) = N(N+1)/2 + 1 for positive intervals.
        return n * (n + 1) // 2 + 1
    elif "perceptron" in hypothesis_name_lower:
        # m_H(N) = 2^N for N <= 3, then breaks at N = 4.
        # For 2D perceptron: m_H(N) = sum_{i=0}^{d} C(N, i) where d = 2.
        # But empirically: m_H(1)=2, m_H(2)=4, m_H(3)=8, m_H(4)=14.
        if n <= 3:
            return 2**n
        else:
            # General formula for d-dimensional perceptron:
            # m_H(N) = 2 * sum_{i=0}^{d} C(N-1, i)
            # For d=2: m_H(N) = 2 * (C(N-1,0) + C(N-1,1) + C(N-1,2))
            # = 2 * (1 + (N-1) + (N-1)(N-2)/2)
            # = 2 * (1 + N - 1 + (N^2 - 3N + 2)/2)
            # = 2 * (N + (N^2 - 3N + 2)/2)
            # = N^2 - N + 2
            return n * n - n + 2
    elif "convex" in hypothesis_name_lower:
        # m_H(N) = 2^N for convex sets (all dichotomies realizable).
        return 2**n
    else:
        # Unknown hypothesis set.
        return -1


def compare_with_theory(
    results_df: pd.DataFrame, hypothesis_name: str
) -> pd.DataFrame:
    """
    Compare computed growth function with theoretical predictions.

    :param results_df: DataFrame from compute_growth_curve()
    :param hypothesis_name: Name of hypothesis set
    :return: DataFrame with additional 'theoretical' and 'error' columns
    """
    # Make a copy to avoid modifying original.
    df = results_df.copy()
    # Compute theoretical values.
    df["theoretical"] = df["n"].apply(
        lambda n: compute_theoretical_growth(hypothesis_name, n)
    )
    # Compute error.
    df["error"] = df["m_h_n_mean"] - df["theoretical"]
    df["error_pct"] = (df["error"] / df["theoretical"] * 100).where(
        df["theoretical"] > 0, np.nan
    )
    return df
