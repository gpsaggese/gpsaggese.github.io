"""
Unit tests for L09_05_02_kalman_filter_utils module.

Import as:

import msml610.tutorials.test.test_L09_05_02_kalman_filter_utils as mtttl002kfiut
"""

import logging

import numpy as np

import helpers.hunit_test as hunitest
import msml610.tutorials.L09_05_02_kalman_filter_utils as mtl0kfiut

_LOG = logging.getLogger(__name__)


# #############################################################################
# TestDogSimulation
# #############################################################################


class TestDogSimulation(hunitest.TestCase):
    """
    Test the DogSimulation class.
    """

    def test1(self) -> None:
        """
        Test initialization with default parameters.
        """
        # Run test.
        dog = mtl0kfiut.DogSimulation()
        # Check output.
        self.assertAlmostEqual(dog.x, 0.0)
        self.assertAlmostEqual(dog.velocity, 1.0)
        self.assertAlmostEqual(dog.measurement_std, 0.0)
        self.assertAlmostEqual(dog.process_std, 0.0)

    def test2(self) -> None:
        """
        Test initialization with custom parameters stores values correctly.
        """
        # Prepare inputs.
        x0 = 5.0
        velocity = 2.0
        measurement_var = 4.0
        process_var = 1.0
        # Run test.
        dog = mtl0kfiut.DogSimulation(
            x0=x0,
            velocity=velocity,
            measurement_var=measurement_var,
            process_var=process_var,
        )
        # Check output.
        # measurement_std = sqrt(4.0) = 2.0; process_std = sqrt(1.0) = 1.0.
        self.assertAlmostEqual(dog.x, 5.0)
        self.assertAlmostEqual(dog.velocity, 2.0)
        self.assertAlmostEqual(dog.measurement_std, 2.0)
        self.assertAlmostEqual(dog.process_std, 1.0)

    def test3(self) -> None:
        """
        Test move() with zero process variance produces deterministic movement.
        """
        # Prepare inputs.
        dog = mtl0kfiut.DogSimulation(x0=0.0, velocity=1.0, process_var=0.0)
        # Run test.
        dog.move()
        # Check output.
        # With zero process variance: x = 0.0 + 1.0 * 1.0 = 1.0.
        self.assertAlmostEqual(dog.x, 1.0)

    def test4(self) -> None:
        """
        Test move() with custom dt and zero process variance.
        """
        # Prepare inputs.
        dog = mtl0kfiut.DogSimulation(x0=0.0, velocity=2.0, process_var=0.0)
        # Run test.
        dog.move(dt=0.5)
        # Check output.
        # With zero process variance: x = 0.0 + 2.0 * 0.5 = 1.0.
        self.assertAlmostEqual(dog.x, 1.0)

    def test5(self) -> None:
        """
        Test move() with negative velocity moves dog backward.
        """
        # Prepare inputs.
        dog = mtl0kfiut.DogSimulation(x0=5.0, velocity=-1.0, process_var=0.0)
        # Run test.
        dog.move()
        # Check output.
        # With negative velocity: x = 5.0 + (-1.0) * 1.0 = 4.0.
        self.assertAlmostEqual(dog.x, 4.0)

    def test6(self) -> None:
        """
        Test sense_position() with zero measurement variance returns exact
        position.
        """
        # Prepare inputs.
        dog = mtl0kfiut.DogSimulation(x0=3.5, measurement_var=0.0)
        # Run test.
        measurement = dog.sense_position()
        # Check output.
        # With no measurement noise, measurement equals actual position.
        self.assertAlmostEqual(measurement, 3.5)

    def test7(self) -> None:
        """
        Test move_and_sense() with zero noise returns exact position after move.
        """
        # Prepare inputs.
        dog = mtl0kfiut.DogSimulation(
            x0=0.0, velocity=1.0, measurement_var=0.0, process_var=0.0
        )
        # Run test.
        measurement = dog.move_and_sense()
        # Check output.
        # After one step: x = 0.0 + 1.0 = 1.0; measurement = 1.0.
        self.assertAlmostEqual(measurement, 1.0)
        self.assertAlmostEqual(dog.x, 1.0)

    def test8(self) -> None:
        """
        Test multiple move_and_sense() calls accumulate position correctly.
        """
        # Prepare inputs.
        dog = mtl0kfiut.DogSimulation(
            x0=0.0, velocity=1.0, measurement_var=0.0, process_var=0.0
        )
        n_steps = 5
        # Run test.
        measurements = [dog.move_and_sense() for _ in range(n_steps)]
        # Check output.
        # With no noise, positions accumulate as 1.0, 2.0, 3.0, 4.0, 5.0.
        expected = [1.0, 2.0, 3.0, 4.0, 5.0]
        for actual_m, exp_m in zip(measurements, expected):
            self.assertAlmostEqual(actual_m, exp_m)

    def test9(self) -> None:
        """
        Test move_and_sense() with process noise using a fixed random seed.
        """
        # Prepare inputs.
        np.random.seed(42)
        dog = mtl0kfiut.DogSimulation(
            x0=0.0, velocity=1.0, measurement_var=0.0, process_var=1.0
        )
        # Run test.
        measurement = dog.move_and_sense()
        # Check output.
        # With seed(42), randn()=0.4967141530112327.
        # dx = 1.0 + 0.4967141530112327 * 1.0 = 1.4967141530112327.
        # x = 0.0 + 1.4967141530112327; measurement (no meas noise) = x.
        expected = 1.4967141530112327
        self.assertAlmostEqual(measurement, expected, places=6)


# #############################################################################
# Test_End_to_End_Continuous_Bayes_Filter1
# #############################################################################


class Test_End_to_End_Continuous_Bayes_Filter1(hunitest.TestCase):
    """
    Test Kalman filter.
    """

    def test1(self) -> None:
        """
        From the notebook `L09_05_02_kalman_filter.ipynb`.
        """
        # Prepare inputs.
        np.random.seed(13)
        process_var = 1.0
        sensor_var = 2.0
        x = mtl0kfiut.Gaussian(0.0, 20.0**2)
        velocity = 1
        dt = 1.0
        process_model = mtl0kfiut.Gaussian(velocity * dt, process_var)
        dog = mtl0kfiut.DogSimulation(
            x0=x.mean,
            velocity=process_model.mean,
            measurement_var=sensor_var,
            process_var=process_model.var,
        )
        zs = [dog.move_and_sense() for _ in range(5)]
        # Run test.
        info = []
        for z in zs:
            prior = mtl0kfiut.predict(x, process_model)
            likelihood = mtl0kfiut.Gaussian(z, sensor_var)
            x = mtl0kfiut.update(prior, likelihood)
            info.append((prior, x, z))
        actual = mtl0kfiut.to_str(info)
        # Check output.
        self.check_string(str(actual))
