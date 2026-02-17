"""
Unit tests for L09_05_01_discrete_bayes_dog_utils module.

Import as:

import msml610.tutorials.test.test_L09_05_01_discrete_bayes_dog_utils as mtttl0dbdu
"""

import logging

import numpy as np

import helpers.hunit_test as hunitest
import msml610.tutorials.L09_05_01_discrete_bayes_dog_utils as mtl00dbdu

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_discrete_bayes_sim
# #############################################################################


class Test_discrete_bayes_sim(hunitest.TestCase):
    """
    Test the discrete_bayes_sim function.
    """

    def test1(self) -> None:
        """
        Test basic simulation with uniform prior and multiple steps.
        """
        # Prepare inputs.
        prior = np.array([0.1] * 10)
        kernel = (0.1, 0.8, 0.1)
        hallway = mtl00dbdu.get_hallway1()
        positions = [0, 1, 2, 3]
        sensor_info = mtl00dbdu.get_sensor_info(positions, hallway)
        z_prob = 0.8
        # Run test.
        priors, posteriors = mtl00dbdu.discrete_bayes_sim(
            prior, kernel, sensor_info, z_prob, hallway
        )
        # Check outputs.
        self.assertEqual(len(priors), 4)
        self.assertEqual(len(posteriors), 4)
        # Verify all beliefs sum to 1.
        for p in priors:
            self.assertAlmostEqual(np.sum(p), 1.0, places=6)
        for p in posteriors:
            self.assertAlmostEqual(np.sum(p), 1.0, places=6)

    def test2(self) -> None:
        """
        Test simulation with single measurement.
        """
        # Prepare inputs.
        prior = np.array([0.1] * 10)
        kernel = (0.1, 0.8, 0.1)
        hallway = mtl00dbdu.get_hallway1()
        positions = [0]
        sensor_info = mtl00dbdu.get_sensor_info(positions, hallway)
        z_prob = 0.75
        # Run test.
        priors, posteriors = mtl00dbdu.discrete_bayes_sim(
            prior, kernel, sensor_info, z_prob, hallway
        )
        # Check outputs.
        self.assertEqual(len(priors), 1)
        self.assertEqual(len(posteriors), 1)
        # Verify beliefs sum to 1.
        self.assertAlmostEqual(np.sum(priors[0]), 1.0, places=6)
        self.assertAlmostEqual(np.sum(posteriors[0]), 1.0, places=6)

    def test3(self) -> None:
        """
        Test simulation with peaked prior distribution.
        """
        # Prepare inputs.
        prior = np.zeros(10)
        prior[3] = 1.0
        kernel = (0.1, 0.8, 0.1)
        hallway = mtl00dbdu.get_hallway1()
        positions = [3, 4, 5]
        sensor_info = mtl00dbdu.get_sensor_info(positions, hallway)
        z_prob = 0.9
        # Run test.
        priors, posteriors = mtl00dbdu.discrete_bayes_sim(
            prior, kernel, sensor_info, z_prob, hallway
        )
        # Check outputs.
        self.assertEqual(len(priors), 3)
        self.assertEqual(len(posteriors), 3)
        # Verify all beliefs sum to 1.
        for p in priors:
            self.assertAlmostEqual(np.sum(p), 1.0, places=6)
        for p in posteriors:
            self.assertAlmostEqual(np.sum(p), 1.0, places=6)

    def test4(self) -> None:
        """
        Test simulation with perfect sensor.
        """
        # Prepare inputs.
        prior = np.array([0.1] * 10)
        kernel = (0.1, 0.8, 0.1)
        hallway = mtl00dbdu.get_hallway1()
        positions = [0, 1, 8]
        sensor_info = mtl00dbdu.get_sensor_info(positions, hallway)
        z_prob = 1.0
        # Run test.
        priors, posteriors = mtl00dbdu.discrete_bayes_sim(
            prior, kernel, sensor_info, z_prob, hallway
        )
        # Check outputs.
        self.assertEqual(len(priors), 3)
        self.assertEqual(len(posteriors), 3)
        # Verify all beliefs sum to 1.
        for p in priors:
            self.assertAlmostEqual(np.sum(p), 1.0, places=6)
        for p in posteriors:
            self.assertAlmostEqual(np.sum(p), 1.0, places=6)

    def test5(self) -> None:
        """
        Test with different hallway configuration.
        """
        # Prepare inputs.
        prior = np.array([0.1] * 10)
        kernel = (0.1, 0.8, 0.1)
        hallway = mtl00dbdu.get_hallway2()
        positions = [0, 1, 2, 3, 4]
        # Create sensor_info manually for different hallway.
        z_doors = [hallway[z] for z in positions]
        z_moves = [0] + [positions[i] - positions[i-1] for i in range(1, len(positions))]
        sensor_info = {
            "positions": positions,
            "z_doors": z_doors,
            "z_moves": z_moves,
        }
        z_prob = 0.7
        # Run test.
        priors, posteriors = mtl00dbdu.discrete_bayes_sim(
            prior, kernel, sensor_info, z_prob, hallway
        )
        # Check outputs.
        self.assertEqual(len(priors), 5)
        self.assertEqual(len(posteriors), 5)
        # Verify beliefs sum to 1.
        for p in priors:
            self.assertAlmostEqual(np.sum(p), 1.0, places=6)
        for p in posteriors:
            self.assertAlmostEqual(np.sum(p), 1.0, places=6)

    def test6(self) -> None:
        """
        Test with extended movement sequence.
        """
        # Prepare inputs.
        prior = np.array([0.1] * 10)
        kernel = (0.1, 0.8, 0.1)
        hallway = mtl00dbdu.get_hallway1()
        positions = mtl00dbdu.get_dog_movements1()
        sensor_info = mtl00dbdu.get_sensor_info(positions, hallway)
        z_prob = 0.85
        # Run test.
        priors, posteriors = mtl00dbdu.discrete_bayes_sim(
            prior, kernel, sensor_info, z_prob, hallway
        )
        # Check outputs.
        self.assertEqual(len(priors), len(positions))
        self.assertEqual(len(posteriors), len(positions))
        # Verify first and last beliefs sum to 1.
        self.assertAlmostEqual(np.sum(priors[0]), 1.0, places=6)
        self.assertAlmostEqual(np.sum(posteriors[0]), 1.0, places=6)
        self.assertAlmostEqual(np.sum(priors[-1]), 1.0, places=6)
        self.assertAlmostEqual(np.sum(posteriors[-1]), 1.0, places=6)
