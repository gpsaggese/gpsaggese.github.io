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

    def _get_default_setup(self) -> tuple:
        """
        Return default prior, kernel, and hallway for standard tests.

        :return: Tuple of (prior, kernel, hallway)
        """
        prior = np.array([0.1] * 10)
        kernel = (0.1, 0.8, 0.1)
        hallway = mtl00dbdu.get_hallway1()
        return prior, kernel, hallway

    def _check_simulation_results(
        self,
        priors: mtl00dbdu.List[mtl00dbdu.Pdf],
        posteriors: mtl00dbdu.List[mtl00dbdu.Pdf],
        expected_length: int,
    ) -> None:
        """
        Validate simulation results for priors and posteriors.

        :param priors: List of prior belief distributions
        :param posteriors: List of posterior belief distributions
        :param expected_length: Expected number of time steps
        """
        # Check lengths.
        self.assertEqual(len(priors), expected_length)
        self.assertEqual(len(posteriors), expected_length)
        # Verify all beliefs sum to 1.
        for p in priors:
            self.assertAlmostEqual(np.sum(p), 1.0, places=6)
        for p in posteriors:
            self.assertAlmostEqual(np.sum(p), 1.0, places=6)

    def _get_most_likely_position(self, belief: mtl00dbdu.Pdf) -> int:
        """
        Extract the most likely position from a belief distribution.

        :param belief: Belief distribution array
        :return: Index of the position with highest probability
        """
        return int(np.argmax(belief))

    def _check_most_likely_positions(
        self,
        posteriors: mtl00dbdu.List[mtl00dbdu.Pdf],
        expected_pos: mtl00dbdu.PosList,
    ) -> None:
        """
        Verify that most likely positions from posteriors match expected values.

        :param posteriors: List of posterior belief distributions
        :param expected_pos: List of expected most likely positions
        """
        actual_pos = []
        # Check most likely position for each posterior.
        for i in range(len(posteriors)):
            most_likely_pos = self._get_most_likely_position(posteriors[i])
            actual_pos.append(most_likely_pos)
        self.assertEqual(actual_pos, expected_pos)

    def test1(self) -> None:
        """
        Test basic simulation with uniform prior and multiple steps.
        """
        # Prepare inputs.
        prior, kernel, hallway = self._get_default_setup()
        positions = [0, 1, 2, 3]
        sensor_info = mtl00dbdu.get_sensor_info(positions, hallway)
        z_prob = 0.8
        # Run test.
        priors, posteriors = mtl00dbdu.discrete_bayes_sim(
            prior, kernel, sensor_info, z_prob, hallway
        )
        # Check outputs.
        self._check_simulation_results(priors, posteriors, 4)
        # Check most likely positions match actual positions.
        self._check_most_likely_positions(posteriors, positions)

    def test2(self) -> None:
        """
        Test simulation with single measurement.
        """
        # Prepare inputs.
        prior, kernel, hallway = self._get_default_setup()
        positions = [0]
        sensor_info = mtl00dbdu.get_sensor_info(positions, hallway)
        z_prob = 0.75
        # Run test.
        priors, posteriors = mtl00dbdu.discrete_bayes_sim(
            prior, kernel, sensor_info, z_prob, hallway
        )
        # Check outputs.
        self._check_simulation_results(priors, posteriors, 1)
        # Check most likely position in final posterior.
        most_likely_pos = self._get_most_likely_position(posteriors[-1])
        # Check that the most likely position is one of the door positions.
        # With single measurement and uniform prior, the filter should identify
        # a door position (0, 1, or 8 in hallway1).
        door_positions = [0, 1, 8]
        self.assertIn(
            most_likely_pos,
            door_positions,
            f"Most likely position {most_likely_pos} should be a door position",
        )

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
        self._check_simulation_results(priors, posteriors, 3)
        # Check most likely position in final posterior.
        most_likely_pos = self._get_most_likely_position(posteriors[-1])
        # Check that the most likely position is the correct one.
        # With high sensor accuracy (z_prob=0.9) and peaked prior, the filter
        # should track to the final position.
        expected_final_pos = positions[-1]
        self.assertEqual(
            most_likely_pos,
            expected_final_pos,
            f"Most likely position {most_likely_pos} should match final position {expected_final_pos}",
        )
        # Check most likely positions match actual positions.
        self._check_most_likely_positions(posteriors, positions)

    def test4(self) -> None:
        """
        Test simulation with perfect sensor.
        """
        # Prepare inputs.
        prior, kernel, hallway = self._get_default_setup()
        positions = [0, 1, 8]
        sensor_info = mtl00dbdu.get_sensor_info(positions, hallway)
        z_prob = 1.0
        # Run test.
        priors, posteriors = mtl00dbdu.discrete_bayes_sim(
            prior, kernel, sensor_info, z_prob, hallway
        )
        # Check outputs.
        self._check_simulation_results(priors, posteriors, 3)
        # Check most likely positions match actual positions.
        # With perfect sensor (z_prob=1.0), tracking should be exact.
        self._check_most_likely_positions(posteriors, positions)

    def test5(self) -> None:
        """
        Test with different hallway configuration.
        """
        # Prepare inputs.
        prior = np.array([0.1] * 10)
        kernel = (0.1, 0.8, 0.1)
        hallway = mtl00dbdu.get_hallway2()
        positions = [0, 1, 2, 3, 4]
        sensor_info = mtl00dbdu.get_sensor_info(positions, hallway)
        z_prob = 0.7
        # Run test.
        priors, posteriors = mtl00dbdu.discrete_bayes_sim(
            prior, kernel, sensor_info, z_prob, hallway
        )
        # Check outputs.
        self._check_simulation_results(priors, posteriors, 5)
        # Check most likely positions match actual positions.
        self._check_most_likely_positions(posteriors, positions)

    def test6(self) -> None:
        """
        Test with extended movement sequence.
        """
        # Prepare inputs.
        prior, kernel, hallway = self._get_default_setup()
        positions = mtl00dbdu.get_dog_movements1()
        sensor_info = mtl00dbdu.get_sensor_info(positions, hallway)
        z_prob = 0.85
        # Run test.
        priors, posteriors = mtl00dbdu.discrete_bayes_sim(
            prior, kernel, sensor_info, z_prob, hallway
        )
        # Check outputs.
        self._check_simulation_results(priors, posteriors, len(positions))
        # Check most likely positions match actual positions.
        self._check_most_likely_positions(posteriors, positions)
