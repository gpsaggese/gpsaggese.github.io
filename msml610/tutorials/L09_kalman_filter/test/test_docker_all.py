"""
Run each notebook in msml610/tutorials/L09_kalman_filter/ inside Docker
using docker_cmd.sh.

Import as:

import msml610.tutorials.L09_kalman_filter.test.test_docker_all as mtl09kfdal
"""

import logging

import pytest

import helpers.hdocker_tests as hdoctest

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_docker
# #############################################################################


class Test_docker(hdoctest.DockerTestCase):
    """
    Run all Docker tests for msml610/tutorials/L09_kalman_filter/.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that L09_04_gh_filter.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L09_04_gh_filter.ipynb"
        # Run test.
        self.helper(notebook_name, generate_html=True)

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that L09_05_01_discrete_bayes_dog.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L09_05_01_discrete_bayes_dog.ipynb"
        # Run test.
        self.helper(notebook_name, generate_html=True)

    @pytest.mark.slow
    def test3(self) -> None:
        """
        Test that L09_05_02_univariate_kalman_filter.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L09_05_02_univariate_kalman_filter.ipynb"
        # Run test.
        self.helper(notebook_name, generate_html=True)

    @pytest.mark.slow
    def test4(self) -> None:
        """
        Test that L09_05_03_multivariate_kalman_filter.ipynb runs without
        error.
        """
        # Prepare inputs.
        notebook_name = "L09_05_03_multivariate_kalman_filter.ipynb"
        # Run test.
        self.helper(notebook_name, generate_html=True)

    @pytest.mark.slow
    def test5(self) -> None:
        """
        Test that L09_05_04_non_linear_kalman_filter.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L09_05_04_non_linear_kalman_filter.ipynb"
        # Run test.
        self.helper(notebook_name, generate_html=True)
