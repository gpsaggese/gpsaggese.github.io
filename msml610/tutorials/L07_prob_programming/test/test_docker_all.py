"""
Run each notebook in msml610/tutorials/L07_prob_programming/ inside Docker
using run_nbconvert.py.

Import as:

import msml610.tutorials.L07_prob_programming.test.test_docker_all as mtl07tdal
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
    Run all Docker tests for msml610/tutorials/L07_prob_programming/.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that L07_01_bayesian_coin.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L07_01_bayesian_coin.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=False, generate_html=True
        )

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that L07_02_probabilistic_programming.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L07_02_probabilistic_programming.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=False, generate_html=True
        )

    @pytest.mark.slow
    def test3(self) -> None:
        """
        Test that L07_02_robust_modeling.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L07_02_robust_modeling.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=False, generate_html=True
        )

    @pytest.mark.slow
    def test4(self) -> None:
        """
        Test that L07_03_hierarchical_models.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L07_03_hierarchical_models.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=False, generate_html=True
        )

    @pytest.mark.slow
    def test5(self) -> None:
        """
        Test that L07_04_generalized_linear_models.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L07_04_generalized_linear_models.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=False, generate_html=True
        )

    @pytest.mark.slow
    def test6(self) -> None:
        """
        Test that L07_05_evaluating_models.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L07_05_evaluating_models.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=False, generate_html=True
        )
