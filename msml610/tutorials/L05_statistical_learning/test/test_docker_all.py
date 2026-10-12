"""
Run each notebook in msml610/tutorials/L05_statistical_learning/ inside Docker
using run_nbconvert.py.

Import as:

import msml610.tutorials.L05_statistical_learning.test.test_docker_all as mtl05tdal
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
    Run all Docker tests for msml610/tutorials/L05_statistical_learning/.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that L05_01_01_hoeffding_inequality.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L05_01_01_hoeffding_inequality.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=False, generate_html=True
        )

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that L05_01_02_bin_analogy_ml.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L05_01_02_bin_analogy_ml.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=False, generate_html=True
        )

    @pytest.mark.slow
    def test3(self) -> None:
        """
        Test that L05_01_03_vc_dimension.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L05_01_03_vc_dimension.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=False, generate_html=True
        )

    @pytest.mark.slow
    def test4(self) -> None:
        """
        Test that L05_01_04_growth_function.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L05_01_04_growth_function.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=False, generate_html=True
        )

    @pytest.mark.slow
    def test5(self) -> None:
        """
        Test that L05_02_01_bias_variance.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L05_02_01_bias_variance.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=False, generate_html=True
        )

    @pytest.mark.slow
    def test6(self) -> None:
        """
        Test that L05_02_02_overfitting.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L05_02_02_overfitting.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=False, generate_html=True
        )
