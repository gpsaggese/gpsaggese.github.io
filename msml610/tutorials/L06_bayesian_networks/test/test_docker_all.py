"""
Run each notebook in msml610/tutorials/L06_bayesian_networks/ inside Docker
using run_nbconvert.py.

Import as:

import msml610.tutorials.L06_bayesian_networks.test.test_docker_all as mtl06bndal
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
    Run all Docker tests for msml610/tutorials/L06_bayesian_networks/.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that L06_01_exact_inference.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L06_01_exact_inference.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=False, generate_html=True
        )

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that L06_02_approximate_inference.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L06_02_approximate_inference.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=False, generate_html=True
        )
