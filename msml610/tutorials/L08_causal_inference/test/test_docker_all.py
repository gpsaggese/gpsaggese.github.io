"""
Run each notebook in msml610/tutorials/L08_causal_inference/ inside Docker
using run_nbconvert.py.

Import as:

import msml610.tutorials.L08_causal_inference.test.test_docker_all as mtl08tdal
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
    Run all Docker tests for msml610/tutorials/L08_causal_inference/.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that L08_04_01_causal_inference.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L08_04_01_causal_inference.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=False, generate_html=True
        )

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that L08_04_02_causal_inference.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L08_04_02_causal_inference.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=False, generate_html=True
        )
