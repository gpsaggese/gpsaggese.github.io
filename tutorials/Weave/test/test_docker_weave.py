"""
Import as:

import tutorials.Weave.test.test_docker_weave as tptdal
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
    Run all Docker tests for `tutorials/Weave`.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that `weave.example.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "weave.example.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that `weave.API.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "weave.API.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )
