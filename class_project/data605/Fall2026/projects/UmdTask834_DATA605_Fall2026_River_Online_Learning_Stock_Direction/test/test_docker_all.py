"""
Import as:

import test.test_docker_all as ttdoal
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
    Run all Docker tests for `the River project`.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that `river.example.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "river.example.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that `river.API.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "river.API.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )
