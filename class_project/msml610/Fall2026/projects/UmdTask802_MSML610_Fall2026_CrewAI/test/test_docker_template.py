"""
Test the CrewAI project notebooks inside Docker.
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
    Run Docker notebook tests for the CrewAI project.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that `crewai.example.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "crewai.example.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that `crewai.API.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "crewai.API.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )
