"""
Import as:

import class_project.msml610.Fall2026.projects.UmdTask657_MSML610_Fall2026_WB_Weave_Project1.test.test_docker_weave as tdweave
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
    Run the Docker tests for the Weave FOMC tone project.
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
