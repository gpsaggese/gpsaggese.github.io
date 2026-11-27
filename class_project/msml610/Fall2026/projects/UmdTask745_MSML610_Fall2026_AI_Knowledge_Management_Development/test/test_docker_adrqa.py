"""
Import as:

import class_project.msml610.Fall2026.projects.UmdTask745_MSML610_Fall2026_AI_Knowledge_Management_Development.test.test_docker_adrqa as tdadrqa
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
    Run the notebooks of the ADR knowledge base project inside Docker.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that the example notebook runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "UmdTask745_MSML610_Fall2026_AI_Knowledge_Management_Development.example.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that the API notebook runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "UmdTask745_MSML610_Fall2026_AI_Knowledge_Management_Development.API.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )
