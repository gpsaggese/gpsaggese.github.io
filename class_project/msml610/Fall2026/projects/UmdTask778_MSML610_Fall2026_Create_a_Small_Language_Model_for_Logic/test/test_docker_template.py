"""
Import as:

import class_project.project_template.test.test_docker_all as tptdal
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
    Run all Docker tests for `class_project/msml610/Fall2026/projects/UmdTask778_MSML610_Fall2026_Create_a_Small_Language_Model_for_Logic`.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that `UmdTask778_MSML610_Fall2026_Create_a_Small_Language_Model_for_Logic.example.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "UmdTask778_MSML610_Fall2026_Create_a_Small_Language_Model_for_Logic.example.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that `UmdTask778_MSML610_Fall2026_Create_a_Small_Language_Model_for_Logic.API.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "UmdTask778_MSML610_Fall2026_Create_a_Small_Language_Model_for_Logic.API.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )
