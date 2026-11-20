"""
Import as:

import class_project.msml610.Fall2026.projects.UmdTask851_MSML610_Fall2026_Kats_Stock_Price_Forecasting.test.test_docker_kats as tdkats
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
    Run all Docker tests for the Kats tutorial.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that `kats.example.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "kats.example.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that `kats.API.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "kats.API.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )
