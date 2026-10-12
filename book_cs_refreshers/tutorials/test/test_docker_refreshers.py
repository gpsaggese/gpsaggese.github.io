"""
Run each notebook in class_project/project_template/ inside Docker using docker_cmd.sh.

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
    Run all Docker tests for class_project/project_template/.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that L91_01_refresher_probability.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "notebooks/L91_01_refresher_probability.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that L92_02_refresher_probability_distributions.ipynb runs
        without error.
        """
        # Prepare inputs.
        notebook_name = (
            "notebooks/L92_02_refresher_probability_distributions.ipynb"
        )
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )

    @pytest.mark.slow
    def test3(self) -> None:
        """
        Test that L94_04_information_theory.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "notebooks/L94_04_information_theory.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )

    @pytest.mark.slow
    def test4(self) -> None:
        """
        Test that L95_05_game_theory.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "notebooks/L95_05_game_theory.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )
