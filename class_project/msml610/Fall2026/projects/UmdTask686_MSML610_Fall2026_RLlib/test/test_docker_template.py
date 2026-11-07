"""
Import as:

import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.test.test_docker_template as ttdote
"""

import logging

import pytest

import helpers.hdocker_tests as hdoctest
import helpers.hprint as hprint

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_docker
# #############################################################################


class Test_docker(hdoctest.DockerTestCase):
    """
    Run all Docker tests for the RLlib project.
    """

    def helper(self, notebook_name: str) -> None:
        """
        Run a notebook of the RLlib project inside Docker.

        :param notebook_name: notebook filename relative to the project dir
        """
        _LOG.debug(hprint.to_str("notebook_name"))
        # Prepare inputs.
        use_docker_cmd = True
        generate_html = False
        # Run test.
        self.run_notebook(
            notebook_name,
            use_docker_cmd=use_docker_cmd,
            generate_html=generate_html,
        )

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that `rllib.example.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "rllib.example.ipynb"
        # Run test.
        self.helper(notebook_name)

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that `rllib.API.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "rllib.API.ipynb"
        # Run test.
        self.helper(notebook_name)
