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
    Run all Docker tests for `class_project/project_template`.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that `template.example.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "template.example.ipynb"
        # Run test.
        self.helper(notebook_name)

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that `template.API.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "template.API.ipynb"
        # Run test.
        self.helper(notebook_name)
