"""
Import as:

import test_docker_template as tdt
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
    Run all Docker tests for the CLIP project.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that `clip.example.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "clip.example.ipynb"
        # Run test.
        self.helper(notebook_name)

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that `clip.API.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "clip.API.ipynb"
        # Run test.
        self.helper(notebook_name)
