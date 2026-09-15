"""
Run each notebook in msml610/tutorials/L12_reinforcement_learning/ inside
Docker using docker_cmd.sh.

Import as:

import msml610.tutorials.L12_reinforcement_learning.test.test_docker_all as mtl12rldal
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
    Run all Docker tests for msml610/tutorials/L12_reinforcement_learning/.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that L12_01_gridworld_4x3.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L12_01_gridworld_4x3.ipynb"
        # Run test.
        self.helper(notebook_name)

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that L12_02_gridworld_4x3_gymnasium.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L12_02_gridworld_4x3_gymnasium.ipynb"
        # Run test.
        self.helper(notebook_name)
