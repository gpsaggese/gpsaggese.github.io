"""
Run each notebook in msml610/tutorials/L09_multi_armed_bandits/ inside Docker
using docker_cmd.sh.

Import as:

import msml610.tutorials.L09_multi_armed_bandits.test.test_docker_all as mtl09mabdal
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
    Run all Docker tests for msml610/tutorials/L09_multi_armed_bandits/.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that L09_03_02_multi_armed_bandits.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L09_03_02_multi_armed_bandits.ipynb"
        # Run test.
        self.helper(notebook_name, generate_html=True)
