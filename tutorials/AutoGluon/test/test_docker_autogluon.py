"""
Run each notebook in tutorials/AutoGluon/ inside Docker using docker_cmd.sh.

Import as:

import tutorials.AutoGluon.test.test_docker_autogluon as tatdoaut
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
    Run all Docker tests for tutorials/AutoGluon/.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that AutoGluon.API.ipynb runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "AutoGluon.API.ipynb"
        # Run test.
        self._helper(notebook_name)

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that AutoGluon.example.ipynb runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "AutoGluon.example.ipynb"
        # Run test.
        self._helper(notebook_name)
