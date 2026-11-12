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
    Run all Docker tests in the `tutorial` dir.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that Lesson10_01_q_learning/q_learning.ipynb runs without error
        inside Docker.
        """
        # Prepare inputs.
        notebook_name = "Lesson10_01_q_learning/q_learning.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )
