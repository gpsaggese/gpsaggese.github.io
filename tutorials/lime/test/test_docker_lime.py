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
        Test that 01.API.lime.ipynb runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "01.API.lime.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )
