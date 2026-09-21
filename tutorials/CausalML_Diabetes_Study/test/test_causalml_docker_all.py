"""
Run each notebook in CausalML_Diabetes_Study/ inside Docker using docker_cmd.sh.

Import as:

import CausalML_Diabetes_Study.test.test_docker_all as causalml_tdal
"""

import pytest

import helpers.hdocker_tests as hdoctest


# #############################################################################
# Test_docker
# #############################################################################


class Test_docker(hdoctest.DockerTestCase):
    """
    Run all Docker tests for CausalML_Diabetes_Study/.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that CausalML.example.ipynb runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "CausalML.example.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that CausalML.API.ipynb runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "CausalML.API.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )
