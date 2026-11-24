"""
Run the notebooks and the unit tests of this folder inside Docker using
docker_cmd.sh.

Import as:

import research.Causal_Analysis_of_Agent_Skill_And_Luck.test.test_docker_all as rcaoasalttda
"""

import logging
import os

import pytest

import helpers.hdocker_tests as hdoctest
import helpers.hgit as hgit

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_docker
# #############################################################################


class Test_docker(hdoctest.DockerTestCase):
    """
    Run all Docker tests for `research/Causal_Analysis_of_Agent_Skill_And_Luck`.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that `causal_success_API.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "causal_success_API.ipynb"
        # Run test.
        self.helper(notebook_name)

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that `causal_success_example.ipynb` runs without error inside
        Docker.
        """
        # Prepare inputs.
        notebook_name = "causal_success_example.ipynb"
        # Run test.
        self.helper(notebook_name)

    @pytest.mark.slow
    def test3(self) -> None:
        """
        Test that `agent_benchmark_API.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "agent_benchmark_API.ipynb"
        # Run test.
        self.helper(notebook_name)

    @pytest.mark.slow
    def test4(self) -> None:
        """
        Test that `agent_benchmark_example.ipynb` runs without error inside
        Docker.
        """
        # Prepare inputs.
        notebook_name = "agent_benchmark_example.ipynb"
        # Run test.
        self.helper(notebook_name)

    @pytest.mark.slow
    def test5(self) -> None:
        """
        Test that the unit tests of this folder pass inside Docker.
        """
        # Prepare inputs.
        docker_script_dir = self._get_docker_docker_script_dir()
        rel_dir = os.path.relpath(
            docker_script_dir, hgit.find_git_root(docker_script_dir)
        )
        test_files = " ".join(
            f"{rel_dir}/test/{test_file}"
            for test_file in (
                "test_agent_benchmark_utils.py",
                "test_causal_success_utils.py",
                "test_causal_success_analysis_utils.py",
            )
        )
        shell_cmd = f"cd /git_root && python -m pytest {test_files}"
        # Run test.
        hdoctest.run_docker_cmd(docker_script_dir, shell_cmd=shell_cmd)
