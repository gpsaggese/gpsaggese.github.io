"""
Run each notebook in tutorials/LangChain_LangGraph/ inside Docker using docker_cmd.sh.

Import as:

import tutorials.LangChain_LangGraph.test.test_docker_all as ltdal
"""

import os

import pytest

import helpers.hdocker_tests as hdoctest

# Marker for tests that require LLM API credentials.
_REQUIRES_LLM_CREDENTIALS = pytest.mark.skipif(
    "LLM_PROVIDER" not in os.environ,
    reason="LLM_PROVIDER environment variable not set",
)


# #############################################################################
# Test_docker
# #############################################################################


class Test_docker(hdoctest.DockerTestCase):
    """
    Run all Docker tests for LangChain_LangGraph tutorials.
    """

    _test_file = __file__

    @pytest.mark.slow
    @_REQUIRES_LLM_CREDENTIALS
    def test1(self) -> None:
        """
        Test that langchain.example.ipynb runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "langchain.example.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )

    @pytest.mark.slow
    @_REQUIRES_LLM_CREDENTIALS
    def test2(self) -> None:
        """
        Test that langchain.API.ipynb runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "langchain.API.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )

    @pytest.mark.slow
    @_REQUIRES_LLM_CREDENTIALS
    def test3(self) -> None:
        """
        Test that langgraph.example.ipynb runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "langgraph.example.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )

    @pytest.mark.slow
    @_REQUIRES_LLM_CREDENTIALS
    def test4(self) -> None:
        """
        Test that deep_agents.example.ipynb runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "deep_agents.example.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )

    @pytest.mark.slow
    @_REQUIRES_LLM_CREDENTIALS
    def test5(self) -> None:
        """
        Test that deep_agents.API.ipynb runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "deep_agents.API.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )
