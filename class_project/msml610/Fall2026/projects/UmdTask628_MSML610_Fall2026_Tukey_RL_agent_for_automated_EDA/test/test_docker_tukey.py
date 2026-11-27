"""
Run the Tukey notebooks inside the project container.

This test runs on the host: it starts its own container through
`docker_cmd.sh`, so it is excluded from the in-container unit tests (the repo's
`pytest.ini` ignores `test_docker_*.py`). Run it with:

> pytest test/test_docker_tukey.py

Import as:

import test.test_docker_tukey as ttedotuk
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
    Run the Tukey notebooks inside Docker.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that `tukey.example.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "tukey.example.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that `tukey.API.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "tukey.API.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )
