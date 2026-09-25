"""
Import as:

import research.Optimal_strategy_for_racket_sports.test.test_docker_racket_strategy as rosfrstdrst
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
    Run all Docker tests for `research/Optimal_strategy_for_racket_sports`.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that `racket_strategy.API.ipynb` runs without error inside
        Docker.
        """
        # Prepare inputs.
        notebook_name = "racket_strategy.API.ipynb"
        # Run test.
        self.helper(notebook_name)

    @pytest.mark.slow
    def test3(self) -> None:
        """
        Test that `racket_strategy.exploration.ipynb` runs without error
        inside Docker.
        """
        # Prepare inputs.
        notebook_name = "racket_strategy.exploration.ipynb"
        # Run test.
        self.helper(notebook_name)
