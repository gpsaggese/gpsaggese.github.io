"""
Run each notebook in msml610/tutorials/L03_knowledge_representation/ inside
Docker using docker_cmd.sh.

Import as:

import msml610.tutorials.L03_knowledge_representation.test.test_docker_all as mtl03tdal
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
    Run all Docker tests for msml610/tutorials/L03_knowledge_representation/.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that L03_01_entailment_implication_inference.ipynb runs without
        error.
        """
        # Prepare inputs.
        notebook_name = "L03_01_entailment_implication_inference.ipynb"
        # Run test.
        self.helper(notebook_name)

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that L03_02_wumpus_world.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L03_02_wumpus_world.ipynb"
        # Run test.
        self.helper(notebook_name)

    @pytest.mark.slow
    def test3(self) -> None:
        """
        Test that L03_03_rule_based_expert_systems.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L03_03_rule_based_expert_systems.ipynb"
        # Run test.
        self.helper(notebook_name)

    @pytest.mark.slow
    def test4(self) -> None:
        """
        Test that L03_04_ontology_reasoning.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L03_04_ontology_reasoning.ipynb"
        # Run test.
        self.helper(notebook_name)

    @pytest.mark.slow
    def test5(self) -> None:
        """
        Test that L03_06_logic_solvers.ipynb runs without error.
        """
        # Prepare inputs.
        notebook_name = "L03_06_logic_solvers.ipynb"
        # Run test.
        self.helper(notebook_name)
