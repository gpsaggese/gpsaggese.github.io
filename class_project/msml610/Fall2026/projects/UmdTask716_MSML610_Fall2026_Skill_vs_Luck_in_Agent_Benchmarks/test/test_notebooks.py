"""
Import as:

import test.test_notebooks as ttenot
"""

import logging
import os
from typing import Any, Dict

import pytest

import helpers.hunit_test as hunitest

_LOG = logging.getLogger(__name__)

# The dir of the project, where the notebooks read their data.
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Scores of the coding runs, downloaded by `fetch_coding_scores.py`.
_PHASE_1_DIR = "data/coding_runs"
_PHASE_2_DIR = "data/coding_runs_phase2"
_VARIANTS_DIR = "data/coding_runs_variants"


# #############################################################################
# Test_notebooks
# #############################################################################


class Test_notebooks(hunitest.TestCase):
    """
    Run each notebook from top to bottom.

    The smoke tests use a small `config` (fewer bootstrap samples, draws, and
    trials), so they check that the code runs, not the results. The superslow
    tests use the full `config` and freeze the main results.
    """

    def helper(
        self,
        name: str,
        overrides: Dict[str, Any],
        *,
        data_dirs: tuple = (),
        last_cell: str = "",
    ) -> str:
        """
        Run the paired script of a notebook, with a changed `config`.

        :param name: notebook name, e.g., `skill_vs_luck.records`
        :param overrides: values that replace those of the `config` of the
            notebook
        :param data_dirs: dirs of scores that the notebook needs; the test is
            skipped when one is missing
        :param last_cell: code to run at the end, e.g., to print a result
        :return: text printed by `last_cell`
        """
        missing = [
            path
            for path in data_dirs
            if not os.path.isdir(os.path.join(_PROJECT_DIR, path))
        ]
        if missing:
            pytest.skip(f"Missing {missing}: run `fetch_coding_scores.py` first")
        jupytext = pytest.importorskip("jupytext")
        nbclient = pytest.importorskip("nbclient")
        nbformat = pytest.importorskip("nbformat")
        notebook = jupytext.read(os.path.join(_PROJECT_DIR, f"{name}.py"))
        if overrides:
            # Change `config` right after the cell that defines it.
            config_cells = [
                index
                for index, cell in enumerate(notebook.cells)
                if cell.cell_type == "code" and "config = {" in cell.source
            ]
            self.assertEqual(len(config_cells), 1)
            notebook.cells.insert(
                config_cells[0] + 1,
                nbformat.v4.new_code_cell(f"config.update({overrides!r})"),
            )
        if last_cell:
            notebook.cells.append(nbformat.v4.new_code_cell(last_cell))
        client = nbclient.NotebookClient(
            notebook,
            timeout=3600,
            kernel_name="python3",
            resources={"metadata": {"path": _PROJECT_DIR}},
        )
        client.execute()
        printed = "".join(
            output.get("text", "")
            for output in notebook.cells[-1].get("outputs", [])
            if output.get("output_type") == "stream"
        )
        return printed

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that the API notebook runs; it has no `config`.
        """
        self.helper("skill_vs_luck.API", {})

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that the SWE-bench notebook runs on a small `config`.
        """
        overrides = {
            "n_boot": 100,
            "logit_gaps": [0.05, 0.25],
            "n_tasks_list": [100, 500],
            "n_recovery_trials": 20,
            "n_splits": 50,
            "curse_n_tasks": [100, 500],
            "curse_n_variants": [1, 2, 5],
            "n_curse_trials": 100,
            "n_svi_steps": 300,
            "n_svi_samples": 50,
            "nuts_subset": (10, 40),
            "n_nuts_samples": 50,
            "n_selection_trials": 10,
        }
        self.helper("skill_vs_luck.example", overrides)

    @pytest.mark.slow
    def test3(self) -> None:
        """
        Test that the notebook of the SWE-bench records runs.
        """
        self.helper("skill_vs_luck.records", {"n_boot": 100})

    @pytest.mark.slow
    def test4(self) -> None:
        """
        Test that the notebook of the local runs runs.
        """
        self.helper(
            "skill_vs_luck.runs", {"n_draws": 100}, data_dirs=(_PHASE_1_DIR,)
        )

    @pytest.mark.slow
    def test5(self) -> None:
        """
        Test that the tuning of the 1-run estimator runs on a small grid.
        """
        overrides = {
            "n_draws": 2,
            "first_grid": (["rasch"], [0.01, 0.1], [False]),
            "finer_grid": [(["rasch"], [0.003], [False])],
            "second_run_draws": 20,
            "check_frozen": False,
        }
        self.helper(
            "skill_vs_luck.one_run", overrides, data_dirs=(_PHASE_1_DIR,)
        )

    @pytest.mark.slow
    def test6(self) -> None:
        """
        Test that the test of the 1-run estimator runs.
        """
        self.helper(
            "skill_vs_luck.one_run_test_set",
            {},
            data_dirs=(_PHASE_1_DIR, _PHASE_2_DIR),
        )

    @pytest.mark.slow
    def test7(self) -> None:
        """
        Test that the notebook of Study C runs.
        """
        self.helper(
            "skill_vs_luck.temperatures",
            {},
            data_dirs=(_PHASE_1_DIR, _PHASE_2_DIR),
        )

    @pytest.mark.slow
    def test8(self) -> None:
        """
        Test that the notebook of Study D runs.
        """
        self.helper(
            "skill_vs_luck.more_models",
            {},
            data_dirs=(_PHASE_1_DIR, _PHASE_2_DIR),
        )

    @pytest.mark.slow
    def test9(self) -> None:
        """
        Test that the notebook of Study E runs on a small `config`.
        """
        self.helper(
            "skill_vs_luck.transfer",
            {
                "n_splits": 50,
                "n_boot": 100,
                "n_svi_steps": 300,
                "n_svi_samples": 50,
            },
            data_dirs=(_PHASE_1_DIR, _PHASE_2_DIR),
        )

    @pytest.mark.slow
    def test10(self) -> None:
        """
        Test that the notebook of Studies F to H runs on a small `config`.
        """
        self.helper(
            "skill_vs_luck.selection",
            {"n_draws": 20, "n_trials": 20, "n_boot": 100},
            data_dirs=(_PHASE_1_DIR, _PHASE_2_DIR, _VARIANTS_DIR),
        )

    @pytest.mark.superslow
    def test11(self) -> None:
        """
        Test that Study E keeps its pre-registered results.
        """
        # Run test.
        actual = self.helper(
            "skill_vs_luck.transfer",
            {},
            data_dirs=(_PHASE_1_DIR, _PHASE_2_DIR),
            last_cell='print(predictions["holds"].to_dict())',
        )
        # Check outputs.
        expected = (
            "{'E1': True, 'E2': True, 'E3': False, 'E4': True, 'E5': False, "
            "'E6': True}\n"
        )
        self.assert_equal(actual, expected)

    @pytest.mark.superslow
    def test12(self) -> None:
        """
        Test that Studies F to H keep their results and their recommendation.
        """
        # Run test.
        actual = self.helper(
            "skill_vs_luck.selection",
            {},
            data_dirs=(_PHASE_1_DIR, _PHASE_2_DIR, _VARIANTS_DIR),
            last_cell='print(predictions["holds"].to_dict(), recommended)',
        )
        # Check outputs.
        expected = (
            "{'F1': False, 'F2': True, 'F3': True, 'G0': True, 'G1': True, "
            "'G2': True, 'G3': True, 'G4': False, 'G5': False} paired\n"
        )
        self.assert_equal(actual, expected)
