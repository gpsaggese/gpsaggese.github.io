"""
Test that all Python tutorial files and paired Jupyter notebooks execute
successfully.

Import as:

import msml610.tutorials.test.test_tutorial_execution as mtttetex
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import pytest
from tqdm import tqdm

import helpers.hdbg as hdbg
import helpers.hjupyter as hjupyte
import helpers.hunit_test as hunitest

_LOG = logging.getLogger(__name__)


# #############################################################################
# TestTutorialExecution
# #############################################################################


@pytest.mark.superslow
class TestTutorialExecution(hunitest.TestCase):
    """
    Test execution of all tutorial Python files and paired Jupyter notebooks.
    """

    def _execute_files(
        self,
        files: List[str],
        *,
        working_dir: str,
        use_docker: bool,
        is_notebook: bool,
    ) -> Dict[str, Tuple[bool, str, float]]:
        """
        Execute a list of files and collect results.

        :param files: list of file paths to execute
        :param working_dir: directory to cd into before execution
        :param use_docker: if True, use invoke docker_cmd
        :param is_notebook: True if files are notebooks, False if Python
        :return: dict mapping file path to (success, error_message,
            elapsed_time)
        """
        results = {}
        file_type = "notebook" if is_notebook else "Python script"
        _LOG.info("Executing %d %ss", len(files), file_type)
        # Use tqdm progress bar for tracking execution.
        for file_path in tqdm(
            files,
            desc=f"Executing {file_type}s",
            unit="file",
        ):
            basename = os.path.basename(file_path)
            if use_docker:
                success, error, elapsed = hjupyte.execute_file_with_docker(
                    file_path,
                    working_dir=working_dir,
                    is_notebook=is_notebook,
                )
            else:
                success, error, elapsed = hjupyte.execute_file_directly(
                    file_path,
                    working_dir=working_dir,
                    is_notebook=is_notebook,
                )
            results[file_path] = (success, error, elapsed)
            # Print results.
            status = "SUCCESS" if success else "FAILED"
            _LOG.info(
                "  %s: %s (%.2f seconds)",
                status,
                basename,
                elapsed,
            )
            if not success:
                _LOG.warning("Error: %s", error)
        return results

    def _run_tutorial_tests(
        self,
        mode: str,
        use_docker: bool,
        *,
        max_tests: Optional[int] = None,
    ) -> None:
        """
        Execute tutorial tests based on specified mode.

        :param mode: execution mode - "run_python", "run_notebook",
            or "run_both"
        :param use_docker: if True, use invoke docker_cmd
        :param max_tests: if provided, limit number of files to test
        """
        # Validate mode.
        valid_modes = ["run_python", "run_notebook", "run_both"]
        hdbg.dassert_in(
            mode,
            valid_modes,
            "Invalid mode:",
            mode,
        )
        # Prepare inputs.
        tutorials_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
        )
        tutorials_dir = os.path.abspath(tutorials_dir)
        # Find files.
        py_files, paired_notebooks, unpaired_notebooks = (
            hjupyte.find_paired_files(
                tutorials_dir,
                pattern="L*.py",
                exclude_pattern="_utils.py",
            )
        )
        # Assert no unpaired notebooks.
        hdbg.dassert_eq(
            len(unpaired_notebooks),
            0,
            "Found unpaired notebooks without corresponding .py files:",
            unpaired_notebooks,
        )
        # Apply max_tests limit if provided.
        if max_tests is not None:
            _LOG.warning("Limiting tests to %d files", max_tests)
            py_files = py_files[:max_tests]
            paired_notebooks = paired_notebooks[:max_tests]
        # Run tests based on mode.
        py_results = {}
        nb_results = {}
        if mode in ["run_python", "run_both"]:
            py_results = self._execute_files(
                py_files,
                working_dir=tutorials_dir,
                use_docker=use_docker,
                is_notebook=False,
            )
        if mode in ["run_notebook", "run_both"]:
            nb_results = self._execute_files(
                paired_notebooks,
                working_dir=tutorials_dir,
                use_docker=use_docker,
                is_notebook=True,
            )
        # Check outputs.
        total_failures, error_message = hjupyte.report_execution_results(
            py_results,
            nb_results,
        )
        if total_failures > 0:
            self.fail(error_message)

    def test1(self) -> None:
        """
        Test execution of Python tutorial / scripts with / without docker.
        """
        # mode = "run_python"
        # mode = "run_notebook"
        mode = "run_both"
        use_docker = True
        # max_tests = None
        max_tests = 2
        self._run_tutorial_tests(
            mode,
            use_docker,
            max_tests=max_tests,
        )
