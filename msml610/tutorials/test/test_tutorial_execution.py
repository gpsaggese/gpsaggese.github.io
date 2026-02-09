"""
Test that all Python tutorial files and paired Jupyter notebooks execute
successfully.

Import as:

import msml610.tutorials.test.test_tutorial_execution as mtttetex
"""

import logging
import os
from typing import Dict, List, Tuple

import pytest

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hjupyter as hjupyte
import helpers.hsystem as hsystem
import helpers.htimer as htimer
import helpers.hunit_test as hunitest

_LOG = logging.getLogger(__name__)


# #############################################################################
# Helper functions
# #############################################################################


# TODO(gp): Move this to hjupyter and generalize, e.g., passing a regex.
def _find_paired_notebooks(
    tutorials_dir: str,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Find all tutorial Python files and paired Jupyter notebooks.

    :param tutorials_dir: path to the tutorials directory
    :return: tuple of (python_files, notebook_files, unpaired_notebooks)
        - python_files: list of .py tutorial files (excluding _utils.py)
        - notebook_files: list of .ipynb files that have corresponding .py
        - unpaired_notebooks: list of .ipynb files without corresponding .py
    """
    hdbg.dassert_path_exists(tutorials_dir)
    # Find all L*.py files excluding utility files.
    py_files = hio.listdir(
        tutorials_dir,
        "L*.py",
        only_files=True,
        use_relative_paths=False,
        maxdepth=1,
    )
    py_files = [f for f in py_files if not f.endswith("_utils.py")]
    py_files = sorted(py_files)
    # Find all L*.ipynb files.
    nb_files = hio.listdir(
        tutorials_dir,
        "L*.ipynb",
        only_files=True,
        use_relative_paths=False,
        maxdepth=1,
    )
    nb_files = sorted(nb_files)
    # Build set of base names from Python files.
    py_basenames = set()
    for py_file in py_files:
        basename = os.path.basename(py_file)
        basename = os.path.splitext(basename)[0]
        py_basenames.add(basename)
    # Check which notebooks have corresponding .py files.
    paired_notebooks = []
    unpaired_notebooks = []
    for nb_file in nb_files:
        basename = os.path.basename(nb_file)
        basename = os.path.splitext(basename)[0]
        if basename in py_basenames:
            paired_notebooks.append(nb_file)
        else:
            unpaired_notebooks.append(nb_file)
    return py_files, paired_notebooks, unpaired_notebooks


def _execute_file_with_docker(
    file_path: str,
    *,
    working_dir: str,
    is_notebook: bool,
) -> Tuple[bool, str, float]:
    """
    Execute a Python file or notebook using docker_cmd.

    :param file_path: path to the file to execute
    :param working_dir: directory to cd into before execution
    :param is_notebook: True if file is a notebook, False if Python script
    :return: tuple of (success, error_message, elapsed_time)
    """
    timer = htimer.Timer()
    try:
        if is_notebook:
            # For notebooks, use hjupyter.run_notebook via docker_cmd.
            scratch_dir = os.path.join(working_dir, "tmp.notebook_scratch")
            # Build Python command to run notebook.
            cmd = (
                f"python -c \""
                f"import helpers.hjupyter as hjupyte; "
                f"import helpers.hio as hio; "
                f"hio.create_dir('{scratch_dir}', incremental=True); "
                f"hjupyte.run_notebook('{file_path}', '{scratch_dir}')\""
            )
        else:
            # For Python scripts, execute directly.
            cmd = f"python {file_path}"
        # Build invoke docker_cmd command.
        docker_cmd = f'invoke docker_cmd --cmd "{cmd}"'
        # Execute in the working directory.
        hsystem.system(
            docker_cmd,
            abort_on_error=False,
            suppress_output=False,
        )
        elapsed = timer.get_elapsed()
        return True, "", elapsed
    except Exception as e:
        elapsed = timer.get_elapsed()
        error_msg = str(e)
        return False, error_msg, elapsed


def _execute_file_directly(
    file_path: str,
    *,
    working_dir: str,
    is_notebook: bool,
) -> Tuple[bool, str, float]:
    """
    Execute a Python file or notebook directly (inside container).

    :param file_path: path to the file to execute
    :param working_dir: directory to cd into before execution
    :param is_notebook: True if file is a notebook, False if Python script
    :return: tuple of (success, error_message, elapsed_time)
    """
    timer = htimer.Timer()
    try:
        if is_notebook:
            # For notebooks, use hjupyter.run_notebook.
            scratch_dir = os.path.join(working_dir, "tmp.notebook_scratch")
            hio.create_dir(scratch_dir, incremental=True)
            hjupyte.run_notebook(
                file_path,
                scratch_dir,
                pre_cmd=f"cd {working_dir}",
            )
        else:
            # For Python scripts, execute directly.
            cmd = f"cd {working_dir} && python {file_path}"
            hsystem.system(
                cmd,
                abort_on_error=True,
                suppress_output=False,
            )
        elapsed = timer.get_elapsed()
        return True, "", elapsed
    except Exception as e:
        elapsed = timer.get_elapsed()
        error_msg = str(e)
        return False, error_msg, elapsed


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
        # TODO(ai_gp): Add tqdm progress bar.
        for idx, file_path in enumerate(files, 1):
            basename = os.path.basename(file_path)
            _LOG.info(
                "Executing %s %d/%d: %s",
                file_type,
                idx,
                len(files),
                basename,
            )
            if use_docker:
                success, error, elapsed = _execute_file_with_docker(
                    file_path,
                    working_dir=working_dir,
                    is_notebook=is_notebook,
                )
            else:
                success, error, elapsed = _execute_file_directly(
                    file_path,
                    working_dir=working_dir,
                    is_notebook=is_notebook,
                )
            results[file_path] = (success, error, elapsed)
            status = "SUCCESS" if success else "FAILED"
            _LOG.info(
                "  %s: %s (%.2f seconds)",
                status,
                basename,
                elapsed,
            )
            if not success:
                _LOG.warning("  Error: %s", error)
        return results

    def _report_results(
        self,
        py_results: Dict[str, Tuple[bool, str, float]],
        nb_results: Dict[str, Tuple[bool, str, float]],
    ) -> None:
        """
        Report execution results and assert if any failures occurred.

        :param py_results: results from Python file execution
        :param nb_results: results from notebook execution
        """
        # Collect failures.
        py_failures = [
            f for f, (success, _, _) in py_results.items() if not success
        ]
        nb_failures = [
            f for f, (success, _, _) in nb_results.items() if not success
        ]
        # Calculate statistics.
        py_total = len(py_results)
        py_success = py_total - len(py_failures)
        nb_total = len(nb_results)
        nb_success = nb_total - len(nb_failures)
        total_files = py_total + nb_total
        total_success = py_success + nb_success
        total_failures = len(py_failures) + len(nb_failures)
        # Calculate timing statistics.
        py_times = [elapsed for _, _, elapsed in py_results.values()]
        nb_times = [elapsed for _, _, elapsed in nb_results.values()]
        py_total_time = sum(py_times) if py_times else 0.0
        nb_total_time = sum(nb_times) if nb_times else 0.0
        total_time = py_total_time + nb_total_time
        # Report summary.
        _LOG.info("=" * 80)
        _LOG.info("EXECUTION SUMMARY")
        _LOG.info("=" * 80)
        _LOG.info("Python scripts: %d total, %d success, %d failed",
                  py_total, py_success, len(py_failures))
        if py_total > 0:
            _LOG.info("  Total time: %.2f seconds", py_total_time)
            _LOG.info("  Average time: %.2f seconds", py_total_time / py_total)
        _LOG.info("Notebooks: %d total, %d success, %d failed",
                  nb_total, nb_success, len(nb_failures))
        if nb_total > 0:
            _LOG.info("  Total time: %.2f seconds", nb_total_time)
            _LOG.info("  Average time: %.2f seconds", nb_total_time / nb_total)
        _LOG.info("-" * 80)
        _LOG.info("TOTAL: %d files, %d success, %d failed",
                  total_files, total_success, total_failures)
        _LOG.info("Total execution time: %.2f seconds", total_time)
        # Report failures if any.
        if total_failures > 0:
            _LOG.error("=" * 80)
            _LOG.error("FAILURES DETECTED")
            _LOG.error("=" * 80)
            if py_failures:
                _LOG.error("Failed Python scripts:")
                for file_path in py_failures:
                    basename = os.path.basename(file_path)
                    _, error, _ = py_results[file_path]
                    _LOG.error("  - %s: %s", basename, error)
            if nb_failures:
                _LOG.error("Failed notebooks:")
                for file_path in nb_failures:
                    basename = os.path.basename(file_path)
                    _, error, _ = nb_results[file_path]
                    _LOG.error("  - %s: %s", basename, error)
            _LOG.error("=" * 80)
            # Assert to fail the test.
            self.fail(
                f"{total_failures} file(s) failed to execute. "
                f"See log for details."
            )

    # TODO(ai_gp): Factor out the common code in a helper function that runs
    # both Python scripts and notebooks, based on a switch "mode", equal to "run_python", "run_notebook"
    # run_both and passing a is_docker flag.
    # Then there is a single test method that sets the variables "mode" and "is_docker" and calls the helper function.
    # Add a max_tests = None or int to limit the number of tests to run.
    def test1(self) -> None:
        """
        Test execution of Python tutorial scripts only.
        """
        # Prepare inputs.
        tutorials_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
        )
        tutorials_dir = os.path.abspath(tutorials_dir)
        use_docker = True
        # Find files.
        py_files, _, unpaired_notebooks = _find_paired_notebooks(
            tutorials_dir
        )
        # Assert no unpaired notebooks.
        hdbg.dassert_eq(
            len(unpaired_notebooks),
            0,
            "Found unpaired notebooks without corresponding .py files:",
            unpaired_notebooks,
        )
        # Run test.
        py_results = self._execute_files(
            py_files,
            working_dir=tutorials_dir,
            use_docker=use_docker,
            is_notebook=False,
        )
        nb_results = {}
        # Check outputs.
        self._report_results(py_results, nb_results)

    def test2(self) -> None:
        """
        Test execution of Jupyter notebooks only.
        """
        # Prepare inputs.
        tutorials_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
        )
        tutorials_dir = os.path.abspath(tutorials_dir)
        use_docker = True
        # Find files.
        _, paired_notebooks, unpaired_notebooks = _find_paired_notebooks(
            tutorials_dir
        )
        # Assert no unpaired notebooks.
        hdbg.dassert_eq(
            len(unpaired_notebooks),
            0,
            "Found unpaired notebooks without corresponding .py files:",
            unpaired_notebooks,
        )
        # Run test.
        nb_results = self._execute_files(
            paired_notebooks,
            working_dir=tutorials_dir,
            use_docker=use_docker,
            is_notebook=True,
        )
        py_results = {}
        # Check outputs.
        self._report_results(py_results, nb_results)

    def test3(self) -> None:
        """
        Test execution of both Python scripts and Jupyter notebooks.
        """
        # Prepare inputs.
        tutorials_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
        )
        tutorials_dir = os.path.abspath(tutorials_dir)
        use_docker = True
        # Find files.
        py_files, paired_notebooks, unpaired_notebooks = _find_paired_notebooks(
            tutorials_dir
        )
        # Assert no unpaired notebooks.
        hdbg.dassert_eq(
            len(unpaired_notebooks),
            0,
            "Found unpaired notebooks without corresponding .py files:",
            unpaired_notebooks,
        )
        # Run test.
        py_results = self._execute_files(
            py_files,
            working_dir=tutorials_dir,
            use_docker=use_docker,
            is_notebook=False,
        )
        nb_results = self._execute_files(
            paired_notebooks,
            working_dir=tutorials_dir,
            use_docker=use_docker,
            is_notebook=True,
        )
        # Check outputs.
        self._report_results(py_results, nb_results)

    def test4(self) -> None:
        """
        Test execution of Python scripts directly (no docker).
        """
        # Prepare inputs.
        tutorials_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
        )
        tutorials_dir = os.path.abspath(tutorials_dir)
        use_docker = False
        # Find files.
        py_files, _, unpaired_notebooks = _find_paired_notebooks(
            tutorials_dir
        )
        # Assert no unpaired notebooks.
        hdbg.dassert_eq(
            len(unpaired_notebooks),
            0,
            "Found unpaired notebooks without corresponding .py files:",
            unpaired_notebooks,
        )
        # Run test.
        py_results = self._execute_files(
            py_files,
            working_dir=tutorials_dir,
            use_docker=use_docker,
            is_notebook=False,
        )
        nb_results = {}
        # Check outputs.
        self._report_results(py_results, nb_results)

    def test5(self) -> None:
        """
        Test execution of notebooks directly (no docker).
        """
        # Prepare inputs.
        tutorials_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
        )
        tutorials_dir = os.path.abspath(tutorials_dir)
        use_docker = False
        # Find files.
        _, paired_notebooks, unpaired_notebooks = _find_paired_notebooks(
            tutorials_dir
        )
        # Assert no unpaired notebooks.
        hdbg.dassert_eq(
            len(unpaired_notebooks),
            0,
            "Found unpaired notebooks without corresponding .py files:",
            unpaired_notebooks,
        )
        # Run test.
        nb_results = self._execute_files(
            paired_notebooks,
            working_dir=tutorials_dir,
            use_docker=use_docker,
            is_notebook=True,
        )
        py_results = {}
        # Check outputs.
        self._report_results(py_results, nb_results)
