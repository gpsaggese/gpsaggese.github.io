"""
Run the Docker checks and each notebook of `tutorials/gymnasium`.

- On the host, `Test_docker` builds the image and runs commands in it
- Inside the image, with `docker_cmd.sh "pytest test"`, `Test_notebooks` runs each
  notebook top to bottom

Import as:

import tutorials.gymnasium.test.test_docker_gymnasium as tgttdogy
"""

import json
import logging
import os
import re
from typing import Any, Dict, List

import pytest

import helpers.hdocker_tests as hdoctest
import helpers.hio as hio
import helpers.hserver as hserver
import helpers.hsystem as hsystem
import helpers.hunit_test as hunitest

_LOG = logging.getLogger(__name__)

# TODO(aahan): Remove when `print_obj_info()` handles installed packages.
_BLOCKED_BY_HELPERS = pytest.mark.xfail(
    raises=RuntimeError,
    strict=False,
    reason="`print_obj_info()` crashes on installed packages (helpers 11dd182e)",
)

# Source text of the code cells whose output changes between runs, by notebook.
# Use text, not indices, so that it survives adding cells.
# - `Text` with its default charset depends on the hash seed
# - The `hdbg.init_logger` cell logs a temp file name
# - `time_vector_env()` prints wall-clock times
_UNSTABLE_CELLS = {
    "gymnasium.03.API.Spaces": ["[t.sample() for", "Text(max_length=5)"],
    "gymnasium.05.API.Vector": ["hdbg.init_logger", "time_vector_env("],
}

# Wall-clock values of `RecordEpisodeStatistics`, replaced in the outputs.
_WALL_CLOCK_REGEXES = [r"'t': [0-9.e-]+", r"time: [0-9.]+s"]


def _get_cell_output(cell: Dict[str, Any]) -> str:
    """
    Return the visible output of a cell, without wall-clock values.

    The kernel can split one stream into chunks in different ways and interleave
    them with display outputs differently, so merge the streams first.

    :param cell: notebook code cell
    :return: stream text followed by the text of the display outputs
    """
    streams: List[str] = []
    displays: List[str] = []
    for output in cell["outputs"]:
        if output["output_type"] == "stream":
            streams.append("".join(output["text"]))
        elif "data" in output:
            displays.append("".join(output["data"].get("text/plain", "")))
    text = "".join(streams) + "\n".join(displays)
    for regex in _WALL_CLOCK_REGEXES:
        text = re.sub(regex, "<wall_clock>", text)
    return text


def _get_stable_outputs(notebook_name: str, path: str) -> str:
    """
    Return the outputs of an executed notebook that repeat between runs.

    :param notebook_name: notebook base name
    :param path: path of the executed notebook
    :return: text with the output of each compared code cell
    """
    notebook = json.loads(hio.from_file(path))
    markers = _UNSTABLE_CELLS.get(notebook_name, [])
    texts = []
    for idx, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"])
        is_unstable = any(marker in source for marker in markers)
        if cell["cell_type"] == "code" and not is_unstable:
            texts.append(f"# Cell {idx}\n{_get_cell_output(cell)}")
    return "\n".join(texts)


# #############################################################################
# Test_docker
# #############################################################################


@pytest.mark.skipif(
    hserver.is_inside_docker(), reason="Needs docker on the host"
)
class Test_docker(hdoctest.DockerTestCase):
    """
    Run the Docker tests for `tutorials/gymnasium`.
    """

    _test_file = __file__


# #############################################################################
# Test_notebooks
# #############################################################################


@pytest.mark.skipif(
    not hserver.is_inside_docker(), reason="Run with docker_cmd.sh 'pytest test'"
)
class Test_notebooks(hunitest.TestCase):
    """
    Run each notebook of `tutorials/gymnasium` twice and compare the outputs.
    """

    def _run_notebook(self, notebook_name: str, output_name: str) -> str:
        """
        Execute a notebook into the scratch space, leaving the original intact.

        :param notebook_name: notebook base name
        :param output_name: base name of the executed notebook
        :return: path of the executed notebook
        """
        tutorial_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        scratch_dir = self.get_scratch_space()
        cmd = (
            f"cd {tutorial_dir} && jupyter nbconvert --to notebook --execute"
            " --ExecutePreprocessor.timeout=600"
            f" --output-dir {scratch_dir} --output {output_name}"
            f" {notebook_name}.ipynb"
        )
        hsystem.system(cmd)
        return os.path.join(scratch_dir, f"{output_name}.ipynb")

    def helper(self, notebook_name: str) -> None:
        """
        Run a notebook twice and compare its outputs.

        :param notebook_name: notebook base name
        """
        # Prepare inputs.
        pytest.importorskip("gymnasium")
        # Run test.
        path_1 = self._run_notebook(notebook_name, "run_1")
        path_2 = self._run_notebook(notebook_name, "run_2")
        # Check outputs.
        actual = _get_stable_outputs(notebook_name, path_2)
        expected = _get_stable_outputs(notebook_name, path_1)
        self.assert_equal(actual, expected)

    @_BLOCKED_BY_HELPERS
    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that notebook 01 runs and repeats inside Docker.
        """
        # Prepare inputs.
        notebook_name = "gymnasium.01.API.Env"
        # Run test.
        self.helper(notebook_name)

    @_BLOCKED_BY_HELPERS
    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that notebook 02 runs and repeats inside Docker.
        """
        # Prepare inputs.
        notebook_name = "gymnasium.02.API.Registry"
        # Run test.
        self.helper(notebook_name)

    @_BLOCKED_BY_HELPERS
    @pytest.mark.slow
    def test3(self) -> None:
        """
        Test that notebook 03 runs and repeats inside Docker.
        """
        # Prepare inputs.
        notebook_name = "gymnasium.03.API.Spaces"
        # Run test.
        self.helper(notebook_name)

    @_BLOCKED_BY_HELPERS
    @pytest.mark.slow
    def test4(self) -> None:
        """
        Test that notebook 04 runs and repeats inside Docker.
        """
        # Prepare inputs.
        notebook_name = "gymnasium.04.API.Wrappers"
        # Run test.
        self.helper(notebook_name)

    @pytest.mark.slow
    def test5(self) -> None:
        """
        Test that notebook 05 runs and repeats inside Docker.
        """
        # Prepare inputs.
        notebook_name = "gymnasium.05.API.Vector"
        # Run test.
        self.helper(notebook_name)
