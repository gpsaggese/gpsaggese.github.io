"""
End-to-end test for `count_lecture_slides.py` run as a `uv` script.

Import as:

import class_scripts.test.test_count_lecture_slides_uv_run as csttclur
"""

import shutil

import pytest

import helpers.hgit as hgit
import helpers.hsystem as hsystem
import helpers.hunit_test as hunitest

# `count_lecture_slides.py` declares its dependencies (e.g., `tabulate`) in a
# `# /// script` block and is meant to be run via `uv run`, which installs
# those dependencies on the fly in an isolated environment. This lets the
# script be exercised end-to-end even when `tabulate` is not installed in the
# test environment itself (the test is skipped only if `uv` itself is
# missing).


# #############################################################################
# Test_count_lecture_slides_py
# #############################################################################


class Test_count_lecture_slides_py(hunitest.TestCase):
    """
    Test `count_lecture_slides.py` executed through `uv run`.
    """

    def test1(self) -> None:
        """
        Test that `--help` runs successfully, exercising the `uv`-managed
        imports (e.g., `tabulate`) without requiring them to be pre-installed.
        """
        # Prepare inputs.
        exec_path = hgit.find_file_in_git_tree("count_lecture_slides.py")
        cmd = f"uv run {exec_path} --help"
        # Run test.
        rc, output = hsystem.system_to_string(cmd)
        # Check outputs.
        self.assertEqual(rc, 0)
        self.assertIn("Course directory", output)
