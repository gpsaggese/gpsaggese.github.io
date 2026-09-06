"""
End-to-end test for `gen_lecture_commentary.py` run as a `uv` script.

Import as:

import class_scripts.test.test_gen_lecture_commentary_uv_run as csttglcur
"""

import shutil

import pytest

import helpers.hgit as hgit
import helpers.hsystem as hsystem
import helpers.hunit_test as hunitest

# `gen_lecture_commentary.py` declares its dependencies (e.g., `pdf2image`,
# `openai`, `requests`) in a `# /// script` block and is meant to be run via
# `uv run`, which installs those dependencies on the fly in an isolated
# environment. This lets the script be exercised end-to-end even when those
# packages are not installed in the test environment itself (the test is
# skipped only if `uv` itself is missing).


# #############################################################################
# Test_gen_lecture_commentary_py
# #############################################################################


class Test_gen_lecture_commentary_py(hunitest.TestCase):
    """
    Test `gen_lecture_commentary.py` executed through `uv run`.
    """

    def test1(self) -> None:
        """
        Test that `--help` runs successfully, exercising the `uv`-managed
        imports (e.g., `pdf2image`) without requiring them to be
        pre-installed.
        """
        # Prepare inputs.
        exec_path = hgit.find_file_in_git_tree("gen_lecture_commentary.py")
        cmd = f"uv run {exec_path} --help"
        # Run test.
        rc, output = hsystem.system_to_string(cmd)
        # Check outputs.
        self.assertEqual(rc, 0)
        self.assertIn("usage", output.lower())
