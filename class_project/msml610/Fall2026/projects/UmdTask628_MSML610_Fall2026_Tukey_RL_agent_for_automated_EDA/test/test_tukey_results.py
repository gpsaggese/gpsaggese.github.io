"""
Test the results script: its parser and the `--dry_run` flag.

Import as:

import test.test_tukey_results as ttetures
"""

import json
import logging
import os
import unittest.mock as umock
from typing import List

import helpers.hunit_test as hunitest
import tukey_results as tresul

_LOG = logging.getLogger(__name__)


def _write_table(path: str) -> None:
    """
    Write a minimal M2 table for `m2_split`: one setting, one method.

    :param path: output path
    """
    table = {
        "table": {"D": {"pc": {"score": -0.5}}},
        "tuned": {"D": {}},
        "per_graph": {
            "D": {
                "pc": {
                    "seeds": [1, 2],
                    "score": [-0.25, -0.75],
                    "shd_target": [1, 3],
                }
            }
        },
    }
    with open(path, "w") as file:
        json.dump(table, file)


# #############################################################################
# Test__parse
# #############################################################################


class Test__parse(hunitest.TestCase):
    """
    Test `tukey_results._parse()`.
    """

    def test1(self) -> None:
        """
        Test that `--dry_run` is a plain flag, off by default.
        """
        # Prepare inputs.
        parser = tresul._parse()
        argv = ["proto_check", "--out", "proto_check.json"]
        # Run test.
        args = parser.parse_args(argv)
        args_dry_run = parser.parse_args(argv + ["--dry_run"])
        # Check outputs.
        self.assertEqual((args.dry_run, args_dry_run.dry_run), (False, True))

    def test2(self) -> None:
        """
        Test that an unknown action is refused.
        """
        # Prepare inputs.
        parser = tresul._parse()
        argv = ["sealed_sets", "--out", "sealed.json"]
        # Run test and check outputs.
        with self.assertRaises(SystemExit):
            parser.parse_args(argv)


# #############################################################################
# Test_tukey_results_py
# #############################################################################


class Test_tukey_results_py(hunitest.TestCase):
    """
    End-to-end tests of the `tukey_results.py` executable.
    """

    def _run_main(self, argv: List[str]) -> None:
        """
        Run `tukey_results._main()` with a mocked `sys.argv`.

        :param argv: command-line arguments after the script name
        """
        parser = tresul._parse()
        with umock.patch("sys.argv", ["tukey_results.py"] + argv):
            tresul._main(parser)

    def test1(self) -> None:
        """
        Test that `m2_split --dry_run` creates no directory and no file.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        table_path = os.path.join(scratch_dir, "table.json")
        _write_table(table_path)
        out_dir = os.path.join(scratch_dir, "m2")
        argv = ["m2_split", "--table", table_path, "--out", out_dir]
        # Run test.
        self._run_main(argv + ["--dry_run"])
        # Check outputs.
        self.assertFalse(os.path.exists(out_dir))

    def test2(self) -> None:
        """
        Test that `m2_split` without `--dry_run` writes its two files.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        table_path = os.path.join(scratch_dir, "table.json")
        _write_table(table_path)
        out_dir = os.path.join(scratch_dir, "m2")
        argv = ["m2_split", "--table", table_path, "--out", out_dir]
        # Prepare outputs.
        expected = ["per_graph.json", "table_summary.json"]
        # Run test.
        self._run_main(argv)
        # Check outputs.
        self.assertEqual(sorted(os.listdir(out_dir)), expected)
