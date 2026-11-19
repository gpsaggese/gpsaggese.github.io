"""
Test the notebook helpers: the Milestone 2 table loader and its PC counts.

Import as:

import test.test_tukey_utils as ttetuuti
"""

import json
import logging
import os
import shutil

import pandas as pd

import helpers.hsystem as hsystem
import helpers.hunit_test as hunitest
import tukey_utils as utils

_LOG = logging.getLogger(__name__)

# The project directory, found from this file: the loaders read paths relative
# to it, as the notebooks do.
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TABLE = "results/m2/table_summary.json"
_CORRECTIONS = "results/post_audit/pc_ci_counts.json"


# #############################################################################
# Test_example_cell2_1_load_m2_table
# #############################################################################


class Test_example_cell2_1_load_m2_table(hunitest.TestCase):
    """
    Test `tukey_utils.example_cell2_1_load_m2_table()` and its PC counts.
    """

    def helper1(
        self, path: str, *, work_dir: str = _PROJECT_DIR
    ) -> pd.DataFrame:
        """
        Load setting D from a working directory, as the notebook does.

        :param path: table file, as the loader receives it
        :param work_dir: directory the relative paths start from
            - Default: the project directory
        :return: the loaded table
        """
        with hsystem.cd(work_dir):
            table = utils.example_cell2_1_load_m2_table("D", path)
        return table

    def helper2(self, table: pd.DataFrame) -> str:
        """
        Format the `calls` of PC, tuned PC and the prototype's rule.

        :param table: output of `example_cell2_1_load_m2_table()`
        :return: e.g., "pc 159.7, pc_tuned 170.005, proto_scripted 76.555"
        """
        calls = table.set_index("method")["calls"]
        text = ", ".join(
            f"{method} {calls[method]}"
            for method in ("pc", "pc_tuned", "proto_scripted")
        )
        return text

    def test1(self) -> None:
        """
        Test that the committed table loads best first with corrected PC counts.
        """
        # Prepare outputs.
        expected = """
        13 methods, best first: ges
        pc 159.7, pc_tuned 170.005, proto_scripted 76.555
        """
        # Run test.
        table = self.helper1(_TABLE)
        # Check outputs.
        actual = (
            f"{len(table)} methods, best first: {table['method'].iloc[0]}\n"
            + self.helper2(table)
        )
        self.assert_equal(actual, expected, dedent=True)
        self.assertTrue(table["score"].is_monotonic_decreasing)

    def test2(self) -> None:
        """
        Test that other spellings of the committed file's path are corrected.
        """
        # Prepare inputs.
        paths = ["./" + _TABLE, os.path.join(_PROJECT_DIR, _TABLE)]
        # Prepare outputs.
        expected = "pc 159.7, pc_tuned 170.005, proto_scripted 76.555"
        # Run test and check outputs.
        for path in paths:
            actual = self.helper2(self.helper1(path))
            self.assert_equal(actual, expected)

    def test3(self) -> None:
        """
        Test that a changed table keeps its own counts, with one warning.

        The changed table has the committed file's relative path, as a table
        regenerated in place would.
        """
        # Prepare inputs.
        work_dir = self.get_scratch_space()
        with open(os.path.join(_PROJECT_DIR, _TABLE)) as file:
            content = json.load(file)
        content["table"]["D"]["pc"]["calls"] = 123.0
        for path in (_TABLE, _CORRECTIONS):
            os.makedirs(os.path.dirname(os.path.join(work_dir, path)))
        with open(os.path.join(work_dir, _TABLE), "w") as file:
            json.dump(content, file)
        shutil.copy(
            os.path.join(_PROJECT_DIR, _CORRECTIONS),
            os.path.join(work_dir, _CORRECTIONS),
        )
        # Prepare outputs.
        expected = "pc 123.0, pc_tuned 173.005, proto_scripted 76.555"
        # Run test.
        with self.assertLogs(utils._LOG, level="WARNING") as logs:
            table = self.helper1(_TABLE, work_dir=work_dir)
        # Check outputs.
        self.assert_equal(self.helper2(table), expected)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("shown as stored", logs.output[0])
