"""
Unit tests for get_lecture_file.py.

Import as:

import class_scripts.test.test_get_lecture_file as csttglfi
"""

# TODO(gp): Make sure this file follows our unit test conventions

import os
from unittest import mock

import helpers.hio as hio
import helpers.hunit_test as hunitest

import class_scripts.get_lecture_file as csgelefi


# #############################################################################
# Test_parse
# #############################################################################


class Test_parse(hunitest.TestCase):
    """
    Test `_parse()` function.
    """

    def test1(self) -> None:
        """
        Test parser accepts the positional `input` argument.
        """
        # Prepare inputs.
        arg_list = ["msml610/01.1"]
        # Prepare outputs.
        expected_input = "msml610/01.1"
        # Run test.
        parser = csgelefi._parse()
        args = parser.parse_args(arg_list)
        # Check outputs.
        self.assert_equal(args.input, expected_input)


# #############################################################################
# Test_main
# #############################################################################


class Test_main(hunitest.TestCase):
    """
    Test `_main()` function.
    """

    def test1(self) -> None:
        """
        Test happy path: finds and logs the path of the matching lecture
        file.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        source_dir = os.path.join(scratch_dir, "lectures_source")
        os.makedirs(source_dir, exist_ok=True)
        hio.to_file(
            os.path.join(source_dir, "Lesson01-Introduction.smd"), "content"
        )
        lecture_file = os.path.join(source_dir, "Lesson01-Introduction.smd")
        arg_list = [lecture_file]
        # Prepare outputs.
        expected_log = f"Lecture file: {lecture_file}"
        # Run test.
        with self.assertLogs(
            "class_scripts.get_lecture_file", level="INFO"
        ) as cm:
            with mock.patch("sys.argv", ["get_lecture_file.py"] + arg_list):
                csgelefi._main(csgelefi._parse())
        # Check outputs.
        # Log records are prefixed with the level and logger name, so only
        # the last (only) record is checked.
        actual_log = cm.output[-1].split(":", 2)[-1]
        self.assert_equal(actual_log, expected_log)
