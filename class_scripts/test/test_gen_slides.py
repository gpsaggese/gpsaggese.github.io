"""
Unit tests for gen_slides.py script.

Import as:

import class_scripts.test.test_gen_slides as cstestgs
"""

import argparse

import helpers.hunit_test as hunitest

import class_scripts.gen_slides as cscgesli


# #############################################################################
# Test_parse
# #############################################################################


class Test_parse(hunitest.TestCase):
    """
    Test _parse() function.
    """

    def _assert_parse_args(self, arg_list: list, expected_values: dict) -> None:
        """
        Test helper for _parse.

        :param arg_list: List of arguments to parse
        :param expected_values: Dictionary of expected argument values
        """
        parser = cscgesli._parse()
        args = parser.parse_args(arg_list)
        for key, value in expected_values.items():
            self.assertEqual(getattr(args, key), value)

    def test1(self) -> None:
        """
        Test parser accepts input in dir/lesson format.
        """
        # Prepare inputs.
        arg_list = ["-i", "msml610/08.1"]
        # Prepare outputs.
        expected_values = {"input": "msml610/08.1", "notes_to_pdf_args": None}
        # Run test.
        parser = cscgesli._parse()
        self.assertIsInstance(parser, argparse.ArgumentParser)
        self._assert_parse_args(arg_list, expected_values)

    def test2(self) -> None:
        """
        Test parser accepts file path.
        """
        # Prepare inputs.
        arg_list = ["-i", "msml610/lectures_source/Lesson10-Name.txt"]
        # Prepare outputs.
        expected_values = {
            "input": "msml610/lectures_source/Lesson10-Name.txt",
            "notes_to_pdf_args": None,
        }
        # Run test.
        self._assert_parse_args(arg_list, expected_values)

    def test3(self) -> None:
        """
        Test parser accepts options to pass through to notes_to_pdf.py.
        """
        # Prepare inputs.
        arg_list = [
            "-i",
            "data605/01.1",
            "--notes_to_pdf_args",
            "extra_arg1 extra_arg2",
        ]
        # Prepare outputs.
        expected_values = {
            "input": "data605/01.1",
            "notes_to_pdf_args": "extra_arg1 extra_arg2",
        }
        # Run test.
        self._assert_parse_args(arg_list, expected_values)

    def test4(self) -> None:
        """
        Test parser has expected description and help.
        """
        # Run test.
        parser = cscgesli._parse()
        # Check outputs.
        description = parser.description
        self.assertIsNotNone(description)
        self.assertIn("Generate lecture slides", description or "")
