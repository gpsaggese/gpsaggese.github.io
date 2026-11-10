"""
Unit tests for gen_slides.py script.

Import as:

import class_scripts.test.test_gen_slides as cstestgs
"""

import argparse
import os

import helpers.hunit_test as hunitest
import helpers.hunit_test_utils as hunteuti
from helpers import hio

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


# #############################################################################
# Test__is_standalone_file
# #############################################################################


class Test__is_standalone_file(hunitest.TestCase):
    """
    Test `gen_slides._is_standalone_file()` function.
    """

    def helper(self, input_spec: str, expected: bool) -> None:
        """
        Test helper for `_is_standalone_file()`.

        :param input_spec: input specification to classify
        :param expected: expected result
        """
        # Run test.
        actual = cscgesli._is_standalone_file(input_spec)
        # Check outputs.
        self.assertEqual(actual, expected)

    def test1(self) -> None:
        """
        Test a markdown file outside of a course dir.
        """
        # Prepare inputs.
        input_spec = "papers/Causal_Analysis_for_Finance/notes.slides.md"
        # Prepare outputs.
        expected = True
        # Run test.
        self.helper(input_spec, expected)

    def test2(self) -> None:
        """
        Test a markdown file in the current dir.
        """
        # Prepare inputs.
        input_spec = "notes.slides.md"
        # Prepare outputs.
        expected = True
        # Run test.
        self.helper(input_spec, expected)

    def test3(self) -> None:
        """
        Test a markdown file under `lectures_source` is a course lecture.
        """
        # Prepare inputs.
        input_spec = "msml610/lectures_source/Lesson10-Name.md"
        # Prepare outputs.
        expected = False
        # Run test.
        self.helper(input_spec, expected)

    def test4(self) -> None:
        """
        Test a `dir/lesson` specification is a course lecture.
        """
        # Prepare inputs.
        input_spec = "msml610/08.1"
        # Prepare outputs.
        expected = False
        # Run test.
        self.helper(input_spec, expected)

    def test5(self) -> None:
        """
        Test a lecture `.smd` file is a course lecture.
        """
        # Prepare inputs.
        input_spec = "msml610/lectures_source/Lesson10.2-Name.smd"
        # Prepare outputs.
        expected = False
        # Run test.
        self.helper(input_spec, expected)


# #############################################################################
# Test__process_lesson_spec
# #############################################################################


class Test__process_lesson_spec(hunitest.TestCase):
    """
    Test `gen_slides._process_lesson_spec()` function.
    """

    def helper(
        self,
        input_spec: str,
        actions: list[str],
        expected_sys_calls: str,
    ) -> None:
        """
        Run `_process_lesson_spec()` and check the commands it would run.

        :param input_spec: input specification passed to `-i`
        :param actions: actions to execute
        :param expected_sys_calls: expected captured system calls
        """
        # Parse the arguments like the CLI does, skipping the PDF viewer.
        parser = cscgesli._parse()
        argv = ["-i", input_spec, "--notes_to_pdf_args=--skip_action open_pdf"]
        args = parser.parse_args(argv)
        # Run test.
        with hunteuti.capture_sys_calls() as sys_calls:
            cscgesli._process_lesson_spec(input_spec, args, actions)
        # Check outputs.
        hunteuti.assert_sys_calls(self, sys_calls, expected_sys_calls)

    def test1(self) -> None:
        """
        Test a standalone markdown file generates the PDF next to the input.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        input_file = os.path.join(scratch_dir, "notes.slides.md")
        hio.to_file(input_file, "* Slide")
        actions = ["generate"]
        # Prepare outputs.
        expected_sys_calls = r"""
        [
            {
            'function': hsystem.system,
            'args': ('notes_to_pdf.py --input=$GIT_ROOT/class_scripts/test/outcomes/Test__process_lesson_spec.test1/tmp.scratch/notes.slides.md --output=$GIT_ROOT/class_scripts/test/outcomes/Test__process_lesson_spec.test1/tmp.scratch/notes.slides.pdf --type=slides --toc_type=navigation --debug_on_error --skip_action=cleanup_before --skip_action=cleanup_after -v INFO --skip_action open_pdf',),
            'kwargs': {'suppress_output': False},
            },
        ]
        """
        # Run test.
        self.helper(input_file, actions, expected_sys_calls)

    def test2(self) -> None:
        """
        Test a course lecture file generates the PDF in the staging dir.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        lecture_dir = os.path.join(scratch_dir, "lectures_source")
        hio.create_dir(lecture_dir, incremental=True)
        input_file = os.path.join(lecture_dir, "Lesson08.1-Intro.smd")
        hio.to_file(input_file, "* Slide")
        actions = ["generate"]
        # Prepare outputs.
        expected_sys_calls = r"""
        [
            {
            'function': hsystem.system,
            'args': ('notes_to_pdf.py --input=$GIT_ROOT/class_scripts/test/outcomes/Test__process_lesson_spec.test2/tmp.scratch/lectures_source/Lesson08.1-Intro.smd --output=$GIT_ROOT/class_scripts/test/outcomes/Test__process_lesson_spec.test2/tmp.scratch/lectures_pdf.tmp/Lesson08.1-Intro.pdf --type=slides --toc_type=navigation --debug_on_error --skip_action=cleanup_before --skip_action=cleanup_after -v INFO --skip_action open_pdf',),
            'kwargs': {'suppress_output': False},
            },
        ]
        """
        # Run test.
        self.helper(input_file, actions, expected_sys_calls)

    def test3(self) -> None:
        """
        Test that a standalone markdown file cannot be released.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        input_file = os.path.join(scratch_dir, "notes.slides.md")
        hio.to_file(input_file, "* Slide")
        actions = ["generate", "release"]
        parser = cscgesli._parse()
        args = parser.parse_args(["-i", input_file])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            cscgesli._process_lesson_spec(input_file, args, actions)
