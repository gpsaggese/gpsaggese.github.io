import os
import pprint
from typing import List, Optional
from unittest import mock

import helpers.hio as hio
import helpers.hunit_test as hunitest
import helpers.hunit_test_utils as hunteuti
import helpers.hprint as hprint

import class_scripts.for_loop_lessons as csfolole


def _create_test_structure(
    self,
    class_name: str,
    dir_name: str,
) -> tuple:
    """
    Create test directory structure.

    :param class_name: Name of class directory (e.g., 'data605', 'msml610')
    :param dir_name: Name of subdirectory (e.g., 'lectures_tex', 'lectures_pdf', 'lectures_video_script')
    :return: Tuple of (class_dir, scratch_dir)
    """
    scratch_dir = self.get_scratch_space()
    class_dir = os.path.join(scratch_dir, class_name)
    subdir = os.path.join(class_dir, dir_name)
    os.makedirs(subdir, exist_ok=True)
    return class_dir, scratch_dir


def _create_test_structure_with_multiple_files(
    self,
    test_files: List[str],
    class_name: str = "data605",
    dir_name: str = "lectures_source",
) -> str:
    """
    Create test directory structure with lecture files in subdirectory.

    :param test_files: List of lecture filenames to create
    :param class_name: Name of class directory (e.g., 'data605', 'msml610')
    :param dir_name: Name of subdirectory (e.g., 'lectures_source')
    :return: Path to `class_dir`
    """
    class_dir, _ = _create_test_structure(self, class_name, dir_name)
    subdir = os.path.join(class_dir, dir_name)
    for filename in test_files:
        filepath = os.path.join(subdir, filename)
        with open(filepath, "w") as f:
            f.write(f"Content of {filename}")
    return class_dir


# #############################################################################
# Test_parse_lecture_patterns
# #############################################################################


class Test_parse_lecture_patterns(hunitest.TestCase):
    """
    Test `_parse_lecture_patterns()` function for parsing lecture patterns and
    ranges.
    """

    def _helper(
        self,
        lectures_arg: str,
        expected_is_range: bool,
        expected_patterns: List[str],
    ) -> None:
        """
        Helper to test `_parse_lecture_patterns()` and assert results.
        """
        # Run test.
        actual_is_range, actual_patterns = csfolole._parse_lecture_patterns(
            lectures_arg
        )
        # Check outputs.
        self.assertEqual(actual_is_range, expected_is_range)
        self.assertEqual(actual_patterns, expected_patterns)

    def test1(self) -> None:
        """
        Test parsing a single lecture pattern.
        """
        # Prepare inputs.
        lectures_arg = "01.1"
        expected_is_range = False
        expected_patterns = ["01.1"]
        # Run test.
        self._helper(lectures_arg, expected_is_range, expected_patterns)

    def test2(self) -> None:
        """
        Test parsing a wildcard pattern.
        """
        # Prepare inputs.
        lectures_arg = "01*"
        expected_is_range = False
        expected_patterns = ["01*"]
        # Run test.
        self._helper(lectures_arg, expected_is_range, expected_patterns)

    def test3(self) -> None:
        """
        Test parsing multiple patterns separated by colons (union syntax).
        """
        # Prepare inputs.
        lectures_arg = "01*:02*:03.1"
        expected_is_range = False
        expected_patterns = ["01*", "02*", "03.1"]
        # Run test.
        self._helper(lectures_arg, expected_is_range, expected_patterns)

    def test4(self) -> None:
        """
        Test parsing a range pattern with hyphen separator.
        """
        # Prepare inputs.
        lectures_arg = "01.1-03.2"
        expected_is_range = True
        expected_range = ["01.1", "03.2"]
        # Run test.
        self._helper(lectures_arg, expected_is_range, expected_range)

    def test5(self) -> None:
        """
        Test that mixing range and union syntax raises `AssertionError`.
        """
        # Prepare inputs.
        lectures_arg = "01.1-03.2:04*"
        # Run test and check output.
        with self.assertRaises(AssertionError) as cm:
            csfolole._parse_lecture_patterns(lectures_arg)
        expected_error = (
            "Cannot mix range syntax (hyphen) with union syntax (colon)"
        )
        self.assertIn(expected_error, str(cm.exception))

    def test6(self) -> None:
        """
        Test that invalid range format raises `AssertionError`.
        """
        # Prepare inputs.
        lectures_arg = "01.1-03.2-05.1"
        # Run test and check output.
        with self.assertRaises(AssertionError) as cm:
            csfolole._parse_lecture_patterns(lectures_arg)
        expected_error = "Range syntax must have exactly two parts (start-end)"
        self.assertIn(expected_error, str(cm.exception))


# #############################################################################
# Test_expand_lecture_range
# #############################################################################


class Test_expand_lecture_range(hunitest.TestCase):
    """
    Test `_expand_lecture_range()` function for finding files in a lesson range.

    Note: These tests require a mock directory structure with lecture files.
    """

    def test1(self) -> None:
        """
        Test expanding a range that includes multiple lecture files.
        """
        # Prepare inputs.
        test_files = [
            "Lesson01.1-Intro.smd",
            "Lesson01.2-BigData.smd",
            "Lesson02.1-Git.smd",
        ]
        class_dir = _create_test_structure_with_multiple_files(self, test_files)
        scratch_dir = self.get_scratch_space()
        start_lesson = "01.1"
        end_lesson = "02.1"
        # Run test.
        actual_files = csfolole._expand_lecture_range(
            class_dir, start_lesson, end_lesson
        )
        # Check outputs.
        actual = str(
            [
                (path.replace(scratch_dir, ""), fname)
                for path, fname in actual_files
            ]
        )
        self.check_string(actual)

    def test2(self) -> None:
        """
        Test expanding a range that includes only one lecture file.
        """
        # Prepare inputs.
        test_files = ["Lesson01.1-Intro.smd"]
        class_dir = _create_test_structure_with_multiple_files(self, test_files)
        scratch_dir = self.get_scratch_space()
        start_lesson = "01.1"
        end_lesson = "01.1"
        # Run test.
        actual_files = csfolole._expand_lecture_range(
            class_dir, start_lesson, end_lesson
        )
        # Check outputs.
        actual = str(
            [
                (path.replace(scratch_dir, ""), fname)
                for path, fname in actual_files
            ]
        )
        self.check_string(actual)

    def test3(self) -> None:
        """
        Test that an empty range raises `AssertionError`.
        """
        # Prepare inputs.
        test_files = ["Lesson01.1-Intro.smd"]
        class_dir = _create_test_structure_with_multiple_files(self, test_files)
        start_lesson = "99.1"
        end_lesson = "99.9"
        # Run test and check output.
        with self.assertRaises(AssertionError) as cm:
            csfolole._expand_lecture_range(class_dir, start_lesson, end_lesson)
        expected_error = "No lecture files found in range"
        self.assertIn(expected_error, str(cm.exception))


# #############################################################################
# Test_find_lecture_files
# #############################################################################


class Test_find_lecture_files(hunitest.TestCase):
    """
    Test `_find_lecture_files()` function for finding lecture files by patterns or range.

    Note: These tests require a mock directory structure with lecture files.
    """

    def _helper(
        self,
        class_dir: str,
        is_range: bool,
        patterns_or_range: List[str],
        expected_count: int,
    ) -> None:
        """
        Helper to test `_find_lecture_files()` and assert result count.
        """
        # Run test.
        actual_files = csfolole._find_lecture_files(
            class_dir, is_range, patterns_or_range
        )
        # Check outputs.
        self.assertEqual(len(actual_files), expected_count)

    def test1(self) -> None:
        """
        Test finding files using range mode.
        """
        # Prepare inputs.
        test_files = [
            "Lesson01.1-Intro.smd",
            "Lesson01.2-BigData.smd",
            "Lesson02.1-Git.smd",
        ]
        class_dir = _create_test_structure_with_multiple_files(self, test_files)
        is_range = True
        patterns_or_range = ["01.1", "02.1"]
        expected_count = 3
        # Run test.
        self._helper(class_dir, is_range, patterns_or_range, expected_count)

    def test2(self) -> None:
        """
        Test finding files using single pattern mode.
        """
        # Prepare inputs.
        test_files = [
            "Lesson01.1-Intro.smd",
            "Lesson01.2-BigData.smd",
            "Lesson02.1-Git.smd",
        ]
        class_dir = _create_test_structure_with_multiple_files(self, test_files)
        is_range = False
        patterns_or_range = ["01*"]
        expected_count = 2
        # Run test.
        self._helper(class_dir, is_range, patterns_or_range, expected_count)

    def test3(self) -> None:
        """
        Test finding files using multiple patterns (union syntax).
        """
        # Prepare inputs.
        test_files = [
            "Lesson01.1-Intro.smd",
            "Lesson01.2-BigData.smd",
            "Lesson02.1-Git.smd",
        ]
        class_dir = _create_test_structure_with_multiple_files(self, test_files)
        is_range = False
        patterns_or_range = ["01*", "02*"]
        expected_count = 3
        # Run test.
        self._helper(class_dir, is_range, patterns_or_range, expected_count)

    def test4(self) -> None:
        """
        Test that invalid range length raises `AssertionError`.
        """
        # Prepare inputs.
        class_dir = _create_test_structure_with_multiple_files(self, [])
        is_range = True
        patterns_or_range = ["01.1", "02.1", "03.1"]
        # Run test and check output.
        with self.assertRaises(AssertionError) as cm:
            csfolole._find_lecture_files(class_dir, is_range, patterns_or_range)
        expected_error = "Range must have exactly two elements"
        self.assertIn(expected_error, str(cm.exception))


# #############################################################################
# Test_generate_tex
# #############################################################################


class Test_generate_tex(hunitest.TestCase):
    """
    Test `_generate_tex()` function for generating TeX files.
    """

    def _helper(
        self,
        class_dir: str,
        source_path: str,
        source_name: str,
        limit: Optional[str] = None,
    ) -> None:
        """
        Helper to test `_generate_tex()` function.

        :param class_dir: class directory
        :param source_path: path to source file
        :param source_name: name of source file
        :param limit: optional limit parameter
        """
        # Run test.
        with hunteuti.capture_sys_calls() as sys_calls:
            csfolole._generate_tex(
                class_dir, source_path, source_name, limit=limit
            )
        # Check outputs.
        self.assertEqual(len(sys_calls), 1)
        cmd_str = sys_calls[0]["args"][0]
        self.check_string(cmd_str, purify_text=True)

    def test1(self) -> None:
        """
        Test `_generate_tex()` with basic inputs generates correct command.
        """
        # Prepare inputs.
        class_dir, scratch_dir = _create_test_structure(
            self, "data605", "lectures_tex"
        )
        source_path = os.path.join(scratch_dir, "Lesson01.1-Intro.smd")
        source_name = "Lesson01.1-Intro.smd"
        hio.to_file(source_path, "Test content")
        # Run test.
        self._helper(class_dir, source_path, source_name)

    def test2(self) -> None:
        """
        Test `_generate_tex()` with limit parameter includes limit in command.
        """
        # Prepare inputs.
        class_dir, scratch_dir = _create_test_structure(
            self, "data605", "lectures_tex"
        )
        source_path = os.path.join(scratch_dir, "Lesson01.1-Intro.smd")
        source_name = "Lesson01.1-Intro.smd"
        hio.to_file(source_path, "Test content")
        limit = "1:3"
        # Run test.
        with hunteuti.capture_sys_calls() as sys_calls:
            csfolole._generate_tex(
                class_dir, source_path, source_name, limit=limit
            )
            # Check outputs.
            self.assertEqual(len(sys_calls), 1)
            cmd_str = sys_calls[0]["args"][0]
            self.assertIn(f"--filter_by_slides {limit}", cmd_str)

    def test3(self) -> None:
        """
        Test `_generate_tex()` with `cmd_opts` appends the extra options to
        the invoked command.
        """
        # Prepare inputs.
        class_dir, scratch_dir = _create_test_structure(
            self, "data605", "lectures_tex"
        )
        source_path = os.path.join(scratch_dir, "Lesson01.1-Intro.smd")
        source_name = "Lesson01.1-Intro.smd"
        hio.to_file(source_path, "Test content")
        cmd_opts = "--no_incremental --open_pdf"
        # Run test.
        with hunteuti.capture_sys_calls() as sys_calls:
            csfolole._generate_tex(
                class_dir, source_path, source_name, cmd_opts=cmd_opts
            )
        # Check outputs.
        self.assertEqual(len(sys_calls), 1)
        cmd_str = sys_calls[0]["args"][0]
        self.assertIn(cmd_opts, cmd_str)


# #############################################################################
# Test_generate_pdf
# #############################################################################


class Test_generate_pdf(hunitest.TestCase):
    """
    Test `_generate_pdf()` function for generating PDF slides.
    """

    def test1(self) -> None:
        """
        Test `_generate_pdf()` with basic inputs generates correct command.
        """
        # Prepare inputs.
        class_dir, scratch_dir = _create_test_structure(
            self, "msml610", "lectures"
        )
        source_path = os.path.join(scratch_dir, "Lesson01.1-Intro.smd")
        source_name = "Lesson01.1-Intro.smd"
        hio.to_file(source_path, "Test content")
        # Run test.
        with hunteuti.capture_sys_calls() as sys_calls:
            csfolole._generate_pdf(
                class_dir, source_path, source_name, skip_action="open_pdf"
            )
        # Check outputs.
        actual_str = pprint.pformat(sys_calls)
        expected_str = hprint.dedent("""
            [{'args': ('notes_to_pdf.py --input '
                       '$GIT_ROOT/class_scripts/test/outcomes/Test_generate_pdf.test1/tmp.scratch/Lesson01.1-Intro.smd '
                       '--output '
                       '$GIT_ROOT/class_scripts/test/outcomes/Test_generate_pdf.test1/tmp.scratch/msml610/lectures/Lesson01.1-Intro.pdf '
                       '--type slides --toc_type navigation --skip_action open_pdf '
                       '--debug_on_error',),
              'function': 'hsystem.system',
              'kwargs': {'suppress_output': False}}]
            """)
        self.assert_equal(actual_str, expected_str, purify_text=True)

    def test2(self) -> None:
        """
        Test `_generate_pdf()` with limit parameter includes limit in command.
        """
        # Prepare inputs.
        class_dir, scratch_dir = _create_test_structure(
            self, "msml610", "lectures"
        )
        source_path = os.path.join(scratch_dir, "Lesson01.1-Intro.smd")
        source_name = "Lesson01.1-Intro.smd"
        hio.to_file(source_path, "Test content")
        limit = "1:5"
        # Run test.
        with hunteuti.capture_sys_calls() as sys_calls:
            csfolole._generate_pdf(
                class_dir, source_path, source_name, limit=limit
            )
        # Check outputs.
        actual_str = pprint.pformat(sys_calls)
        expected_str = hprint.dedent("""
            [{'args': ('notes_to_pdf.py --input '
                       '$GIT_ROOT/class_scripts/test/outcomes/Test_generate_pdf.test2/tmp.scratch/Lesson01.1-Intro.smd '
                       '--output '
                       '$GIT_ROOT/class_scripts/test/outcomes/Test_generate_pdf.test2/tmp.scratch/msml610/lectures/Lesson01.1-Intro.pdf '
                       '--type slides --toc_type navigation --skip_action open_pdf '
                       '--debug_on_error --filter_by_slides 1:5',),
              'function': 'hsystem.system',
              'kwargs': {'suppress_output': False}}]
            """)
        self.assert_equal(actual_str, expected_str, purify_text=True)

    def test3(self) -> None:
        """
        Test `_generate_pdf()` with `cmd_opts` appends the extra options to
        the invoked command.
        """
        # Prepare inputs.
        class_dir, scratch_dir = _create_test_structure(
            self, "msml610", "lectures"
        )
        source_path = os.path.join(scratch_dir, "Lesson01.1-Intro.smd")
        source_name = "Lesson01.1-Intro.smd"
        hio.to_file(source_path, "Test content")
        cmd_opts = "--no_incremental --open_pdf"
        # Run test.
        with hunteuti.capture_sys_calls() as sys_calls:
            csfolole._generate_pdf(
                class_dir, source_path, source_name, cmd_opts=cmd_opts
            )
        # Check outputs.
        self.assertEqual(len(sys_calls), 1)
        cmd_str = sys_calls[0]["args"][0]
        self.assertIn(cmd_opts, cmd_str)


# #############################################################################
# Test_generate_toc
# #############################################################################


class Test_generate_toc(hunitest.TestCase):
    """
    Test `_generate_toc()` function for extracting TOC from a single lecture file.
    """

    def test1(self) -> None:
        """
        Test _generate_toc extracts TOC and adds lesson header.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        source_path = os.path.join(scratch_dir, "Lesson01.1-Intro.smd")
        source_name = "Lesson01.1-Intro.smd"
        hio.to_file(source_path, "# Main\n## Section 1\n### Subsection")
        # Mock `system_to_string()` to return TOC content.
        with mock.patch(
            "helpers.hsystem.system_to_string"
        ) as mock_system_to_string:
            mock_system_to_string.return_value = (
                0,
                "## Section 1\n### Subsection",
            )
            result = csfolole._generate_toc(source_path, source_name)
        # Check outputs.
        self.assertIsNotNone(result)
        self.assertIn("# Lesson01.1-Intro.smd", result)
        self.assertIn("## Section 1", result)

    def test2(self) -> None:
        """
        Test `_generate_toc()` calls `extract_toc_from_txt.py` with correct parameters.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        source_path = os.path.join(scratch_dir, "Lesson02.1-Advanced.smd")
        source_name = "Lesson02.1-Advanced.smd"
        hio.to_file(source_path, "# Title\n## Content")
        # Capture system calls.
        with hunteuti.capture_sys_calls() as sys_calls:
            csfolole._generate_toc(source_path, source_name)
        # Check outputs.
        actual_str = pprint.pformat(sys_calls)
        expected_str = hprint.dedent("""
            [{'args': ('extract_toc_from_txt.py -i '
                       '$GIT_ROOT/class_scripts/test/outcomes/Test_generate_toc.test2/tmp.scratch/Lesson02.1-Advanced.smd '
                       '--max_level 5 --warn_on_malformed',),
              'function': 'hsystem.system_to_string',
              'kwargs': {'suppress_output': True}}]
            """)
        self.assert_equal(actual_str, expected_str, purify_text=True)


# #############################################################################
# Test_generate_lecture_commentary
# #############################################################################


class Test_generate_lecture_commentary(hunitest.TestCase):
    """
    Test `_generate_lecture_commentary()` function.
    """

    def test1(self) -> None:
        """
        Test `_generate_lecture_commentary()` with basic inputs generates the
        correct `gen_lecture_commentary.py` command.
        """
        # Prepare inputs.
        class_dir = "data605"
        source_path = "data605/lectures_source/Lesson01.1-Intro.smd"
        source_name = "Lesson01.1-Intro.smd"
        # Run test.
        with hunteuti.capture_sys_calls() as sys_calls:
            csfolole._generate_lecture_commentary(
                class_dir, source_path, source_name
            )
        # Check outputs.
        self.assertEqual(len(sys_calls), 1)
        cmd_str = sys_calls[0]["args"][0]
        self.assertEqual(cmd_str, "gen_lecture_commentary.py data605/01.1")

    def test2(self) -> None:
        """
        Test `_generate_lecture_commentary()` with `cmd_opts` appends the
        extra options to the invoked `gen_lecture_commentary.py` command.
        """
        # Prepare inputs.
        class_dir = "data605"
        source_path = "data605/lectures_source/Lesson01.1-Intro.smd"
        source_name = "Lesson01.1-Intro.smd"
        cmd_opts = "--no_incremental --open_pdf"
        # Run test.
        with hunteuti.capture_sys_calls() as sys_calls:
            csfolole._generate_lecture_commentary(
                class_dir, source_path, source_name, cmd_opts=cmd_opts
            )
        # Check outputs.
        self.assertEqual(len(sys_calls), 1)
        cmd_str = sys_calls[0]["args"][0]
        self.assertEqual(
            cmd_str,
            "gen_lecture_commentary.py data605/01.1 --no_incremental --open_pdf",
        )


# #############################################################################
# Test_generate_pdf_e2e
# #############################################################################


class Test_generate_pdf_e2e(hunitest.TestCase):
    """
    End-to-end tests for `_generate_pdf()` function.

    These tests execute the actual command line using `hsystem.system()`
    to verify the complete integration of the PDF generation pipeline.
    """

    def test1(self) -> None:
        """
        Fast test: `_generate_pdf()` executes successfully with minimal source file.
        """
        # Prepare inputs.
        class_dir, scratch_dir = _create_test_structure(
            self, "msml610", "lectures"
        )
        source_path = os.path.join(scratch_dir, "Lesson01.1-Intro.smd")
        source_name = "Lesson01.1-Intro.smd"
        source_content = """
        # Lesson 01.1: Introduction

        ## Slide 1
        Content.
        """
        source_content = hprint.dedent(source_content)
        hio.to_file(source_path, source_content)
        csfolole._generate_pdf(class_dir, source_path, source_name)

    def test2(self) -> None:
        """
        Fast test: `_generate_pdf()` with limit parameter completes successfully.
        """
        # Prepare inputs.
        class_dir, scratch_dir = _create_test_structure(
            self, "msml610", "lectures"
        )
        source_path = os.path.join(scratch_dir, "Lesson02.1-Advanced.smd")
        source_name = "Lesson02.1-Advanced.smd"
        source_content = """
        # Lesson 02.1: Advanced Topics

        * Slide 1
        Content.

        * Slide 2
        More content.
        """
        source_content = hprint.dedent(source_content)
        hio.to_file(source_path, source_content)
        limit = "0:1"
        csfolole._generate_pdf(class_dir, source_path, source_name, limit=limit)


# #############################################################################
# Test_generate_tex_e2e
# #############################################################################


class Test_generate_tex_e2e(hunitest.TestCase):
    """
    End-to-end tests for `_generate_tex()` function.

    These tests execute the actual command line using `hsystem.system()`
    to verify TeX file generation.
    No output verification - only checks that the command completes.
    """

    def test1(self) -> None:
        """
        Fast test: `_generate_tex()` executes successfully with minimal source file.
        """
        # Prepare inputs.
        class_dir, scratch_dir = _create_test_structure(
            self, "data605", "lectures_tex"
        )
        source_path = os.path.join(scratch_dir, "Lesson03.1-Distributed.smd")
        source_name = "Lesson03.1-Distributed.smd"
        source_content = """
        # Lesson 03.1: Distributed Systems

        ## Introduction
        Overview.
        """
        source_content = hprint.dedent(source_content)
        hio.to_file(source_path, source_content)
        csfolole._generate_tex(class_dir, source_path, source_name)


# #############################################################################
# Test_generate_script_e2e
# #############################################################################


class Test_generate_script_e2e(hunitest.TestCase):
    """
    End-to-end tests for `_generate_script()` function.

    These tests execute the actual command line using `hsystem.system()`
    to verify script file generation.
    No output verification - only checks that the command completes.
    """

    def test1(self) -> None:
        """
        Fast test: `_generate_script()` executes successfully with minimal source file.
        """
        # Prepare inputs.
        class_dir, scratch_dir = _create_test_structure(
            self, "data605", "lectures_video_script"
        )
        source_path = os.path.join(scratch_dir, "Lesson04.1-Scripts.smd")
        source_name = "Lesson04.1-Scripts.smd"
        source_content = """
        # Lesson 04.1: Scripts

        ## Transition: Start
        Beginning.

        ## Content
        Body.
        """
        source_content = hprint.dedent(source_content)
        hio.to_file(source_path, source_content)
        csfolole._generate_script(class_dir, source_path, source_name)


# #############################################################################
# Test_process_lecture_file_e2e
# #############################################################################


class Test_process_lecture_file_e2e(hunitest.TestCase):
    """
    End-to-end integration tests for `_process_lecture_file()` function.

    These tests execute actual command pipelines via `hsystem.system()`.
    No output verification - only checks that execution completes.
    """

    def test1(self) -> None:
        """
        Fast test: Process single file with `generate_pdf` action.
        """
        # Prepare inputs.
        class_dir, scratch_dir = _create_test_structure(
            self, "msml610", "lectures"
        )
        source_path = os.path.join(scratch_dir, "Lesson01.1-Test.smd")
        source_name = "Lesson01.1-Test.smd"
        source_content = """
        # Lesson 01.1: Test

        ## Slide
        Content.
        """
        source_content = hprint.dedent(source_content)
        hio.to_file(source_path, source_content)
        actions = ["generate_pdf"]
        # Run test - only verify it completes without exception.
        csfolole._process_lecture_file(
            class_dir, source_path, source_name, actions
        )

    def test2(self) -> None:
        """
        Fast test: Process single file with multiple actions.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        class_dir = os.path.join(scratch_dir, "data605")
        os.makedirs(os.path.join(class_dir, "lectures"), exist_ok=True)
        os.makedirs(os.path.join(class_dir, "lectures_tex"), exist_ok=True)
        source_path = os.path.join(scratch_dir, "Lesson02.1-Multi.smd")
        source_name = "Lesson02.1-Multi.smd"
        source_content = """
        # Lesson 02.1: Multi Action

        ## Slide 1
        Content 1.

        ## Slide 2
        Content 2.
        """
        source_content = hprint.dedent(source_content)
        hio.to_file(source_path, source_content)
        actions = ["generate_pdf", "generate_tex"]
        # Run test - only verify it completes without exception.
        csfolole._process_lecture_file(
            class_dir, source_path, source_name, actions
        )

    def test3(self) -> None:
        """
        Fast test: Process file with `generate_script` action.
        """
        # Prepare inputs.
        class_dir, scratch_dir = _create_test_structure(
            self, "data605", "lectures_video_script"
        )
        source_path = os.path.join(scratch_dir, "Lesson03.1-Script.smd")
        source_name = "Lesson03.1-Script.smd"
        source_content = """
        # Lesson 03.1: Script Test

        ## Transition: Start
        Begin.

        ## Content
        Body text.
        """
        source_content = hprint.dedent(source_content)
        hio.to_file(source_path, source_content)
        actions = ["generate_script"]
        # Run test - only verify it completes without exception.
        csfolole._process_lecture_file(
            class_dir, source_path, source_name, actions
        )


# #############################################################################
# Test_process_lecture_file_with_generate_toc
# #############################################################################


class Test_process_lecture_file_with_generate_toc(hunitest.TestCase):
    """
    Test `_process_lecture_file()` function with `generate_toc` action.

    Verifies that the function returns TOC content when `generate_toc` is specified.
    """

    def test1(self) -> None:
        """
        Test `_process_lecture_file()` returns TOC content for `generate_toc` action.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        class_dir = os.path.join(scratch_dir, "msml610")
        source_path = os.path.join(scratch_dir, "Lesson01.1-Intro.smd")
        source_name = "Lesson01.1-Intro.smd"
        hio.to_file(source_path, "# Title\n## Section 1")
        actions = ["generate_toc"]
        # Mock system_to_string.
        with mock.patch(
            "helpers.hsystem.system_to_string"
        ) as mock_system_to_string:
            mock_system_to_string.return_value = (0, "## Section 1")
            result = csfolole._process_lecture_file(
                class_dir, source_path, source_name, actions
            )
        # Check outputs.
        self.assertIsNotNone(result)
        self.assertIn("# Lesson01.1-Intro.smd", result)
        self.assertIn("## Section 1", result)

    def test2(self) -> None:
        """
        Test `_process_lecture_file()` returns None for non-`generate_toc` actions.

        Input:
        - class_dir: test class directory
        - source_path: path to source file
        - source_name: name of source file
        - actions: ['generate_pdf']

        Expected:
        - Returns None
        """
        # Prepare inputs.
        class_dir, scratch_dir = _create_test_structure(
            self, "msml610", "lectures"
        )
        source_path = os.path.join(scratch_dir, "Lesson01.1-Intro.smd")
        source_name = "Lesson01.1-Intro.smd"
        hio.to_file(source_path, "# Title")
        actions = ["generate_pdf"]
        # Capture system calls.
        with hunteuti.capture_sys_calls() as sys_calls:
            result = csfolole._process_lecture_file(
                class_dir, source_path, source_name, actions
            )
        # Check outputs.
        actual_str = pprint.pformat(sys_calls)
        expected_str = hprint.dedent("""
            [{'args': ('notes_to_pdf.py --input '
                       '$GIT_ROOT/class_scripts/test/outcomes/Test_process_lecture_file_with_generate_toc.test2/tmp.scratch/Lesson01.1-Intro.smd '
                       '--output '
                       '$GIT_ROOT/class_scripts/test/outcomes/Test_process_lecture_file_with_generate_toc.test2/tmp.scratch/msml610/lectures/Lesson01.1-Intro.pdf '
                       '--type slides --toc_type navigation --skip_action open_pdf '
                       '--debug_on_error',),
              'function': 'hsystem.system',
              'kwargs': {'suppress_output': False}}]
            """)
        self.assert_equal(actual_str, expected_str, purify_text=True)

    def test3(self) -> None:
        """
        Test `_process_lecture_file()` passes `cmd_opts` through to the
        underlying command.
        """
        # Prepare inputs.
        class_dir, scratch_dir = _create_test_structure(
            self, "msml610", "lectures"
        )
        source_path = os.path.join(scratch_dir, "Lesson01.1-Intro.smd")
        source_name = "Lesson01.1-Intro.smd"
        hio.to_file(source_path, "# Title")
        actions = ["generate_pdf"]
        cmd_opts = "--no_incremental"
        # Capture system calls.
        with hunteuti.capture_sys_calls() as sys_calls:
            csfolole._process_lecture_file(
                class_dir,
                source_path,
                source_name,
                actions,
                cmd_opts=cmd_opts,
            )
        # Check outputs.
        self.assertEqual(len(sys_calls), 1)
        cmd_str = sys_calls[0]["args"][0]
        self.assertIn(cmd_opts, cmd_str)
