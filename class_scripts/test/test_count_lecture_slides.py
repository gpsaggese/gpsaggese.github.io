"""
Unit tests for count_lecture_slides.

Import as:

import class_scripts.test.test_count_lecture_slides as cstestcls
"""

import os
import pprint
from typing import List

import pytest

import helpers.hio as hio
import helpers.hprint as hprint
import helpers.hunit_test as hunitest

# `count_lecture_slides.py` is a `uv run` script that requires `tabulate`.
pytest.importorskip("tabulate")
import class_scripts.count_lecture_slides as cscolesl


# #############################################################################
# Test__count_slides
# #############################################################################


class Test__count_slides(hunitest.TestCase):
    """
    Test `_count_slides()`.
    """

    def test1(self) -> None:
        """
        Test counting slides with multiple slide markers.

        Input: content with 3 lines starting with "* "
        Expected: 3
        """
        # Prepare inputs.
        content = """
        # Header
        * Slide 1
        Some text
        * Slide 2
        * Slide 3
        More text
        """
        content = hprint.dedent(content)
        # Prepare outputs.
        expected = 3
        # Run test.
        actual = cscolesl._count_slides(content)
        # Check outputs.
        self.assertEqual(actual, expected)

    def test2(self) -> None:
        """
        Test counting slides with no slide markers.

        Input: content with no lines starting with "* "
        Expected: 0
        """
        # Prepare inputs.
        content = """
        # Header
        Some text
        More text
        *No space after asterisk
        """
        content = hprint.dedent(content)
        # Prepare outputs.
        expected = 0
        # Run test.
        actual = cscolesl._count_slides(content)
        # Check outputs.
        self.assertEqual(actual, expected)

    def test3(self) -> None:
        """
        Test counting slides with asterisk not at start of line.

        Input: content with "* " not at start of line
        Expected: 0
        """
        # Prepare inputs.
        content = """
        This is text with * in the middle
        Not * at start
        """
        content = hprint.dedent(content)
        # Prepare outputs.
        expected = 0
        # Run test.
        actual = cscolesl._count_slides(content)
        # Check outputs.
        self.assertEqual(actual, expected)


# #############################################################################
# Test__count_headers
# #############################################################################


class Test__count_headers(hunitest.TestCase):
    """
    Test `_count_headers()`.
    """

    def test1(self) -> None:
        """
        Test counting headers at all three levels.

        Input: content with 1 H1, 2 H2, 1 H3
        Expected: (1, 2, 1)
        """
        # Prepare inputs.
        content = """
        # Main Title
        Some text
        ## Section 1
        Text
        ## Section 2
        Text
        ### Subsection
        """
        content = hprint.dedent(content)
        # Prepare outputs.
        expected = (1, 2, 1)
        # Run test.
        actual = cscolesl._count_headers(content)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test2(self) -> None:
        """
        Test counting headers with no headers.

        Input: content with no header markers at start of line
        Expected: (0, 0, 0)
        """
        # Prepare inputs.
        content = """
        This is just text
        No headers here
        Text with # in middle
        """
        content = hprint.dedent(content)
        # Prepare outputs.
        expected = (0, 0, 0)
        # Run test.
        actual = cscolesl._count_headers(content)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test3(self) -> None:
        """
        Test counting headers distinguishing between levels.

        Input: content with mixed header syntax
        Expected: correct count at each level (0, 1, 0)
        """
        # Prepare inputs.
        content = """
        ## Level 2
        Not a header: ## in text
        ####Too many hashes
        """
        content = hprint.dedent(content)
        # Prepare outputs.
        expected = (0, 1, 0)
        # Run test.
        actual = cscolesl._count_headers(content)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))


# #############################################################################
# Test__count_text_stats
# #############################################################################


class Test__count_text_stats(hunitest.TestCase):
    """
    Test `_count_text_stats()`.
    """

    def test1(self) -> None:
        """
        Test counting lines, words, and characters.

        Input: 3-line content
        Expected: (3, 7, 36)
        """
        # Prepare inputs.
        content = """
        Hello world
        Line two here
        Final line
        """
        content = hprint.dedent(content)
        # Prepare outputs.
        expected = (3, 7, 36)
        # Run test.
        actual = cscolesl._count_text_stats(content)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test2(self) -> None:
        """
        Test counting stats for empty content.

        Input: empty string
        Expected: (1, 0, 0)
        """
        # Prepare inputs.
        content = ""
        # Prepare outputs.
        expected = (1, 0, 0)
        # Run test.
        actual = cscolesl._count_text_stats(content)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test3(self) -> None:
        """
        Test counting stats with single word.

        Input: single word
        Expected: (1, 1, word_length)
        """
        # Prepare inputs.
        content = "word"
        # Prepare outputs.
        expected = (1, 1, 4)
        # Run test.
        actual = cscolesl._count_text_stats(content)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))


# #############################################################################
# Test__collect_stats
# #############################################################################


class Test__collect_stats(hunitest.TestCase):
    """
    Test `_collect_stats()`.
    """

    def test1(self) -> None:
        """
        Test collecting stats from multiple files.

        Creates two fake lecture files and verifies stats collection.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        course_dir = os.path.join(scratch, "testcourse")
        lectures_source = os.path.join(course_dir, "lectures_source")
        os.makedirs(lectures_source, exist_ok=True)
        lesson1_content = "# Intro\n* Slide 1\n* Slide 2\n## Section\nText here"
        lesson2_content = "# Chapter\n* Slide A\n## Part 1\n### Sub\nMore text"
        hio.to_file(
            os.path.join(lectures_source, "Lesson01-Intro.txt"), lesson1_content
        )
        hio.to_file(
            os.path.join(lectures_source, "Lesson02-Chapter.txt"),
            lesson2_content,
        )
        # Prepare outputs.
        expected = """
        [{'Chars': 48,
          'File': 'Lesson01-Intro.txt',
          'H1': 1,
          'H2': 1,
          'H3': 0,
          'Lines': 5,
          'Slides': 2,
          'Words': 12},
         {'Chars': 47,
          'File': 'Lesson02-Chapter.txt',
          'H1': 1,
          'H2': 1,
          'H3': 1,
          'Lines': 5,
          'Slides': 1,
          'Words': 12}]
        """
        expected = hprint.dedent(expected)
        # Run test.
        actual = cscolesl._collect_stats(course_dir)
        # Check outputs.
        actual_str = pprint.pformat(actual)
        self.assert_equal(actual_str, expected)


# #############################################################################
# Test__format_table
# #############################################################################


class Test__format_table(hunitest.TestCase):
    """
    Test `_format_table()`.
    """

    def _make_test_rows(self) -> List[dict]:
        """Helper to create test rows."""
        return [
            {
                "File": "Lesson01.txt",
                "Slides": 5,
                "H1": 1,
                "H2": 2,
                "H3": 1,
                "Lines": 50,
                "Words": 300,
                "Chars": 2000,
            },
        ]

    def test1(self) -> None:
        """
        Test markdown format output.

        Input: single row
        Expected: markdown table with GitHub format
        """
        # Prepare inputs.
        rows = self._make_test_rows()
        # Prepare outputs.
        expected = r"""
        | File         |   Slides |   H1 |   H2 |   H3 |   Lines |   Words |   Chars |
        |--------------|----------|------|------|------|---------|---------|---------|
        | Lesson01.txt |        5 |    1 |    2 |    1 |      50 |     300 |    2000 |
        """
        expected = hprint.dedent(expected)
        # Run test.
        actual = cscolesl._format_table(rows, format_type="markdown")
        # Check outputs.
        self.assert_equal(actual.strip(), expected)

    def test2(self) -> None:
        """
        Test TSV format output.

        Input: single row
        Expected: tab-separated values
        """
        # Prepare inputs.
        rows = self._make_test_rows()
        # Prepare outputs.
        expected = r"""
        File        	  Slides	  H1	  H2	  H3	  Lines	  Words	  Chars
        Lesson01.txt	       5	   1	   2	   1	     50	    300	   2000
        """
        expected = hprint.dedent(expected)
        # Run test.
        actual = cscolesl._format_table(rows, format_type="tsv")
        # Check outputs.
        self.assert_equal(actual.strip(), expected)

    def test3(self) -> None:
        """
        Test CSV format output.

        Input: single row
        Expected: comma-separated values
        """
        # Prepare inputs.
        rows = self._make_test_rows()
        # Prepare outputs.
        expected = """
        File,Slides,H1,H2,H3,Lines,Words,Chars
        Lesson01.txt,5,1,2,1,50,300,2000
        """
        expected = hprint.dedent(expected)
        # Run test.
        actual = cscolesl._format_table(rows, format_type="csv")
        # Check outputs.
        self.assert_equal(actual.strip(), expected)
