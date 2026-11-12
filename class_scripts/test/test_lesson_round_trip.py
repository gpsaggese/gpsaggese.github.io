"""
Unit tests for round-trip parsing of markdown lesson files.

Tests that lesson files can be read, parsed, reassembled, and match the original.
"""

import glob
import os

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hunit_test as hunitest
import helpers.hmarkdown_slide_iterator as hmaslite

import logging

_LOG = logging.getLogger(__name__)


# #############################################################################
# TestLessonRoundTrip
# #############################################################################


class TestLessonRoundTrip(hunitest.TestCase):
    """
    Test round-trip parsing of markdown lesson files.

    Verifies that lesson files can be read, parsed into structured items,
    reassembled back to markdown, and match the original content exactly.
    """

    def helper_test_round_trip(self, lesson_file: str) -> None:
        """
        Test helper for lesson round-trip parsing.

        :param lesson_file: Path to the lesson file to test
        """
        _LOG.info("Processing %s", lesson_file)
        # Read original content.
        original_content = hio.from_file(lesson_file)
        # Remove trailing empty lines before round-trip test.
        original_content = original_content.rstrip() + "\n"
        # Parse the lesson file.
        items = list(hmaslite.read_lesson_file(lesson_file))
        # Reassemble from parsed items.
        reassembled_content = hmaslite.reassemble_from_items(items)
        # Remove trailing empty lines from reassembled content.
        reassembled_content = reassembled_content.rstrip() + "\n"
        # Verify round-trip: reassembled must match original.
        self.assert_equal(
            original_content,
            reassembled_content,
        )

    def helper_lesson_files_round_trip(self, lesson_dir: str) -> None:
        """
        Test round-trip parsing of all Lesson*.smd files in `lesson_dir/`.

        Reads each lesson file, parses it, reassembles it, and verifies the
        reassembled content matches the original byte-for-byte.
        """
        # Find all lesson files.
        lesson_pattern = os.path.join(lesson_dir, "Lesson*.smd")
        lesson_files = sorted(glob.glob(lesson_pattern))
        hdbg.dassert_ne(
            len(lesson_files),
            0,
            "Lesson files must be found matching pattern: '%s'",
            lesson_pattern,
        )
        # Test each lesson file.
        for lesson_file in lesson_files:
            self.helper_test_round_trip(lesson_file)

    def test_data605(self) -> None:
        lesson_dir = "data605/lectures_source"
        self.helper_lesson_files_round_trip(lesson_dir)

    def test_msml610(self) -> None:
        lesson_dir = "msml610/lectures_source"
        self.helper_lesson_files_round_trip(lesson_dir)
