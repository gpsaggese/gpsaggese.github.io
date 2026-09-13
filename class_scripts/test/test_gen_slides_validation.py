"""
Cross-course validation tests for slide generation.

Tests common to both data605 and msml610 courses.

Import as:

import class_scripts.test.test_gen_slides_validation as cstestgsval
"""

import logging

import class_scripts.common_utils as csccouti
import helpers.hunit_test as hunitest

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_gen_slides_batch_validation
# #############################################################################


class Test_gen_slides_batch_validation(hunitest.TestCase):
    """
    Batch validation tests for slide generation across all courses.
    """

    def test1(self) -> None:
        """
        Test that both courses have lessons.
        """
        # Prepare inputs.
        expected_courses = ["msml610", "data605"]
        # Run test.
        all_lessons = csccouti.collect_all_lessons()
        # Check outputs.
        for course in expected_courses:
            self.assertIn(course, all_lessons)
            self.assertGreater(len(all_lessons[course]), 0)

    def test2(self) -> None:
        """
        Test that msml610 has expected number of lessons.
        """
        # Prepare inputs.
        course_dir = "msml610"
        min_expected_lessons = 35
        # Run test.
        all_lessons = csccouti.collect_all_lessons()
        lesson_count = len(all_lessons[course_dir])
        # Check outputs.
        self.assertGreaterEqual(
            lesson_count,
            min_expected_lessons,
            f"msml610 should have at least {min_expected_lessons} "
            f"lessons, found {lesson_count}",
        )
        _LOG.info("msml610 has %d lessons", lesson_count)

    def test3(self) -> None:
        """
        Test that data605 has expected number of lessons.
        """
        # Prepare inputs.
        course_dir = "data605"
        min_expected_lessons = 35
        # Run test.
        all_lessons = csccouti.collect_all_lessons()
        lesson_count = len(all_lessons[course_dir])
        # Check outputs.
        self.assertGreaterEqual(
            lesson_count,
            min_expected_lessons,
            f"data605 should have at least {min_expected_lessons} "
            f"lessons, found {lesson_count}",
        )
        _LOG.info("data605 has %d lessons", lesson_count)

    def test4(self) -> None:
        """
        Test that lesson numbers are well-formed across all courses.
        """
        # Prepare inputs.
        # TODO(ai_gp): Add examples of what this expression matches, also this should
        # become a standing function in csgsteut.
        valid_lesson_pattern = r"^\d+(\.[0-9A-Za-z]+)?$"
        error_msg_template = "Invalid lesson format '{lesson}' in {course}"
        # Run test.
        all_lessons = csccouti.collect_all_lessons()
        # Check outputs.
        for course_dir, lessons in all_lessons.items():
            for lesson in lessons:
                error_msg = error_msg_template.format(
                    lesson=lesson, course=course_dir
                )
                self.assertRegex(
                    lesson,
                    valid_lesson_pattern,
                    error_msg,
                )
