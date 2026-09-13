"""
Test gen_slides.py script for data605 course.

Import as:

import data605.test.test_gen_slides as d6ttestgs
"""

import class_scripts.gen_slides_test_utils as csgsteut


# #############################################################################
# Test_Data605_LessonDiscovery
# #############################################################################


class Test_Data605_LessonDiscovery(csgsteut.LessonDiscovery_TestCase):
    COURSE_DIR = "data605"
    FIRST_LESSON_FILENAME = "Lesson01.1-Intro.smd"


# #############################################################################
# Test_Data605_Run_preprocess_notes_py
# #############################################################################


class Test_Data605_Run_preprocess_notes_py(
    csgsteut.Run_preprocess_notes_py_TestCase
):
    COURSE_DIR = "data605"


# #############################################################################
# Test_Data605_Run_notes_to_pdf_py
# #############################################################################


class Test_Data605_Run_notes_to_pdf_py(csgsteut.Run_notes_to_pdf_py_TestCase):
    COURSE_DIR = "data605"


# #############################################################################
# Test_Data605_Run_gen_slides_py
# #############################################################################


class Test_Data605_Run_gen_slides_py(csgsteut.Run_gen_slides_py_TestCase):
    COURSE_DIR = "data605"
    FIRST_LESSON = "01.1"
    SECOND_LESSON = "08.2"
