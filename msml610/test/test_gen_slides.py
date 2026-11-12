"""
Test gen_slides.py script for msml610 course.

Import as:

import msml610.test.test_gen_slides as mttestgs
"""

import class_scripts.gen_slides_test_utils as csgsteut


# #############################################################################
# Test_Msml610_LessonDiscovery
# #############################################################################


class Test_Msml610_LessonDiscovery(csgsteut.LessonDiscovery_TestCase):
    COURSE_DIR = "msml610"
    FIRST_LESSON_FILENAME = "Lesson01.1-AI_and_Machine_Learning.smd"


# #############################################################################
# Test_Msml610_Run_preprocess_notes_py
# #############################################################################


class Test_Msml610_Run_preprocess_notes_py(
    csgsteut.Run_preprocess_notes_py_TestCase
):
    COURSE_DIR = "msml610"


# #############################################################################
# Test_Msml610_Run_notes_to_pdf_py
# #############################################################################


class Test_Msml610_Run_notes_to_pdf_py(csgsteut.Run_notes_to_pdf_py_TestCase):
    COURSE_DIR = "msml610"


# #############################################################################
# Test_Msml610_Run_gen_slides_py
# #############################################################################


class Test_Msml610_Run_gen_slides_py(csgsteut.Run_gen_slides_py_TestCase):
    COURSE_DIR = "msml610"
    FIRST_LESSON = "01.1"
    SECOND_LESSON = "08.1"
