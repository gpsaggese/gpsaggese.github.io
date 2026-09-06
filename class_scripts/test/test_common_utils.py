"""
Unit tests for common_utils.py.

Import as:

import class_scripts.test.test_common_utils as csttcout
"""

import logging
import os
import sys
from unittest import mock

import helpers.hio as hio
import helpers.hunit_test as hunitest

import class_scripts.common_utils as csccouti

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_validate_dir_lesson_args
# #############################################################################


class Test_validate_dir_lesson_args(hunitest.TestCase):
    """
    Test `validate_dir_lesson_args()` function.
    """

    def helper(self, dir_arg: str, lesson_arg: str, expected: str) -> None:
        """
        Test helper for `validate_dir_lesson_args()` error cases.

        :param dir_arg: course directory
        :param lesson_arg: lesson number
        :param expected: expected substring in error message
        """
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            csccouti.validate_dir_lesson_args(dir_arg, lesson_arg)
        # Check outputs.
        self.assertIn(expected, str(cm.exception))

    def test1(self) -> None:
        """
        Test happy path: non-empty dir and lesson pass validation.
        """
        # Prepare inputs.
        dir_arg = "msml610"
        lesson_arg = "01.1"
        # Prepare outputs.
        expected = None
        # Run test.
        actual = csccouti.validate_dir_lesson_args(dir_arg, lesson_arg)
        # Check outputs.
        self.assertEqual(actual, expected)

    def test2(self) -> None:
        """
        Test edge case: empty `dir_arg` raises AssertionError.
        """
        # Prepare inputs.
        dir_arg = ""
        lesson_arg = "01.1"
        # Prepare outputs.
        expected = "DIR argument cannot be empty"
        # Run test.
        self.helper(dir_arg, lesson_arg, expected)

    def test3(self) -> None:
        """
        Test edge case: empty `lesson_arg` raises AssertionError.
        """
        # Prepare inputs.
        dir_arg = "msml610"
        lesson_arg = ""
        # Prepare outputs.
        expected = "LESSON argument cannot be empty"
        # Run test.
        self.helper(dir_arg, lesson_arg, expected)


# #############################################################################
# Test_extract_lesson_from_file
# #############################################################################


class Test_extract_lesson_from_file(hunitest.TestCase):
    """
    Test `extract_lesson_from_file()` function.
    """

    def helper1(
        self, file_path: str, expected_dir: str, expected_lesson: str
    ) -> None:
        """
        Test helper for `extract_lesson_from_file()`.

        :param file_path: file path to test
        :param expected_dir: expected directory from extraction
        :param expected_lesson: expected lesson number from extraction
        """
        actual_dir, actual_lesson = csccouti.extract_lesson_from_file(file_path)
        self.assert_equal(actual_dir, expected_dir)
        self.assert_equal(actual_lesson, expected_lesson)

    def helper2(self, file_path: str, expected_error_msg: str) -> None:
        """
        Test helper for `extract_lesson_from_file()` error cases.

        :param file_path: file path to test
        :param expected_error_msg: expected substring in error message
        """
        with self.assertRaises(AssertionError) as cm:
            csccouti.extract_lesson_from_file(file_path)
        self.assertIn(expected_error_msg, str(cm.exception))

    def test1(self) -> None:
        """
        Test extraction from valid file path with single digit lesson.
        """
        # Prepare inputs.
        file_path = "msml610/lectures_source/Lesson10-Introduction.md"
        # Prepare outputs.
        expected_dir = "msml610"
        expected_lesson = "10"
        # Run test.
        self.helper1(file_path, expected_dir, expected_lesson)

    def test2(self) -> None:
        """
        Test extraction from valid file path with dotted lesson number.
        """
        # Prepare inputs.
        file_path = "data605/lectures_source/Lesson02.3-MapReduce.txt"
        # Prepare outputs.
        expected_dir = "data605"
        expected_lesson = "02.3"
        # Run test.
        self.helper1(file_path, expected_dir, expected_lesson)

    def test3(self) -> None:
        """
        Test extraction with lesson number containing multiple dots.
        """
        # Prepare inputs.
        file_path = "msml610/lectures_source/Lesson10.2.1-Complex.md"
        # Prepare outputs.
        expected_dir = "msml610"
        expected_lesson = "10.2"
        # Run test.
        self.helper1(file_path, expected_dir, expected_lesson)

    def test4(self) -> None:
        """
        Test that invalid filename without Lesson prefix raises
        AssertionError.
        """
        # Prepare inputs.
        file_path = "msml610/lectures_source/InvalidName.md"
        # Prepare outputs.
        expected_error_msg = "Could not extract lesson number"
        # Run test.
        self.helper2(file_path, expected_error_msg)

    def test5(self) -> None:
        """
        Test that invalid directory in path raises AssertionError.
        """
        # Prepare inputs.
        file_path = "invalid_dir/lectures_source/Lesson01-Name.md"
        # Prepare outputs.
        expected_error_msg = "invalid"
        # Run test.
        self.helper2(file_path, expected_error_msg)

    def test6(self) -> None:
        """
        Test extraction from an absolute path (e.g., a test scratch dir).
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        source_dir = os.path.join(scratch_dir, "lectures_source")
        os.makedirs(source_dir, exist_ok=True)
        file_path = os.path.join(source_dir, "Lesson01.1-Intro.smd")
        hio.to_file(file_path, "content")
        # Prepare outputs.
        expected_dir = scratch_dir
        expected_lesson = "01.1"
        # Run test.
        self.helper1(file_path, expected_dir, expected_lesson)


# #############################################################################
# Test_parse_lesson_spec
# #############################################################################


class Test_parse_lesson_spec(hunitest.TestCase):
    """
    Test `parse_lesson_spec()` function.
    """

    def helper1(self, arg: str, expected_dir: str, expected_lesson: str) -> None:
        """
        Test helper for `parse_lesson_spec()`.

        :param arg: input argument to parse
        :param expected_dir: expected directory
        :param expected_lesson: expected lesson
        """
        actual_dir, actual_lesson = csccouti.parse_lesson_spec(arg)
        self.assert_equal(actual_dir, expected_dir)
        self.assert_equal(actual_lesson, expected_lesson)

    def helper2(self, arg: str, expected_error_msg: str) -> None:
        """
        Test helper for `parse_lesson_spec()` error cases.

        :param arg: input argument to parse
        :param expected_error_msg: expected substring in error message
        """
        with self.assertRaises(AssertionError) as cm:
            csccouti.parse_lesson_spec(arg)
        self.assertIn(expected_error_msg, str(cm.exception))

    def test1(self) -> None:
        """
        Test parsing dir/lesson format with msml610.
        """
        # Prepare inputs.
        arg = "msml610/08.1"
        # Prepare outputs.
        expected_dir = "msml610"
        expected_lesson = "08.1"
        # Run test.
        self.helper1(arg, expected_dir, expected_lesson)

    def test2(self) -> None:
        """
        Test parsing dir/lesson format with data605.
        """
        # Prepare inputs.
        arg = "data605/01.1"
        # Prepare outputs.
        expected_dir = "data605"
        expected_lesson = "01.1"
        # Run test.
        self.helper1(arg, expected_dir, expected_lesson)

    def test3(self) -> None:
        """
        Test parsing file path with lectures_source.
        """
        # Prepare inputs.
        arg = "msml610/lectures_source/Lesson10-Introduction.md"
        # Prepare outputs.
        expected_dir = "msml610"
        expected_lesson = "10"
        # Run test.
        self.helper1(arg, expected_dir, expected_lesson)

    def test4(self) -> None:
        """
        Test parsing file path with .txt extension.
        """
        # Prepare inputs.
        arg = "data605/lectures_source/Lesson02.3-MapReduce.txt"
        # Prepare outputs.
        expected_dir = "data605"
        expected_lesson = "02.3"
        # Run test.
        self.helper1(arg, expected_dir, expected_lesson)

    def test5(self) -> None:
        """
        Test that invalid directory in dir/lesson format raises
        AssertionError.
        """
        # Prepare inputs.
        arg = "invalid/08.1"
        # Prepare outputs.
        expected_error_msg = "doesn't exist"
        # Run test.
        self.helper2(arg, expected_error_msg)

    def test6(self) -> None:
        """
        Test that invalid format without / raises AssertionError.
        """
        # Prepare inputs.
        arg = "msml610"
        # Prepare outputs.
        expected_error_msg = "Invalid input"
        # Run test.
        self.helper2(arg, expected_error_msg)

    def test7(self) -> None:
        """
        Test that too many slashes raises AssertionError.
        """
        # Prepare inputs.
        arg = "msml610/extra/08.1"
        # Prepare outputs.
        expected_error_msg = "Expected dir/lesson format"
        # Run test.
        self.helper2(arg, expected_error_msg)


# #############################################################################
# Test_find_lecture_file
# #############################################################################


class Test_find_lecture_file(hunitest.TestCase):
    """
    Test `find_lecture_file()` function.
    """

    def helper(self, filenames: list) -> str:
        """
        Create a `lectures_source/` scratch dir with the given filenames.

        :param filenames: file names to create under `lectures_source/`
        :return: path to the parent (course) directory
        """
        scratch_dir = self.get_scratch_space()
        source_dir = os.path.join(scratch_dir, "lectures_source")
        os.makedirs(source_dir, exist_ok=True)
        for filename in filenames:
            hio.to_file(os.path.join(source_dir, filename), "content")
        return scratch_dir

    def test1(self) -> None:
        """
        Test happy path: exactly one matching file is found.
        """
        # Prepare inputs.
        filenames = ["Lesson01-Introduction.smd"]
        dir_path = self.helper(filenames)
        # Prepare outputs.
        expected = os.path.join(
            dir_path, "lectures_source", "Lesson01-Introduction.smd"
        )
        # Run test.
        actual = csccouti.find_lecture_file(dir_path, "01")
        # Check outputs.
        self.assert_equal(str(actual), expected)

    def test2(self) -> None:
        """
        Test edge case: no matching file raises AssertionError.
        """
        # Prepare inputs.
        filenames = ["Lesson02-Other.smd"]
        dir_path = self.helper(filenames)
        # Prepare outputs.
        expected = "Expected exactly one file"
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            csccouti.find_lecture_file(dir_path, "01")
        # Check outputs.
        self.assertIn(expected, str(cm.exception))

    def test3(self) -> None:
        """
        Test edge case: two matching files raises AssertionError.
        """
        # Prepare inputs.
        filenames = ["Lesson01-First.smd", "Lesson01-Second.smd"]
        dir_path = self.helper(filenames)
        # Prepare outputs.
        expected = "Expected exactly one file"
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            csccouti.find_lecture_file(dir_path, "01")
        # Check outputs.
        self.assertIn(expected, str(cm.exception))


# #############################################################################
# Test_get_source_name
# #############################################################################


class Test_get_source_name(hunitest.TestCase):
    """
    Test `get_source_name()` function.
    """

    def test1(self) -> None:
        """
        Test happy path: returns the file name without the directory path.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        source_dir = os.path.join(scratch_dir, "lectures_source")
        os.makedirs(source_dir, exist_ok=True)
        hio.to_file(
            os.path.join(source_dir, "Lesson01-Introduction.smd"), "content"
        )
        # Prepare outputs.
        expected = "Lesson01-Introduction.smd"
        # Run test.
        actual = csccouti.get_source_name(scratch_dir, "01")
        # Check outputs.
        self.assert_equal(actual, expected)


# #############################################################################
# Test_get_output_name
# #############################################################################


class Test_get_output_name(hunitest.TestCase):
    """
    Test `get_output_name()` function.
    """

    def helper(self, source_name: str, extension: str, expected: str) -> None:
        """
        Test helper for `get_output_name()`.

        :param source_name: source file name
        :param extension: new extension
        :param expected: expected output file name
        """
        # Run test.
        actual = csccouti.get_output_name(source_name, extension)
        # Check outputs.
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test happy path: replace extension on a simple file name.
        """
        # Prepare inputs.
        source_name = "Lesson01-Introduction.md"
        extension = ".pdf"
        # Prepare outputs.
        expected = "Lesson01-Introduction.pdf"
        # Run test.
        self.helper(source_name, extension, expected)

    def test2(self) -> None:
        """
        Test edge case: source name without an extension.
        """
        # Prepare inputs.
        source_name = "Lesson01-Introduction"
        extension = ".pdf"
        # Prepare outputs.
        expected = "Lesson01-Introduction.pdf"
        # Run test.
        self.helper(source_name, extension, expected)

    def test3(self) -> None:
        """
        Test edge case: source name with multiple dots keeps all but the
        last extension.
        """
        # Prepare inputs.
        source_name = "Lesson02.3-MapReduce.txt"
        extension = ".md"
        # Prepare outputs.
        expected = "Lesson02.3-MapReduce.md"
        # Run test.
        self.helper(source_name, extension, expected)


# #############################################################################
# Test_get_comment_prefix
# #############################################################################


class Test_get_comment_prefix(hunitest.TestCase):
    """
    Test `get_comment_prefix()` function.
    """

    def test1(self) -> None:
        """
        Test happy path: "tex" extension maps to the "%" LaTeX comment.
        """
        # Prepare inputs.
        extension = "tex"
        # Prepare outputs.
        expected = "%"
        # Run test.
        actual = csccouti.get_comment_prefix(extension)
        # Check outputs.
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Test happy path: "typ" extension maps to the "//" Typst comment.
        """
        # Prepare inputs.
        extension = "typ"
        # Prepare outputs.
        expected = "//"
        # Run test.
        actual = csccouti.get_comment_prefix(extension)
        # Check outputs.
        self.assert_equal(actual, expected)

    def test3(self) -> None:
        """
        Test edge case: unsupported extension raises AssertionError.
        """
        # Prepare inputs.
        extension = "md"
        # Prepare outputs.
        expected = "'md' in '{'tex': '%', 'typ': '//'}'"
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            csccouti.get_comment_prefix(extension)
        # Check outputs.
        self.assertIn(expected, str(cm.exception))


# #############################################################################
# Test_call_llm
# #############################################################################


class Test_call_llm(hunitest.TestCase):
    """
    Test `call_llm()` function.
    """

    def test1(self) -> None:
        """
        Test happy path: "hllm" backend dispatches to
        `helpers.hllm.get_completion()` and forwards its response.
        """
        # Prepare inputs.
        user_prompt = "What is the capital of France?"
        system_prompt = "You are a helpful assistant."
        model = "gpt-4"
        llm_backend = "hllm"
        # Prepare outputs.
        expected = "Paris"
        # Run test.
        # Note: the "hllm" backend tests inject a fake module into
        # `sys.modules["helpers.hllm"]` instead of using
        # `mock.patch("helpers.hllm.get_completion")` directly.
        # - Problem: `helpers.hllm` imports `openai` and `pydantic` at module
        #   level. `mock.patch("helpers.hllm.get_completion", ...)` needs to
        #   import the real `helpers.hllm` module to find `get_completion` to
        #   patch. In an environment where `openai` isn't installed that import
        #   raises `ModuleNotFoundError`.
        # - Solution: pre-populate `sys.modules["helpers.hllm"]` with a
        #   `MagicMock` via `mock.patch.dict()` before `call_llm()` runs.
        #   This keeps the test hermetic without adding `openai` as a
        #   dependency of the CI job
        # TODO(gp): Maybe factor this out.
        mock_hllm = mock.MagicMock()
        mock_hllm.get_completion.return_value = expected
        with mock.patch.dict(sys.modules, {"helpers.hllm": mock_hllm}):
            actual = csccouti.call_llm(
                user_prompt, system_prompt, model, llm_backend
            )
        # Check outputs.
        self.assert_equal(actual, expected)
        mock_hllm.get_completion.assert_called_once_with(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=model,
            cache_mode="NORMAL",
            temperature=0.1,
            images_as_base64=(),
        )

    def test2(self) -> None:
        """
        Test happy path: "hllm" backend forwards `images_as_base64` for
        multi-modal context.
        """
        # Prepare inputs.
        user_prompt = "Describe this slide."
        system_prompt = "You are a helpful assistant."
        model = ""
        llm_backend = "hllm"
        images_as_base64 = ("base64_image_data",)
        # Prepare outputs.
        expected = "A slide about causal inference."
        mock_hllm = mock.MagicMock()
        mock_hllm.get_completion.return_value = expected
        # Run test.
        with mock.patch.dict(sys.modules, {"helpers.hllm": mock_hllm}):
            actual = csccouti.call_llm(
                user_prompt,
                system_prompt,
                model,
                llm_backend,
                images_as_base64=images_as_base64,
            )
        # Check outputs.
        self.assert_equal(actual, expected)
        mock_hllm.get_completion.assert_called_once_with(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=model,
            cache_mode="NORMAL",
            temperature=0.1,
            images_as_base64=images_as_base64,
        )

    def test3(self) -> None:
        """
        Test happy path: "hllm_cli_lib" backend dispatches to
        `helpers.hllm_cli.apply_llm(backend="library")` and forwards its
        response.
        """
        # Prepare inputs.
        user_prompt = "What is the capital of France?"
        system_prompt = "You are a helpful assistant."
        model = "gpt-4"
        llm_backend = "hllm_cli_lib"
        # Prepare outputs.
        expected = "Paris"
        apply_llm_return_value = (expected, None)
        # Run test.
        with mock.patch(
            "helpers.hllm_cli.apply_llm", return_value=apply_llm_return_value
        ) as mock_apply_llm:
            actual = csccouti.call_llm(
                user_prompt, system_prompt, model, llm_backend
            )
        # Check outputs.
        self.assert_equal(actual, expected)
        mock_apply_llm.assert_called_once_with(
            user_prompt,
            system_prompt=system_prompt,
            model=model,
            backend="library",
        )

    def test4(self) -> None:
        """
        Test edge case: unsupported `llm_backend` raises AssertionError.
        """
        # Prepare inputs.
        user_prompt = "What is the capital of France?"
        system_prompt = "You are a helpful assistant."
        model = ""
        llm_backend = "invalid_backend"
        # Prepare outputs.
        expected = (
            "'invalid_backend' in '('hllm', 'hllm_cli_lib', 'hllm_cli_exec')'"
        )
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            csccouti.call_llm(user_prompt, system_prompt, model, llm_backend)
        # Check outputs.
        self.assertIn(expected, str(cm.exception))

    def test5(self) -> None:
        """
        Test happy path: "hllm_cli_exec" backend dispatches to
        `helpers.hllm_cli.apply_llm(backend="executable")` and forwards its
        response.
        """
        # Prepare inputs.
        user_prompt = "What is the capital of France?"
        system_prompt = "You are a helpful assistant."
        model = "gpt-4"
        llm_backend = "hllm_cli_exec"
        # Prepare outputs.
        expected = "Paris"
        apply_llm_return_value = (expected, None)
        # Run test.
        with mock.patch(
            "helpers.hllm_cli.apply_llm", return_value=apply_llm_return_value
        ) as mock_apply_llm:
            actual = csccouti.call_llm(
                user_prompt, system_prompt, model, llm_backend
            )
        # Check outputs.
        self.assert_equal(actual, expected)
        mock_apply_llm.assert_called_once_with(
            user_prompt,
            system_prompt=system_prompt,
            model=model,
            backend="executable",
        )


# #############################################################################
# Test_get_lesson_numbers
# #############################################################################


class Test_get_lesson_numbers(hunitest.TestCase):
    """
    Test `get_lesson_numbers()` function.
    """

    def helper(self, filenames: list) -> str:
        """
        Create a `lectures_source/` scratch dir with the given filenames.

        :param filenames: file names to create under `lectures_source/`
        :return: path to the parent (course) directory
        """
        scratch_dir = self.get_scratch_space()
        source_dir = os.path.join(scratch_dir, "lectures_source")
        os.makedirs(source_dir, exist_ok=True)
        for filename in filenames:
            hio.to_file(os.path.join(source_dir, filename), "content")
        return scratch_dir

    def test1(self) -> None:
        """
        Test happy path: lesson numbers are extracted and sorted.
        """
        # Prepare inputs.
        filenames = [
            "Lesson02-Intro.smd",
            "Lesson01.1-BigData.smd",
            "Lesson01.2-History.smd",
        ]
        course_dir = self.helper(filenames)
        # Prepare outputs.
        expected = ["01.1", "01.2", "02"]
        # Run test.
        actual = csccouti.get_lesson_numbers(course_dir)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test2(self) -> None:
        """
        Test edge case: duplicate lesson numbers (e.g., a .smd and a .pdf
        for the same lesson) are de-duplicated.
        """
        # Prepare inputs.
        filenames = ["Lesson01-Intro.smd", "Lesson01-Intro.pdf"]
        course_dir = self.helper(filenames)
        # Prepare outputs.
        expected = ["01"]
        # Run test.
        actual = csccouti.get_lesson_numbers(course_dir)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test3(self) -> None:
        """
        Test edge case: missing `lectures_source/` dir raises
        AssertionError.
        """
        # Prepare inputs.
        course_dir = self.get_scratch_space()
        # Run test.
        with self.assertRaises(AssertionError):
            csccouti.get_lesson_numbers(course_dir)


# #############################################################################
# Test_get_lesson_files
# #############################################################################


class Test_get_lesson_files(hunitest.TestCase):
    """
    Test `get_lesson_files()` function.
    """

    def helper(self, filenames: list) -> str:
        """
        Create a `lectures_source/` scratch dir with the given filenames.

        :param filenames: file names to create under `lectures_source/`
        :return: path to the parent (course) directory
        """
        scratch_dir = self.get_scratch_space()
        source_dir = os.path.join(scratch_dir, "lectures_source")
        os.makedirs(source_dir, exist_ok=True)
        for filename in filenames:
            hio.to_file(os.path.join(source_dir, filename), "content")
        return scratch_dir

    def test1(self) -> None:
        """
        Test happy path: lesson file paths are returned sorted.
        """
        # Prepare inputs.
        filenames = ["Lesson02-Intro.smd", "Lesson01-BigData.smd"]
        course_dir = self.helper(filenames)
        # Prepare outputs.
        expected = [
            os.path.join(course_dir, "lectures_source", "Lesson01-BigData.smd"),
            os.path.join(course_dir, "lectures_source", "Lesson02-Intro.smd"),
        ]
        # Run test.
        actual = csccouti.get_lesson_files(course_dir)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test2(self) -> None:
        """
        Test edge case: non-lesson files are ignored.
        """
        # Prepare inputs.
        filenames = ["Lesson01-Intro.smd", "README.md"]
        course_dir = self.helper(filenames)
        # Prepare outputs.
        expected = [
            os.path.join(course_dir, "lectures_source", "Lesson01-Intro.smd"),
        ]
        # Run test.
        actual = csccouti.get_lesson_files(course_dir)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))


# #############################################################################
# Test_collect_all_lessons
# #############################################################################


class Test_collect_all_lessons(hunitest.TestCase):
    """
    Test `collect_all_lessons()` function.
    """

    def test1(self) -> None:
        """
        Test happy path: lessons are collected per course dir via
        `get_lesson_numbers()`.
        """
        # Prepare outputs.
        expected = {
            "data605": ["01"],
            "msml610": ["02"],
            "book_springer": ["03"],
        }
        # Run test.
        with mock.patch(
            "class_scripts.common_utils.get_lesson_numbers",
            side_effect=lambda course_dir: expected[course_dir],
        ):
            actual = csccouti.collect_all_lessons()
        # Check outputs.
        self.assert_equal(str(actual), str(expected))


# #############################################################################
# Test_get_pdf_page_counts
# #############################################################################


class Test_get_pdf_page_counts(hunitest.TestCase):
    """
    Test `get_pdf_page_counts()` function.
    """

    def test1(self) -> None:
        """
        Test happy path: returns a page count per matching PDF file.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        hio.to_file(os.path.join(scratch_dir, "Lesson01.pdf"), "")
        hio.to_file(os.path.join(scratch_dir, "Lesson02.pdf"), "")
        mdls_outputs = [
            (0, "kMDItemNumberOfPages = 10"),
            (0, "kMDItemNumberOfPages = 20"),
        ]
        # Prepare outputs.
        expected = {"Lesson01.pdf": 10, "Lesson02.pdf": 20}
        # Run test.
        # TODO(ai_gp): Use the sys call mocking.
        with mock.patch(
            "helpers.hsystem.system_to_string",
            side_effect=mdls_outputs,
        ):
            actual = csccouti.get_pdf_page_counts(scratch_dir)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test2(self) -> None:
        """
        Test edge case: no matching PDF files returns an empty dict.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        # Prepare outputs.
        expected: dict = {}
        # Run test.
        actual = csccouti.get_pdf_page_counts(scratch_dir)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))
