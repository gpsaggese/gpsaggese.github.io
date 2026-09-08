"""
Unit tests for gen_book_chapter.py.

Import as:

import class_scripts.test.test_gen_book_chapter as csttgeboch
"""

import logging
import os
from unittest import mock

import helpers.hio as hio
import helpers.hprint as hprint
import helpers.hunit_test as hunitest

import class_scripts.gen_book_chapter as csgeboch

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_add_line_numbers
# #############################################################################


class Test_add_line_numbers(hunitest.TestCase):
    """
    Test `_add_line_numbers()` function.
    """

    def test1(self) -> None:
        """
        Test happy path: each line is prefixed with its 1-based line number.
        """
        # Prepare inputs.
        content = hprint.dedent(
            """
            first
            second
            third
            """
        )
        # Prepare outputs.
        # Note: `_add_line_numbers()` right-justifies each line number to
        # width 5, so a single-digit number gets a 4-space pad (see test3);
        # this is data, not code indentation, so it is not dedented.
        expected = """    1 | first
    2 | second
    3 | third"""
        # Run test.
        actual = csgeboch._add_line_numbers(content)
        # Check outputs.
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Test edge case: empty content returns an empty string.
        """
        # Prepare inputs.
        content = ""
        # Prepare outputs.
        expected = ""
        # Run test.
        actual = csgeboch._add_line_numbers(content)
        # Check outputs.
        self.assert_equal(actual, expected)

    def test3(self) -> None:
        """
        Test edge case: single-line content is prefixed with line number 1.
        """
        # Prepare inputs.
        content = "only line"
        # Prepare outputs.
        expected = "    1 | only line"
        # Run test.
        actual = csgeboch._add_line_numbers(content)
        # Check outputs.
        self.assert_equal(actual, expected)


# #############################################################################
# Test_strip_code_fence
# #############################################################################


class Test_strip_code_fence(hunitest.TestCase):
    """
    Test `_strip_code_fence()` function.
    """

    def test1(self) -> None:
        """
        Test happy path: a fenced block with a language tag is unwrapped.
        """
        # Prepare inputs.
        text = hprint.dedent(
            r"""
            ```latex
            \section{Intro}
            ```
            """
        )
        # Prepare outputs.
        expected = r"\section{Intro}"
        # Run test.
        actual = csgeboch._strip_code_fence(text)
        # Check outputs.
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Test edge case: a fenced block without a language tag is unwrapped.
        """
        # Prepare inputs.
        text = hprint.dedent(
            """
            ```
            plain text
            ```
            """
        )
        # Prepare outputs.
        expected = "plain text"
        # Run test.
        actual = csgeboch._strip_code_fence(text)
        # Check outputs.
        self.assert_equal(actual, expected)

    def test3(self) -> None:
        """
        Test edge case: text without an enclosing fence is left unchanged.
        """
        # Prepare inputs.
        text = hprint.dedent(
            """
            # Chapter 1
            Some content.
            """
        )
        # Prepare outputs.
        expected = text
        # Run test.
        actual = csgeboch._strip_code_fence(text)
        # Check outputs.
        self.assert_equal(actual, expected)

    def test4(self) -> None:
        """
        Test edge case: empty fence block (no content between delimiters).
        """
        # Prepare inputs.
        text = hprint.dedent(
            """
            ```
            ```
            """
        )
        # Prepare outputs.
        expected = ""
        # Run test.
        actual = csgeboch._strip_code_fence(text)
        # Check outputs.
        self.assert_equal(actual, expected)


# #############################################################################
# Test_insert_provenance_tag
# #############################################################################


class Test_insert_provenance_tag(hunitest.TestCase):
    """
    Test `_insert_provenance_tag()` function.
    """

    def helper(self, text: str, mode: str, expected: str) -> None:
        """
        Test helper for _insert_provenance_tag.

        :param text: Input text
        :param mode: Mode for tag insertion
        :param expected: Expected output with {tag} placeholder
        """
        text = hprint.dedent(text)
        expected = hprint.dedent(expected)
        tag = "git_hash=abc1234 timestamp=20250101_000000"
        expected_with_tag = expected.format(tag=tag)
        # Run test.
        with mock.patch.object(
            csgeboch.hgit, "get_generation_tag", return_value=tag
        ):
            actual = csgeboch._insert_provenance_tag(text, mode)
        # Check outputs.
        # TODO(ai_gp): Use self.assert_equal() instead of assertEqual() for string comparison (testing.rules.md:## Assertion Patterns)
        self.assertEqual(actual, expected_with_tag)

    def test1(self) -> None:
        """
        Test happy path: "md" mode without YAML front matter inserts an
        HTML comment at the top of the text.
        """
        # Prepare inputs.
        text = """
        # Chapter 1
        Some content.
        """
        mode = "md"
        # Prepare outputs.
        expected = """
        <!-- {tag} -->

        # Chapter 1
        Some content.
        """
        # Run test.
        self.helper(text, mode, expected)

    def test2(self) -> None:
        """
        Test edge case: "md" mode with YAML front matter inserts the
        comment right after the closing `---`.
        """
        # Prepare inputs.
        text = """
        ---
        title: "Chapter 1"
        ---
        # Chapter 1
        Some content.
        """
        mode = "md"
        # Prepare outputs.
        expected = """
        ---
        title: "Chapter 1"
        ---
        <!-- {tag} -->

        # Chapter 1
        Some content.
        """
        # Run test.
        self.helper(text, mode, expected)

    def test3(self) -> None:
        """
        Test happy path: "springer_latex" mode prefixes the tag with a
        LaTeX `%` comment.
        """
        # Prepare inputs.
        text = r"\section{Intro}"
        mode = "springer_latex"
        # Prepare outputs.
        expected = r"""
        % {tag}
        \section{{Intro}}
        """
        # Run test.
        self.helper(text, mode, expected)

    def test4(self) -> None:
        """
        Test happy path: "typst_aima" mode prefixes the tag with a Typst
        `//` comment.
        """
        # Prepare inputs.
        text = "= Chapter 1"
        mode = "typst_aima"
        # Prepare outputs.
        expected = """
        // {tag}
        = Chapter 1
        """
        # Run test.
        self.helper(text, mode, expected)

    def test5(self) -> None:
        """
        Test edge case: empty text in "md" mode still receives the
        provenance tag.
        """
        # Prepare inputs.
        text = ""
        mode = "md"
        # Prepare outputs.
        expected = """
        <!-- {tag} -->
        """
        expected += "\n\n"
        # Run test.
        self.helper(text, mode, expected)


# #############################################################################
# Test_extract_course_and_title
# #############################################################################


class Test_extract_course_and_title(hunitest.TestCase):
    """
    Test `_extract_course_and_title()` function.
    """

    def helper(self, filename: str, content: str, expected: tuple) -> None:
        """
        Test helper for _extract_course_and_title.

        :param filename: Input file name
        :param content: Input file content
        :param expected: Expected (course_title, lesson_title) tuple
        """
        content = hprint.dedent(content)
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        input_file = os.path.join(scratch_dir, filename)
        hio.to_file(input_file, content)
        # Run test.
        actual = csgeboch._extract_course_and_title(input_file, content)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test1(self) -> None:
        """
        Test happy path: course and lesson titles are extracted from the
        `course_title`/`lesson_title` metadata directives.
        """
        # Prepare inputs.
        filename = "Lesson01.1-Intro.smd"
        content = """
        // course_title=MSML610: Advanced Machine Learning
        // lesson_title=L01.1: Class Introduction
        * Slide 1
        """
        # Prepare outputs.
        expected = (
            "MSML610: Advanced Machine Learning",
            "L01.1: Class Introduction",
        )
        # Run test.
        self.helper(filename, content, expected)

    def test2(self) -> None:
        """
        Test edge case: no metadata directives falls back to the input's
        base name for the chapter title and an empty course title.
        """
        # Prepare inputs.
        filename = "Lesson02.1-NoTitle.smd"
        content = """
        * Slide 1
        Some content.
        """
        # Prepare outputs.
        expected = ("", "Lesson02.1-NoTitle")
        # Run test.
        self.helper(filename, content, expected)

    def test3(self) -> None:
        """
        Test edge case: only course_title directive present, lesson_title
        falls back to the input's base name.
        """
        # Prepare inputs.
        filename = "Lesson03.1-Partial.smd"
        content = """
        // course_title=MSML610: Advanced Machine Learning
        * Slide 1
        """
        # Prepare outputs.
        expected = ("MSML610: Advanced Machine Learning", "Lesson03.1-Partial")
        # Run test.
        self.helper(filename, content, expected)

    def test4(self) -> None:
        """
        Test edge case: only lesson_title directive present, course_title
        is empty.
        """
        # Prepare inputs.
        filename = "Lesson04.1-LessonOnly.smd"
        content = """
        // lesson_title=L04.1: Advanced Topics
        * Slide 1
        """
        # Prepare outputs.
        expected = ("", "L04.1: Advanced Topics")
        # Run test.
        self.helper(filename, content, expected)


# #############################################################################
# Test_build_user_prompt
# #############################################################################


class Test_build_user_prompt(hunitest.TestCase):
    """
    Test `_build_user_prompt()` function.
    """

    def test1(self) -> None:
        """
        Test happy path: "md" mode builds a header without Typst-specific
        metadata.
        """
        # Prepare inputs.
        input_file = "msml610/lectures_source/Lesson01.1-Intro.smd"
        content = "Some content."
        mode = "md"
        course_title = "MSML610: Advanced Machine Learning"
        chapter_title = "Class Introduction"
        lesson = "01.1"
        # Prepare outputs.
        expected = hprint.dedent(
            """
            Source file: msml610/lectures_source/Lesson01.1-Intro.smd
            Chapter title: Class Introduction
            Course title: MSML610: Advanced Machine Learning

            ---

                1 | Some content."""
        )
        # Run test.
        actual = csgeboch._build_user_prompt(
            input_file, content, mode, course_title, chapter_title, lesson
        )
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Test happy path: "typst_aima" mode adds the chapter number and the
        Typst import line to the header.
        """
        # Prepare inputs.
        input_file = "msml610/lectures_source/Lesson10.2-Name.smd"
        content = "Some content."
        mode = "typst_aima"
        course_title = "MSML610: Advanced Machine Learning"
        chapter_title = "Name"
        lesson = "10.2"
        # Prepare outputs.
        expected = hprint.dedent(
            """
            Source file: msml610/lectures_source/Lesson10.2-Name.smd
            Chapter title: Name
            Course title: MSML610: Advanced Machine Learning
            Chapter number: 10
            Typst import line: #import "/helpers_root/dev_scripts_helpers/typst/aima_style.typ": aima-style, algorithm, chapter, glossary

            ---

                1 | Some content."""
        )
        # Run test.
        actual = csgeboch._build_user_prompt(
            input_file, content, mode, course_title, chapter_title, lesson
        )
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)

    def test3(self) -> None:
        """
        Test edge case: empty content still includes header and separator.
        """
        # Prepare inputs.
        input_file = "msml610/lectures_source/Lesson05.1-Empty.smd"
        content = ""
        mode = "md"
        course_title = "MSML610: Advanced Machine Learning"
        chapter_title = "Empty Chapter"
        lesson = "05.1"
        # Prepare outputs.
        expected = hprint.dedent(
            """
            Source file: msml610/lectures_source/Lesson05.1-Empty.smd
            Chapter title: Empty Chapter
            Course title: MSML610: Advanced Machine Learning

            ---

            """
        )
        # Run test.
        actual = csgeboch._build_user_prompt(
            input_file, content, mode, course_title, chapter_title, lesson
        )
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_get_system_prompt
# #############################################################################


class Test_get_system_prompt(hunitest.TestCase):
    """
    Test `_get_system_prompt()` function.
    """

    def test1(self) -> None:
        """
        Test happy path: the common and mode-specific prompt files are
        concatenated with a blank line in between.
        """
        # Prepare inputs.
        mode = "md"
        common_prompt = "Common style guide."
        mode_prompt = "Markdown-specific instructions."
        # Prepare outputs.
        expected = hprint.dedent(
            f"""
            {common_prompt}

            {mode_prompt}
            """
        )
        # Run test.
        with mock.patch.object(
            csgeboch.hio,
            "from_file",
            side_effect=[common_prompt, mode_prompt],
        ):
            actual = csgeboch._get_system_prompt(mode)
        # Check outputs.
        self.assert_equal(actual, expected)


# #############################################################################
# Test_diagram_fence_re
# #############################################################################


class Test_diagram_fence_re(hunitest.TestCase):
    """
    Test `_DIAGRAM_FENCE_RE` recognizes every diagram fence language
    `render_images.py` supports.
    """

    # TODO(ai_gp): Factor out an helper
    def test1(self) -> None:
        """
        Test happy path: a plain `raw_latex` fence (no name/size suffix) is
        recognized, with an empty `suffix` group.
        """
        # Prepare inputs.
        body = hprint.dedent(
            r"""
            ```raw_latex
            \begin{tikzpicture}
            \end{tikzpicture}
            ```
            """
        )
        # Prepare outputs.
        expected = {
            "lang": "raw_latex",
            "suffix": "",
            "code": hprint.dedent(
                r"""
                \begin{tikzpicture}
                \end{tikzpicture}"""
            ),
        }
        # Run test.
        match = csgeboch._DIAGRAM_FENCE_RE.search(body)
        self.assertIsNotNone(match, f"No match found in: {body}")
        # Check outputs.
        actual = {
            "lang": match.group("lang"),
            "suffix": match.group("suffix"),
            "code": match.group("code"),
        }
        self.assert_equal(str(actual), str(expected))

    def test2(self) -> None:
        """
        Test edge case: a `raw_latex` fence with a `[width=X%]` suffix is
        recognized, and the suffix is captured separately from the code.
        """
        # Prepare inputs.
        body = hprint.dedent(
            r"""
            ```raw_latex[width=62%]
            \begin{tikzpicture}
            \end{tikzpicture}
            ```
            """
        )
        # Prepare outputs.
        expected = {
            "lang": "raw_latex",
            "suffix": "[width=62%]",
        }
        # Run test.
        match = csgeboch._DIAGRAM_FENCE_RE.search(body)
        self.assertIsNotNone(match, f"No match found in: {body}")
        # Check outputs.
        actual = {
            "lang": match.group("lang"),
            "suffix": match.group("suffix"),
        }
        self.assert_equal(str(actual), str(expected))

    def test3(self) -> None:
        """
        Test edge case: `graphviz`, `mermaid`, and `tikz` (the original,
        pre-fix set) are still recognized.
        """
        for expected_lang in ("graphviz", "mermaid", "tikz"):
            # Prepare inputs.
            body = f"```{expected_lang}\nsome code\n```"
            # Run test.
            match = csgeboch._DIAGRAM_FENCE_RE.search(body)
            self.assertIsNotNone(match, f"No match found in: {body}")
            # Check outputs.
            actual_lang = match.group("lang")
            self.assert_equal(actual_lang, expected_lang)


# #############################################################################
# Test_render_diagram_placeholder
# #############################################################################


class Test_render_diagram_placeholder(hunitest.TestCase):
    """
    Test `_render_diagram_placeholder()` function.
    """

    def helper(
        self,
        fence_language: str,
        code: str,
        label: str,
        description: str,
        suffix: str = "",
        expected_output: str = "",
    ) -> None:
        """
        Test helper for _render_diagram_placeholder.

        :param fence_language: Fence language (e.g., "raw_latex")
        :param code: Code content
        :param label: Figure label
        :param description: Figure description
        :param suffix: Optional fence suffix (e.g., "[width=62%]")
        :param expected_output: Expected output string (will be dedented)
        """
        expected = hprint.dedent(expected_output)
        # Run test.
        actual = csgeboch._render_diagram_placeholder(
            fence_language,
            code,
            suffix=suffix,
            label=label,
            description=description,
        )
        # Check outputs.
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test happy path: no suffix, fence language emitted bare.
        """
        # Prepare inputs.
        fence_language = "raw_latex"
        code = "code"
        label = "fig:aitimeline"
        description = "Diagram illustrating AI Timeline"
        # Prepare outputs.
        expected = """
        ```raw_latex
        code
        ```
        label=fig:aitimeline
        caption=Diagram illustrating AI Timeline
        """
        # Run test.
        self.helper(fence_language, code, label, description, "", expected)

    def test2(self) -> None:
        """
        Test edge case: a `[width=X%]` suffix is preserved on the re-emitted
        fence line, so `render_images.py` still sees it.
        """
        # Prepare inputs.
        fence_language = "raw_latex"
        code = "code"
        suffix = "[width=62%]"
        label = "fig:aihypecycle"
        description = "Diagram illustrating The AI Hype Cycle"
        # Prepare outputs.
        expected = """
        ```raw_latex[width=62%]
        code
        ```
        label=fig:aihypecycle
        caption=Diagram illustrating The AI Hype Cycle
        """
        # Run test.
        self.helper(fence_language, code, label, description, suffix, expected)
