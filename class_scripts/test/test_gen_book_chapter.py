"""
Unit tests for gen_book_chapter.py.

Import as:

import class_scripts.test.test_gen_book_chapter as csttgeboch
"""

import logging
import os
from unittest import mock

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hprint as hprint
import helpers.hunit_test as hunitest

import class_scripts.gen_book_chapter as csgeboch

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test__add_line_numbers
# #############################################################################

# TODO(ai_gp): Add tests for public-facing functions or classes from gen_book_chapter before testing internal helpers (testing.rules.md:## Test From the Outside-In)

class Test__add_line_numbers(hunitest.TestCase):
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
        # TODO(ai_gp): Use triple-quote with hprint.dedent() instead of escaped \n (testing.rules.md:## Use Triple-Quote Assignment with `hprint.dedent` for Multi-line Strings)
        expected = "    1 | first\n    2 | second\n    3 | third"
        # Run test.
        actual = csgeboch._add_line_numbers(content)
        # Check outputs.
        # TODO(ai_gp): Use self.assert_equal() instead of assertEqual() for string comparison (testing.rules.md:## Assertion Patterns)
        self.assertEqual(actual, expected)

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
        self.assertEqual(actual, expected)

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
        self.assertEqual(actual, expected)


# #############################################################################
# Test__strip_code_fence
# #############################################################################


class Test__strip_code_fence(hunitest.TestCase):
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
        self.assertEqual(actual, expected)

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
        self.assertEqual(actual, expected)

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
        self.assertEqual(actual, expected)

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
        self.assertEqual(actual, expected)


# #############################################################################
# Test__insert_provenance_tag
# #############################################################################


class Test__insert_provenance_tag(hunitest.TestCase):
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
        tag = "git_hash=abc1234 timestamp=20250101_000000"
        expected_with_tag = expected.format(tag=tag)
        # Run test.
        # TODO(ai_gp): Patch at the call site (csgeboch module namespace) not where defined (helpers module); verify mock.patch target matches how gen_book_chapter imports get_generation_tag (testing.rules.md:## Mock at the Call Site)
        with mock.patch("helpers.hgit.get_generation_tag", return_value=tag):
            actual = csgeboch._insert_provenance_tag(text, mode)
        # Check outputs.
        # TODO(ai_gp): Use self.assert_equal() instead of assertEqual() for string comparison (testing.rules.md:## Assertion Patterns)
        self.assertEqual(actual, expected_with_tag)

    def test1(self) -> None:
        """
        Test happy path: "md" mode without YAML front matter inserts an
        HTML comment at the top of the text.
        """
        # TODO(ai_gp): Move hprint.dedent() calls to helper method instead of calling in test method (testing.rules.md:## Move Dedent and Checking into the Helper Method)
        # Prepare inputs.
        text = hprint.dedent(
            """
            # Chapter 1
            Some content.
            """
        )
        mode = "md"
        # Prepare outputs.
        expected = hprint.dedent(
            """
            <!-- {tag} -->

            # Chapter 1
            Some content.
            """
        )
        # Run test.
        self.helper(text, mode, expected)

    def test2(self) -> None:
        """
        Test edge case: "md" mode with YAML front matter inserts the
        comment right after the closing `---`.
        """
        # TODO(ai_gp): Move hprint.dedent() calls to helper method instead of calling in test method (testing.rules.md:## Move Dedent and Checking into the Helper Method)
        # Prepare inputs.
        text = hprint.dedent(
            """
            ---
            title: "Chapter 1"
            ---
            # Chapter 1
            Some content.
            """
        )
        mode = "md"
        # Prepare outputs.
        expected = hprint.dedent(
            """
            ---
            title: "Chapter 1"
            ---
            <!-- {tag} -->

            # Chapter 1
            Some content.
            """
        )
        # Run test.
        self.helper(text, mode, expected)

    def test3(self) -> None:
        """
        Test happy path: "springer_latex" mode prefixes the tag with a
        LaTeX `%` comment.
        """
        # TODO(ai_gp): Move hprint.dedent() calls to helper method instead of calling in test method (testing.rules.md:## Move Dedent and Checking into the Helper Method)
        # Prepare inputs.
        text = r"\section{Intro}"
        mode = "springer_latex"
        # Prepare outputs.
        expected = hprint.dedent(
            r"""
            % {tag}
            \section{{Intro}}
            """
        )
        # Run test.
        self.helper(text, mode, expected)

    def test4(self) -> None:
        """
        Test happy path: "typst_aima" mode prefixes the tag with a Typst
        `//` comment.
        """
        # TODO(ai_gp): Move hprint.dedent() calls to helper method instead of calling in test method (testing.rules.md:## Move Dedent and Checking into the Helper Method)
        # Prepare inputs.
        text = "= Chapter 1"
        mode = "typst_aima"
        # Prepare outputs.
        expected = hprint.dedent(
            """
            // {tag}
            = Chapter 1
            """
        )
        # Run test.
        self.helper(text, mode, expected)

    def test5(self) -> None:
        """
        Test edge case: empty text in "md" mode still receives the
        provenance tag.
        """
        # TODO(ai_gp): Move hprint.dedent() calls to helper method instead of calling in test method (testing.rules.md:## Move Dedent and Checking into the Helper Method)
        # Prepare inputs.
        text = ""
        mode = "md"
        # Prepare outputs.
        expected = hprint.dedent(
            """
            <!-- {tag} -->
            """
        )
        expected += "\n\n"
        # Run test.
        self.helper(text, mode, expected)


# #############################################################################
# Test__extract_course_and_title
# #############################################################################


class Test__extract_course_and_title(hunitest.TestCase):
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
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        input_file = os.path.join(scratch_dir, filename)
        hio.to_file(input_file, content)
        # Run test.
        actual = csgeboch._extract_course_and_title(input_file, content)
        # Check outputs.
        self.assertEqual(actual, expected)

    def test1(self) -> None:
        """
        Test happy path: course and lesson titles are extracted from the
        `course_title`/`lesson_title` metadata directives.
        """
        # TODO(ai_gp): Move hprint.dedent() calls to helper method instead of calling in test method (testing.rules.md:## Move Dedent and Checking into the Helper Method)
        # Prepare inputs.
        filename = "Lesson01.1-Intro.smd"
        content = hprint.dedent(
            """
            // course_title=MSML610: Advanced Machine Learning
            // lesson_title=L01.1: Class Introduction
            * Slide 1
            """
        )
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
        # TODO(ai_gp): Move hprint.dedent() calls to helper method instead of calling in test method (testing.rules.md:## Move Dedent and Checking into the Helper Method)
        # Prepare inputs.
        filename = "Lesson02.1-NoTitle.smd"
        content = hprint.dedent(
            """
            * Slide 1
            Some content.
            """
        )
        # Prepare outputs.
        expected = ("", "Lesson02.1-NoTitle")
        # Run test.
        self.helper(filename, content, expected)

    def test3(self) -> None:
        """
        Test edge case: only course_title directive present, lesson_title
        falls back to the input's base name.
        """
        # TODO(ai_gp): Move hprint.dedent() calls to helper method instead of calling in test method (testing.rules.md:## Move Dedent and Checking into the Helper Method)
        # Prepare inputs.
        filename = "Lesson03.1-Partial.smd"
        content = hprint.dedent(
            """
            // course_title=MSML610: Advanced Machine Learning
            * Slide 1
            """
        )
        # Prepare outputs.
        expected = ("MSML610: Advanced Machine Learning", "Lesson03.1-Partial")
        # Run test.
        self.helper(filename, content, expected)

    def test4(self) -> None:
        """
        Test edge case: only lesson_title directive present, course_title
        is empty.
        """
        # TODO(ai_gp): Move hprint.dedent() calls to helper method instead of calling in test method (testing.rules.md:## Move Dedent and Checking into the Helper Method)
        # Prepare inputs.
        filename = "Lesson04.1-LessonOnly.smd"
        content = hprint.dedent(
            """
            // lesson_title=L04.1: Advanced Topics
            * Slide 1
            """
        )
        # Prepare outputs.
        expected = ("", "L04.1: Advanced Topics")
        # Run test.
        self.helper(filename, content, expected)


# #############################################################################
# Test__build_user_prompt
# #############################################################################


class Test__build_user_prompt(hunitest.TestCase):
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
# Test__get_system_prompt
# #############################################################################


class Test__get_system_prompt(hunitest.TestCase):
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
        # TODO(ai_gp): Patch at the call site (csgeboch module namespace) not where defined (helpers module); verify mock.patch target matches how gen_book_chapter imports from_file (testing.rules.md:## Mock at the Call Site)
        with mock.patch(
            "helpers.hio.from_file",
            side_effect=[common_prompt, mode_prompt],
        ):
            actual = csgeboch._get_system_prompt(mode)
        # Check outputs.
        self.assertEqual(actual, expected)
    # TODO(ai_gp): Remove test for error condition (invalid input and assertion); focus on happy paths and edge cases instead (testing.rules.md:## What not to Test)
    def test2(self) -> None:
        """
        Test edge case: an unsupported mode raises AssertionError.
        """
        # Prepare inputs.
        mode = "bogus"
        # Prepare outputs.
        expected = "'bogus'"
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            csgeboch._get_system_prompt(mode)
        # Check outputs.
        self.assertIn(expected, str(cm.exception))


# #############################################################################
# Test__diagram_fence_re
# #############################################################################


class Test__diagram_fence_re(hunitest.TestCase):
    """
    Test `_DIAGRAM_FENCE_RE` recognizes every diagram fence language
    `render_images.py` supports.
    """

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
        # Run test.
        match = csgeboch._DIAGRAM_FENCE_RE.search(body)
        # Check outputs.
        # TODO(ai_gp): Use self.assert*() instead of hdbg.dassert_re_match(); do not use hdbg.dassert for test assertions (testing.rules.md:## Do Not Use `hdbg.dassert` to Test Assertions)
        # TODO(ai_gp): Compare whole match output with assert_equal, not piecewise—combine multiple assertEqual() calls on match groups into single assert_equal() (testing.rules.md:## Compare Whole Output with `assert_equal`, Not Piecewise)
        match = hdbg.dassert_re_match(match, "No match found in: %s", body)
        self.assertEqual(match.group("lang"), "raw_latex")
        self.assertEqual(match.group("suffix"), "")
        expected_code = hprint.dedent(
            r"""
            \begin{tikzpicture}
            \end{tikzpicture}"""
        )
        self.assertEqual(match.group("code"), expected_code)

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
        # Run test.
        match = csgeboch._DIAGRAM_FENCE_RE.search(body)
        # Check outputs.
        # TODO(ai_gp): Use self.assert*() instead of hdbg.dassert_re_match(); do not use hdbg.dassert for test assertions (testing.rules.md:## Do Not Use `hdbg.dassert` to Test Assertions)
        # TODO(ai_gp): Compare whole match output with assert_equal, not piecewise—combine multiple assertEqual() calls on match groups into single assert_equal() (testing.rules.md:## Compare Whole Output with `assert_equal`, Not Piecewise)
        match = hdbg.dassert_re_match(match, "No match found in: %s", body)
        self.assertEqual(match.group("lang"), "raw_latex")
        self.assertEqual(match.group("suffix"), "[width=62%]")

    def test3(self) -> None:
        """
        Test edge case: `graphviz`, `mermaid`, and `tikz` (the original,
        pre-fix set) are still recognized.
        """
        for lang in ("graphviz", "mermaid", "tikz"):
            # Prepare inputs.
            body = f"```{lang}\nsome code\n```"
            # Run test.
            match = csgeboch._DIAGRAM_FENCE_RE.search(body)
            # Check outputs.
            # TODO(ai_gp): Use self.assert*() instead of hdbg.dassert_re_match(); do not use hdbg.dassert for test assertions (testing.rules.md:## Do Not Use `hdbg.dassert` to Test Assertions)
            match = hdbg.dassert_re_match(match, "No match found in: %s", body)
            self.assertEqual(match.group("lang"), lang)


# #############################################################################
# Test__render_diagram_placeholder
# #############################################################################


class Test__render_diagram_placeholder(hunitest.TestCase):
    """
    Test `_render_diagram_placeholder()` function.
    """

    def test1(self) -> None:
        """
        Test happy path: no suffix, fence language emitted bare.
        """
        # TODO(ai_gp): Add "Prepare inputs" section before "Run test" - assign hardcoded parameters to variables first (testing.rules.md:## Use Three Sections in Testing Methods)
        # Run test.
        actual = csgeboch._render_diagram_placeholder(
            "raw_latex",
            "code",
            label="fig:aitimeline",
            description="Diagram illustrating AI Timeline",
        )
        # Prepare outputs.
        expected = hprint.dedent(
            """
            ```raw_latex
            code
            ```
            label=fig:aitimeline
            caption=Diagram illustrating AI Timeline
            """
        )
        # Check outputs.
        self.assertEqual(actual, expected)

    def test2(self) -> None:
        """
        Test edge case: a `[width=X%]` suffix is preserved on the re-emitted
        fence line, so `render_images.py` still sees it.
        """
        # TODO(ai_gp): Add "Prepare inputs" section before "Run test" - assign hardcoded parameters to variables first (testing.rules.md:## Use Three Sections in Testing Methods)
        # Run test.
        actual = csgeboch._render_diagram_placeholder(
            "raw_latex",
            "code",
            suffix="[width=62%]",
            label="fig:aihypecycle",
            description="Diagram illustrating The AI Hype Cycle",
        )
        # Prepare outputs.
        expected = hprint.dedent(
            """
            ```raw_latex[width=62%]
            code
            ```
            label=fig:aihypecycle
            caption=Diagram illustrating The AI Hype Cycle
            """
        )
        # Check outputs.
        self.assertEqual(actual, expected)
