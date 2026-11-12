"""
Unit tests for inline_content_in_skeleton_slides.py.

Import as:

import class_scripts.test.test_inline_content_in_skeleton_slides as csttinsl
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
from unittest import mock

import helpers.hio as hio
import helpers.hprint as hprint
import helpers.hunit_test as hunitest

import class_scripts.inline_content_in_skeleton_slides as csicisksl

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_inline_content_in_skeleton_slides_py
# #############################################################################


class Test_inline_content_in_skeleton_slides_py(hunitest.TestCase):
    """
    End-to-end tests for `_main()`.
    """

    def helper(
        self,
        skeleton_txt: str,
        extra_args: List[str],
        expected_txt: str,
    ) -> None:
        """
        Helper for running `_main()` end-to-end and checking the rewritten
        file.

        :param skeleton_txt: skeleton `.smd` content written to the input
            file
        :param extra_args: additional CLI arguments appended after `-i
            <input_file>`
        :param expected_txt: expected content of the input file after
            `_main()` runs
        """
        # Prepare inputs.
        skeleton_txt = hprint.dedent(skeleton_txt)
        expected_txt = hprint.dedent(expected_txt)
        scratch_dir = self.get_scratch_space()
        skeleton_file = os.path.join(scratch_dir, "test.smd")
        hio.to_file(skeleton_file, skeleton_txt)
        argv = [
            "inline_content_in_skeleton_slides.py",
            "-i",
            skeleton_file,
        ] + extra_args
        # Run test.
        parser = csicisksl._parse()
        with mock.patch("sys.argv", argv):
            csicisksl._main(parser)
        # Check outputs.
        actual_txt = hio.from_file(skeleton_file)
        self.assert_equal(actual_txt, expected_txt)

    def test1(self) -> None:
        """
        Test happy path: a slide with no `// From` tag round-trips
        unchanged through the full CLI pipeline.
        """
        # Prepare inputs.
        skeleton_txt = """
        * Slide A
        Some content, no `// From` tag.
        """
        extra_args: List[str] = []
        # Prepare outputs.
        expected_txt = skeleton_txt
        # Run test and check outputs.
        self.helper(skeleton_txt, extra_args, expected_txt)

    def test2(self) -> None:
        """
        Test `--no_abort_on_issue` leaves a placeholder instead of raising,
        end to end, for a missing referenced file.
        """
        # Prepare inputs.
        skeleton_txt = """
        * Slide A
        // From does_not_exist.smd:Some Title
        """
        extra_args = ["--no_abort_on_issue"]
        # Prepare outputs.
        expected_txt = skeleton_txt
        # Run test and check outputs.
        self.helper(skeleton_txt, extra_args, expected_txt)


# #############################################################################
# Test_inline_content_in_skeleton_slides
# #############################################################################


class Test_inline_content_in_skeleton_slides(hunitest.TestCase):
    """
    Test `inline_content_in_skeleton_slides()` function.
    """

    def helper(
        self,
        skeleton_txt: str,
        source_files: Dict[str, str],
        no_abort_on_issue: bool,
        expected_txt: str,
        expected_num_inlined: int,
        expected_num_missing: int,
    ) -> None:
        """
        Helper for testing `inline_content_in_skeleton_slides()`.

        :param skeleton_txt: skeleton `.smd` content
        :param source_files: `{relative path: content}` of `// From` target
            files, materialized under a scratch dir used as `root_dir`
        :param no_abort_on_issue: passed through to the function under test
        :param expected_txt: expected transformed skeleton content
        :param expected_num_inlined: expected count of inlined tags
        :param expected_num_missing: expected count of placeholder tags
        """
        # Prepare inputs.
        skeleton_txt = hprint.dedent(skeleton_txt)
        root_dir = self.get_scratch_space()
        for rel_path, content in source_files.items():
            hio.to_file(os.path.join(root_dir, rel_path), hprint.dedent(content))
        lines = skeleton_txt.split("\n")
        # Run test.
        (
            actual_lines,
            actual_num_inlined,
            actual_num_missing,
        ) = csicisksl.inline_content_in_skeleton_slides(
            lines, root_dir=root_dir, no_abort_on_issue=no_abort_on_issue
        )
        # Check outputs.
        actual_txt = hprint.dedent("\n".join(actual_lines))
        self.assert_equal(actual_txt, expected_txt, dedent=True)
        self.assertEqual(actual_num_inlined, expected_num_inlined)
        self.assertEqual(actual_num_missing, expected_num_missing)

    def test1(self) -> None:
        """
        Test happy path: a `* ` slide-level match is inlined.
        """
        # Prepare inputs.
        skeleton_txt = """
        * Slide A
        // From src.smd:Title X
        """
        source_files = {
            "src.smd": """
            * Title X
            Body line 1
            Body line 2
            * Title Y
            Other
            """
        }
        no_abort_on_issue = False
        # Prepare outputs.
        expected_txt = """
        * Slide A
        // From src.smd:Title X

        Body line 1
        Body line 2
        """
        expected_num_inlined = 1
        expected_num_missing = 0
        # Run test and check outputs.
        self.helper(
            skeleton_txt,
            source_files,
            no_abort_on_issue,
            expected_txt,
            expected_num_inlined,
            expected_num_missing,
        )

    def test2(self) -> None:
        """
        Test happy path: a header-level match inlines nested content.
        """
        # Prepare inputs.
        skeleton_txt = """
        * Slide A
        // From src.smd:Section A
        """
        source_files = {
            "src.smd": """
            ## Section A
            ### Sub A
            * Slide A1
            Content A1
            ## Section B
            Content B
            """
        }
        no_abort_on_issue = False
        # Prepare outputs.
        expected_txt = """
        * Slide A
        // From src.smd:Section A

        ### Sub A
        * Slide A1
        Content A1
        """
        expected_num_inlined = 1
        expected_num_missing = 0
        # Run test and check outputs.
        self.helper(
            skeleton_txt,
            source_files,
            no_abort_on_issue,
            expected_txt,
            expected_num_inlined,
            expected_num_missing,
        )

    def test3(self) -> None:
        """
        Test a missing file with `no_abort_on_issue=True` leaves a
        placeholder.
        """
        # Prepare inputs.
        skeleton_txt = """
        * Slide A
        // From does_not_exist.smd:Some Title
        """
        source_files: Dict[str, str] = {}
        no_abort_on_issue = True
        # Prepare outputs.
        expected_txt = skeleton_txt
        expected_num_inlined = 0
        expected_num_missing = 1
        # Run test and check outputs.
        self.helper(
            skeleton_txt,
            source_files,
            no_abort_on_issue,
            expected_txt,
            expected_num_inlined,
            expected_num_missing,
        )

    def test4(self) -> None:
        """
        Test a missing title with `no_abort_on_issue=True` leaves a
        placeholder.
        """
        # Prepare inputs.
        skeleton_txt = """
        * Slide A
        // From src.smd:Missing Title
        """
        source_files = {
            "src.smd": """
            * Title X
            Body
            """
        }
        no_abort_on_issue = True
        # Prepare outputs.
        expected_txt = skeleton_txt
        expected_num_inlined = 0
        expected_num_missing = 1
        # Run test and check outputs.
        self.helper(
            skeleton_txt,
            source_files,
            no_abort_on_issue,
            expected_txt,
            expected_num_inlined,
            expected_num_missing,
        )

    def test5(self) -> None:
        """
        Test a `* ` slide with no `// From` tag is left untouched.
        """
        # Prepare inputs.
        skeleton_txt = """
        * Slide A
        Some content here.
        * Slide B
        // From src.smd:Title X
        """
        source_files = {
            "src.smd": """
            * Title X
            Body
            """
        }
        no_abort_on_issue = False
        # Prepare outputs.
        expected_txt = """
        * Slide A
        Some content here.
        * Slide B
        // From src.smd:Title X

        Body
        """
        expected_num_inlined = 1
        expected_num_missing = 0
        # Run test and check outputs.
        self.helper(
            skeleton_txt,
            source_files,
            no_abort_on_issue,
            expected_txt,
            expected_num_inlined,
            expected_num_missing,
        )

    def test6(self) -> None:
        """
        Test a title that only exists inside a `/* ... */` comment block is
        treated as not found.
        """
        # Prepare inputs.
        skeleton_txt = """
        * Slide A
        // From src.smd:Hidden Title
        """
        source_files = {
            "src.smd": """
            /*
            * Hidden Title
            Body
            */
            * Visible Title
            Other
            """
        }
        no_abort_on_issue = True
        # Prepare outputs.
        expected_txt = skeleton_txt
        expected_num_inlined = 0
        expected_num_missing = 1
        # Run test and check outputs.
        self.helper(
            skeleton_txt,
            source_files,
            no_abort_on_issue,
            expected_txt,
            expected_num_inlined,
            expected_num_missing,
        )

    def test7(self) -> None:
        """
        Test default mode (`no_abort_on_issue=False`) raises when a
        referenced file is missing.
        """
        # Prepare inputs.
        skeleton_txt = hprint.dedent(
            """
            * Slide A
            // From does_not_exist.smd:Some Title
            """
        )
        lines = skeleton_txt.split("\n")
        root_dir = self.get_scratch_space()
        # Run test and check output.
        with self.assertRaises(AssertionError):
            csicisksl.inline_content_in_skeleton_slides(
                lines, root_dir=root_dir, no_abort_on_issue=False
            )


# #############################################################################
# Test__parse
# #############################################################################


class Test__parse(hunitest.TestCase):
    """
    Test `_parse()` function.
    """

    def helper(
        self,
        arg_list: List[str],
        expected_input_file: str,
        expected_no_abort_on_issue: bool,
    ) -> None:
        """
        Helper for testing `_parse()`.

        :param arg_list: command-line argument list to parse
        :param expected_input_file: expected `args.input_file`
        :param expected_no_abort_on_issue: expected `args.no_abort_on_issue`
        """
        # Run test.
        parser = csicisksl._parse()
        args = parser.parse_args(arg_list)
        # Check outputs.
        self.assert_equal(args.input_file, expected_input_file)
        self.assertEqual(args.no_abort_on_issue, expected_no_abort_on_issue)

    def test1(self) -> None:
        """
        Test the parser accepts `-i`/`--input_file` and defaults
        `no_abort_on_issue` to `False`.
        """
        # Prepare inputs.
        arg_list = ["-i", "test.smd"]
        # Prepare outputs.
        expected_input_file = "test.smd"
        expected_no_abort_on_issue = False
        # Run test and check outputs.
        self.helper(arg_list, expected_input_file, expected_no_abort_on_issue)

    def test2(self) -> None:
        """
        Test `--no_abort_on_issue` sets the flag to `True`.
        """
        # Prepare inputs.
        arg_list = ["-i", "test.smd", "--no_abort_on_issue"]
        # Prepare outputs.
        expected_input_file = "test.smd"
        expected_no_abort_on_issue = True
        # Run test and check outputs.
        self.helper(arg_list, expected_input_file, expected_no_abort_on_issue)


# #############################################################################
# Test__get_title_level
# #############################################################################


class Test__get_title_level(hunitest.TestCase):
    """
    Test `_get_title_level()` function.
    """

    def helper(self, line: str, expected: Tuple[bool, int, str]) -> None:
        """
        Helper for testing `_get_title_level()`.

        :param line: candidate title line
        :param expected: expected `(is_title, level, title)` tuple
        """
        # Run test.
        actual = csicisksl._get_title_level(line)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test1(self) -> None:
        """
        Test a plain content line is not a title.
        """
        # Prepare inputs.
        line = "Some content"
        # Prepare outputs.
        expected = (False, 0, "")
        # Run test and check outputs.
        self.helper(line, expected)

    def test2(self) -> None:
        """
        Test an H1 Markdown header.
        """
        # Prepare inputs.
        line = "# Title"
        # Prepare outputs.
        expected = (True, 1, "Title")
        # Run test and check outputs.
        self.helper(line, expected)

    def test3(self) -> None:
        """
        Test an H2 Markdown header.
        """
        # Prepare inputs.
        line = "## Subtitle"
        # Prepare outputs.
        expected = (True, 2, "Subtitle")
        # Run test and check outputs.
        self.helper(line, expected)

    def test4(self) -> None:
        """
        Test a `* ` slide bullet, deeper than any Markdown header level.
        """
        # Prepare inputs.
        line = "* Slide Title"
        # Prepare outputs.
        expected = (True, 100, "Slide Title")
        # Run test and check outputs.
        self.helper(line, expected)

    def test5(self) -> None:
        """
        Test a `#`-framed section-divider comment is not mistaken for a
        header.
        """
        # Prepare inputs.
        line = "# " + "#" * 40
        # Prepare outputs.
        expected = (False, 0, "")
        # Run test and check outputs.
        self.helper(line, expected)


# #############################################################################
# Test__strip_comment_blocks
# #############################################################################


class Test__strip_comment_blocks(hunitest.TestCase):
    """
    Test `_strip_comment_blocks()` function.
    """

    def helper(self, txt: str, expected: str) -> None:
        """
        Helper for testing `_strip_comment_blocks()`.

        :param txt: input file content
        :param expected: expected content with comment blocks removed
        """
        # Prepare inputs.
        txt = hprint.dedent(txt)
        lines = txt.split("\n")
        # Run test.
        actual = "\n".join(csicisksl._strip_comment_blocks(lines))
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)

    def test1(self) -> None:
        """
        Test text with no comment blocks is unchanged.
        """
        # Prepare inputs.
        txt = """
        before
        after
        """
        # Prepare outputs.
        expected = txt
        # Run test and check outputs.
        self.helper(txt, expected)

    def test2(self) -> None:
        """
        Test a `/* ... */` block is removed.
        """
        # Prepare inputs.
        txt = """
        before
        /*
        hidden1
        hidden2
        */
        after
        """
        # Prepare outputs.
        expected = """
        before
        after
        """
        # Run test and check outputs.
        self.helper(txt, expected)

    def test3(self) -> None:
        """
        Test a `<!-- ... -->` block is removed.
        """
        # Prepare inputs.
        txt = """
        before
        <!--
        hidden
        -->
        after
        """
        # Prepare outputs.
        expected = """
        before
        after
        """
        # Run test and check outputs.
        self.helper(txt, expected)

    def test4(self) -> None:
        """
        Test empty input is unchanged.
        """
        # Prepare inputs.
        txt = ""
        # Prepare outputs.
        expected = txt
        # Run test and check outputs.
        self.helper(txt, expected)

    def test5(self) -> None:
        """
        Test multiple, separate comment blocks are all removed.
        """
        # Prepare inputs.
        txt = """
        before
        /*
        hidden1
        */
        middle
        /*
        hidden2
        */
        after
        """
        # Prepare outputs.
        expected = """
        before
        middle
        after
        """
        # Run test and check outputs.
        self.helper(txt, expected)

    def test6(self) -> None:
        """
        Test a comment block at the very start of the file is removed.
        """
        # Prepare inputs.
        txt = """
        /*
        hidden
        */
        after
        """
        # Prepare outputs.
        expected = """
        after
        """
        # Run test and check outputs.
        self.helper(txt, expected)

    def test7(self) -> None:
        """
        Test a comment block at the very end of the file is removed.
        """
        # Prepare inputs.
        txt = """
        before
        /*
        hidden
        */
        """
        # Prepare outputs.
        expected = """
        before
        """
        # Run test and check outputs.
        self.helper(txt, expected)

    def test8(self) -> None:
        """
        Test two adjacent comment blocks, with no content between them, are
        both removed.
        """
        # Prepare inputs.
        txt = """
        before
        /*
        hidden1
        */
        /*
        hidden2
        */
        after
        """
        # Prepare outputs.
        expected = """
        before
        after
        """
        # Run test and check outputs.
        self.helper(txt, expected)


# #############################################################################
# Test__extract_titled_content
# #############################################################################


class Test__extract_titled_content(hunitest.TestCase):
    """
    Test `_extract_titled_content()` function.
    """

    def helper(
        self, txt: str, title: str, expected: Optional[List[str]]
    ) -> None:
        """
        Helper for testing `_extract_titled_content()`.

        :param txt: source file content
        :param title: title to extract
        :param expected: expected extracted body, or `None` if `title` is
            not found
        """
        # Prepare inputs.
        txt = hprint.dedent(txt)
        lines = txt.split("\n")
        # Run test.
        actual = csicisksl._extract_titled_content(lines, title)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test1(self) -> None:
        """
        Test a `* ` slide-level match stops at the next slide.
        """
        # Prepare inputs.
        txt = """
        * Slide A
        Content A line 1
        Content A line 2
        * Slide B
        Content B
        """
        title = "Slide A"
        # Prepare outputs.
        expected = ["Content A line 1", "Content A line 2"]
        # Run test and check outputs.
        self.helper(txt, title, expected)

    def test2(self) -> None:
        """
        Test a header-level match includes nested headers/slides and stops
        at the next same-level header.
        """
        # Prepare inputs.
        txt = """
        ## Section A
        ### Sub A
        * Slide A1
        Content A1
        * Slide A2
        Content A2
        ## Section B
        Content B
        """
        title = "Section A"
        # Prepare outputs.
        expected = [
            "### Sub A",
            "* Slide A1",
            "Content A1",
            "* Slide A2",
            "Content A2",
        ]
        # Run test and check outputs.
        self.helper(txt, title, expected)

    def test3(self) -> None:
        """
        Test a title that doesn't exist returns `None`.
        """
        # Prepare inputs.
        txt = """
        * Slide A
        Content A
        """
        title = "Missing Title"
        # Prepare outputs.
        expected = None
        # Run test and check outputs.
        self.helper(txt, title, expected)

    def test4(self) -> None:
        """
        Test a duplicate title uses the first occurrence.
        """
        # Prepare inputs.
        txt = """
        * Dup
        First
        * Dup
        Second
        """
        title = "Dup"
        # Prepare outputs.
        expected = ["First"]
        # Run test and check outputs.
        self.helper(txt, title, expected)

    def test5(self) -> None:
        """
        Test leading/trailing blank lines are stripped from the extracted
        body.
        """
        # Prepare inputs.
        txt = """
        ## Section A

        Content line

        ## Section B
        """
        title = "Section A"
        # Prepare outputs.
        expected = ["Content line"]
        # Run test and check outputs.
        self.helper(txt, title, expected)


# #############################################################################
# Test__parse_from_tag
# #############################################################################


class Test__parse_from_tag(hunitest.TestCase):
    """
    Test `_parse_from_tag()` function.
    """

    def helper(self, line: str, expected: Optional[Tuple[str, str]]) -> None:
        """
        Helper for testing `_parse_from_tag()`.

        :param line: candidate tag line
        :param expected: expected `(file, slide_title)` tuple, or `None`
        """
        # Run test.
        actual = csicisksl._parse_from_tag(line)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test1(self) -> None:
        """
        Test a well-formed `// From` tag.
        """
        # Prepare inputs.
        line = "// From msml610/lectures_source/Lesson06.1.smd:Some Title"
        # Prepare outputs.
        expected = (
            "msml610/lectures_source/Lesson06.1.smd",
            "Some Title",
        )
        # Run test and check outputs.
        self.helper(line, expected)

    def test2(self) -> None:
        """
        Test a line that is not a `// From` tag.
        """
        # Prepare inputs.
        line = "// Just a regular comment"
        # Prepare outputs.
        expected = None
        # Run test and check outputs.
        self.helper(line, expected)

    def test3(self) -> None:
        """
        Test a slide title that itself contains a colon: only the first
        colon separates the file from the title.
        """
        # Prepare inputs.
        line = "// From a/b.smd:Chemical Shift: Use Student's t-dist (1/3)"
        # Prepare outputs.
        expected = ("a/b.smd", "Chemical Shift: Use Student's t-dist (1/3)")
        # Run test and check outputs.
        self.helper(line, expected)


# #############################################################################
# Test__find_next_title_line
# #############################################################################


class Test__find_next_title_line(hunitest.TestCase):
    """
    Test `_find_next_title_line()` function.
    """

    def helper(self, txt: str, start_idx: int, expected: int) -> None:
        """
        Helper for testing `_find_next_title_line()`.

        :param txt: source file content
        :param start_idx: index to start searching from
        :param expected: expected index of the next title line
        """
        # Prepare inputs.
        txt = hprint.dedent(txt)
        lines = txt.split("\n")
        # Run test.
        actual = csicisksl._find_next_title_line(lines, start_idx)
        # Check outputs.
        self.assertEqual(actual, expected)

    def test1(self) -> None:
        """
        Test the index of the next title line is found.
        """
        # Prepare inputs.
        txt = """
        not a title

        ## Next Title
        more content
        """
        start_idx = 1
        # Prepare outputs.
        expected = 2
        # Run test and check outputs.
        self.helper(txt, start_idx, expected)

    def test2(self) -> None:
        """
        Test that `len(lines)` is returned when there's no more title.
        """
        # Prepare inputs.
        txt = """
        not a title
        still not a title
        """
        start_idx = 0
        # Prepare outputs.
        expected = 2
        # Run test and check outputs.
        self.helper(txt, start_idx, expected)
