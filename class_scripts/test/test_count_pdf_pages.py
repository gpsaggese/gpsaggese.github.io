"""
Unit tests for count_pdf_pages.py.

Import as:

import class_scripts.test.test_count_pdf_pages as csttcppa
"""

# TODO(gp): Make sure this file follows our unit test conventions

from unittest import mock

import helpers.hunit_test as hunitest

import class_scripts.count_pdf_pages as cscopdpa


# #############################################################################
# Test_parse
# #############################################################################


class Test_parse(hunitest.TestCase):
    """
    Test `_parse()` function.
    """

    def test1(self) -> None:
        """
        Test parser accepts the `--dir` argument.
        """
        # Prepare inputs.
        arg_list = ["--dir", "msml610/lectures"]
        # Prepare outputs.
        expected = "msml610/lectures"
        # Run test.
        parser = cscopdpa._parse()
        args = parser.parse_args(arg_list)
        # Check outputs.
        self.assert_equal(args.dir, expected)

    def test2(self) -> None:
        """
        Test parser accepts the `--input` argument.
        """
        # Prepare inputs.
        arg_list = ["--input", "msml610/lectures/Lesson01.pdf"]
        # Prepare outputs.
        expected = "msml610/lectures/Lesson01.pdf"
        # Run test.
        parser = cscopdpa._parse()
        args = parser.parse_args(arg_list)
        # Check outputs.
        self.assert_equal(args.input, expected)

    def test3(self) -> None:
        """
        Test parser rejects passing both `--dir` and `--input`.
        """
        # Prepare inputs.
        arg_list = [
            "--dir",
            "msml610/lectures",
            "--input",
            "msml610/lectures/Lesson01.pdf",
        ]
        # Run test.
        parser = cscopdpa._parse()
        with self.assertRaises(SystemExit):
            parser.parse_args(arg_list)

    def test4(self) -> None:
        """
        Test parser rejects passing neither `--dir` nor `--input`.
        """
        # Prepare inputs.
        arg_list = []
        # Run test.
        parser = cscopdpa._parse()
        with self.assertRaises(SystemExit):
            parser.parse_args(arg_list)


# #############################################################################
# Test_main
# #############################################################################


class Test_main(hunitest.TestCase):
    """
    Test `_main()` function.
    """

    def test1(self) -> None:
        """
        Test happy path with `--dir`: `get_pdf_page_counts()` is called with
        the directory and pattern, and results are logged.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        arg_list = ["--dir", scratch_dir]
        page_counts = {"Lesson01.pdf": 10, "Lesson02.pdf": 20}
        # Prepare outputs.
        expected_log = "Lesson01.pdf\t10\nLesson02.pdf\t20"
        # Run test.
        with mock.patch(
            "class_scripts.common_utils.get_pdf_page_counts",
            return_value=page_counts,
        ) as mock_get_counts:
            with self.assertLogs(
                "class_scripts.count_pdf_pages", level="INFO"
            ) as cm:
                with mock.patch("sys.argv", ["count_pdf_pages.py"] + arg_list):
                    cscopdpa._main(cscopdpa._parse())
        # Check outputs.
        mock_get_counts.assert_called_once_with(
            scratch_dir, pattern="Lesson*.pdf"
        )
        # Log records are prefixed with the level and logger name, so only
        # the last two records (one per page count) are checked.
        actual_log = "\n".join(
            record.split(":", 2)[-1] for record in cm.output[-2:]
        )
        self.assert_equal(actual_log, expected_log)

    def test2(self) -> None:
        """
        Test happy path with `--input`: `count_pdf_pages()` is called with
        the file and the result is logged.
        """
        # Prepare inputs.
        pdf_path = "data605/lectures/Lesson01.pdf"
        arg_list = ["--input", pdf_path]
        # Prepare outputs.
        expected_log = f"{pdf_path}\t7"
        # Run test.
        with mock.patch(
            "class_scripts.common_utils.count_pdf_pages",
            return_value=7,
        ) as mock_count_pages:
            with self.assertLogs(
                "class_scripts.count_pdf_pages", level="INFO"
            ) as cm:
                with mock.patch("sys.argv", ["count_pdf_pages.py"] + arg_list):
                    cscopdpa._main(cscopdpa._parse())
        # Check outputs.
        mock_count_pages.assert_called_once_with(pdf_path)
        # Log records are prefixed with the level and logger name, so only
        # the last record (the single page count) is checked.
        actual_log = cm.output[-1].split(":", 2)[-1]
        self.assert_equal(actual_log, expected_log)
