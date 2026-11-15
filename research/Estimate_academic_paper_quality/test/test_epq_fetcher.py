import logging
import os
from typing import List
from unittest import mock

import helpers.hio as hio
import helpers.hprint as hprint
import helpers.hunit_test as hunitest

import research.Estimate_academic_paper_quality.epq_fetcher as epqfetch
import research.Estimate_academic_paper_quality.test.epq_test_utils as epqteuti

_LOG = logging.getLogger(__name__)

# #############################################################################
# Test_pdf_readers
# #############################################################################


class Test_pdf_readers(hunitest.TestCase):
    """
    Test `read_pdf_title()`, `read_pdf_pages()`, and `count_pages()`.
    """

    def test1(self) -> None:
        """
        Test the title is the text with the largest font of page 1.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "a.pdf")
        epqteuti.create_pdf(
            path, ["Body text of the paper."], title_text="Lazy Prices"
        )
        # Run test.
        actual = epqfetch.read_pdf_title(path)
        # Check outputs.
        self.assert_equal(actual, "Lazy Prices")

    def test2(self) -> None:
        """
        Test the metadata title is used when page 1 has no text.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "a.pdf")
        epqteuti.create_pdf(path, [""], meta_title="Title in Metadata")
        # Run test.
        actual = epqfetch.read_pdf_title(path)
        # Check outputs.
        self.assert_equal(actual, "Title in Metadata")

    def test3(self) -> None:
        """
        Test an empty page with no metadata gives an empty title.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "a.pdf")
        epqteuti.create_pdf(path, [""])
        # Run test.
        actual = epqfetch.read_pdf_title(path)
        # Check outputs.
        self.assert_equal(actual, "")

    def test4(self) -> None:
        """
        Test the page count and the text of each page.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "a.pdf")
        epqteuti.create_pdf(path, ["first page", "second page", "third page"])
        # Run test.
        n_pages = epqfetch.count_pages(path)
        pages = epqfetch.read_pdf_pages(path)
        # Check outputs.
        self.assertEqual(n_pages, 3)
        self.assert_equal(
            str([p.strip() for p in pages]),
            str(["first page", "second page", "third page"]),
        )


# #############################################################################
# Test_is_fetched
# #############################################################################


class Test_is_fetched(hunitest.TestCase):
    """
    Test `epq_fetcher.is_fetched()`.
    """

    def helper(self, pdf_content: str, md_content: str, expected: bool) -> None:
        """
        Test helper for `is_fetched()`.

        :param pdf_content: content of the PDF, "" for no file
        :param md_content: content of the markdown, "" for no file
        :param expected: expected result
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        pdf_dir = os.path.join(scratch, "paper_pdf")
        md_dir = os.path.join(scratch, "paper_md")
        hio.create_dir(pdf_dir, incremental=True)
        hio.create_dir(md_dir, incremental=True)
        if pdf_content:
            hio.to_file(os.path.join(pdf_dir, "b.pdf"), pdf_content)
        if md_content:
            hio.to_file(os.path.join(md_dir, "b.md"), md_content)
        # Run test.
        actual = epqfetch.is_fetched(pdf_dir, md_dir, "b")
        # Check outputs.
        self.assertEqual(actual, expected)

    def test1(self) -> None:
        """
        Test a valid PDF and a markdown with text are fetched.
        """
        self.helper("%PDF-1.4", "text " * 200, True)

    def test2(self) -> None:
        """
        Test a missing markdown is not fetched.
        """
        self.helper("%PDF-1.4", "", False)

    def test3(self) -> None:
        """
        Test an HTML file named `.pdf` is not fetched.
        """
        self.helper("<html>", "text " * 200, False)

    def test4(self) -> None:
        """
        Test a markdown with almost no text is not fetched.
        """
        self.helper("%PDF-1.4", "short", False)


# #############################################################################
# Test__read_last_line
# #############################################################################


class Test__read_last_line(hunitest.TestCase):
    """
    Test `epq_fetcher._read_last_line()`.
    """

    def helper(self, log_text: str, expected: str) -> None:
        """
        Test helper for `_read_last_line()`.

        :param log_text: content of the log, "" for no file
        :param expected: expected reason
        """
        # Prepare inputs.
        log_path = os.path.join(self.get_scratch_space(), "tool.log")
        if log_text:
            hio.to_file(log_path, log_text)
        # Run test.
        actual = epqfetch._read_last_line(log_path)
        # Check outputs.
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test the notes of `hsystem` after a failed command are skipped, and the
        last exception line of the traceback is the reason.
        """
        log_text = """
        Traceback (most recent call last):
          File "download_with_chrome.py", line 337, in _download
        playwright._impl._errors.Error: BrowserType.connect_over_cdp: Protocol error
        Call log:
          - <ws preparing> retrieving websocket url from http://127.0.0.1:9222
          - <ws disconnected> ws://127.0.0.1:9222/devtools/browser/x code=1000

        '
        - Output saved in 'tmp.system_output.txt'
        - Command saved in 'tmp.system_cmd.sh'
        """
        expected = (
            "playwright._impl._errors.Error: BrowserType.connect_over_cdp: "
            "Protocol error"
        )
        self.helper(hprint.dedent(log_text), expected)

    def test2(self) -> None:
        """
        Test a log with no exception line gives its last line.
        """
        log_text = "Start\nDownloading\nHTTP 404 for the input\n\n"
        self.helper(log_text, "HTTP 404 for the input")

    def test3(self) -> None:
        """
        Test a missing log.
        """
        self.helper("", "no output")

    def test4(self) -> None:
        """
        Test a log with only the notes of `hsystem`.
        """
        log_text = "'\n- Output saved in 'o.txt'\n- Command saved in 'c.sh'\n"
        self.helper(log_text, "no output")


# #############################################################################
# Test_fetch_paper
# #############################################################################


class Test_fetch_paper(hunitest.TestCase):
    """
    Test `epq_fetcher.fetch_paper()` with a stub of the download tool.
    """

    def helper(
        self,
        download_inputs: List[str],
        expected: str,
        *,
        pdf_content: str = "",
    ) -> str:
        """
        Test helper for `fetch_paper()`.

        :param download_inputs: ordered download inputs of the paper
        :param expected: expected `status|reason`
        :param pdf_content: content of a PDF that is already in the dir
        :return: scratch dir, with the dirs `paper_pdf` and `paper_md`
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        tool_path = epqteuti.make_stub_tool(scratch)
        calls_path = os.path.join(scratch, "calls.txt")
        pdf_dir = os.path.join(scratch, "paper_pdf")
        md_dir = os.path.join(scratch, "paper_md")
        hio.create_dir(pdf_dir, incremental=True)
        if pdf_content:
            hio.to_file(os.path.join(pdf_dir, "b.pdf"), pdf_content)
        paper = epqteuti.make_paper(
            base_name="b", download_inputs=tuple(download_inputs)
        )
        # Run test.
        with mock.patch.dict(os.environ, {"EPQ_STUB_CALLS": calls_path}):
            result = epqfetch.fetch_paper(
                paper,
                pdf_dir,
                md_dir,
                log_dir=os.path.join(scratch, "logs"),
                tool_path=tool_path,
            )
        # Check outputs.
        actual = f"{result.status}|{result.reason}"
        self.assert_equal(actual, expected)
        return scratch

    def test1(self) -> None:
        """
        Test the first input that works gives the PDF and the markdown.
        """
        # Run test.
        scratch = self.helper(["https://x/good.pdf"], "ok|")
        # Check outputs.
        pdf_dir = os.path.join(scratch, "paper_pdf")
        md_dir = os.path.join(scratch, "paper_md")
        self.assertTrue(epqfetch.is_fetched(pdf_dir, md_dir, "b"))
        # The markdown is not left next to the PDF.
        self.assertEqual(os.listdir(pdf_dir), ["b.pdf"])
        self.assertEqual(os.listdir(md_dir), ["b.md"])

    def test2(self) -> None:
        """
        Test an HTML page is rejected and the next input is tried.
        """
        # Run test.
        scratch = self.helper(["https://x/html", "https://x/good"], "ok|")
        # Check outputs.
        pdf_dir = os.path.join(scratch, "paper_pdf")
        md_dir = os.path.join(scratch, "paper_md")
        self.assertTrue(epqfetch.is_fetched(pdf_dir, md_dir, "b"))

    def test3(self) -> None:
        """
        Test all the inputs failing gives the reason of each input.
        """
        # Prepare outputs.
        expected = (
            "download_failed|https://x/fail1: Error: HTTP 404 for the input; "
            "https://x/html: not a PDF (HTML page?)"
        )
        # Run test.
        self.helper(["https://x/fail1", "https://x/html"], expected)

    def test4(self) -> None:
        """
        Test a paper with no download input.
        """
        self.helper([], "no_download_link|no public PDF found")

    def test5(self) -> None:
        """
        Test a markdown with no text is a conversion failure.
        """
        expected = "conversion_failed|markdown has no text (scanned PDF?)"
        self.helper(["https://x/scanned"], expected)

    def test6(self) -> None:
        """
        Test a crash of the conversion keeps the last line of the log.
        """
        expected = "conversion_failed|Error: conversion crashed"
        self.helper(["https://x/convfail"], expected)

    def test7(self) -> None:
        """
        Test a valid PDF with no markdown only runs the conversion.
        """
        # Run test.
        scratch = self.helper(
            ["https://x/good"], "ok|", pdf_content="%PDF-1.4 old"
        )
        # Check outputs.
        calls = hio.from_file(os.path.join(scratch, "calls.txt"))
        self.assertIn("clear=True", calls)
        self.assertNotIn("https://x/good", calls)
        # The markdown is in its own dir.
        self.assertTrue(
            os.path.isfile(os.path.join(scratch, "paper_md", "b.md"))
        )
        self.assertFalse(
            os.path.isfile(os.path.join(scratch, "paper_pdf", "b.md"))
        )

    def test8(self) -> None:
        """
        Test a failed conversion of an existing PDF.
        """
        expected = "conversion_failed|Error: conversion crashed"
        self.helper(
            ["https://x/good"], expected, pdf_content="%PDF-1.4 NOCONVERT"
        )
