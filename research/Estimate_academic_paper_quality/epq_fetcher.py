"""
Own the files of each paper in `<output_dir>/papers/`.

This module:
- Creates `<BASE>.pdf` and `<BASE>.md` with the existing download tool
  `download_academic_paper_to_md.py`
- Records why a download or a conversion fails
- Reads the PDF for the other components: page texts, page count, and title of
  the first page

The design is in `docs/02.4-design.fetcher.md`.

Import as:

import research.Estimate_academic_paper_quality.epq_fetcher as epqfetch
"""

import logging
import os
import re
import shlex
from typing import List, Tuple

import helpers.hdbg as hdbg
import helpers.hgit as hgit
import helpers.hio as hio
import helpers.hprint as hprint
import helpers.hsystem as hsystem
import pymupdf

import research.Estimate_academic_paper_quality.epq_types as epqtypes

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

# A markdown file shorter than this has no text, e.g., a scanned PDF.
_MIN_MARKDOWN_CHARS = 500

# Name of the download tool, found in the Git tree.
_TOOL_NAME = "download_academic_paper_to_md.py"

# Lines that `hsystem` prints after a failed command, with no error in them.
_HSYSTEM_NOTE_PREFIXES = ("- Output saved in", "- Command saved in")

# Last line of a Python traceback, e.g., `ValueError: bad` or `Error: bad`.
_EXCEPTION_RE = re.compile(r"^[\w.]*(Error|Exception)\b")


# #############################################################################
# Paths and checks
# #############################################################################


def get_tool_path() -> str:
    """
    Find the download tool in the Git tree.

    :return: absolute path of `download_academic_paper_to_md.py`
    """
    return hgit.find_file_in_git_tree(_TOOL_NAME)


def _get_paths(papers_dir: str, base_name: str) -> Tuple[str, str]:
    """
    Build the paths of the PDF and of the markdown of a paper.

    :param papers_dir: dir of the files of the papers
    :param base_name: base name of the paper
    :return: paths of the PDF and of the markdown
    """
    pdf_path = os.path.join(papers_dir, f"{base_name}.pdf")
    md_path = os.path.join(papers_dir, f"{base_name}.md")
    return pdf_path, md_path


def _check_pdf(pdf_path: str) -> bool:
    """
    Check that a file exists and starts with the PDF header.

    The download tool saves the body of any answer `200` as the PDF, so a
    landing page becomes an HTML file named `.pdf`.

    :param pdf_path: path of the file
    :return: True if the file starts with `%PDF`
    """
    is_pdf = False
    if os.path.isfile(pdf_path):
        with open(pdf_path, "rb") as f:
            is_pdf = f.read(4) == b"%PDF"
    return is_pdf


def _check_markdown(md_path: str) -> bool:
    """
    Check that a markdown file exists and has text.

    :param md_path: path of the file
    :return: True if the file has at least `_MIN_MARKDOWN_CHARS` chars
    """
    has_text = False
    if os.path.isfile(md_path):
        text = hio.from_file(md_path)
        has_text = len(text.strip()) >= _MIN_MARKDOWN_CHARS
    return has_text


def is_fetched(papers_dir: str, base_name: str) -> bool:
    """
    Check if a paper has a valid PDF and a markdown file.

    :param papers_dir: dir of the files of the papers
    :param base_name: base name of the paper
    :return: True if the download and the conversion can be skipped
    """
    pdf_path, md_path = _get_paths(papers_dir, base_name)
    return _check_pdf(pdf_path) and _check_markdown(md_path)


# #############################################################################
# Download and conversion
# #############################################################################


def _read_last_line(log_path: str) -> str:
    """
    Read the line of a log file that tells why the tool failed.

    The tool runs other commands with `hsystem`. When one of them fails,
    `hsystem` prints lines that say where it saved the command and the output.
    These lines come after the real error and are skipped. If the log has an
    exception line, e.g., `playwright._impl._errors.Error: ...`, the last one is
    the reason. Otherwise the reason is the last line that is not skipped.

    :param log_path: path of the log
    :return: the line, or `no output` if the log is missing or has no line to
        show
    """
    last_line = "no output"
    if os.path.isfile(log_path):
        lines = [line.strip() for line in hio.from_file(log_path).split("\n")]
        # Skip the empty lines, the lines with only quotes, and the notes of
        # `hsystem` on the files that it saved.
        lines = [
            line
            for line in lines
            if line.strip("'\"") and not line.startswith(_HSYSTEM_NOTE_PREFIXES)
        ]
        exception_lines = [line for line in lines if _EXCEPTION_RE.match(line)]
        if exception_lines:
            last_line = exception_lines[-1]
        elif lines:
            last_line = lines[-1]
    return last_line


def _run_tool(
    tool_path: str,
    tool_input: str,
    base_path: str,
    log_path: str,
    *,
    only_convert: bool,
    log_level: str,
) -> int:
    """
    Run the download tool on one input.

    :param tool_path: path of `download_academic_paper_to_md.py`
    :param tool_input: local PDF, URL, or SSRN URL to give to `--input`
    :param base_path: path of the files without extension, for `--output`
    :param log_path: file that gets the output of the tool
    :param only_convert: if True, run only the `convert` action
    :param log_level: logging level to forward to the tool
    :return: exit code of the tool
    """
    cmd = [
        tool_path,
        f"--input {shlex.quote(tool_input)}",
        f"--output {shlex.quote(base_path)}",
    ]
    if only_convert:
        cmd.append("--clear_actions --action convert")
    else:
        # The `summarize` action calls an LLM and the pipeline does not need it.
        cmd.append("--skip_action summarize")
    cmd.append("--no_incremental")
    cmd.append(f"-v {log_level}")
    # Send both stdout and stderr to the log: `output_file` of `hsystem` keeps
    # only stdout, and the error of the tool is on stderr.
    cmd.append(f"> {shlex.quote(log_path)} 2>&1")
    cmd_str = " ".join(cmd)
    hio.create_dir(os.path.dirname(log_path), incremental=True)
    rc = hsystem.system(cmd_str, abort_on_error=False, suppress_output=True)
    return rc


def _convert_existing_pdf(
    paper: epqtypes.Paper,
    papers_dir: str,
    log_path: str,
    *,
    tool_path: str,
    log_level: str,
) -> epqtypes.FetchResult:
    """
    Convert a valid PDF that has no markdown.

    :param paper: paper with a PDF in `papers_dir`
    :param papers_dir: dir of the files of the papers
    :param log_path: file that gets the output of the tool
    :param tool_path: path of the download tool
    :param log_level: logging level to forward to the tool
    :return: result of the conversion
    """
    pdf_path, md_path = _get_paths(papers_dir, paper.base_name)
    base_path = os.path.splitext(pdf_path)[0]
    rc = _run_tool(
        tool_path,
        pdf_path,
        base_path,
        log_path,
        only_convert=True,
        log_level=log_level,
    )
    if rc == 0 and _check_markdown(md_path):
        result = epqtypes.FetchResult(status="ok", reason="")
    else:
        reason = _read_last_line(log_path)
        result = epqtypes.FetchResult(status="conversion_failed", reason=reason)
    return result


def fetch_paper(
    paper: epqtypes.Paper,
    papers_dir: str,
    *,
    log_dir: str,
    tool_path: str = "",
    log_level: str = "INFO",
) -> epqtypes.FetchResult:
    """
    Download and convert one paper, trying each download input in order.

    The first input that gives a valid PDF and a markdown file wins. A failure
    is returned as a result with its reason and never raised, so that the run
    goes on with the other papers.

    :param paper: paper with its ordered download inputs
    :param papers_dir: dir of the files of the papers
    :param log_dir: dir of the logs of the tool
    :param tool_path: path of the download tool
        - Default: find `download_academic_paper_to_md.py` in the Git tree
    :param log_level: logging level to forward to the tool
    :return: status `ok`, `no_download_link`, `download_failed`, or
        `conversion_failed`, with the reason
    """
    _LOG.debug(hprint.to_str("paper.base_name papers_dir log_dir"))
    hdbg.dassert_ne(paper.base_name, "", "A paper needs a base name")
    hio.create_dir(papers_dir, incremental=True)
    tool_path = tool_path or get_tool_path()
    pdf_path, md_path = _get_paths(papers_dir, paper.base_name)
    base_path = os.path.splitext(pdf_path)[0]
    log_path = os.path.join(log_dir, f"{paper.base_name}.download.log")
    if _check_pdf(pdf_path):
        # A valid PDF is there, e.g., from an earlier run that failed to
        # convert: only the conversion is missing.
        result = _convert_existing_pdf(
            paper, papers_dir, log_path, tool_path=tool_path, log_level=log_level
        )
        return result
    if not paper.download_inputs:
        result = epqtypes.FetchResult(
            status="no_download_link", reason="no public PDF found"
        )
        return result
    reasons: List[str] = []
    for tool_input in paper.download_inputs:
        # Remove a leftover that is not a PDF, so that it does not look like a
        # result of this input.
        if os.path.isfile(pdf_path):
            os.remove(pdf_path)
        rc = _run_tool(
            tool_path,
            tool_input,
            base_path,
            log_path,
            only_convert=False,
            log_level=log_level,
        )
        if _check_pdf(pdf_path):
            # The download worked: the markdown tells if the conversion did.
            if rc == 0 and _check_markdown(md_path):
                result = epqtypes.FetchResult(status="ok", reason="")
            else:
                reason = _read_last_line(log_path)
                if rc == 0:
                    reason = "markdown has no text (scanned PDF?)"
                result = epqtypes.FetchResult(
                    status="conversion_failed", reason=reason
                )
            return result
        # The download failed for this input: keep the reason, try the next.
        if os.path.isfile(pdf_path):
            # The tool saved an answer that is not a PDF, e.g., a landing page.
            reason = "not a PDF (HTML page?)"
            os.remove(pdf_path)
        else:
            reason = _read_last_line(log_path)
        _LOG.warning("Input '%s' failed: %s", tool_input, reason)
        reasons.append(f"{tool_input}: {reason}")
    result = epqtypes.FetchResult(
        status="download_failed", reason="; ".join(reasons)
    )
    return result


# #############################################################################
# Readers of the files of a paper
# #############################################################################


def read_markdown(md_path: str) -> str:
    """
    Read the markdown of a paper.

    :param md_path: path of the markdown file
    :return: text of the file
    """
    hdbg.dassert_file_exists(md_path)
    return hio.from_file(md_path)


def read_pdf_pages(pdf_path: str) -> List[str]:
    """
    Read the text of each page of a PDF.

    :param pdf_path: path of the PDF
    :return: text of each page, in order
    """
    hdbg.dassert_file_exists(pdf_path)
    with pymupdf.open(pdf_path) as doc:
        pages = [page.get_text("text") for page in doc]
    return pages


def count_pages(pdf_path: str) -> int:
    """
    Count all the pages of a PDF, including references and appendices.

    :param pdf_path: path of the PDF
    :return: number of pages
    """
    hdbg.dassert_file_exists(pdf_path)
    with pymupdf.open(pdf_path) as doc:
        n_pages = len(doc)
    return n_pages


def read_pdf_title(pdf_path: str) -> str:
    """
    Read the title of a PDF from its first page.

    The title is the text with the largest font size of the first page. A PDF
    with a cover page can give a wrong title. If the page has no text, the
    title in the metadata of the PDF is used.

    :param pdf_path: path of the PDF
    :return: title, or an empty string if the PDF has none
    """
    hdbg.dassert_file_exists(pdf_path)
    title = ""
    with pymupdf.open(pdf_path) as doc:
        if len(doc) > 0:
            # Collect the text spans of the first page with their font size.
            spans = []
            blocks = doc[0].get_text("dict", sort=True)["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            spans.append((span["size"], text))
            if spans:
                # Join the spans whose size is the largest, up to a rounding.
                max_size = max(size for size, _ in spans)
                title = " ".join(
                    text for size, text in spans if size >= max_size - 0.5
                )
        if not title:
            title = (doc.metadata or {}).get("title", "") or ""
    title = re.sub(r"\s+", " ", title).strip()
    return title
