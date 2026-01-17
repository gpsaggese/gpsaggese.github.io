"""
Shared utility functions for class scripts.

Import as:

import classes2.common_utils as clcomuut
"""

import glob
import logging
from pathlib import Path
from typing import Dict, Optional

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################


# Valid course directories.
VALID_DIRS = ["data605", "msml610"]


# #############################################################################
# Helper functions
# #############################################################################


def validate_dir_lesson_args(dir_arg: str, lesson_arg: str) -> None:
    """
    Validate DIR and LESSON arguments.

    :param dir_arg: course directory
    :param lesson_arg: lesson number
    """
    # Validate DIR is not empty.
    hdbg.dassert_ne(dir_arg, "", "DIR argument cannot be empty")
    # Validate LESSON is not empty.
    hdbg.dassert_ne(lesson_arg, "", "LESSON argument cannot be empty")
    # Log the validated arguments.
    _LOG.debug("Validated DIR='%s', LESSON='%s'", dir_arg, lesson_arg)


def find_lecture_file(dir_path: str, lesson: str) -> Path:
    """
    Find the lecture file matching the lesson pattern.

    Searches for exactly one file matching {dir_path}/lectures_source/Lesson{lesson}*.

    :param dir_path: course directory
    :param lesson: lesson number
    :return: path to the found lecture file
    """
    # Build the search pattern.
    pattern = f"{dir_path}/lectures_source/Lesson{lesson}*"
    _LOG.debug("Searching for files matching pattern='%s'", pattern)
    # Find matching files.
    files = glob.glob(pattern)
    # Validate exactly one file found.
    hdbg.dassert_eq(
        len(files),
        1,
        "Expected exactly one file, found %d files: %s",
        len(files),
        files,
    )
    file_path = Path(files[0])
    _LOG.info("Found lecture file: %s", file_path)
    return file_path


def get_source_name(dir_path: str, lesson: str) -> str:
    """
    Get the source file name for a lesson.

    :param dir_path: course directory
    :param lesson: lesson number
    :return: source file name without directory path
    """
    file_path = find_lecture_file(dir_path, lesson)
    source_name = file_path.name
    _LOG.debug("Source name='%s'", source_name)
    return source_name


def get_output_name(source_name: str, extension: str) -> str:
    """
    Generate output file name by replacing the extension.

    :param source_name: source file name
    :param extension: new extension
    :return: output file name
    """
    # Remove .txt extension and add new extension.
    output_name = source_name.replace(".txt", extension)
    _LOG.debug(
        "Generated output name='%s' from source='%s'", output_name, source_name
    )
    return output_name


def ensure_dir_exists(dir_path: str, *, from_scratch: bool = False) -> None:
    """
    Ensure a directory exists, optionally creating it from scratch.

    :param dir_path: directory path to create
    :param from_scratch: if True, remove existing directory first
    """
    if from_scratch and Path(dir_path).exists():
        _LOG.debug("Removing existing directory: %s", dir_path)
        hio.delete_dir(dir_path)
    # Create directory if it doesn't exist.
    hio.create_dir(dir_path, incremental=True)
    _LOG.debug("Ensured directory exists: %s", dir_path)


def count_pdf_pages(pdf_path: str) -> int:
    """
    Count the number of pages in a PDF file using mdls.

    This function uses the macOS-specific mdls command to extract PDF metadata.

    :param pdf_path: path to the PDF file
    :return: number of pages in the PDF
    """
    hdbg.dassert(Path(pdf_path).exists(), "PDF file does not exist:", pdf_path)
    # Use mdls to get page count.
    cmd = f"mdls -name kMDItemNumberOfPages '{pdf_path}'"
    _LOG.debug("Running command: %s", cmd)
    output = hsystem.system_to_string(cmd)
    # Parse output like "kMDItemNumberOfPages = 42".
    parts = output.strip().split("=")
    hdbg.dassert_eq(len(parts), 2, "Unexpected mdls output format:", output)
    page_count_str = parts[1].strip()
    page_count = int(page_count_str)
    _LOG.debug("PDF '%s' has %d pages", pdf_path, page_count)
    return page_count


def get_pdf_page_counts(directory: str, pattern: str = "Lesson*.pdf") -> Dict[str, int]:
    """
    Get page counts for all PDF files matching a pattern in a directory.

    :param directory: directory to search
    :param pattern: glob pattern for PDF files
    :return: dictionary mapping file names to page counts
    """
    hdbg.dassert(
        Path(directory).exists(), "Directory does not exist:", directory
    )
    # Find all matching PDF files.
    dir_path = Path(directory)
    pdf_files = sorted(dir_path.glob(pattern))
    _LOG.info("Found %d PDF files in %s", len(pdf_files), directory)
    # Count pages for each PDF.
    page_counts = {}
    for pdf_file in pdf_files:
        page_count = count_pdf_pages(str(pdf_file))
        page_counts[pdf_file.name] = page_count
    return page_counts
