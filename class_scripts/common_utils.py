"""
Shared utility functions for class scripts.

Import as:

import class_scripts.common_utils as csccouti
"""

import glob
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dev_scripts_helpers.documentation.preprocess_notes as dshdprno
import helpers.hcache_simple as hcacsimp
import helpers.hdbg as hdbg
import helpers.hgit as hgit
import helpers.hio as hio
import helpers.hprint as hprint
import helpers.hretry as hretry
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)


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


def extract_lesson_from_file(file_path_str: str) -> Tuple[str, str]:
    """
    Extract lesson number and course directory from a lecture source path.

    Parses filenames like "Lesson10.2-Causal_Discovery.smd" to extract
    "10.2". Also extracts the course directory (e.g., "data605", "msml610")
    from the path.

    :param file_path_str: file path like
        "msml610/lectures_source/Lesson10.2-Name.smd"
    :return: tuple of (dir, lesson), e.g. ("msml610", "10.2")
    """
    filename = os.path.basename(file_path_str)
    match = re.match(r"Lesson(\d+(?:\.\d+)?)", filename)
    hdbg.dassert_is_not(
        match,
        None,
        "Could not extract lesson number from filename: %s",
        filename,
    )
    lesson = match.group(1)  # type: ignore[union-attr]
    # The course dir is everything before `.../lectures_source/...`, so this
    # works for both course-relative paths (e.g., "msml610/lectures_source/
    # ...") and absolute paths (e.g., test scratch dirs).
    if "lectures_source" in file_path_str:
        dir_name = file_path_str.split("lectures_source")[0].rstrip(os.sep)
    else:
        dir_name = file_path_str.split(os.sep)[0]
    hdbg.dassert_dir_exists(dir_name)
    _LOG.debug(
        "Extracted lesson='%s', dir='%s' from path='%s'",
        lesson,
        dir_name,
        file_path_str,
    )
    return dir_name, lesson


def parse_lesson_spec(arg: str) -> Tuple[str, str]:
    """
    Parse a lecture specification into (dir, lesson).

    Handles:
    - "data605/08.1" or "msml610/08.1" -> ("data605", "08.1")
    - "data605/lectures_source/Lesson10.2-Name.smd" -> extracted via
      `extract_lesson_from_file()`

    This is the standard input format for all `class_scripts` CLIs that
    operate on a single lesson.

    :param arg: lecture specification, e.g. "data605/08.1" or a file path
    :return: tuple of (directory, lesson)
    """
    if "lectures_source" in arg or arg.endswith(".smd"):
        return extract_lesson_from_file(arg)
    hdbg.dassert_in(
        "/",
        arg,
        f"Invalid input '{arg}'. Use 'data605/08.1' or "
        "'data605/lectures_source/Lesson08.1-Name.smd'",
    )
    parts = arg.split("/")
    hdbg.dassert_eq(
        len(parts),
        2,
        f"Expected dir/lesson format, got '{arg}'. Use 'data605/08.1'",
    )
    dir_input, lesson = parts
    hdbg.dassert_dir_exists(dir_input)
    return dir_input, lesson


def find_lecture_file(dir_path: str, lesson: str) -> Path:
    """
    Find the lecture file matching the lesson pattern.

    Searches for exactly one file matching {dir_path}/lectures_source/Lesson{lesson}*.

    :param dir_path: course directory
    :param lesson: lesson number
    :return: path to the found lecture file
    """
    # Build the search pattern.
    pattern = f"{dir_path}/lectures_source/Lesson{lesson}*.smd"
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
    _LOG.debug("Found lecture file: %s", file_path)
    return file_path


def get_source_name(dir_path: str, lesson: str) -> str:
    """
    Get the source file name for a lesson.

    Example:
    - Input: dir_path="msml610", lesson="01"
    - Output: "Lesson01-Introduction.md" (if file exists)

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

    Example:
    - Input: source_name="Lesson01-Introduction.md", extension=".pdf"
    - Output: "Lesson01-Introduction.pdf"

    :param source_name: source file name
    :param extension: new extension
    :return: output file name
    """
    # Remove file extension and add new extension.
    base_name = os.path.splitext(source_name)[0]
    output_name = base_name + extension
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
    hdbg.dassert_file_exists(pdf_path)
    # Use mdls to get page count.
    cmd = f"mdls -name kMDItemNumberOfPages '{pdf_path}'"
    _LOG.debug("Running command: %s", cmd)
    _, output = hsystem.system_to_string(cmd)
    # Parse output like "kMDItemNumberOfPages = 42".
    parts = output.strip().split("=")
    hdbg.dassert_eq(len(parts), 2, "Unexpected mdls output format:", output)
    page_count_str = parts[1].strip()
    page_count = int(page_count_str)
    _LOG.debug("PDF '%s' has %d pages", pdf_path, page_count)
    return page_count


def get_pdf_page_counts(
    directory: str, pattern: str = "Lesson*.pdf"
) -> Dict[str, int]:
    """
    Get page counts for all PDF files matching a pattern in a directory.

    Example:
    - Output: {"Lesson01.pdf": 45, "Lesson02.pdf": 38, "Lesson03.pdf": 52}

    :param directory: directory to search
    :param pattern: glob pattern for PDF files
    :return: dictionary mapping file names to page counts
    """
    hdbg.dassert_dir_exists(directory)
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


# #############################################################################
# Lesson discovery
# #############################################################################


def get_lesson_numbers(course_dir: str) -> List[str]:
    """
    Get all lesson numbers in a course.

    :param course_dir: Course directory (data605 or msml610)
    :return: Sorted list of lesson numbers like
        ```
        ["01.1", "01.2", "02", "03", "04.1"]
        ```
    """
    _LOG.debug(hprint.to_str("course_dir"))
    lectures_source_dir = os.path.join(course_dir, "lectures_source")
    hdbg.dassert_dir_exists(lectures_source_dir)
    lessons = []
    for file in os.listdir(lectures_source_dir):
        # Match lesson numbers from filenames like "Lesson01.1-BigData.smd"
        # -> "01.1".
        match = re.match(r"Lesson(\d+(?:\.[0-9A-Za-z]+)?)", file)
        if match:
            lesson_num = match.group(1)
            lessons.append(lesson_num)
    # De-dup since e.g. both "Lesson01.1-*.smd" and "Lesson01.1-*.pdf" match.
    lessons = sorted(set(lessons))
    _LOG.debug("return=%s", lessons)
    return lessons


def get_lesson_files(course_dir: str) -> List[str]:
    """
    Discover all lesson files in a course directory.

    :param course_dir: Course directory (data605 or msml610)
    :return: Sorted list of lesson file paths, e.g.,
        ```
        ["data605/lectures_source/Lesson01.1-BigData.smd",
        "data605/lectures_source/Lesson01.2-History.smd"]
        ```
    """
    _LOG.debug(hprint.to_str("course_dir"))
    lectures_source_dir = os.path.join(course_dir, "lectures_source")
    hdbg.dassert_dir_exists(lectures_source_dir)
    lesson_files = []
    for file in os.listdir(lectures_source_dir):
        # Match files like "Lesson01.1-BigData.smd" or
        # "Lesson02-Introduction.smd".
        if re.match(r"^Lesson\d+", file):
            file_path = os.path.join(lectures_source_dir, file)
            lesson_files.append(file_path)
    lesson_files = sorted(lesson_files)
    _LOG.debug("return=%s", lesson_files)
    return lesson_files


def collect_all_lessons() -> Dict[str, List[str]]:
    """
    Collect all lessons organized by course.

    :return: Dict with course dirs as keys and lesson lists as values, e.g.,
        ```
        {"data605": ["01.1", "01.2", "02"], "msml610": ["01", "02.1"]}
        ```
    """
    all_lessons = {}
    # TODO(gp): Look for all dirs that have a lectures_source.
    valid_dirs = ["data605", "msml610", "book_springer"]
    for course_dir in valid_dirs:
        all_lessons[course_dir] = get_lesson_numbers(course_dir)
    _LOG.debug("return=%s", all_lessons)
    return all_lessons


# #############################################################################
# LLM-generated artifacts
# #############################################################################


def extract_title_from_markdown(input_file: str) -> Optional[str]:
    r"""
    Extract title from markdown file.

    First looks for a `lesson_title` metadata directive (e.g.,
    `// lesson_title=...`) at the top of the file. If not present, falls
    back to looking for patterns like:
    \text{\blue{Lesson 2.1: Git}}

    :param input_file: path to input markdown file
    :return: extracted title or None if not found
    """
    hdbg.dassert_file_exists(input_file)
    content = hio.from_file(input_file)
    # Check for a `lesson_title` metadata directive at the top of the file.
    lines = content.split("\n")
    metadata, _ = dshdprno.extract_slide_metadata(lines)
    if "lesson_title" in metadata:
        title = metadata["lesson_title"].strip()
        _LOG.info("Extracted title from metadata template: %s", title)
        return title
    # Pattern to match \text{\blue{...}} or \text{...} or similar LaTeX constructs.
    pattern = r"\\text\{(?:\\blue\{)?([^}]+)\}?"
    match = re.search(pattern, content)
    if match:
        title = match.group(1)
        # Clean up the title.
        title = title.strip()
        _LOG.info("Extracted title: %s", title)
        return title
    _LOG.warning("Could not extract title from markdown file")
    return None


@hretry.sync_retry(
    num_attempts=5,
    exceptions=(RuntimeError,),
    retry_delay_in_sec=2,
)
def git_add_with_retry(file_name: str, *, dry_run: bool) -> None:
    """
    Run `git add` on `file_name`, retrying on failure.

    This is needed because concurrent Git commands (e.g., another
    generation script, or an IDE) can hold `.git/index.lock`, causing
    `git add` to fail with "Unable to create '.git/index.lock': File
    exists.".

    :param file_name: path of the file to add
    :param dry_run: print the command without executing it
    """
    cmd = f"git add {file_name}"
    hsystem.system(cmd, print_command=True, dry_run=dry_run)


# Backends supported by `call_llm()`.
# - "hllm": `helpers.hllm.get_completion()`, supports passing images as
#   multi-modal context
# - "hllm_cli_lib": `helpers.hllm_cli.apply_llm(backend="library")`,
#   text-only, calls the `llm` Python library in-process
# - "hllm_cli_exec": `helpers.hllm_cli.apply_llm(backend="executable")`,
#   text-only, shells out to simonw's `llm` CLI executable
LLM_BACKENDS = ("hllm", "hllm_cli_lib", "hllm_cli_exec")

# Map a `LLM_BACKENDS` entry to the `backend` argument `hllm_cli.apply_llm()`
# expects.
_LLM_BACKEND_TO_HLLM_CLI_BACKEND = {
    "hllm_cli_lib": "library",
    "hllm_cli_exec": "executable",
}


def call_llm(
    user_prompt: str,
    system_prompt: str,
    model: str,
    llm_backend: str,
    *,
    images_as_base64: Tuple[str, ...] = (),
) -> str:
    """
    Generate a completion from an LLM through one of `LLM_BACKENDS`.

    :param user_prompt: user message
    :param system_prompt: system prompt (style guide / instructions)
    :param model: LLM model to use, or "" to use the backend's default
    :param llm_backend: which backend to use, one of `LLM_BACKENDS`
    :param images_as_base64: images to pass as multi-modal context (only
        supported by the "hllm" backend; ignored by "hllm_cli_lib" /
        "hllm_cli_exec")
    :return: generated text
    """
    hdbg.dassert_in(llm_backend, LLM_BACKENDS)
    if llm_backend == "hllm":
        import helpers.hllm as hllm

        response = hllm.get_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=model,
            cache_mode="NORMAL",
            temperature=0.1,
            images_as_base64=images_as_base64,
        )
    else:
        import helpers.hllm_cli as hllmcli

        hllm_cli_backend = _LLM_BACKEND_TO_HLLM_CLI_BACKEND[llm_backend]
        response, _ = hllmcli.apply_llm(
            user_prompt,
            system_prompt=system_prompt,
            model=model,
            backend=hllm_cli_backend,
        )
    return str(response)


@hcacsimp.simple_cache(cache_type="json")
def call_llm_cached(
    user_prompt: str,
    system_prompt: str,
    model: str,
    llm_backend: str,
    *,
    images_as_base64: Tuple[str, ...] = (),
) -> str:
    """
    Like `call_llm()`, but cache the completion on disk keyed by its
    arguments, so re-running a script for the same input (e.g., an
    incremental / per-unit generation loop) does not re-pay for an
    unchanged LLM call.

    :param user_prompt: user message
    :param system_prompt: system prompt (style guide / instructions)
    :param model: LLM model to use, or "" to use the backend's default
    :param llm_backend: which backend to use, one of `LLM_BACKENDS`
    :param images_as_base64: images to pass as multi-modal context (only
        supported by the "hllm" backend; ignored by "hllm_cli_lib" /
        "hllm_cli_exec")
    :return: generated text
    """
    return call_llm(
        user_prompt,
        system_prompt,
        model,
        llm_backend,
        images_as_base64=images_as_base64,
    )


# Map a file extension (without the leading dot) to the character(s) that
# start a line comment in that file type. Used to stamp provenance tags (git
# hash/timestamp) into LLM-generated artifacts.
_EXTENSION_TO_COMMENT_PREFIX = {
    "tex": "%",
    "typ": "//",
}


def get_comment_prefix(extension: str) -> str:
    """
    Get the line-comment prefix for a file extension.

    :param extension: file extension, without the leading dot (e.g.,
        "tex", "typ")
    :return: comment prefix (e.g., "%" for "tex", "//" for "typ")
    """
    hdbg.dassert_in(extension, _EXTENSION_TO_COMMENT_PREFIX)
    return _EXTENSION_TO_COMMENT_PREFIX[extension]


def convert_markdown_to_pdf(
    md_file: str, pdf_file: str, script_dir: str, *, dry_run: bool = False
) -> None:
    r"""
    Convert a markdown file to PDF using pandoc.

    Uses the LaTeX macros from `latex_abbrevs.sty` and header style shared by
    the `class_scripts` LLM-generated markdown output (lecture commentary,
    book chapters), so that macros used in the generated text (e.g., `\vmu`)
    resolve correctly.

    :param md_file: input markdown file
    :param pdf_file: output PDF file
    :param script_dir: directory containing `header-style.tex` (typically
        `class_scripts/`)
    :param dry_run: print the command without executing it
    """
    latex_abbrevs_file = os.path.join(
        hgit.find_file("dev_scripts_helpers"),
        "documentation",
        "latex_abbrevs.sty",
    )
    hdbg.dassert_file_exists(latex_abbrevs_file)
    cmd = (
        f"pandoc {md_file} -o {pdf_file} "
        f"--pdf-engine=xelatex "
        f"-V geometry:margin=1in "
        f"-V fontsize=11pt "
        f"--highlight-style=tango "
        f"--include-in-header={latex_abbrevs_file} "
        f"--include-in-header={script_dir}/header-style.tex"
    )
    hsystem.system(cmd, print_command=True, dry_run=dry_run)
