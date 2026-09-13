#!/usr/bin/env python3
r"""
Generate comprehensive table of contents for book from lecture slides.

- Reads a `book_map.md` file which contains chapters and their associated
  lesson files
- Extracts the table of contents from each lesson file
- Combines them into a single output markdown file.

# Usage Example

- Generate a table of contents with headers up to level 2 (H1-H2 only):
> create_book_toc_from_slides.py --output book_toc.md --max_level 2

- Generate a table of contents with headers up to level 5 (full depth):
> create_book_toc_from_slides.py --output book_toc.md --max_level 5
"""

import argparse
import logging
import re
import subprocess
from typing import List, Tuple

from tqdm import tqdm

import helpers.hdbg as hdbg
import helpers.hgit as hgit
import helpers.hparser as hparser
import helpers.hprint as hprint
import helpers.hselect_input_output as hseinout

_LOG = logging.getLogger(__name__)


def _extract_chapters_and_lessons(
    book_map_file: str,
) -> List[Tuple[str, str, List[str]]]:
    """
    Extract chapters and their associated lesson files from book_map.md.

    :param book_map_file: Path to the book_map.md file
    :return: List of (chapter_title, chapter_header, lesson_files) tuples
        where each tuple contains the chapter title, markdown header line,
        and list of associated lesson file paths
    """
    _LOG.debug(hprint.to_str("book_map_file"))
    hdbg.dassert_file_exists(book_map_file)
    # Read book_map.md file.
    # TODO(ai_gp): Use hio.from_file
    with open(book_map_file, "r") as f:
        lines = f.readlines()
    # Parse chapters and lessons via state machine:
    # - Track chapter headers (## N: Title) and switch to new chapter
    # - Within chapter, look for **Lessons** section
    # - Extract bullet-point lesson files within lessons section
    chapters = []
    current_chapter_title = ""
    current_chapter_header = ""
    current_lessons = []
    in_lessons_section = False
    for line in lines:
        line = line.rstrip()
        # Check for chapter header (## N: Title).
        match = re.match(r"^## (\d+): (.+)$", line)
        if match:
            # Save previous chapter.
            if current_chapter_title:
                chapters.append(
                    (
                        current_chapter_title,
                        current_chapter_header,
                        current_lessons,
                    )
                )
            current_chapter_title = match.group(2)
            current_chapter_header = line
            current_lessons = []
            in_lessons_section = False
            continue
        # Check for **Lessons** section.
        if line.startswith("**Lessons**"):
            in_lessons_section = True
            continue
        # Check for next section (e.g., **Tutorials**).
        if line.startswith("**") and line != "**Lessons**":
            in_lessons_section = False
            continue
        # Extract lesson files.
        if in_lessons_section and line.strip().startswith("- "):
            lesson_file = line.strip()[2:].strip().strip("`")
            current_lessons.append(lesson_file)
    # Save last chapter.
    if current_chapter_title:
        chapters.append(
            (current_chapter_title, current_chapter_header, current_lessons)
        )
    _LOG.debug("return=%s chapters", len(chapters))
    return chapters


def _extract_toc_from_lesson(lesson_file: str, *, max_level: int) -> str:
    """
    Extract table of contents from a single lesson file.

    Calls extract_toc_from_txt.py to extract headers up to max_level from
    the lesson file and returns the markdown-formatted output.

    :param lesson_file: Path to the lesson file
    :param max_level: Maximum header level to extract
    :return: Extracted table of contents as markdown string
    """
    _LOG.debug(hprint.to_str("lesson_file max_level"))
    hdbg.dassert_file_exists(lesson_file)
    # Build and execute command to extract TOC via external script.
    # Script is located in git tree and outputs to stdout.
    extract_toc_script = hgit.find_file_in_git_tree("extract_toc_from_txt.py")
    cmd_parts = [
        extract_toc_script,
        f"--input={lesson_file}",
        "--output=-",
        f"--max_level={max_level}",
        "--warn_on_malformed",
        "--count_slides",
    ]
    cmd = " ".join(cmd_parts)
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        check=True,
    )
    _LOG.debug("return=%d chars", len(result.stdout))
    return result.stdout


def _insert_toc_in_file(
    file_path: str,
    *,
    max_level: int,
) -> None:
    """
    Insert table of contents after lessons in `### Lessons` section.

    Reads the file, finds lessons under '### Lessons', extracts TOC from each
    lesson file, then inserts a '### Current TOC' section after all lesson
    entries with each lesson's TOC prefixed by a comment line.
    Removes any existing '### Actual TOC' or '### Current TOC' sections.

    :param file_path: Path to the file to process
    :param max_level: Maximum header level to extract
    """
    _LOG.debug(hprint.to_str("file_path max_level"))
    hdbg.dassert_file_exists(file_path)
    with open(file_path, "r") as f:
        lines = f.readlines()
    # Remove Current TOC sections and collect lessons in one pass.
    processed = []
    all_lessons = []
    in_lessons = False
    skip_toc = False
    for line in lines:
        line_s = line.rstrip()
        # Skip Current TOC sections.
        if line_s.startswith("### Current TOC"):
            skip_toc = True
            continue
        if skip_toc:
            if line_s.startswith("###"):
                skip_toc = False
            else:
                continue
        # Track lessons and collect them.
        if line_s.startswith("### Lessons"):
            in_lessons = True
            processed.append(line)
            continue
        if in_lessons and line_s.startswith("###"):
            in_lessons = False
        elif in_lessons and line_s.strip().startswith("- "):
            lesson = line_s.strip()[2:].strip().strip("`")
            all_lessons.append(lesson)
        processed.append(line)
    hdbg.dassert(len(all_lessons) > 0, "No lessons found in '%s'", file_path)
    # Extract TOC for each lesson.
    lesson_tocs = {}
    for lesson_file in all_lessons:
        hdbg.dassert_file_exists(lesson_file)
        toc = _extract_toc_from_lesson(lesson_file, max_level=max_level)
        hdbg.dassert_ne(toc, "")
        lesson_tocs[lesson_file] = toc
    # Rebuild with Current TOC sections inserted after each lessons section.
    output = []
    in_lessons = False
    current_chapter_lessons = []
    for line in processed:
        line_s = line.rstrip()
        output.append(line)
        if line_s.startswith("### Lessons"):
            in_lessons = True
            current_chapter_lessons = []
            continue
        if in_lessons and line_s.strip().startswith("- "):
            lesson = line_s.strip()[2:].strip().strip("`")
            current_chapter_lessons.append(lesson)
        if in_lessons and line_s.startswith("###") and line_s != "### Lessons":
            in_lessons = False
            # Remove this section heading we just appended.
            output.pop()
            # Remove trailing blanks and insert Current TOC.
            while output and output[-1].strip() == "":
                output.pop()
            output.append("\n")
            output.append("### Current TOC\n")
            for lesson_file in current_chapter_lessons:
                output.append(f"// `{lesson_file}`\n")
                output.append(lesson_tocs[lesson_file])
                output.append("\n")
            output.append(line)
    # Handle file ending in lessons section.
    if in_lessons and current_chapter_lessons:
        while output and output[-1].strip() == "":
            output.pop()
        output.append("\n")
        output.append("### Current TOC\n")
        for lesson_file in current_chapter_lessons:
            output.append(f"// `{lesson_file}`\n")
            output.append(lesson_tocs[lesson_file])
            output.append("\n")
    with open(file_path, "w") as f:
        f.write("".join(output))
    _LOG.info("Inserted TOC section into '%s'", file_path)


def _create_book_toc(
    book_map_file: str,
    output_file: str,
    *,
    max_level: int,
    max_number: int = 0,
) -> None:
    """
    Create combined table of contents for all chapters and lessons.

    Reads book_map.md, extracts chapters and lessons, then extracts the
    table of contents from each lesson file and combines them into a single
    output file with proper markdown structure.

    :param book_map_file: Path to the book_map.md file
    :param output_file: Path to the output file
    :param max_level: Maximum header level to extract
    :param max_number: Maximum number of chapters (h2 headers) to include
        - Default: 0 (include all chapters)
    """
    _LOG.debug(hprint.to_str("book_map_file output_file max_level max_number"))
    # Validate input file exists.
    hdbg.dassert_file_exists(book_map_file)
    # Extract chapters and lessons.
    chapters = _extract_chapters_and_lessons(book_map_file)
    _LOG.info("Found '%d' chapters", len(chapters))
    # Validate all lesson files exist before processing.
    for _, _, lessons in chapters:
        for lesson_file in lessons:
            hdbg.dassert_file_exists(lesson_file)
    # Limit chapters if max_number is specified.
    if max_number > 0:
        chapters = chapters[:max_number]
        _LOG.warning("Limited to '%d' chapters", len(chapters))
    # Build output content.
    output_lines = []
    output_lines.append("# Book Table of Contents")
    output_lines.append("")
    # Flatten chapter structure (chapter_idx, header, lesson_file) for progress bar.
    # Allows iteration with chapter context while tracking total lessons count.
    lessons_with_chapters = [
        (chapter_idx, chapter_header, lesson_file)
        for chapter_idx, (_, chapter_header, lessons) in enumerate(chapters)
        for lesson_file in lessons
    ]
    _LOG.debug("total_lessons=%d", len(lessons_with_chapters))
    # Process lessons with progress bar.
    current_chapter_idx = -1
    for chapter_idx, chapter_header, lesson_file in tqdm(
        lessons_with_chapters, desc="Processing lectures"
    ):
        # Add chapter header when moving to a new chapter.
        if chapter_idx != current_chapter_idx:
            output_lines.append(chapter_header)
            output_lines.append("")
            current_chapter_idx = chapter_idx
        _LOG.info("Extracting TOC from '%s'", lesson_file)
        output_lines.append(f"### {lesson_file}")
        output_lines.append("")
        # Extract TOC from lesson.
        toc_content = _extract_toc_from_lesson(lesson_file, max_level=max_level)
        hdbg.dassert_ne(toc_content, "")
        output_lines.append(toc_content)
    # Write output file.
    output_content = "\n".join(output_lines)
    hseinout.to_file(output_content, output_file)
    _LOG.info("Wrote output to '%s'", output_file)


def _parse() -> argparse.ArgumentParser:
    """
    Create and return the argument parser.

    :return: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="book.From_Data_To_Decisions/book_map.md",
        help="Path to the book_map.md file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="book_toc.md",
        help="Path to the output file",
    )
    parser.add_argument(
        "--max_level",
        type=int,
        default=5,
        help="Maximum header level to extract",
    )
    parser.add_argument(
        "--max_number",
        type=int,
        default=0,
        help="Maximum number of chapters (h2 headers) to include (0 = all)",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Insert TOC sections into input file instead of creating separate output",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    """
    Main entry point.

    :param parser: Argument parser
    """
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    _LOG.debug(
        hprint.to_str("args.in_place args.input args.output args.max_level")
    )
    # Dispatch to either in-place insertion or full TOC generation.
    if args.in_place:
        _insert_toc_in_file(args.input, max_level=args.max_level)
    else:
        _create_book_toc(
            book_map_file=args.input,
            output_file=args.output,
            max_level=args.max_level,
            max_number=args.max_number,
        )


if __name__ == "__main__":
    parser = _parse()
    _main(parser)
