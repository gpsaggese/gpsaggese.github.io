#!/usr/bin/env -S uv run

# /// script
# dependencies = ["tabulate"]
# ///

"""
Count slides and text statistics in lecture source files.

This script counts slides, headers, lines, words, and characters in each lecture
source file in {DIR}/lectures_source/.

# Usage Example

- Count slides and text statistics for the data605 course:
> count_lecture_slides.py data605

- Count slides and text statistics for the msml610 course:
> count_lecture_slides.py msml610

- Print the statistics as a TSV table:
> count_lecture_slides.py msml610 --format tsv

- Print the statistics as a CSV table:
> count_lecture_slides.py msml610 --format csv

Import as:

import class_scripts.count_lecture_slides as clcoulsl
"""

import argparse
import logging
import os
import re
from typing import Any, Dict, List, Tuple

import tabulate

import class_scripts.common_utils as csccouti
import helpers.hdbg as hdbg
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)

# #############################################################################
# Helper functions
# #############################################################################


def _count_slides(content: str) -> int:
    """
    Count the number of slides (lines starting with "* ") in content.

    :param content: file content as string
    :return: number of slides
    """
    lines = content.split("\n")
    slide_count = sum(1 for line in lines if re.match(r"^\* ", line))
    return slide_count


def _count_headers(content: str) -> Tuple[int, int, int]:
    """
    Count headers at each level (# , ## , ### ) in content.

    :param content: file content as string
    :return: tuple of (h1_count, h2_count, h3_count)
    """
    lines = content.split("\n")
    h1_count = sum(1 for line in lines if re.match(r"^# ", line))
    h2_count = sum(1 for line in lines if re.match(r"^## ", line))
    h3_count = sum(1 for line in lines if re.match(r"^### ", line))
    return (h1_count, h2_count, h3_count)


def _count_text_stats(content: str) -> Tuple[int, int, int]:
    """
    Count lines, words, and characters in content.

    :param content: file content as string
    :return: tuple of (line_count, word_count, char_count)
    """
    lines = content.split("\n")
    line_count = len(lines)
    word_count = len(content.split())
    char_count = len(content)
    return (line_count, word_count, char_count)


def _collect_stats(directory: str) -> List[Dict[str, Any]]:
    """
    Collect statistics for all lecture files in a directory.

    :param directory: course directory containing lectures_source/
    :return: list of dicts with file stats
    """
    lectures_dir = os.path.join(directory, "lectures_source")
    hdbg.dassert_dir_exists(lectures_dir)
    _LOG.info("Scanning directory: %s", lectures_dir)
    lesson_files = csccouti.get_lesson_files(directory)
    rows = []
    for file_path in lesson_files:
        _LOG.debug("Processing file: %s", file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        filename = os.path.basename(file_path)
        slides = _count_slides(content)
        h1, h2, h3 = _count_headers(content)
        lines, words, chars = _count_text_stats(content)
        row = {
            "File": filename,
            "Slides": slides,
            "H1": h1,
            "H2": h2,
            "H3": h3,
            "Lines": lines,
            "Words": words,
            "Chars": chars,
        }
        rows.append(row)
    return rows


def _format_table(
    rows: List[Dict[str, Any]], *, format_type: str = "markdown"
) -> str:
    """
    Format rows as a table in the specified format.

    :param rows: list of dicts with stats
    :param format_type: output format (markdown, tsv, csv)
    :return: formatted table as string
    """
    headers = ["File", "Slides", "H1", "H2", "H3", "Lines", "Words", "Chars"]
    table_data = [[row[h] for h in headers] for row in rows]
    hdbg.dassert_in(format_type, ["markdown", "tsv", "csv"])
    if format_type == "markdown":
        return tabulate.tabulate(table_data, headers=headers, tablefmt="github")
    if format_type == "tsv":
        return tabulate.tabulate(table_data, headers=headers, tablefmt="tsv")
    output = []
    output.append(",".join(headers))
    for row_data in table_data:
        output.append(",".join(str(x) for x in row_data))
    return "\n".join(output)


# #############################################################################
# Argument parsing and main
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "dir",
        type=str,
        help="Course directory (e.g., data605, msml610)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="markdown",
        choices=["markdown", "csv", "tsv"],
        help="Output format (default: markdown)",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    _LOG.info("DIR=%s, FORMAT=%s", args.dir, args.format)
    rows = _collect_stats(args.dir)
    table_str = _format_table(rows, format_type=args.format)
    print(table_str)


if __name__ == "__main__":
    _main(_parse())
