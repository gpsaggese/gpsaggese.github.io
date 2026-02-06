#!/usr/bin/env python

"""
Count pages in lecture PDF files.

This script counts the number of pages in each PDF file in {DIR}/lectures/.

Usage:
> count_lecture_pages.py data605
> count_lecture_pages.py msml610

Import as:

import class_scripts.count_lecture_pages as clcolepa
"""

import argparse
import logging

import class_scripts.common_utils as clcomuut
import helpers.hdbg as hdbg
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)

# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dir",
        type=str,
        help="Course directory (e.g., data605, msml610)",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Build the directory path.
    lectures_dir = f"{args.dir}/lectures"
    _LOG.info("DIR=%s", lectures_dir)
    # Get page counts for all PDFs.
    page_counts = clcomuut.get_pdf_page_counts(
        lectures_dir, pattern="Lesson*.pdf"
    )
    # Print results tab-separated.
    for filename, page_count in page_counts.items():
        _LOG.info("%s\t%d", filename, page_count)


if __name__ == "__main__":
    _main(_parse())
