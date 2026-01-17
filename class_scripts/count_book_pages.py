#!/usr/bin/env python

"""
Count pages in book PDF files.

This script counts the number of pages in each PDF file in {DIR}/book/.

Usage:
> count_book_pages.py data605
> count_book_pages.py msml610

Import as:

import classes2.count_book_pages as clcobopa
"""

import argparse
import logging

import classes2.common_utils as clcomuut
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
    book_dir = f"{args.dir}/book"
    _LOG.info("DIR=%s", book_dir)
    # Get page counts for all PDFs.
    page_counts = clcomuut.get_pdf_page_counts(book_dir, pattern="Lesson*.pdf")
    # Print results tab-separated.
    for filename, page_count in page_counts.items():
        _LOG.info("%s\t%d", filename, page_count)


if __name__ == "__main__":
    _main(_parse())
