#!/usr/bin/env python

"""
Count pages in PDF files.

This script counts the number of pages in PDF files matching `Lesson*.pdf` in
a directory (`--dir`), or in a single PDF file (`--input`).

# Usage Example

- Count pages in all lecture PDF files for a course:
> count_pdf_pages.py --dir data605/lectures

- Count pages in all book chapter PDF files for a course:
> count_pdf_pages.py --dir msml610/book

- Count pages in a single PDF file:
> count_pdf_pages.py --input data605/lectures/Lesson01.1-BigData.pdf

Import as:

import class_scripts.count_pdf_pages as clcopdpa
"""

import argparse
import logging

import class_scripts.common_utils as csccouti
import helpers.hdbg as hdbg
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)

# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dir",
        type=str,
        default="",
        help="Directory to scan for PDF files matching 'Lesson*.pdf'",
    )
    group.add_argument(
        "-i",
        "--input",
        type=str,
        default="",
        help="Single PDF file to count pages for",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    if args.input:
        # Count pages in the single PDF file passed via `--input`.
        _LOG.info("INPUT=%s", args.input)
        page_count = csccouti.count_pdf_pages(args.input)
        _LOG.info("%s\t%d", args.input, page_count)
    else:
        # Count pages for all PDFs matching the pattern in `--dir`.
        _LOG.info("DIR=%s", args.dir)
        page_counts = csccouti.get_pdf_page_counts(
            args.dir, pattern="Lesson*.pdf"
        )
        for filename, page_count in page_counts.items():
            _LOG.info("%s\t%d", filename, page_count)


if __name__ == "__main__":
    _main(_parse())
