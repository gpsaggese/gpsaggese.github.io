#!/usr/bin/env python

"""
Count words in lecture script files.

This script counts the number of words in each file in {DIR}/lectures_script/.

Usage:
> count_words.py data605
> count_words.py msml610

Import as:

import classes2.count_words as clcowoor
"""

import argparse
import logging
from pathlib import Path

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


def _count_words_in_file(file_path: Path) -> int:
    """
    Count the number of words in a text file.

    :param file_path: path to the file
    :return: number of words in the file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    word_count = len(content.split())
    return word_count


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Build the directory path.
    lectures_script_dir = Path(args.dir) / "lectures_script"
    hdbg.dassert(
        lectures_script_dir.exists(),
        "Directory does not exist:",
        str(lectures_script_dir),
    )
    _LOG.debug("Scanning directory: %s", lectures_script_dir)
    # Process all files in the directory.
    files = sorted(lectures_script_dir.iterdir())
    for file_path in files:
        if file_path.is_file():
            word_count = _count_words_in_file(file_path)
            # Print filename and word count, tab-separated.
            _LOG.info("%s\t%d", file_path.name, word_count)


if __name__ == "__main__":
    _main(_parse())
