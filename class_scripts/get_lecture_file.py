#!/usr/bin/env python

"""
Find and print the path to a lecture file.

This script finds exactly one lecture file matching the pattern:
{DIR}/lectures_source/Lesson{LESSON}*

Usage:
> get_lecture_file.py data605 01.1
> get_lecture_file.py msml610 02.3

Import as:

import class_scripts.get_lecture_file as clgelifi
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
    parser.add_argument(
        "lesson",
        type=str,
        help="Lesson number (e.g., 01.1, 02.3)",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Validate arguments.
    clcomuut.validate_dir_lesson_args(args.dir, args.lesson)
    # Find the lecture file.
    lecture_file = clcomuut.find_lecture_file(args.dir, args.lesson)
    # Print the file path.
    _LOG.info("Lecture file: %s", lecture_file)


if __name__ == "__main__":
    _main(_parse())
