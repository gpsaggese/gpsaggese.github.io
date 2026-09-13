#!/usr/bin/env python

"""
Find and print the path to a lecture file.

This script finds exactly one lecture file matching the pattern:
{DIR}/lectures_source/Lesson{LESSON}*

# Usage Example

- Find the lecture file for lesson 01.1 in the data605 course:
> get_lecture_file.py data605/01.1

- Find the lecture file for lesson 02.3 in the msml610 course:
> get_lecture_file.py msml610/02.3

Import as:

import class_scripts.get_lecture_file as clgelifi
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
    parser.add_argument(
        "input",
        type=str,
        help="Lecture specification: 'data605/08.1', 'msml610/08.1', "
        "or file path 'msml610/lectures_source/Lesson10.2-Name.smd'",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=False)
    # Parse and validate arguments.
    dir_arg, lesson_arg = csccouti.parse_lesson_spec(args.input)
    csccouti.validate_dir_lesson_args(dir_arg, lesson_arg)
    # Find the lecture file.
    lecture_file = csccouti.find_lecture_file(dir_arg, lesson_arg)
    # Print the file path.
    _LOG.info("Lecture file: %s", lecture_file)


if __name__ == "__main__":
    _main(_parse())
