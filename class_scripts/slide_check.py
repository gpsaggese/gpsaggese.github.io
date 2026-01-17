#!/usr/bin/env python

"""
Check and fix text in lecture slides.

This script checks and fixes text in lecture slides using process_slides.py
with the text_check_fix action.

Usage:
> slide_check.py data605 01.1
> slide_check.py msml610 02.3

Import as:

import class_scripts.slide_check as clslchec
"""

import argparse
import logging

import class_scripts.common_utils as clcomuut
import helpers.hdbg as hdbg
import helpers.hparser as hparser
import helpers.hsystem as hsystem

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
    parser.add_argument(
        "extra_opts",
        nargs="*",
        help="Additional options to pass to process_slides.py",
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
    src_name = str(lecture_file)
    dst_name = src_name
    # Build the command.
    cmd_parts = [
        "process_slides.py",
        f"--in_file {src_name}",
        "--action text_check_fix",
        f"--out_file {dst_name}",
        "--use_llm_transform",
    ]
    # Add extra options if provided.
    if args.extra_opts:
        cmd_parts.extend(args.extra_opts)
    cmd = " ".join(cmd_parts)
    _LOG.info("Running command: %s", cmd)
    # Execute the command.
    hsystem.system(cmd)


if __name__ == "__main__":
    _main(_parse())
