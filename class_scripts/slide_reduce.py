#!/usr/bin/env python

"""
Reduce lecture slides using LLM.

This script reduces lecture slides using process_slides.py with the
slide_reduce action.

Usage:
> slide_reduce.py data605 01.1
> slide_reduce.py msml610 02.3

Import as:

import class_scripts.slide_reduce as clslredu
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
    # Build the command.
    cmd_parts = [
        "process_slides.py",
        f"--in_file {src_name}",
        "--action slide_reduce",
        f"--out_file {src_name}",
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
