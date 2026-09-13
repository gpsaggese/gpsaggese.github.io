#!/usr/bin/env python

"""
Improve lecture slides using LLM.

This script improves lecture slides using process_slides.py with the
slide_improve action.

# Usage Example

- Improve DATA605 lesson 01.1 slides using LLM:
> slide_improve.py -i data605/01.1

- Improve MSML610 lesson 02.3 slides using LLM:
> slide_improve.py -i msml610/02.3

Import as:

import class_scripts.slide_improve as clslimpr
"""

import argparse
import logging

import class_scripts.common_utils as csccouti
import helpers.hdbg as hdbg
import helpers.hparser as hparser
import helpers.hprint as hprint
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)

# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Lecture specification: 'data605/08.1', 'msml610/08.1', "
        "or file path 'msml610/lectures_source/Lesson10.2-Name.smd'",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the command that would be executed without running it",
    )
    parser.add_argument(
        "--process_slides_args",
        action="store",
        default=None,
        help="Additional options string passed through verbatim to "
        "process_slides.py",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Parse and validate arguments.
    dir_arg, lesson_arg = csccouti.parse_lesson_spec(args.input)
    csccouti.validate_dir_lesson_args(dir_arg, lesson_arg)
    # Find the lecture file.
    lecture_file = csccouti.find_lecture_file(dir_arg, lesson_arg)
    src_name = str(lecture_file)
    dst_name = src_name
    # Build the command.
    cmd_parts = [
        "process_slides.py",
        f"--in_file {src_name}",
        "--action slide_improve",
        f"--out_file {dst_name}",
        "--use_llm_transform",
    ]
    if args.dry_run:
        cmd_parts.append("--dry_run")
    # Add extra options if provided.
    if args.process_slides_args:
        cmd_parts.append(args.process_slides_args)
    cmd = " ".join(cmd_parts)
    _LOG.info("%s", hprint.color_highlight(f"> {cmd}", "green"))
    # Execute the command.
    hsystem.system(cmd)


if __name__ == "__main__":
    _main(_parse())
