#!/usr/bin/env python

"""
Generate lecture slides PDF.

This script generates a PDF from lecture source files using notes_to_pdf.py.

Usage:
> gen_slides.py data605 01.1
> gen_slides.py msml610 02.3

Import as:

import classes2.gen_slides as clgeslio
"""

import argparse
import logging

import classes2.common_utils as clcomuut
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
        help="Additional options to pass to notes_to_pdf.py",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Validate arguments.
    clcomuut.validate_dir_lesson_args(args.dir, args.lesson)
    # Get source and destination names.
    src_name = clcomuut.get_source_name(args.dir, args.lesson)
    dst_name = clcomuut.get_output_name(src_name, ".pdf")
    # Build paths.
    input_file = f"{args.dir}/lectures_source/{src_name}"
    output_file = f"{args.dir}/lectures/{dst_name}"
    # Ensure output directory exists.
    clcomuut.ensure_dir_exists(f"{args.dir}/lectures")
    # Build the command with debug options.
    opts_debug = "--skip_action cleanup_before --skip_action cleanup_after"
    cmd_parts = [
        "notes_to_pdf.py",
        f"--input {input_file}",
        f"--output {output_file}",
        "--type slides",
        "--toc_type navigation",
        "--debug_on_error",
        opts_debug,
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
