#!/usr/bin/env python

"""
Generate quizzes for a lecture using LLM.

This script generates multiple choice questions from lecture content using
llm_cli.py.

Usage:
> gen_quizzes.py data605 01.1
> gen_quizzes.py msml610 02.3

Import as:

import classes2.gen_quizzes as clgequiz
"""

import argparse
import logging
from pathlib import Path

import classes2.common_utils as clcomuut
import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################


# System prompt for quiz generation.
QUIZ_PROMPT = """
You are a college professor teaching a class.

Given the content below:
- Write 20 multiple choice questions
- Each question has 5 possible answers with only one correct answer
- Make sure to focus on concept and understanding of the material rather than memorization.
- Mark the correct answer in bold.

The output should be in Markdown code without having page separators, any
comment, or divved fence, just the questions and the answers.
"""


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
        help="Additional options to pass to llm_cli.py",
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
    dst_name = clcomuut.get_output_name(src_name, ".quizzes.md")
    # Build paths.
    input_file = f"{args.dir}/lectures_source/{src_name}"
    output_file = f"{args.dir}/lectures_quizzes/{dst_name}"
    # Ensure output directory exists.
    clcomuut.ensure_dir_exists(f"{args.dir}/lectures_quizzes")
    # Save the prompt to a temporary file.
    prompt_file = "tmp.gen_quizzes_prompt.txt"
    hio.to_file(prompt_file, QUIZ_PROMPT)
    _LOG.debug("Saved prompt to: %s", prompt_file)
    # Build the command.
    cmd_parts = [
        "llm_cli.py",
        f"--input {input_file}",
        f"--output {output_file}",
        f"--system_prompt_file {prompt_file}",
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
