#!/usr/bin/env python

"""
Generate quizzes for a lecture using LLM.

This script generates questions from lecture content using llm_cli.py.
Two modes are available:
- Multiple choice quizzes (--for_class_quizzes): 20 questions with 5 answers each
  - Saved to: <class_dir>/lectures_quizzes/<lesson>.quizzes.md
- Discussion/review questions (--for_class_recap): 3-6 open-ended questions
  - Saved to: <class_dir>/lectures_recap/<lesson>.recap.md

By default, the output file is automatically formatted using lint_txt.py with
prettier. Use --no_lint to skip formatting.

Usage:
> gen_quizzes.py --for_class_quizzes data605 01.1
> gen_quizzes.py --for_class_recap msml610 02.3
> gen_quizzes.py --for_class_recap data605 01.2 --no_lint

Import as:

import class_scripts.gen_quizzes as clgequiz
"""

import argparse
import logging

import class_scripts.common_utils as csccouti
import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################


# System prompt for quiz generation (multiple choice).
CLASS_QUIZZES_PROMPT = """
You are a college professor teaching a class.

Given the content below:
- Write 20 multiple choice questions
- Each question has 5 possible answers with only one correct answer
- Make sure to focus on concept and understanding of the material rather than memorization.
- Mark the correct answer in bold.

The output should be in Markdown code without having page separators, any
comment, or divved fence, just the questions and the answers.
"""


# System prompt for class recap questions (open-ended discussion).
CLASS_RECAP_PROMPT = """
You are a college professor teaching a class.

Given the content below:
- Write 4 discussion/review questions for students to answer after watching the videos
- These should be open-ended questions that require synthesis of information, e.g.,
  - For task-formulation questions, explicitly define the task, experience, and
    performance metrics.
  - For example-based questions, give concrete real-world systems and briefly
    justify why they fit.
  - For comparison or reflection questions, explain why one approach is
    preferable in certain situations.
  - For conceptual "how and why" questions, describe the underlying mechanism and
    its practical significance.
  - You can also include questions with more right/wrong answers to emphasize key points

- Focus on deeper understanding and application of concepts rather than memorization

The output should be in Markdown code without having page separators, any
comment, or divved fence, just the questions.
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
        "--for_class_quizzes",
        action="store_true",
        help="Generate multiple choice quizzes (20 questions with 5 answers each)",
    )
    parser.add_argument(
        "--for_class_recap",
        action="store_true",
        help="Generate open-ended discussion/review questions (3-6 questions)",
    )
    parser.add_argument(
        "--lint",
        action="store_true",
        default=True,
        help="Run lint_txt.py with prettier action on output file (default: True)",
    )
    parser.add_argument(
        "--no_lint",
        action="store_false",
        dest="lint",
        help="Skip running lint_txt.py on output file",
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
    csccouti.validate_dir_lesson_args(args.dir, args.lesson)
    # Validate that only one option is specified.
    hdbg.dassert(
        args.for_class_quizzes or args.for_class_recap,
        "Must specify either --for_class_quizzes or --for_class_recap",
    )
    hdbg.dassert(
        not (args.for_class_quizzes and args.for_class_recap),
        "Cannot specify both --for_class_quizzes and --for_class_recap",
    )
    # Select the appropriate prompt, output directory, and file extension.
    if args.for_class_quizzes:
        prompt = CLASS_QUIZZES_PROMPT
        output_dir = f"{args.dir}/lectures_quizzes"
        file_extension = ".quizzes.md"
        _LOG.info("Using CLASS_QUIZZES_PROMPT for multiple choice questions")
    else:
        prompt = CLASS_RECAP_PROMPT
        output_dir = f"{args.dir}/lectures_recap"
        file_extension = ".recap.md"
        _LOG.info("Using CLASS_RECAP_PROMPT for discussion/review questions")
    # Get source and destination names.
    src_name = csccouti.get_source_name(args.dir, args.lesson)
    dst_name = csccouti.get_output_name(src_name, file_extension)
    # Build paths.
    input_file = f"{args.dir}/lectures_source/{src_name}"
    output_file = f"{output_dir}/{dst_name}"
    # Ensure output directory exists.
    csccouti.ensure_dir_exists(output_dir)
    # Save the prompt to a temporary file.
    prompt_file = "tmp.gen_quizzes_prompt.txt"
    hio.to_file(prompt_file, prompt)
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
    # Run linting if requested.
    if args.lint:
        _LOG.info("Running lint_txt.py on output file: %s", output_file)
        lint_cmd = f"lint_txt.py -i {output_file} --action prettier"
        _LOG.info("Executing: %s", lint_cmd)
        hsystem.system(lint_cmd)


if __name__ == "__main__":
    _main(_parse())
