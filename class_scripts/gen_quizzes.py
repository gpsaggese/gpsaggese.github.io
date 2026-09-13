#!/usr/bin/env python

"""
Generate quizzes for a lecture using LLM.

This script generates questions from lecture content using `llm_cli.py`.

- Two modes are available:
  - Multiple choice quizzes (--for_class_quizzes): 20 questions with 5 answers each
    - Saved to: `<class_dir>/lectures_quizzes/<lesson>.quizzes.md`
  - Discussion/review questions (--for_class_recap): 3-6 open-ended questions
    - Saved to: `<class_dir>/lectures_recap/<lesson>.recap.md`

- By default, the output file is automatically formatted using `lint_text.py`
  with prettier. Use --no_lint to skip formatting.

# Usage Example

- Generate multiple-choice quizzes for DATA605 lesson 01.1:
> gen_quizzes.py --for_class_quizzes -i data605/01.1

- Generate discussion/review questions for MSML610 lesson 02.3:
> gen_quizzes.py --for_class_recap -i msml610/02.3

- Generate discussion/review questions for DATA605 lesson 01.2 without auto-formatting the output:
> gen_quizzes.py --for_class_recap -i data605/01.2 --no_lint

Import as:

import class_scripts.gen_quizzes as clgequiz
"""

import argparse
import logging

import class_scripts.common_utils as csccouti
import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser
import helpers.hprint as hprint
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################


# System prompt for quiz generation (multiple choice).
CLASS_QUIZZES_PROMPT = """
## Role
You are an experienced college professor designing assessment questions that
test conceptual understanding.

## Task Using the provided course material, generate **20 multiple-choice
questions**.

## Question Requirements
- Each question must:
  - Assess understanding, reasoning, or application — **not simple
    memorization**
  - Be clearly written and unambiguous
  - Every 5 questions include negative-logic questions (e.g., "Which of the
    following is NOT true?") where appropriate
  - Every 5 questions include answers like "none of the above" or "all of the
    above"
- Avoid trivial wording changes from the source text
- Do not quote long phrases directly from the material

## Answer Requirements
- Each question must have **exactly 5 options (A–E)**
- **Only one correct answer**
- The correct answer must be **bolded**
- Correct answers must be reasonably balanced across A, B, C, D, and E (no
  obvious patterns)

## Formatting Rules
- Do not number the questions
- Follow this exact structure for every question:
  ```
  # What is the meaning of "datafication"?
    - A) Collecting data for storage
    - B) **Turning all aspects of life into data**
    - C) Cleaning and refining existing data
    - D) Summarizing data in reports
    - E) Storing data in physical format
  ```

## Output Constraints
- Output ONLY the questions and answer choices
- Do NOT include explanations, commentary, separators, headings, or extra text
- Do NOT include page breaks or code fences
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
        help="Run lint_text.py with prettier action on output file (default: True)",
    )
    parser.add_argument(
        "--no_lint",
        action="store_false",
        dest="lint",
        help="Skip running lint_text.py on output file",
    )
    parser.add_argument(
        "--llm_cli_args",
        action="store",
        default=None,
        help="Additional options string passed through verbatim to llm_cli.py",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Parse and validate arguments.
    dir_arg, lesson_arg = csccouti.parse_lesson_spec(args.input)
    csccouti.validate_dir_lesson_args(dir_arg, lesson_arg)
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
        output_dir = f"{dir_arg}/lectures_quizzes"
        file_extension = ".quizzes.md"
        _LOG.info("Using CLASS_QUIZZES_PROMPT for multiple choice questions")
    else:
        prompt = CLASS_RECAP_PROMPT
        output_dir = f"{dir_arg}/lectures_recap"
        file_extension = ".recap.md"
        _LOG.info("Using CLASS_RECAP_PROMPT for discussion/review questions")
    # Get source and destination names.
    src_name = csccouti.get_source_name(dir_arg, lesson_arg)
    dst_name = csccouti.get_output_name(src_name, file_extension)
    # Build paths.
    input_file = f"{dir_arg}/lectures_source/{src_name}"
    output_file = f"{output_dir}/{dst_name}"
    # Ensure output directory exists.
    csccouti.ensure_dir_exists(output_dir)
    # Save the prompt to a temporary file.
    prompt_file = "tmp.gen_quizzes_prompt.txt"
    hio.to_file(prompt_file, prompt)
    _LOG.debug("Saved prompt to: %s", prompt_file)
    # Prepare command arguments.
    script_name = "llm_cli.py"
    input_arg = f"--input {input_file}"
    output_arg = f"--output {output_file}"
    prompt_file_arg = f"--system_prompt_file {prompt_file}"
    # Build the command.
    cmd_parts = [
        script_name,
        input_arg,
        output_arg,
        prompt_file_arg,
    ]
    # Add extra options if provided.
    if args.llm_cli_args:
        cmd_parts.append(args.llm_cli_args)
    cmd = " ".join(cmd_parts)
    _LOG.info("%s", hprint.color_highlight(f"> {cmd}", "green"))
    # Execute the command.
    hsystem.system(cmd)
    # Run linting if requested.
    if args.lint:
        _LOG.info("Running lint_text.py on output file: %s", output_file)
        # Prepare linting command.
        lint_action = "prettier"
        lint_cmd = f"lint_text.py -i {output_file} --action {lint_action}"
        _LOG.info("Executing: %s", lint_cmd)
        hsystem.system(lint_cmd)


if __name__ == "__main__":
    _main(_parse())
