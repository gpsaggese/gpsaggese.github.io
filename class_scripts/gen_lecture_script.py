#!/usr/bin/env python

"""
Generate lecture script from slides.

This script performs multiple steps:
1. Generate script using generate_slide_script.py
2. Generate intro using llm_cli.py
3. Generate outro using llm_cli.py
4. Combine intro, script, and outro
5. Lint the final script

Usage:
> gen_lecture_script.py data605 01.1
> gen_lecture_script.py msml610 02.3

Import as:

import class_scripts.gen_lecture_script as clgelesc
"""

import argparse
import logging
from pathlib import Path

import class_scripts.common_utils as clcomuut
import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################


# Prompt for generating intro.
INTRO_PROMPT = (
    "You are a college professor and you need to do an introduction in 50 word "
    "the content of the slides starting with In this lesson we will discuss"
)

# Prompt for generating outro.
OUTRO_PROMPT = (
    "You are a college professor and you need to summarize what was discussed "
    "in less than 50 word in the slides like In this lesson we have discussed"
)


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
        help="Additional options to pass to generate_slide_script.py",
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
    dst_name = clcomuut.get_output_name(src_name, ".script.txt")
    # Build paths.
    input_file = f"{args.dir}/lectures_source/{src_name}"
    output_dir = f"{args.dir}/lectures_script"
    output_file = f"{output_dir}/{dst_name}"
    # Ensure output directory exists.
    clcomuut.ensure_dir_exists(output_dir)
    # Step 1: Generate script.
    _LOG.info("Step 1: Generating script using generate_slide_script.py")
    cmd_parts = [
        "generate_slide_script.py",
        f"--in_file {input_file}",
        f"--out_file {output_file}",
        "--slides_per_group 3",
    ]
    if args.extra_opts:
        cmd_parts.extend(args.extra_opts)
    cmd = " ".join(cmd_parts)
    hsystem.system(cmd)
    # Step 2: Generate intro.
    _LOG.info("Step 2: Generating intro")
    intro_file = "tmp.gen_lecture_script.intro.txt"
    cmd = f"llm_cli.py -i {output_file} -p '{INTRO_PROMPT}' -o {intro_file}"
    hsystem.system(cmd)
    # Step 3: Generate outro.
    _LOG.info("Step 3: Generating outro")
    outro_file = "tmp.gen_lecture_script.outro.txt"
    cmd = f"llm_cli.py -i {output_file} -p '{OUTRO_PROMPT}' -o {outro_file}"
    hsystem.system(cmd)
    # Step 4: Combine intro, script, and outro.
    _LOG.info("Step 4: Combining intro, script, and outro")
    intro_text = hio.from_file(intro_file)
    script_text = hio.from_file(output_file)
    outro_text = hio.from_file(outro_file)
    # Build combined content.
    combined_parts = [
        "# Intro",
        intro_text.strip(),
        "",
        script_text.strip(),
        "",
        "# Outro",
        outro_text.strip(),
    ]
    combined_text = "\n".join(combined_parts)
    # Write to temporary file.
    tmp_file = "tmp.gen_lecture_script.combined.txt"
    hio.to_file(tmp_file, combined_text)
    # Move to final location.
    hio.to_file(output_file, combined_text)
    # Step 5: Lint the final script.
    _LOG.info("Step 5: Linting the final script")
    cmd = (
        f"lint_txt.py -i {output_file} -o {output_file} "
        f"--use_dockerized_prettier --action prettier --action frame_chapters"
    )
    hsystem.system(cmd)
    _LOG.info("Lecture script generated: %s", output_file)


if __name__ == "__main__":
    _main(_parse())
