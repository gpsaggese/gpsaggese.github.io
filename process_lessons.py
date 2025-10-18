#!/usr/bin/env python

"""
Generate PDF slides and/or a reading scripts for lecture materials.

# Generate PDFs for specific lectures:
> process_lessons.py --lectures 01.1 --class data605 --action pdf

# Generate scripts for multiple lectures:
> process_lessons.py --lectures 01*:02* --class data605 --action script

# Generate both PDFs and scripts:
> process_lessons.py --lectures 01* --class msml610 --action pdf --action script

# Generate all default actions:
> process_lessons.py --lectures 01* --class msml610

# Generate specific slides from a lecture:
> process_lessons.py --lectures 01.1 --limit 1:3 --class data605 --action pdf

Import as:

import process_lessons as prlssn
"""

import argparse
import glob
import logging
import os
from typing import List, Tuple

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)

# #############################################################################

_VALID_ACTIONS = ["pdf", "script"]
_DEFAULT_ACTIONS = ["pdf"]

# #############################################################################


def _parse() -> argparse.ArgumentParser:
    """
    Parse command line arguments.

    :return: configured argument parser
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lectures",
        action="store",
        required=True,
        help="Lecture pattern(s) to process (e.g., '01*', '01.1', '01*:03*')",
    )
    parser.add_argument(
        "--limit",
        action="store",
        help="Slide range to process when single lecture specified (e.g., '1:3')",
    )
    parser.add_argument(
        "--class",
        dest="class_name",
        action="store",
        required=True,
        choices=["data605", "msml610"],
        help="Class directory name",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the commands that would be executed without running them",
    )
    hparser.add_action_arg(parser, _VALID_ACTIONS, _DEFAULT_ACTIONS)
    hparser.add_verbosity_arg(parser)
    return parser


def _parse_lecture_patterns(lectures_arg: str) -> List[str]:
    """
    Parse the lectures argument into a list of patterns.

    The lectures argument can be:
    - A single pattern: '01.1' or '01*'
    - Multiple patterns separated by colon: '01*:02*:03.1'

    :param lectures_arg: lectures argument from command line
    :return: list of lecture patterns
    """
    patterns = lectures_arg.split(":")
    _LOG.debug("Parsed lecture patterns: %s", patterns)
    return patterns


def _find_lecture_files(
    class_dir: str, patterns: List[str]
) -> List[Tuple[str, str]]:
    """
    Find all lecture source files matching the given patterns.

    :param class_dir: class directory (data605 or msml610)
    :param patterns: list of lecture patterns to match
    :return: list of tuples (source_path, source_name)
    """
    lectures_source_dir = os.path.join(class_dir, "lectures_source")
    hdbg.dassert(
        os.path.isdir(lectures_source_dir),
        "Lectures source directory does not exist:",
        lectures_source_dir,
    )
    # Find all matching files.
    all_files = []
    for pattern in patterns:
        pattern_path = os.path.join(lectures_source_dir, f"Lesson{pattern}*")
        matched_files = sorted(glob.glob(pattern_path))
        _LOG.debug("Pattern '%s' matched %d files", pattern, len(matched_files))
        all_files.extend(matched_files)
    # Convert to tuples of (path, basename).
    result = [(f, os.path.basename(f)) for f in all_files]
    _LOG.info("Found %d lecture files", len(result))
    return result


def _generate_pdf(
    class_dir: str,
    source_path: str,
    source_name: str,
    *,
    limit: str = None,
    skip_action: str = "open",
) -> None:
    """
    Generate PDF slides from a lecture source file.

    Calls notes_to_pdf.py with appropriate arguments to convert a text source
    file into PDF slides.

    :param class_dir: class directory (data605 or msml610)
    :param source_path: path to source .txt file
    :param source_name: name of source file
    :param limit: optional slide range to process (e.g., '1:3')
    :param skip_action: action to skip (default: 'open')
    """
    # Compute output path.
    dst_name = source_name.replace(".txt", ".pdf")
    lectures_dir = os.path.join(class_dir, "lectures")
    hio.create_dir(lectures_dir, incremental=True)
    output_path = os.path.join(lectures_dir, dst_name)
    # Build command.
    _LOG.info("Processing %s -> %s", source_name, dst_name)
    cmd = [
        "notes_to_pdf.py",
        "--input",
        source_path,
        "--output",
        output_path,
        "--type",
        "slides",
        "--toc_type",
        "navigation",
        "--skip_action",
        skip_action,
        "--debug_on_error",
    ]
    if limit:
        cmd.extend(["--limit", limit])
    # Execute command.
    cmd_str = " ".join(cmd)
    _LOG.info("Executing: %s", cmd_str)
    hsystem.system(cmd_str, suppress_output=False)


def _generate_script(class_dir: str, source_path: str, source_name: str) -> None:
    """
    Generate script from a lecture source file.

    Performs the following steps:
    1. Calls generate_slide_script.py to create the script
    2. Removes 'Transition: ' prefix using perl
    3. Lints the output using lint_txt.py

    :param class_dir: class directory (data605 or msml610)
    :param source_path: path to source .txt file
    :param source_name: name of source file
    """
    # Compute output path.
    dst_name = source_name.replace(".txt", ".script.txt")
    lectures_script_dir = os.path.join(class_dir, "lectures_script")
    hio.create_dir(lectures_script_dir, incremental=True)
    output_path = os.path.join(lectures_script_dir, dst_name)
    # Step 1: Generate slide script.
    _LOG.info("Generating script for %s -> %s", source_name, dst_name)
    cmd = [
        "generate_slide_script.py",
        "--in_file",
        source_path,
        "--out_file",
        output_path,
        "--slides_per_group",
        "3",
    ]
    cmd_str = " ".join(cmd)
    _LOG.info("Executing: %s", cmd_str)
    hsystem.system(cmd_str, suppress_output=False)
    # Step 2: Remove 'Transition: ' prefix.
    cmd_str = f"perl -pi -e 's/^Transition: //g' {output_path}"
    _LOG.info("Executing: %s", cmd_str)
    hsystem.system(cmd_str, suppress_output=False)
    # Step 3: Lint the output.
    cmd_str = f"lint_txt.py -i {output_path} --use_dockerized_prettier"
    _LOG.info("Executing: %s", cmd_str)
    hsystem.system(cmd_str, suppress_output=False)


def _process_lecture_file(
    class_dir: str,
    source_path: str,
    source_name: str,
    actions: List[str],
    *,
    limit: str = None,
) -> None:
    """
    Process a single lecture file for specified actions.

    :param class_dir: class directory (data605 or msml610)
    :param source_path: path to source .txt file
    :param source_name: name of source file
    :param actions: list of actions to execute ('pdf' and/or 'script')
    :param limit: optional slide range to process
    """
    _LOG.info("Processing file: %s", source_path)
    # Process each action.
    for action in actions:
        if action == "pdf":
            _generate_pdf(class_dir, source_path, source_name, limit=limit)
        elif action == "script":
            if limit:
                _LOG.warning("Ignoring --limit for script generation")
            _generate_script(class_dir, source_path, source_name)
        else:
            hdbg.dfatal("Unknown action: %s", action)


def _main(parser: argparse.ArgumentParser) -> None:
    """
    Main execution function.

    Orchestrates the lesson generation process:
    1. Parse and validate arguments
    2. Find matching lecture files
    3. Process each file for specified actions

    :param parser: configured argument parser
    """
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Parse arguments.
    patterns = _parse_lecture_patterns(args.lectures)
    actions = hparser.select_actions(args, _VALID_ACTIONS, _DEFAULT_ACTIONS)
    _LOG.info("Selected actions: %s", actions)
    # Find matching lecture files.
    files = _find_lecture_files(args.class_name, patterns)
    hdbg.dassert_lt(0, len(files), "No lecture files found for patterns:", patterns)
    # Validate if --limit is specified.
    if args.limit:
        hdbg.dassert_eq(len(files), 1, "Need exactly one file when using --limit")
    # Print the commands that would be executed without running them.
    if args.dry_run:
        _LOG.info("Dry run mode enabled. Will print the commands that would be executed without running them.")
        for source_path, source_name in files:
            _LOG.info("Processing file: %s", source_path)
        return
    # Process each file.
    for source_path, source_name in files:
        _process_lecture_file(
            args.class_name, source_path, source_name, actions, limit=args.limit
        )
    _LOG.info("All files processed successfully")


if __name__ == "__main__":
    _main(_parse())
