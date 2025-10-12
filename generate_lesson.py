#!/usr/bin/env python

"""
Generate PDF slides and/or a reading scripts for lecture materials.

# Generate PDFs for specific lectures:
> generate_lesson.py --lectures 01.1 --class data605 --target pdf

# Generate scripts for multiple lectures:
> generate_lesson.py --lectures 01*:02* --class data605 --target script

# Generate both PDFs and scripts:
> generate_lesson.py --lectures 01* --class msml610 --target pdf,script

# Generate specific slides from a lecture:
> generate_lesson.py --lectures 01.1 --limit 1:3 --class data605 --target pdf

Import as:

import generate_lesson as genlssn
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
        "--target",
        action="store",
        required=True,
        help="Target output types: 'pdf', 'script', or 'pdf,script'",
    )
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


def _validate_single_lecture_file(files: List[Tuple[str, str]]) -> None:
    """
    Validate that exactly one lecture file was found.

    Used when processing a single lecture with --limit option.

    :param files: list of file tuples
    """
    hdbg.dassert_eq(len(files), 1, "Need exactly one file when using --limit")


def _parse_target_arg(target_arg: str) -> List[str]:
    """
    Parse the target argument into a list of target types.

    :param target_arg: target argument from command line (e.g., 'pdf', 'pdf,script')
    :return: list of target types
    """
    targets = [t.strip() for t in target_arg.split(",")]
    # Validate targets.
    for target in targets:
        hdbg.dassert_in(target, ["pdf", "script"], "Invalid target:", target)
    _LOG.debug("Parsed targets: %s", targets)
    return targets


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
    hsystem.system(" ".join(cmd))


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
    hsystem.system(" ".join(cmd))
    # Step 2: Remove 'Transition: ' prefix.
    _LOG.debug("Removing 'Transition: ' prefix from %s", output_path)
    hsystem.system(f"perl -pi -e 's/^Transition: //g' {output_path}")
    # Step 3: Lint the output.
    _LOG.debug("Linting %s", output_path)
    hsystem.system(f"lint_txt.py -i {output_path} --use_dockerized_prettier")


def _process_lecture_file(
    class_dir: str,
    source_path: str,
    source_name: str,
    targets: List[str],
    *,
    limit: str = None,
) -> None:
    """
    Process a single lecture file for specified targets.

    :param class_dir: class directory (data605 or msml610)
    :param source_path: path to source .txt file
    :param source_name: name of source file
    :param targets: list of targets to generate ('pdf' and/or 'script')
    :param limit: optional slide range to process
    """
    _LOG.info("Processing file: %s", source_path)
    # Process each target.
    for target in targets:
        if target == "pdf":
            _generate_pdf(class_dir, source_path, source_name, limit=limit)
        elif target == "script":
            if limit:
                _LOG.warning("Ignoring --limit for script generation")
            _generate_script(class_dir, source_path, source_name)
        else:
            hdbg.dfatal("Unknown target: %s", target)


def _main(parser: argparse.ArgumentParser) -> None:
    """
    Main execution function.

    Orchestrates the lesson generation process:
    1. Parse and validate arguments
    2. Find matching lecture files
    3. Process each file for specified targets

    :param parser: configured argument parser
    """
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Parse arguments.
    patterns = _parse_lecture_patterns(args.lectures)
    targets = _parse_target_arg(args.target)
    # Find matching lecture files.
    files = _find_lecture_files(args.class_name, patterns)
    hdbg.dassert_lt(0, len(files), "No lecture files found for patterns:", patterns)
    # Validate if --limit is specified.
    if args.limit:
        _validate_single_lecture_file(files)
    # Process each file.
    for source_path, source_name in files:
        _process_lecture_file(
            args.class_name, source_path, source_name, targets, limit=args.limit
        )
    _LOG.info("All files processed successfully")


if __name__ == "__main__":
    _main(_parse())
