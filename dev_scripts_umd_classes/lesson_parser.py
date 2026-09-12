#!/usr/bin/env -S uv run

# /// script
# dependencies = []
# ///
"""
Parse Markdown lesson files and iterate through slides, headers, and comments.

# Usage Example

- Parse a lesson file and print all extracted items:
> lesson_parser.py -i /path/to/lesson.txt
"""

import argparse
import logging

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hmarkdown_slide_iterator as hmaslite
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)


def _parse() -> argparse.ArgumentParser:
    """
    Create argument parser for lesson parser script.

    :return: argument parser with lesson parsing options
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    hparser.add_input_file_arg(parser)
    parser.add_argument(
        "--item_type",
        type=str,
        choices=["slide", "header", "comment", "all"],
        default="all",
        help="Filter items by type (default: all)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for results (default: stdout)",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    """
    Parse lesson file and output extracted items.

    Reads a lesson file, iterates through items (slides, headers, comments),
    optionally filters by type, and outputs results to file or stdout.

    :param parser: argument parser with input_file, item_type, output_file options
    """
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    input_file = args.input
    item_type = args.item_type
    output_file = args.output_file
    hdbg.dassert_file_exists(
        input_file,
        "Lesson file to parse does not exist at: '%s'",
        input_file,
    )
    # Iterate through items in the lesson file.
    items = hmaslite.read_lesson_file(input_file)
    output_lines = []
    item_count = 0
    for item in items:
        if item_type != "all" and item["type"] != item_type:
            continue
        item_count += 1
        output_lines.append(
            f"Item #{item_count}: type={item['type']}, line={item['line_number']}"
        )
        output_lines.append(f"Content ({len(item['content'])} lines):")
        for line in item["content"][:5]:
            output_lines.append(f"  {line}")
        if len(item["content"]) > 5:
            output_lines.append(f"  ... ({len(item['content']) - 5} more lines)")
        output_lines.append("")
    # Write output.
    output_text = "\n".join(output_lines)
    if output_file:
        hio.to_file(output_file, output_text)
        _LOG.info("Wrote output to '%s'", output_file)
    else:
        _LOG.info("%s", output_text)


if __name__ == "__main__":
    parser = _parse()
    _main(parser)
