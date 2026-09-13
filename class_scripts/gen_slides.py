#!/usr/bin/env python

"""
Generate lecture slides PDF.

This script generates a PDF from lecture source files using notes_to_pdf.py.
The PDF is built in the staging dir `{DIR}/lectures_pdf.tmp/`. Use the
`release` action to copy the built PDF from `{DIR}/lectures_pdf.tmp/` to the
published dir `{DIR}/lectures_pdf/`.

# Usage Example

- Generate the slides PDF for msml610 lesson 08.1:
> gen_slides.py -i msml610/08.1

- Generate the slides PDF for data605 lesson 01.1:
> gen_slides.py -i data605/01.1

- Generate the slides PDF for msml610 lesson 08.1, skipping the
  cleanup_before action:
> gen_slides.py -i msml610/08.1 --notes_to_pdf_args="--skip_action cleanup_before"

- Generate the slides PDF by specifying the lecture source file path
  directly:
> gen_slides.py -i msml610/lectures_source/Lesson10.2-Causal_Discovery.smd

- Release the slides PDF already built for msml610 lesson 08.1, i.e., copy
  it from `msml610/lectures_pdf.tmp/` to `msml610/lectures_pdf/`:
> gen_slides.py -i msml610/08.1 --action release
"""

import argparse
import logging
import shlex
import shutil

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
        "--daemon",
        action="store_true",
        help="Watch input file for changes and regenerate PDF on change",
    )
    parser.add_argument(
        "--action",
        action="store",
        default="generate",
        choices=["generate", "release"],
        help="'generate' builds the slides PDF in the staging dir "
        "lectures_pdf.tmp (default); 'release' copies the built PDF from "
        "lectures_pdf.tmp to lectures_pdf",
    )
    parser.add_argument(
        "--slides_engine",
        action="store",
        default=None,
        choices=["beamer", "typst"],
        help="Engine used to render slides: 'beamer' (default) or 'typst'",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the commands that would be executed without running them",
    )
    parser.add_argument(
        "--notes_to_pdf_args",
        action="store",
        default=None,
        help="Additional options string passed through verbatim to "
        "notes_to_pdf.py, e.g., '--skip_action cleanup_before'",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _extra_opts_mention_open_pdf(notes_to_pdf_args: str) -> bool:
    """
    Check if `notes_to_pdf_args` already specifies an action for "open_pdf".

    E.g., `--skip_action open_pdf`, `--skip_action=open_pdf`, or
    `--action=open_pdf`. If so, the caller is managing that action
    explicitly, so we should not also force our own `--action=open_pdf`,
    which would make `notes_to_pdf.py` fail with an assertion since the
    same action can't be in both `--action` and `--skip_action`.

    :param notes_to_pdf_args: extra options string passed through to
        `notes_to_pdf.py`
    :return: whether "open_pdf" already appears in `notes_to_pdf_args`
    """
    return bool(notes_to_pdf_args) and "open_pdf" in notes_to_pdf_args


def _release(dir_arg: str, dst_name: str) -> None:
    """
    Copy a built slides PDF from the staging dir to the published dir.

    :param dir_arg: course directory, e.g. "msml610"
    :param dst_name: PDF file name, e.g. "Lesson08.1-Causal_AI_intro.pdf"
    """
    src_file = f"{dir_arg}/lectures_pdf.tmp/{dst_name}"
    hdbg.dassert_file_exists(src_file)
    dst_dir = f"{dir_arg}/lectures_pdf"
    csccouti.ensure_dir_exists(dst_dir)
    dst_file = f"{dst_dir}/{dst_name}"
    shutil.copy2(src_file, dst_file)
    msg = f"Released: {src_file} -> {dst_file}"
    _LOG.info("%s", hprint.color_highlight(msg, "green"))


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    dir_arg, lesson_arg = csccouti.parse_lesson_spec(args.input)
    csccouti.validate_dir_lesson_args(dir_arg, lesson_arg)
    # Get source and destination names.
    src_name = csccouti.get_source_name(dir_arg, lesson_arg)
    dst_name = csccouti.get_output_name(src_name, ".pdf")
    if args.action == "release":
        if args.dry_run:
            src_file = f"{dir_arg}/lectures_pdf.tmp/{dst_name}"
            dst_file = f"{dir_arg}/lectures_pdf/{dst_name}"
            _LOG.info(
                "%s",
                hprint.color_highlight(
                    f"[dry run] Would release: {src_file} -> {dst_file}",
                    "green",
                ),
            )
            return
        _release(dir_arg, dst_name)
        return
    # Build paths.
    input_file = f"{dir_arg}/lectures_source/{src_name}"
    output_file = f"{dir_arg}/lectures_pdf.tmp/{dst_name}"
    # Ensure output directory exists.
    csccouti.ensure_dir_exists(f"{dir_arg}/lectures_pdf.tmp")
    # Build the command with debug options.
    cmd_parts = [
        "notes_to_pdf.py",
        f"--input={input_file}",
        f"--output={output_file}",
        "--type=slides",
        "--toc_type=navigation",
        "--debug_on_error",
        "--skip_action=cleanup_before",
        "--skip_action=cleanup_after",
    ]
    # Add slides engine if specified.
    if args.slides_engine:
        cmd_parts.append(f"--slides_engine={args.slides_engine}")
    if not args.daemon and not _extra_opts_mention_open_pdf(
        args.notes_to_pdf_args
    ):
        # `notes_to_pdf.py`'s default actions don't include "open_pdf", so
        # add it explicitly to open the PDF after a one-shot generation,
        # unless the caller already specified how to handle that action
        # (e.g. `--skip_action open_pdf` to build without opening a viewer).
        cmd_parts.append("--action=open_pdf")
    # Prepare command by quoting all arguments to preserve special characters.
    quoted_parts = [shlex.quote(part) for part in cmd_parts]
    cmd = " ".join(quoted_parts)
    # Append the extra options verbatim (i.e., not quoted) so a caller can
    # pass multiple options to `notes_to_pdf.py` in a single string.
    if args.notes_to_pdf_args:
        cmd += f" {args.notes_to_pdf_args}"
    if args.dry_run:
        preview_cmd = cmd + (" --daemon" if args.daemon else "")
        _LOG.info(
            "%s",
            hprint.color_highlight(f"[dry run] > {preview_cmd}", "green"),
        )
        return
    if args.daemon:
        # `notes_to_pdf.py`'s default actions don't include "open_pdf", so
        # build once upfront and open the PDF; then hand off to its own
        # `--daemon` watch loop, which regenerates on change without
        # reopening the viewer (it skips "open_pdf" on watch runs since the
        # viewer auto-reloads).
        initial_cmd = cmd + " --action=open_pdf"
        _LOG.info("%s", hprint.color_highlight(f"> {initial_cmd}", "green"))
        hsystem.system(initial_cmd, suppress_output=False)
        cmd += " --daemon"
    # Execute the command.
    _LOG.info("%s", hprint.color_highlight(f"> {cmd}", "green"))
    hsystem.system(cmd, suppress_output=False)


if __name__ == "__main__":
    _main(_parse())
