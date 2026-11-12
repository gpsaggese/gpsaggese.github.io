#!/usr/bin/env python

"""
Generate lecture slides PDF.

This script:
- generates a PDF from lecture source files using `notes_to_pdf.py`. The PDF is
  built in the staging dir `{DIR}/lectures_pdf.tmp/`
- generates a PDF from a standalone `.md` file outside of any course
  `lectures_source` dir (e.g., the slides of a paper). The PDF is written next
  to the input file, since there is no staging dir; the `release` action is not
  available for these files
- uses the `release` action to copy the built PDF from
  `{DIR}/lectures_pdf.tmp/` to the published dir `{DIR}/lectures_pdf/` and
  compress it in place via `compress_pdf.py`; a plain `generate` run is left
  uncompressed so preview iterations stay fast.

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

- Generate the slides PDF from a markdown file outside of any course
  `lectures_source` dir, e.g., the slides of a paper. The PDF is written next
  to the input file, i.e., `papers/Causal_Analysis_for_Finance/notes.slides.pdf`:
> gen_slides.py -i papers/Causal_Analysis_for_Finance/notes.slides.md \
      --slides_engine typst

- Release the slides PDF already built for msml610 lesson 08.1, i.e., copy
  it from `msml610/lectures_pdf.tmp/` to `msml610/lectures_pdf/` and
  compress it (without rebuilding it):
> gen_slides.py -i msml610/08.1 --only_action release

- Generate and release the slides PDF in a single command:
> gen_slides.py -i msml610/08.1 --action release

- Generate the slides PDFs for multiple lessons in one run:
> gen_slides.py --files "msml610/08.1 msml610/08.2"

- Generate the slides PDFs for many lessons, continuing past any failures and
  printing a summary of what failed at the end:
> gen_slides.py --files "msml610/08.1 msml610/08.2 msml610/09.1" \
      --no_abort_on_error
"""

import argparse
import logging
import os
import shlex
import shutil

import helpers.hselect_action as hselacti
from helpers import hdbg, hgit, hparser, hprint, hsystem

import class_scripts.common_utils as csccouti

_LOG = logging.getLogger(__name__)

# `generate` must run before `release`: `hselacti.select_actions()` reorders
# whatever the user passes to match this order, so a single
# `--action release` (which adds to the `generate` default) always builds
# before releasing.
_VALID_ACTIONS = ["generate", "release"]
_DEFAULT_ACTIONS = ["generate"]

# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i",
        "--input",
        type=str,
        help="Lecture specification: 'data605/08.1', 'msml610/08.1', "
        "file path 'msml610/lectures_source/Lesson10.2-Name.smd', or a "
        "standalone markdown file outside of `lectures_source`, e.g., "
        "'papers/Causal_Analysis_for_Finance/notes.slides.md'",
    )
    input_group.add_argument(
        "-f",
        "--files",
        type=str,
        help="Space-separated list of lecture specifications to process in "
        "one run, e.g., 'msml610/08.1 msml610/08.2' (each element follows "
        "the same format as -i/--input)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Watch input file for changes and regenerate PDF on change",
    )
    hselacti.add_action_arg(parser, _VALID_ACTIONS, _DEFAULT_ACTIONS)
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
        "--no_abort_on_error",
        action="store_true",
        help="Continue processing the remaining targets if one fails "
        "(instead of aborting on the first failure) and print a summary of "
        "the failed targets at the end",
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


def _is_standalone_file(input_spec: str) -> bool:
    """
    Check if `input_spec` is a markdown file outside of a course lecture dir.

    A course lecture is specified as 'msml610/08.1' or as a path under
    `lectures_source`. Anything else ending in `.md` is a standalone file, e.g.,
    the slides of a paper. The existence of the file is checked later, so a typo
    in the path gets a clear "file not found" error instead of a confusing
    "Expected dir/lesson format" one.

    :param input_spec: input specification, e.g.,
        "papers/Causal_Analysis_for_Finance/notes.slides.md"
    :return: whether `input_spec` is a standalone markdown file
    """
    _LOG.debug(hprint.to_str("input_spec"))
    is_standalone = "lectures_source" not in input_spec and input_spec.endswith(
        ".md"
    )
    _LOG.debug("return=%s", is_standalone)
    return is_standalone


def _compress_pdf(pdf_file: str, log_level: str) -> None:
    """
    Compress a PDF in place via `compress_pdf.py`.

    :param pdf_file: path to the PDF to compress
    :param log_level: verbosity level (e.g., "DEBUG") to forward to
        `compress_pdf.py`
    """
    exec_file = hgit.find_file("compress_pdf.py")
    cmd = f"{exec_file} --input {pdf_file} -v {log_level}"
    hsystem.system(cmd, suppress_output=False)


def _release(dir_arg: str, dst_name: str, log_level: str) -> None:
    """
    Copy a built slides PDF from the staging dir to the published dir and
    compress it in place.

    :param dir_arg: course directory, e.g. "msml610"
    :param dst_name: PDF file name, e.g. "Lesson08.1-Causal_AI_intro.pdf"
    :param log_level: verbosity level forwarded to `compress_pdf.py`
    """
    src_file = f"{dir_arg}/lectures_pdf.tmp/{dst_name}"
    hdbg.dassert_file_exists(src_file)
    dst_dir = f"{dir_arg}/lectures_pdf"
    csccouti.ensure_dir_exists(dst_dir)
    dst_file = f"{dst_dir}/{dst_name}"
    shutil.copy2(src_file, dst_file)
    msg = f"Released: {src_file} -> {dst_file}"
    _LOG.info("%s", hprint.color_highlight(msg, "green"))
    # Compress the PDF only on release, so preview (`generate`) renders stay
    # fast and untouched.
    _compress_pdf(dst_file, log_level)


def _generate(
    input_file: str, output_file: str, args: argparse.Namespace
) -> None:
    """
    Build the slides PDF via `notes_to_pdf.py`.

    :param input_file: slides source file, e.g.
        "msml610/lectures_source/Lesson08.1-Causal_AI_intro.smd"
    :param output_file: PDF file to build, e.g.
        "msml610/lectures_pdf.tmp/Lesson08.1-Causal_AI_intro.pdf"
    :param args: parsed command-line arguments
    """
    # Ensure output directory exists. A bare file name (e.g., "notes.pdf")
    # has an empty dir name, which stands for the current dir.
    output_dir = os.path.dirname(output_file) or "."
    csccouti.ensure_dir_exists(output_dir)
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
        "-v",
        args.log_level,
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


def _process_standalone_file(
    input_file: str, args: argparse.Namespace, actions: list[str]
) -> None:
    """
    Generate the slides PDF for a markdown file outside of any course dir.

    There is no staging dir nor published dir, so the PDF is written next to
    the input file and the `release` action is not supported.

    :param input_file: markdown file, e.g.
        "papers/Causal_Analysis_for_Finance/notes.slides.md"
    :param args: parsed command-line arguments
    :param actions: actions to execute, e.g. ["generate"]
    """
    _LOG.debug(hprint.to_str("input_file actions"))
    hdbg.dassert_file_exists(input_file, "Standalone slides file not found")
    hdbg.dassert_not_in(
        "release",
        actions,
        "A standalone file has no published dir to release to: "
        "use a course lesson spec, e.g., 'msml610/08.1'",
    )
    # E.g., "papers/foo/notes.slides.md" -> "papers/foo/notes.slides.pdf".
    output_file = os.path.splitext(input_file)[0] + ".pdf"
    remaining_actions = list(actions)
    to_execute, remaining_actions = hselacti.mark_action(
        "generate", remaining_actions
    )
    if to_execute:
        _generate(input_file, output_file, args)
    hdbg.dassert_eq(
        len(remaining_actions or []),
        0,
        "There are unprocessed actions: %s",
        remaining_actions,
    )


def _process_course_lesson(
    input_spec: str, args: argparse.Namespace, actions: list[str]
) -> None:
    """
    Generate and/or release the slides PDF for a course lecture.

    :param input_spec: lecture specification, e.g. 'msml610/08.1' or a
        lecture source file path
    :param args: parsed command-line arguments
    :param actions: actions to execute, e.g. ["generate"], ["release"], or
        ["generate", "release"]
    """
    dir_arg, lesson_arg = csccouti.parse_lesson_spec(input_spec)
    csccouti.validate_dir_lesson_args(dir_arg, lesson_arg)
    # Get source and destination names.
    src_name = csccouti.get_source_name(dir_arg, lesson_arg)
    dst_name = csccouti.get_output_name(src_name, ".pdf")
    # `generate` always runs before `release` since `actions` is already
    # ordered to match `_VALID_ACTIONS`.
    remaining_actions = list(actions)
    to_execute, remaining_actions = hselacti.mark_action(
        "generate", remaining_actions
    )
    if to_execute:
        # `_generate()` handles `args.dry_run` itself, since it needs to
        # print the exact `notes_to_pdf.py` command it would run.
        input_file = f"{dir_arg}/lectures_source/{src_name}"
        output_file = f"{dir_arg}/lectures_pdf.tmp/{dst_name}"
        _generate(input_file, output_file, args)
    to_execute, remaining_actions = hselacti.mark_action(
        "release", remaining_actions
    )
    if to_execute:
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
        else:
            _release(dir_arg, dst_name, args.log_level)
    hdbg.dassert_eq(
        len(remaining_actions or []),
        0,
        "There are unprocessed actions: %s",
        remaining_actions,
    )


def _process_lesson_spec(
    input_spec: str, args: argparse.Namespace, actions: list[str]
) -> None:
    """
    Generate and/or release the slides PDF for a single input specification.

    :param input_spec: course lecture specification (e.g. 'msml610/08.1' or a
        lecture source file path) or a standalone markdown file outside of
        `lectures_source`
    :param args: parsed command-line arguments
    :param actions: actions to execute, e.g. ["generate"], ["release"], or
        ["generate", "release"]
    """
    if _is_standalone_file(input_spec):
        _process_standalone_file(input_spec, args, actions)
    else:
        _process_course_lesson(input_spec, args, actions)


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    actions = hselacti.select_actions(args, _VALID_ACTIONS, _DEFAULT_ACTIONS)
    _LOG.info(
        "%s",
        hselacti.actions_to_string(actions, _VALID_ACTIONS, add_frame=True),
    )
    input_specs = args.files.split() if args.files else [args.input]
    if args.daemon:
        hdbg.dassert_eq(
            len(input_specs),
            1,
            "`--daemon` only supports a single lecture, got: %s",
            input_specs,
        )
        hdbg.dassert_eq(
            actions,
            ["generate"],
            "`--daemon` only supports the 'generate' action, got: %s",
            actions,
        )
    csccouti.process_targets(
        input_specs,
        lambda input_spec: _process_lesson_spec(input_spec, args, actions),
        no_abort_on_error=args.no_abort_on_error,
    )


if __name__ == "__main__":
    _main(_parse())
