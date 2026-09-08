#!/usr/bin/env python

"""
Render a book chapter Typst file to PDF.

This script:
- Compiles the book chapter `.typ` file for a lesson to PDF via
  `run_typst.py`, writing the PDF to the staging dir `{DIR}/book.tmp/`. This
  way a preview render never touches the PDF checked into `{DIR}/book/`,
  mirroring how `gen_slides.py` stages a slide PDF in
  `{DIR}/lectures_pdf.tmp/`
- Uses the `release` action to copy the built PDF from `{DIR}/book.tmp/` to
  the published dir `{DIR}/book/` and compress it in place via
  `compress_pdf.py`; a plain `generate` render is left uncompressed so
  preview iterations stay fast

# Usage Example

- Render the book chapter for msml610 lesson 03.2:
> render_book_chapter.py -i msml610/03.2

- Render the book chapter for data605 lesson 01.1:
> render_book_chapter.py -i data605/01.1

- Render by specifying the book chapter file path directly:
> render_book_chapter.py -i msml610/book/Lesson03.2-Name.typ

- Watch the book chapter file for changes and re-render on save:
> render_book_chapter.py -i msml610/03.2 --daemon

- Render without opening the PDF viewer:
> render_book_chapter.py -i msml610/03.2 --run_typst_args="--skip_action open_pdf"

- Release the book chapter PDF already built for msml610 lesson 03.2, i.e.,
  copy it from `msml610/book.tmp/` to `msml610/book/` and compress it:
> render_book_chapter.py -i msml610/03.2 --action release

- Render the book chapter PDFs for multiple lessons in one run:
> render_book_chapter.py --files "msml610/03.1 msml610/03.2"

Import as:

import class_scripts.render_book_chapter as clrenboch
"""

import argparse
import logging
import os
import shlex
import shutil
from typing import Tuple

import class_scripts.common_utils as csccouti
import helpers.hdbg as hdbg
import helpers.hgit as hgit
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
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i",
        "--input",
        type=str,
        help="Lecture specification: 'data605/08.1', 'msml610/08.1', "
        "or book chapter file path 'msml610/book/Lesson10.2-Name.typ'",
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
        help="Watch the book chapter file for changes and re-render on change",
    )
    parser.add_argument(
        "--action",
        action="store",
        default="generate",
        choices=["generate", "release"],
        help="'generate' compiles the book chapter PDF in the staging dir "
        "book.tmp (default); 'release' copies the built PDF from book.tmp "
        "to book",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the commands that would be executed without running them",
    )
    parser.add_argument(
        "--no_abort_on_warnings",
        action="store_true",
        help="Don't assert if `typst compile` emits warnings (forwarded to "
        "run_typst.py)",
    )
    parser.add_argument(
        "--run_typst_args",
        action="store",
        default=None,
        help="Additional options string passed through verbatim to "
        "run_typst.py, e.g., '--skip_action open_pdf'",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _resolve_book_chapter_file(input_spec: str) -> Tuple[str, str]:
    """
    Resolve a lecture specification to its course dir and book chapter file.

    :param input_spec: lecture specification, e.g. 'msml610/03.2', or a
        direct path to a book chapter `.typ` file, e.g.
        'msml610/book/Lesson03.2-Name.typ'
    :return: tuple of (course directory, path to the book chapter `.typ`
        file)
    """
    if input_spec.endswith(".typ"):
        hdbg.dassert_file_exists(input_spec)
        dir_arg = input_spec.split(os.sep)[0]
        hdbg.dassert_dir_exists(dir_arg)
        return dir_arg, input_spec
    dir_arg, lesson_arg = csccouti.parse_lesson_spec(input_spec)
    csccouti.validate_dir_lesson_args(dir_arg, lesson_arg)
    typ_file = str(
        csccouti.find_lecture_file(
            dir_arg, lesson_arg, sub_dir="book", extension="typ"
        )
    )
    return dir_arg, typ_file


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


def _release(dir_arg: str, basename: str, log_level: str) -> None:
    """
    Copy a built book chapter PDF from the staging dir to the published dir
    and compress it in place.

    :param dir_arg: course directory, e.g. "msml610"
    :param basename: PDF file name without extension, e.g.
        "Lesson03.2-Name"
    :param log_level: verbosity level forwarded to `compress_pdf.py`
    """
    src_file = f"{dir_arg}/book.tmp/{basename}.pdf"
    hdbg.dassert_file_exists(src_file)
    dst_dir = f"{dir_arg}/book"
    csccouti.ensure_dir_exists(dst_dir)
    dst_file = f"{dst_dir}/{basename}.pdf"
    shutil.copy2(src_file, dst_file)
    msg = f"Released: {src_file} -> {dst_file}"
    _LOG.info("%s", hprint.color_highlight(msg, "green"))
    # Compress the PDF only on release, so preview (`generate`) renders stay
    # fast and untouched.
    _compress_pdf(dst_file, log_level)


def _process_lesson_spec(input_spec: str, args: argparse.Namespace) -> None:
    """
    Render (or release) the book chapter PDF for a single lecture
    specification.

    :param input_spec: lecture specification, e.g. 'msml610/03.2' or a book
        chapter file path
    :param args: parsed command-line arguments
    """
    dir_arg, typ_file = _resolve_book_chapter_file(input_spec)
    basename = os.path.splitext(os.path.basename(typ_file))[0]
    if args.action == "release":
        if args.dry_run:
            src_file = f"{dir_arg}/book.tmp/{basename}.pdf"
            dst_file = f"{dir_arg}/book/{basename}.pdf"
            _LOG.info(
                "%s",
                hprint.color_highlight(
                    f"[dry run] Would release: {src_file} -> {dst_file}",
                    "green",
                ),
            )
            return
        _release(dir_arg, basename, args.log_level)
        return
    # Stage the rendered PDF in `book.tmp/`, next to the published `book/`
    # dir, so a preview render never touches the tracked PDF.
    out_dir = f"{dir_arg}/book.tmp"
    output_file = f"{out_dir}/{basename}.pdf"
    csccouti.ensure_dir_exists(out_dir)
    # Build the command, quoting all arguments to preserve special characters.
    run_typst_exec = hgit.find_file("run_typst.py")
    cmd_parts = [
        run_typst_exec,
        f"--input={typ_file}",
        f"--output={output_file}",
    ]
    if args.no_abort_on_warnings:
        cmd_parts.append("--no_abort_on_warnings")
    if args.daemon:
        cmd_parts.append("--daemon")
    quoted_parts = [shlex.quote(part) for part in cmd_parts]
    cmd = " ".join(quoted_parts)
    # Append the extra options verbatim (i.e., not quoted) so a caller can
    # pass multiple options to `run_typst.py` in a single string.
    if args.run_typst_args:
        cmd += f" {args.run_typst_args}"
    if args.dry_run:
        _LOG.info(
            "%s", hprint.color_highlight(f"[dry run] > {cmd}", "green")
        )
        return
    _LOG.info("%s", hprint.color_highlight(f"> {cmd}", "green"))
    hsystem.system(cmd, suppress_output=False)


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    input_specs = args.files.split() if args.files else [args.input]
    if args.daemon:
        hdbg.dassert_eq(
            len(input_specs),
            1,
            "`--daemon` only supports a single lecture, got: %s",
            input_specs,
        )
    for input_spec in input_specs:
        _process_lesson_spec(input_spec, args)


if __name__ == "__main__":
    _main(_parse())
