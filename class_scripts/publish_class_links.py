#!/usr/bin/env python

r"""
Generate an HTML page with links to lesson materials for a class or book.

For each lesson source file found in `{DIR}/lectures_source/`, builds an HTML
page linking to that lesson's generated artifacts:
- Slides PDF:
    - `{DIR}/lectures_pdf/{LESSON}.pdf`
- Lecture commentary:
    - `{DIR}/lectures_commentary/{LESSON}.book_chapter.html`
    - `{DIR}/lectures_commentary/{LESSON}.book_chapter.pdf`
- Lesson recap:
    - `{DIR}/lectures_recap/{LESSON}.recap.md`

Each existing artifact is linked to its GitHub URL (via `to_github.py`) so the
page works when shared outside of a local checkout.

By default the script stops with an error as soon as an expected artifact is
missing. Pass `--do_not_fail_on_warnings` to instead log a warning, list the
lesson without the missing link, and keep going.

# Usage Example

- Generate the links page for a class, failing on the first missing artifact:
> publish_class_links.py \
        --dir data605 \
        --out_file data605/class_links.html

- Generate the links page while tolerating missing artifacts, linking to the
  `master` branch instead of the current branch:
> publish_class_links.py \
        --dir msml610 \
        --out_file /tmp/links.html \
        --do_not_fail_on_warnings --use_master

- Generate the links page and open it in the default browser:
> publish_class_links.py \
        --dir data605 \
        --out_file data605/class_links.html \
        --open_html

Import as:

import class_scripts.publish_class_links as cspucli
"""

import argparse
import glob
import logging
import os
from typing import List, NamedTuple

import dev_scripts_helpers.github.to_github as dshgtogi
import helpers.hdbg as hdbg
import helpers.hgit as hgit
import helpers.hio as hio
import helpers.hparser as hparser
import helpers.hprint as hprint
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

_LECTURES_SOURCE_SUBDIR = "lectures_source"
_LECTURES_PDF_SUBDIR = "lectures_pdf"
_LECTURES_COMMENTARY_SUBDIR = "lectures_commentary"
_LECTURES_RECAP_SUBDIR = "lectures_recap"

_LABEL_SLIDES_PDF = "Slides (PDF)"
_LABEL_COMMENTARY_HTML = "Commentary (HTML)"
_LABEL_COMMENTARY_PDF = "Commentary (PDF)"
_LABEL_RECAP = "Recap (TXT)"


# #############################################################################
# LessonArtifact
# #############################################################################


class LessonArtifact(NamedTuple):
    """
    Store one expected lesson artifact and whether it exists on disk.
    """

    label: str
    file_path: str
    exists: bool


# #############################################################################
# LessonPage
# #############################################################################


class LessonPage(NamedTuple):
    """
    Store a lesson name and its list of `LessonArtifact` in display order.
    """

    lesson_name: str
    artifacts: List[LessonArtifact]


# #############################################################################
# Helper functions
# #############################################################################


def _to_media_github_url(github_blob_url: str) -> str:
    """
    Convert a GitHub blob URL to its LFS-aware `media.githubusercontent.com`
    equivalent.

    GitHub's raw-content endpoints (`raw.githubusercontent.com`, and anything
    built on top of them like `htmlpreview.github.io`) serve Git-LFS-tracked
    files as their small pointer text instead of the real content.
    `media.githubusercontent.com` is the GitHub endpoint that resolves LFS
    pointers to the actual file content, so links to `*.html` files (which
    `.gitattributes` tracks via LFS) must go through it instead.

    :param github_blob_url: GitHub URL of the form
        `https://github.com/<owner>/<repo>/blob/<branch>/<path>`
    :return: equivalent `https://media.githubusercontent.com/media/<owner>/
        <repo>/<branch>/<path>` URL
    """
    media_url = github_blob_url.replace(
        "https://github.com/", "https://media.githubusercontent.com/media/"
    ).replace("/blob/", "/")
    return media_url


def _get_short_label(label: str) -> str:
    """
    Strip the trailing "(TYPE)" qualifier from an artifact label.

    The full label (e.g., `Commentary (HTML)`) is used in the table header,
    where the artifact type needs to be called out; the short label (e.g.,
    `Commentary`) is used in the table cells, which are already grouped by
    column.

    :param label: full artifact label
    :return: label without a trailing "(TYPE)" qualifier
    """
    return label.split(" (")[0]


def _find_lesson_names(dir_: str) -> List[str]:
    """
    Find the sorted lesson names from the lecture source files.

    :param dir_: course/book directory (e.g., `data605`)
    :return: sorted lesson names
        - Example: `['Lesson01.1-Intro', 'Lesson01.2-Big_Data']`
    """
    lectures_source_dir = os.path.join(dir_, _LECTURES_SOURCE_SUBDIR)
    hdbg.dassert_dir_exists(lectures_source_dir)
    pattern = os.path.join(lectures_source_dir, "Lesson*.smd")
    source_files = sorted(glob.glob(pattern))
    hdbg.dassert_lt(
        0,
        len(source_files),
        "No lecture source files found matching '%s'",
        pattern,
    )
    lesson_names = [
        os.path.splitext(os.path.basename(f))[0] for f in source_files
    ]
    _LOG.debug(hprint.to_str("lesson_names"))
    return lesson_names


def _get_lesson_artifacts(
    dir_: str, lesson_name: str, *, do_not_fail_on_warnings: bool
) -> List[LessonArtifact]:
    """
    Build and check the expected artifact paths for a lesson.

    :param dir_: course/book directory
    :param lesson_name: lesson name (e.g., `Lesson01.1-Intro`)
    :param do_not_fail_on_warnings: if `True`, log missing artifacts as
        warnings and keep going; if `False`, stop on the first missing
        artifact
    :return: one `LessonArtifact` per expected artifact, in display order
        (slides PDF, commentary HTML, commentary PDF, recap)
    """
    artifact_specs = [
        (
            _LABEL_SLIDES_PDF,
            os.path.join(dir_, _LECTURES_PDF_SUBDIR, f"{lesson_name}.pdf"),
        ),
        (
            _LABEL_COMMENTARY_HTML,
            os.path.join(
                dir_,
                _LECTURES_COMMENTARY_SUBDIR,
                f"{lesson_name}.book_chapter.html",
            ),
        ),
        (
            _LABEL_COMMENTARY_PDF,
            os.path.join(
                dir_,
                _LECTURES_COMMENTARY_SUBDIR,
                f"{lesson_name}.book_chapter.pdf",
            ),
        ),
        (
            _LABEL_RECAP,
            os.path.join(
                dir_, _LECTURES_RECAP_SUBDIR, f"{lesson_name}.recap.md"
            ),
        ),
    ]
    artifacts = []
    for label, file_path in artifact_specs:
        exists = os.path.isfile(file_path)
        if not exists:
            # Report the missing artifact: stop unless the caller asked to
            # only warn and keep going.
            msg = f"Lesson '{lesson_name}': missing '{label}' at '{file_path}'"
            if do_not_fail_on_warnings:
                _LOG.warning(msg)
            else:
                raise FileNotFoundError(msg)
        artifacts.append(LessonArtifact(label, file_path, exists))
    return artifacts


# #############################################################################
# HTML generation
# #############################################################################


def _generate_html_page(
    dir_: str, lessons: List[LessonPage], *, use_master: bool
) -> str:
    """
    Generate the HTML page content listing artifact links for each lesson.

    Each existing artifact is linked to its GitHub URL (via
    `to_github.py`'s `_get_github_url()`) so that the page works when
    shared outside of a local checkout. The stylesheet from
    `link-page-style.css` is embedded (not linked) for the same reason.

    :param dir_: course/book directory, used for the page title
    :param lessons: one `LessonPage` per lesson, in display order
    :param use_master: if `True`, link to the `master` branch instead of
        the current branch
    :return: HTML page content
    """
    course_name = os.path.basename(dir_.rstrip("/"))
    # Stylesheet embedded (not linked) into the generated page so it stays
    # self-contained and portable regardless of where it is opened from.
    css_file_path = hgit.find_file_in_git_tree("link-page-style.css")
    css_content = hio.from_file(css_file_path)
    # Build one table row per lesson.
    rows = []
    for lesson in lessons:
        cells = [f"    <td>{lesson.lesson_name}</td>"]
        for artifact in lesson.artifacts:
            short_label = _get_short_label(artifact.label)
            if artifact.exists:
                href = dshgtogi._get_github_url(
                    file_path=artifact.file_path, use_master=use_master
                )
                if artifact.label == _LABEL_COMMENTARY_HTML:
                    # GitHub serves `.html` files as raw source rather than
                    # rendering them, and `.html` is Git-LFS tracked (see
                    # `.gitattributes`): route through the LFS-aware media
                    # endpoint first, then preview via htmlpreview.github.io.
                    media_href = _to_media_github_url(href)
                    htmlpreview_prefix = "https://htmlpreview.github.io/?"
                    href = f"{htmlpreview_prefix}{media_href}"
                cell = f'    <td><a href="{href}">{short_label}</a></td>'
            else:
                cell = f'    <td class="missing">{short_label}</td>'
            cells.append(cell)
        cells_str = "\n".join(cells)
        rows.append(f"  <tr>\n{cells_str}\n  </tr>")
    rows_str = "\n".join(rows)
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <title>{course_name} - Class Links</title>
    <style>
    {css_content}
    </style>
    </head>
    <body>
    <h1>{course_name} - Class Links</h1>
    <table>
    <tr>
      <th>Lesson</th>
      <th>{_LABEL_SLIDES_PDF}</th>
      <th>{_LABEL_COMMENTARY_HTML}</th>
      <th>{_LABEL_COMMENTARY_PDF}</th>
      <th>{_LABEL_RECAP}</th>
    </tr>
    {rows_str}
    </table>
    </body>
    </html>
    """
    html = hprint.dedent(html)
    return html


# #############################################################################
# Script
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Course/book directory to scan for lessons (e.g., data605)",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        required=True,
        help="Path to the output HTML file",
    )
    parser.add_argument(
        "--do_not_fail_on_warnings",
        action="store_true",
        help=(
            "Log missing lesson artifacts as warnings and keep going "
            "instead of stopping at the first missing file"
        ),
    )
    parser.add_argument(
        "--use_master",
        action="store_true",
        help=(
            "Link to the 'master' branch instead of the current branch "
            "when building the GitHub URLs for lesson artifacts"
        ),
    )
    parser.add_argument(
        "--open_html",
        action="store_true",
        help="Open the generated HTML file in the default browser",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    dir_ = args.dir
    out_file = args.out_file
    do_not_fail_on_warnings = args.do_not_fail_on_warnings
    # Find the lessons to publish.
    lesson_names = _find_lesson_names(dir_)
    _LOG.info("Found %d lessons in '%s'", len(lesson_names), dir_)
    # Check each lesson's artifacts and collect the results.
    lessons = []
    for lesson_name in lesson_names:
        artifacts = _get_lesson_artifacts(
            dir_, lesson_name, do_not_fail_on_warnings=do_not_fail_on_warnings
        )
        lessons.append(LessonPage(lesson_name, artifacts))
    # Generate and write the HTML page.
    html = _generate_html_page(dir_, lessons, use_master=args.use_master)
    hio.to_file(out_file, html)
    _LOG.info("Wrote HTML page to '%s'", out_file)
    if args.open_html:
        cmd = f"open {out_file}"
        hsystem.system(cmd, print_command=True)


if __name__ == "__main__":
    _main(_parse())
