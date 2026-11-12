#!/usr/bin/env python

"""
Inline slide content referenced by `// From <file>:<slide title>` tags.

Parses a skeleton slide deck (`.smd`) and, for each

```
* <slide title>
// From <file>:<slide title>`
```

looks up `<slide title>` inside `<file>` (resolved relative to the Git client
root) and inlines its body right after the idiom, replacing whatever
content was already there.

`<slide title>` is matched against either a `* ` slide bullet or a `#`/`##`/
`###` Markdown header in `<file>`; the body is everything after the matched
title up to the next title at the same or a shallower level.

NOTE: run this only on a fresh skeleton file, where the `// From` tags have
no content yet. Running it a second time on an already-inlined file is
unsound and can mangle previously inlined content

By default, a missing `<file>` or a missing `<slide title>` aborts the
script (unless `--no_abort_on_issue` is passed and continue with the remaining
slides).

# Usage Example

- Inline content in place for a skeleton file:
> inline_content_in_skeleton_slides.py -i book_springer/lectures_source/Lesson05.1_Probabilistic_ML.smd

- Inline content, logging errors and continuing instead of aborting on a
  missing file or slide title:
  > inline_content_in_skeleton_slides.py -i book_springer/lectures_source/Lesson05.1_Probabilistic_ML.smd --no_abort_on_issue

Import as:

import class_scripts.inline_content_in_skeleton_slides as clincoss
"""

import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import helpers.hdbg as hdbg
import helpers.hgit as hgit
import helpers.hio as hio
import helpers.hmarkdown_comments as hmarcomm
import helpers.hmarkdown_headers as hmarhead
import helpers.hparser as hparser
import helpers.hprint as hprint

_LOG = logging.getLogger(__name__)

# #############################################################################
# Title parsing
# #############################################################################


def _get_title_level(line: str) -> Tuple[bool, int, str]:
    """
    Check if `line` is a title line, i.e., a Markdown header or a `* ` slide
    bullet.

    :param line: line to check
    :return: tuple containing:
        - whether `line` is a title line
        - level of the title (`_SLIDE_LEVEL` for a `* ` bullet, number of `#`
          for a header, `0` if not a title)
        - title text (empty string if not a title)
    """
    # Level assigned to a `* <title>` slide bullet: deeper than any
    # `#`...`######` Markdown header, so a slide's body stops at the next
    # slide or header.
    _SLIDE_LEVEL = 100
    if hmarhead.is_markdown_line_separator(line):
        return False, 0, ""
    is_title, level, title = hmarhead.is_header(line)
    if not is_title and line.startswith("* "):
        is_title, level, title = True, _SLIDE_LEVEL, line[2:]
    return is_title, level, title


def _strip_comment_blocks(lines: List[str]) -> List[str]:
    """
    Drop lines inside `/* ... */` or `<!-- ... -->` comment blocks.

    Commented-out slides are disabled by the lecture author, so they must not
    be matched as `// From` targets, and their content (including the
    dangling `*/`/`-->` terminator) must not leak into inlined output.

    :param lines: content of a `.smd` file, as a list of lines
    :return: `lines` with comment blocks removed
    """
    out_lines: List[str] = []
    in_skip_block = False
    for line in lines:
        do_continue, in_skip_block = hmarcomm.process_comment_block(
            line, in_skip_block
        )
        if not do_continue:
            out_lines.append(line)
    return out_lines


def _extract_titled_content(lines: List[str], title: str) -> Optional[List[str]]:
    """
    Extract the body of the section titled `title` from `lines`.

    The body is made of all the lines strictly after the matching title line,
    up to (but not including) the next title line at the same or a shallower
    level. Leading/trailing blank lines are stripped, since the caller adds
    its own blank-line separators around the inlined content. If `title`
    occurs more than once, the first occurrence is used.

    :param lines: content of a `.smd` file, as a list of lines
    :param title: exact title of the header or `* ` slide to extract (e.g.,
        `"Bayes' Theorem: Recap"`)
    :return: extracted body, or `None` if `title` is not found
    """
    content: List[str] = []
    current_level = 0
    inside = False
    found = False
    for line in lines:
        is_title, level, line_title = _get_title_level(line)
        if is_title:
            if inside and level <= current_level:
                # Reached the next sibling (or shallower) title: done.
                break
            if not found and line_title.strip() == title:
                found = True
                inside = True
                current_level = level
                # Skip the title line itself.
                continue
        if inside:
            content.append(line)
    if not found:
        return None
    # Strip leading/trailing blank lines.
    while content and content[-1].strip() == "":
        content.pop()
    while content and content[0].strip() == "":
        content.pop(0)
    return content


# #############################################################################
# `// From` resolution
# #############################################################################


def _parse_from_tag(line: str) -> Optional[Tuple[str, str]]:
    """
    Parse a `// From <file>:<slide title>` tag line.

    :param line: candidate tag line
    :return: `(file, slide_title)`, or `None` if `line` is not a `// From` tag
    """
    # Prefix of a `// From <file>:<slide title>` tag line.
    _FROM_TAG_PREFIX = "// From "
    ret = None
    if line.startswith(_FROM_TAG_PREFIX):
        rest = line[len(_FROM_TAG_PREFIX) :]
        hdbg.dassert_in(
            ":", rest, "Malformed '// From' tag (missing ':'): '%s'", line
        )
        from_file, _, slide_title = rest.partition(":")
        ret = from_file, slide_title
    return ret


def _resolve_content(
    from_file: str,
    slide_title: str,
    *,
    root_dir: str,
    source_cache: Dict[str, List[str]],
    no_abort_on_issue: bool,
) -> Optional[List[str]]:
    """
    Resolve the content referenced by a `// From <from_file>:<slide_title>`
    tag.

    :param from_file: path of the referenced file, relative to `root_dir`
    :param slide_title: title of the slide/header to extract from `from_file`
    :param root_dir: Git client root that `from_file` is relative to
    :param source_cache: cache of `{absolute path: lines}` for files already
        read, updated in place
    :param no_abort_on_issue: if True, log an error and return `None` instead
        of aborting when `from_file` or `slide_title` can't be found
    :return: extracted content, or `None` if it couldn't be resolved
    """
    _LOG.debug(hprint.to_str("from_file slide_title"))
    from_file_abs = os.path.join(root_dir, from_file)
    file_exists = os.path.exists(from_file_abs)
    hdbg.dassert(
        file_exists,
        "Referenced file doesn't exist: '%s' (from '%s')",
        from_file_abs,
        from_file,
        only_warning=no_abort_on_issue,
    )
    if not file_exists:
        return None
    if from_file_abs not in source_cache:
        raw_lines = hio.from_file(from_file_abs).split("\n")
        source_cache[from_file_abs] = _strip_comment_blocks(raw_lines)
    content = _extract_titled_content(source_cache[from_file_abs], slide_title)
    hdbg.dassert(
        content is not None,
        "Slide title '%s' not found in '%s'",
        slide_title,
        from_file,
        only_warning=no_abort_on_issue,
    )
    return content


# #############################################################################
# Inlining
# #############################################################################


def _find_next_title_line(lines: List[str], start_idx: int) -> int:
    """
    Find the index of the next title line at or after `start_idx`.

    :param lines: content of the skeleton file, as a list of lines
    :param start_idx: index to start scanning from
    :return: index of the next title line, or `len(lines)` if there's none
    """
    idx = len(lines)
    for i in range(start_idx, len(lines)):
        is_title, _, _ = _get_title_level(lines[i])
        if is_title:
            idx = i
            break
    return idx


def inline_content_in_skeleton_slides(
    lines: List[str], *, root_dir: str, no_abort_on_issue: bool
) -> Tuple[List[str], int, int]:
    """
    Inline the content referenced by `// From <file>:<slide title>` tags.

    Everything between a `// From` line and the next title line (a header or
    a `* ` slide bullet) is treated as stale/placeholder content and
    replaced with freshly resolved content.

    NOTE: this function is meant to run once, on a fresh skeleton file where
    the `// From` tags are still empty (per the template in
    `prompt.03.create_lesson_slides_from_skeleton_slides.md`). Re-running it
    on an already-inlined file is unsound: previously inlined content can
    itself contain header/`* ` title lines (e.g., a header-level `// From`
    match pulls in nested subsections and slides), so there is no way to
    tell those apart from the *next* skeleton slide, and re-running would
    scan into (and mangle) already-inlined content.

    :param lines: content of the skeleton `.smd` file, as a list of lines
    :param root_dir: Git client root that `// From` paths are relative to
    :param no_abort_on_issue: if True, continue past a missing file/slide
        title instead of aborting
    :return: tuple containing:
        - the transformed lines
        - number of `// From` tags successfully inlined
        - number of `// From` tags left as a placeholder
    """
    _LOG.debug(hprint.to_str("root_dir no_abort_on_issue"))
    out_lines: List[str] = []
    source_cache: Dict[str, List[str]] = {}
    num_inlined = 0
    num_missing = 0
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        out_lines.append(line)
        from_tag = None
        if line.startswith("* ") and i + 1 < n:
            from_tag = _parse_from_tag(lines[i + 1])
        if from_tag is None:
            i += 1
            continue
        from_file, slide_title = from_tag
        out_lines.append(lines[i + 1])
        # Drop the stale (placeholder, or previously inlined) content up to
        # the next title line, and regenerate it.
        boundary = _find_next_title_line(lines, i + 2)
        content = _resolve_content(
            from_file,
            slide_title,
            root_dir=root_dir,
            source_cache=source_cache,
            no_abort_on_issue=no_abort_on_issue,
        )
        out_lines.append("")
        if content is None:
            num_missing += 1
        else:
            num_inlined += 1
            out_lines.extend(content)
            out_lines.append("")
        i = boundary
    _LOG.info(
        "Inlined %d slide(s), %d left as placeholder", num_inlined, num_missing
    )
    return out_lines, num_inlined, num_missing


# #############################################################################
# CLI
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_file",
        action="store",
        required=True,
        help="Skeleton `.smd` file to process; the result is saved in place",
    )
    parser.add_argument(
        "--no_abort_on_issue",
        action="store_true",
        help=(
            "Log an error and leave a placeholder instead of aborting when a "
            "referenced file or slide title can't be found"
        ),
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    root_dir = hgit.get_client_root(super_module=True)
    lines = hio.from_file(args.input_file).split("\n")
    out_lines, _, _ = inline_content_in_skeleton_slides(
        lines, root_dir=root_dir, no_abort_on_issue=args.no_abort_on_issue
    )
    hio.to_file(args.input_file, "\n".join(out_lines))


if __name__ == "__main__":
    _main(_parse())
