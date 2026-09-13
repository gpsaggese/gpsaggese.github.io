#!/usr/bin/env python

r"""
Generate a book chapter from lecture slides using an LLM.

Converts a lecture source file in `.smd` format into a book chapter, in one
of three formats selected via `--mode`:
- `springer_latex`: a Springer LaTeX chapter (`.tex`)
- `typst_aima`: a Typst/AIMA-style chapter (`.typ`)
- `md`: a plain Markdown chapter (`.md`)

The style guide for all three modes is split across:
- `prompt.generate_book_chapter_common.md`: shared style guide (audience,
  tone, content rules, constraints) common to all modes
- `prompt.generate_latex_book_chapter.md`, `prompt.generate_typst_book_chapter.md`,
  `prompt.generate_md_book_chapter.md`: mode-specific syntax and structure

The output is written next to the input's course directory, in a sibling
`book/` directory (e.g., `msml610/lectures_source/Lesson10.2-Name.smd` ->
`msml610/book/Lesson10.2-Name.<ext>`), with the same base name as the input
and the extension for the selected `--mode`.

This script performs the following steps:
1. Generate the book chapter text via LLM
2. `git_add`: add the generated file to Git (optional, off by default)
3. `lint`: lint the generated file (on by default)
4. `render_pdf`: compile the chapter to PDF (`typst_aima` via
   `run_typst.py`, `md` via pandoc; not supported for `springer_latex`)
   (on by default)
5. `open_pdf`: open the compiled PDF in Skim (optional, off by default)

Steps 2-5 are actions and can be selected with `--action` / `--skip_action`
/ `--only_action` (see `helpers.hselect_action`); `lint` and `render_pdf`
run by default, `git_add` and `open_pdf` don't.

# Usage Example

- Generate a Springer LaTeX chapter for MSML610 lesson 08.1:
> gen_book_chapter.py --mode springer_latex msml610/08.1

- Generate a Typst chapter for MSML610 lesson 10.2 (compiles to PDF by
  default, without opening it or adding it to Git):
> gen_book_chapter.py --mode typst_aima msml610/10.2

- Generate a Typst chapter, add it to Git, and open the PDF once compiled:
> gen_book_chapter.py --mode typst_aima msml610/10.2 --action git_add --action open_pdf

- Generate a Markdown chapter for DATA605 lesson 01.1:
> gen_book_chapter.py --mode md data605/01.1

Import as:

import class_scripts.gen_book_chapter as clgeboch
"""

import argparse
import logging
import os
import re
import shutil
from typing import Dict, List, Optional, Tuple

from tqdm.auto import tqdm

import class_scripts.common_utils as csccouti
import dev_scripts_helpers.documentation.preprocess_notes as dshdprno
import helpers.hdbg as hdbg
import helpers.hgit as hgit
import helpers.hio as hio
import helpers.hmarkdown_slide_iterator as hmaslite
import helpers.hparser as hparser
import helpers.hprint as hprint
import helpers.hselect_action as hselacti
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)

# #############################################################################
# Modes
# #############################################################################

# Map `--mode` to the output file extension.
_MODE_TO_EXTENSION = {
    "springer_latex": "tex",
    "typst_aima": "typ",
    "md": "md",
}

# Backends supported by `_call_llm()`.
# - "hllm": `helpers.hllm.get_completion()`
# - "hllm_cli_lib": `helpers.hllm_cli.apply_llm(backend="library")`,
#   text-only
# - "hllm_cli_exec": `helpers.hllm_cli.apply_llm(backend="executable")`,
#   text-only, shells out to simonw's `llm` CLI executable
_LLM_BACKENDS = csccouti.LLM_BACKENDS

# #############################################################################
# Actions
# #############################################################################

# Post-generation steps, selectable via `--action` / `--skip_action` /
# `--only_action` (see `helpers.hselect_action`). Generating the chapter text
# itself (step 1) is not an action: it's gated by `--no_incremental` /
# `--dry_run` instead, since it's the one step producing the file the other
# four act on. `git_add` and `open_pdf` are optional (off by default);
# `lint` and `render_pdf` run by default.
_VALID_ACTIONS = ["git_add", "lint", "render_pdf", "open_pdf"]
_DEFAULT_ACTIONS = ["lint", "render_pdf"]


# #############################################################################
# Prompt building
# #############################################################################

# Map `--mode` to the mode-specific prompt file, sitting next to this script.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_PROMPT_FILE = os.path.join(
    _SCRIPT_DIR, "prompt.generate_book_chapter_common.md"
)
_MODE_TO_PROMPT_FILE = {
    "springer_latex": os.path.join(
        _SCRIPT_DIR, "prompt.generate_latex_book_chapter.md"
    ),
    "typst_aima": os.path.join(
        _SCRIPT_DIR, "prompt.generate_typst_book_chapter.md"
    ),
    "md": os.path.join(_SCRIPT_DIR, "prompt.generate_md_book_chapter.md"),
}

# Per-slide prompt file used only by the `typst_aima` per-slide generation
# path (see "Per-slide Typst generation" below): narrower than
# `prompt.generate_typst_book_chapter.md` since headings, columns, and
# figures are built deterministically in Python. The LLM never sees a
# figure/diagram/table's raw markup, but it is given a one-line label +
# description for each one (a "figure manifest", see
# `_build_manifest_block()`) so it can still write a real cross-reference
# and explanation for it in the surrounding prose.
_TYPST_SLIDE_PROMPT_FILE = os.path.join(
    _SCRIPT_DIR, "prompt.generate_typst_book_chapter_slide.md"
)


def _get_system_prompt(mode: str) -> str:
    """
    Build the system prompt for `mode` by concatenating the shared style
    guide with the mode-specific instructions.

    :param mode: generation mode, one of `_MODE_TO_EXTENSION`
    :return: combined system prompt
    """
    hdbg.dassert_in(mode, _MODE_TO_PROMPT_FILE)
    common_prompt = hio.from_file(_COMMON_PROMPT_FILE)
    mode_prompt = hio.from_file(_MODE_TO_PROMPT_FILE[mode])
    system_prompt = f"{common_prompt}\n\n{mode_prompt}"
    return system_prompt


def _get_system_prompt_slide() -> str:
    """
    Build the system prompt used by the `typst_aima` per-slide generation
    path (see "Per-slide Typst generation" below).

    :return: combined system prompt
    """
    common_prompt = hio.from_file(_COMMON_PROMPT_FILE)
    slide_prompt = hio.from_file(_TYPST_SLIDE_PROMPT_FILE)
    system_prompt = f"{common_prompt}\n\n{slide_prompt}"
    return system_prompt


def _add_line_numbers(content: str) -> str:
    """
    Prefix each line of `content` with its 1-based line number.

    Used so the LLM can populate accurate `From: <file>:<line num> ...`
    source-attribution comments (see "Source Input Format" in
    `prompt.generate_book_chapter_common.md`).

    :param content: text to number
    :return: `content` with a `"<line num> | "` prefix on every line
    """
    lines = content.splitlines()
    numbered_lines = [
        f"{i:>5} | {line}" for i, line in enumerate(lines, start=1)
    ]
    numbered_content = "\n".join(numbered_lines)
    return numbered_content


def _extract_course_and_title(input_file: str, content: str) -> Tuple[str, str]:
    """
    Extract the course title and chapter title from the lecture source.

    :param input_file: path to the input markdown slides file
    :param content: content of `input_file`
    :return: tuple of (course_title, chapter_title); `course_title` is ""
        if not found, `chapter_title` falls back to the input's base name
    """
    lines = content.split("\n")
    metadata, _ = dshdprno.extract_slide_metadata(lines)
    course_title = metadata.get("course_title", "")
    chapter_title = csccouti.extract_title_from_markdown(input_file)
    if not chapter_title:
        chapter_title = os.path.splitext(os.path.basename(input_file))[0]
    return course_title, chapter_title


def _build_user_prompt(
    input_file: str,
    content: str,
    mode: str,
    course_title: str,
    chapter_title: str,
    lesson: str,
) -> str:
    """
    Build the user prompt: a header with the context the LLM needs (source
    file path, titles, mode-specific metadata), followed by the numbered
    source content.

    :param input_file: path to the input markdown slides file
    :param content: content of `input_file`
    :param mode: generation mode, one of `_MODE_TO_EXTENSION`
    :param course_title: course title (e.g., "MSML610: Advanced Machine
        arning"), empty if not found in the source
    :param chapter_title: chapter title to use verbatim in the output
    :param lesson: lesson number (e.g., "10.2"), used to derive the Typst
        chapter number
    :return: user prompt to send to the LLM
    """
    header_lines = [
        f"Source file: {input_file}",
        f"Chapter title: {chapter_title}",
        f"Course title: {course_title}",
    ]
    if mode == "typst_aima":
        # The chapter number is the integer part of the lesson number
        # (e.g., "10.2" -> "10").
        chapter_num = lesson.split(".")[0]
        header_lines.append(f"Chapter number: {chapter_num}")
        # Use a root-absolute path (resolved against `--root`, not the
        # output file's directory) so the import works regardless of how
        # deep the output file lives (e.g. `<dir>/book/` vs `sweep_results/`).
        header_lines.append(
            "Typst import line: "
            '#import "/helpers_root/dev_scripts_helpers/typst/'
            'aima_style.typ": aima-style, algorithm, chapter, glossary'
        )
    header = "\n".join(header_lines)
    numbered_content = _add_line_numbers(content)
    user_prompt = f"{header}\n\n---\n\n{numbered_content}"
    return user_prompt


# #############################################################################
# LLM call
# #############################################################################


def _call_llm(
    user_prompt: str, system_prompt: str, model: str, llm_backend: str
) -> str:
    """
    Generate the book chapter text using an LLM.

    :param user_prompt: user message (source content plus context)
    :param system_prompt: system prompt (style guide)
    :param model: LLM model to use, or "" to use the backend's default
    :param llm_backend: which LLM backend to use, one of `_LLM_BACKENDS`
    :return: generated chapter text
    """
    hdbg.dassert_in(llm_backend, _LLM_BACKENDS)
    return csccouti.call_llm_cached(
        user_prompt, system_prompt, model, llm_backend
    )


def _get_model_id(model: str, llm_backend: str) -> str:
    """
    Resolve the model id that `llm_backend` will actually use.

    :param model: LLM model to use, or "" to use the backend's default
    :param llm_backend: which LLM backend to use, one of `_LLM_BACKENDS`
    :return: resolved model id
    """
    hdbg.dassert_in(llm_backend, _LLM_BACKENDS)
    if llm_backend == "hllm":
        import helpers.hllm as hllm

        model_id = hllm.get_model_id(model)
    else:
        import helpers.hllm_cli as hllmcli

        model_id = hllmcli.get_model_id(model)
    return model_id


# #############################################################################
# Post-processing
# #############################################################################


# TODO(ai_gp): Convert it into a verbose one with explanation.
# TODO(ai_gp): Move inside the function
# Matches an entire response wrapped in a single fenced code block, e.g. the
# LLM answering with "```latex\n...\n```" despite being asked not to.
# The `\n?` before the closing fence makes the content group optional, so an
# empty fence (e.g., "```\n```", with no content line at all) also matches.
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\n(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """
    Strip a single Markdown code fence wrapping the entire `text`, if any.

    :param text: raw LLM output
    :return: `text` without an enclosing code fence
    """
    text = text.strip()
    match = _CODE_FENCE_RE.match(text)
    if match:
        text = match.group(1)
    return text


def _insert_provenance_tag(text: str, mode: str) -> str:
    """
    Insert a comment with the git hash and timestamp of generation, so that
    one can tell from which commit and when the file was generated.

    :param text: generated chapter text
    :param mode: generation mode, one of `_MODE_TO_EXTENSION`
    :return: `text` with the provenance tag inserted
    """
    tag = hgit.get_generation_tag()
    if mode == "md":
        # Insert after the YAML front matter, if any, since `---` must
        # stay the first line of the file for pandoc to recognize it.
        comment = f"<!-- {tag} -->"
        yaml_end = text.find("\n---\n") if text.startswith("---\n") else -1
        if yaml_end == -1:
            result = f"{comment}\n\n{text}"
        else:
            split_at = yaml_end + len("\n---\n")
            result = f"{text[:split_at]}{comment}\n\n{text[split_at:]}"
    else:
        prefix = csccouti.get_comment_prefix(_MODE_TO_EXTENSION[mode])
        result = f"{prefix} {tag}\n{text}"
    return result


# #############################################################################
# Per-slide Typst generation
# #############################################################################

# Only `typst_aima` uses this path (see the `--mode` choice in `_parse()`).
# It follows the same approach as `gen_lecture_commentary.py`: structure
# (headings, figures/diagrams/tables) is emitted deterministically by
# Python, and the LLM is called once per slide, only to convert that
# slide's *prose* into Typst body markup. This keeps each LLM call small
# and focused, instead of asking one call to transform an entire
# multi-hundred-line lecture (and every construct in it) at once, which is
# what left artifacts like stray `::: columns` markers and `@Tag@` labels
# in the whole-document path (`_generate_book_chapter()` below, still used
# by `springer_latex` and `md`).
#
# A `::: columns` two-column layout is a page-layout choice made in the
# `.smd` source, not semantic content: it is stripped out (see
# `_strip_column_markup()`) before the slide's body reaches the LLM, and
# the LLM always converts the whole slide as one flowing passage. The
# source's layout never affects the generated chapter.

# Matches a `::: columns` / `:::: {.column width=X%}` ... `::::` / `:::`
# pandoc fenced-div layout (see "Use Lists"/columns in the `.smd` source
# convention). These lines carry only layout information (which column a
# line sits in, and how wide it is), so `_strip_column_markup()` just
# deletes them.
_COLUMNS_OPEN_RE = re.compile(r"^:::\s*columns\s*$")
_COLUMN_PANEL_OPEN_RE = re.compile(r"^::::\s*\{\.column[^}]*\}\s*$")
_COLUMN_PANEL_CLOSE_RE = re.compile(r"^::::\s*$")
_COLUMNS_CLOSE_RE = re.compile(r"^:::\s*$")

# A raw diagram-source code fence: rendered by `render_images.py` into a
# PNG (see `run_typst.py`), never by the LLM.
_DIAGRAM_FENCE_RE = re.compile(
    r"```(graphviz|mermaid|tikz)\n(.*?)\n```", re.DOTALL
)
# A pandoc `{=typst}` raw block (e.g. a `#styled-table(...)`/`#references(...)`
# call already written in native Typst in the `.smd` source): valid only
# when spliced in unwrapped, since a Typst document (unlike a Pandoc
# markdown-to-Typst conversion) never executes a `{=typst}` fence.
_TYPST_RAW_FENCE_RE = re.compile(r"```\{=typst\}\n(.*?)\n```", re.DOTALL)
# A bare Markdown image, optionally followed (after a blank line) by an
# italicized caption line, e.g. `\footnotesize _Caption text_` or
# `_Caption text_`.
_IMAGE_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)(?P<attrs>\{[^}]*\})?"
    r"(?:\n\n?(?:\\footnotesize\s+)?_(?P<caption>[^_]+)_)?"
)
# Strips a leading course/lesson prefix off a figure basename, e.g.
# "L01.2.Richard_Feynman" -> "Richard_Feynman".
_FIGURE_PREFIX_RE = re.compile(r"^L\d+(?:\.\d+)*\.")
# A Pandoc inline raw-Typst span, e.g. `` `#cite("key")`{=typst} `` ->
# `#cite("key")`. This is Pandoc's markdown convention for embedding a
# literal Typst call inline in prose (so slides, which go through Pandoc,
# render it); a native `.typ` file has no such convention and would
# otherwise show the backticks and `{=typst}` suffix as literal text.
_INLINE_TYPST_RAW_RE = re.compile(r"`([^`]+)`\{=typst\}")

# A `label="..."` node/edge attribute in a Graphviz diagram, used by
# `_describe_diagram()` to derive a fallback one-line description (and hence
# a manifest entry and Typst caption) for a diagram that has no caption of
# its own in the `.smd` source.
_GRAPHVIZ_NODE_LABEL_RE = re.compile(r'label\s*=\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
# Bracketed/parenthesized node text in a Mermaid diagram (e.g. `A[Some
# Node]`, `B(Round node)`), used the same way as `_GRAPHVIZ_NODE_LABEL_RE`.
_MERMAID_NODE_LABEL_RE = re.compile(r"[\[({]([^\[\](){}]{2,60})[\])}]")
# The `headers: (...)` argument of a `#styled-table(...)` call (see
# `styled-table` in `aima_style.typ`), used by `_describe_table()` to derive
# a fallback one-line description for a table.
_STYLED_TABLE_HEADERS_RE = re.compile(
    r"#styled-table\(\s*headers:\s*\(([^)]*)\)"
)

# A stray Markdown `**bold**` left in the LLM output instead of the
# `#strong[...]` call that `prompt.generate_typst_book_chapter_slide.md`
# ("Highlighting and Emphasis") instructs the LLM to always emit. Typst's
# bold delimiter is a single `*`, not `**`, so a leftover `**text**` is
# parsed as two empty `*...*` runs with nothing between the pair of
# stars, which is what makes `typst compile` warn "no text within
# stars". This is a safety net for LLM non-compliance, not a substitute
# for the prompt rule.
#
# `(?<!\w)` / lack of a mirrored `(?!\w)` after the closing `**`: only
# the opening delimiter is required to sit at a word boundary, which is
# enough to rule out `x**2`/`f(**kwargs)`-style Python power/unpack
# operators (immediately preceded by an identifier character) while
# still matching ordinary Markdown bold (`**word`, preceded by
# whitespace/punctuation/start-of-line). The captured text may not
# contain a backtick (would mean the match crossed an inline-code span)
# or a blank line (would mean it crossed a paragraph/bullet boundary).
_STRAY_MARKDOWN_BOLD_RE = re.compile(
    r"(?<!\w)\*\*(?!\s)((?:(?!\*\*|\n\n|`).)+?)(?<!\s)\*\*(?!\*)",
    re.DOTALL,
)


def _fix_stray_markdown_bold(text: str, *, output_file: str) -> str:
    """
    Replace stray Markdown `**bold**` left by the LLM with `#strong[...]`
    and warn about each fix.

    :param text: full generated `.typ` document text
    :param output_file: path being written to, included in the warning
        so the fix can be traced back to a specific chapter
    :return: `text` with stray `**bold**` converted to `#strong[...]`
    """
    for match in _STRAY_MARKDOWN_BOLD_RE.finditer(text):
        _LOG.warning(
            "'%s': fixing stray Markdown bold left by the LLM: '%s' -> "
            "'#strong[%s]'",
            output_file,
            match.group(0),
            match.group(1),
        )
    text = _STRAY_MARKDOWN_BOLD_RE.sub(r"#strong[\1]", text)
    return text


# Matches a whole `#figure(...) <label>` block (image or table), non-greedy
# so it stops at the first closing `) <label>`, which is how
# `_render_image_placeholder()`/`_wrap_table_placeholder()` always end one.
_FIGURE_BLOCK_RE = re.compile(
    r"(?P<indent>[ \t]*)#figure\([\s\S]*?\) <(?P<label>(?:fig|tab):[a-z0-9_-]+)>"
)
# Matches a whole diagram fence followed by its `label=`/`caption=`
# metadata (see `_render_diagram_placeholder()`), before `render_images.py`
# (run separately, later) turns it into a `#figure(image(...))` call.
_DIAGRAM_BLOCK_RE = re.compile(
    r"(?P<indent>[ \t]*)```(?:graphviz|mermaid|tikz)\n"
    r"[\s\S]*?\n```\n"
    r"[ \t]*label=(?P<label>fig:[a-z0-9_-]+)\n"
    r"[ \t]*caption=(?P<caption>[^\n]*)"
)


def _ensure_visual_references(text: str, manifest: Dict[str, str]) -> str:
    """
    Guarantee every figure, table, and diagram is referenced and explained
    in the surrounding prose, as a safety net for LLM non-compliance with
    "Figure, Table, and Diagram Identification Process" (see
    `prompt.generate_book_chapter_common.md`). This is a safety net for LLM
    non-compliance, not a substitute for the prompt rule (same idiom as
    `_fix_stray_markdown_bold()`).

    If no `@<label>` cross-reference to a visual appears anywhere else in
    `text`, insert one deterministic sentence immediately before it.

    :param text: full generated `.typ` document text
    :param manifest: full Typst label (e.g. "fig:richardfeynman") -> its
        one-line description, for every visual in the document
    :return: `text` with a guaranteed reference + explanation sentence
        before every visual that lacked one
    """
    offset = 0
    matches = list(_FIGURE_BLOCK_RE.finditer(text)) + list(
        _DIAGRAM_BLOCK_RE.finditer(text)
    )
    matches.sort(key=lambda m: m.start())
    for match in matches:
        label = match.group("label")
        description = manifest.get(label)
        if description is None:
            # Not one of ours (shouldn't happen, but don't guess a caption).
            continue
        if f"@{label}" in text:
            # Already referenced somewhere (by the LLM, following the
            # prompt), no fallback needed.
            continue
        indent = match.group("indent")
        sentence = f"{indent}As @{label} shows, {description}.\n\n"
        insert_at = match.start() + offset
        text = text[:insert_at] + sentence + text[insert_at:]
        offset += len(sentence)
    return text


def _strip_column_markup(body: str) -> str:
    """
    Remove `::: columns` / `:::: {.column width=X%}` pandoc div markers
    from a slide body, keeping the content of every column, in the same
    order it appeared in the source.

    The two-column layout is a page-layout choice made in the `.smd`
    source; it carries no semantic content, so the LLM never sees it (see
    "What you will NEVER see in the input" in
    `prompt.generate_typst_book_chapter_slide.md`) and the whole slide is
    converted as a single flowing body, regardless of how many columns the
    source used.

    :param body: raw slide body text (everything after the `* Title`
        line), possibly wrapped in a `::: columns` div
    :return: `body` with column-div marker lines removed
    """
    lines = body.splitlines()
    kept = [
        line
        for line in lines
        if not (
            _COLUMNS_OPEN_RE.match(line.strip())
            or _COLUMN_PANEL_OPEN_RE.match(line.strip())
            or _COLUMN_PANEL_CLOSE_RE.match(line.strip())
            or _COLUMNS_CLOSE_RE.match(line.strip())
        )
    ]
    return "\n".join(kept)


def _slugify_figure_name(path: str) -> str:
    """
    Turn an image path into a Typst label slug, e.g.
    ".../L01.2.Richard_Feynman.jpg" -> "richardfeynman".

    :param path: image path from the `.smd` source
    :return: lowercase alphanumeric label slug
    """
    base = os.path.splitext(os.path.basename(path))[0]
    base = _FIGURE_PREFIX_RE.sub("", base)
    slug = re.sub(r"[^a-zA-Z0-9]", "", base).lower()
    return slug


def _humanize_figure_name(path: str) -> str:
    """
    Derive a fallback caption from an image's filename, e.g.
    ".../L01.2.Richard_Feynman.jpg" -> "Richard Feynman".

    :param path: image path from the `.smd` source
    :return: human-readable caption derived from the filename
    """
    base = os.path.splitext(os.path.basename(path))[0]
    base = _FIGURE_PREFIX_RE.sub("", base)
    return base.replace("_", " ")


def _slugify_text(text: str) -> str:
    """
    Turn arbitrary text into a lowercase alphanumeric label slug, e.g.
    "AI and Economics (1/2)" -> "aiandeconomics12".

    Used for a diagram or table label, which (unlike an image) has no
    source filename to slugify (see `_slugify_figure_name()`).

    :param text: text to slugify (typically a slide title)
    :return: lowercase alphanumeric slug
    """
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


def _next_visual_label(
    kind: str, base_text: str, counter: Dict[str, int]
) -> str:
    """
    Compute a document-unique Typst label for a diagram or table.

    :param kind: "fig" for a diagram, "tab" for a table
    :param base_text: text to derive the slug from (the slide title)
    :param counter: mutable `slug -> count` map, shared across the whole
        document, so repeated slide titles don't collide
    :return: a label such as "fig:aiandeconomics12", or
        "fig:aiandeconomics12-2" for a second visual derived from the same
        slug
    """
    slug = _slugify_text(base_text) or "visual"
    count = counter.get(slug, 0) + 1
    counter[slug] = count
    suffix = "" if count == 1 else f"-{count}"
    return f"{kind}:{slug}{suffix}"


def _describe_diagram(lang: str, code: str, slide_title: str) -> str:
    """
    Derive a one-line description of a diagram from its source, for use as
    both its Typst caption and its manifest entry (see "Figure, Table, and
    Diagram Identification Process" in `prompt.generate_book_chapter_common.md`).

    :param lang: fence language (`graphviz`, `mermaid`, or `tikz`)
    :param code: diagram source code
    :param slide_title: title of the slide the diagram belongs to, used as
        a fallback when no node text can be extracted (e.g. `tikz`, or a
        diagram with unlabeled nodes)
    :return: one-line, human-readable description of the diagram
    """
    label_re = {
        "graphviz": _GRAPHVIZ_NODE_LABEL_RE,
        "mermaid": _MERMAID_NODE_LABEL_RE,
    }.get(lang)
    labels: List[str] = []
    if label_re is not None:
        for match in label_re.finditer(code):
            text = match.group(1).replace("\\n", " ").strip()
            if text and text not in labels:
                labels.append(text)
    if labels:
        shown = labels[:4]
        if len(shown) == 1:
            joined = shown[0]
        else:
            joined = ", ".join(shown[:-1]) + f" and {shown[-1]}"
        description = f"Diagram relating {joined}"
    else:
        description = f"Diagram illustrating {slide_title}"
    return description


def _describe_table(raw_typst: str, slide_title: str) -> Optional[str]:
    """
    Derive a one-line description of a `#styled-table(...)` call from its
    `headers` argument.

    :param raw_typst: raw Typst source from a `{=typst}` fence
    :param slide_title: title of the slide the table belongs to, used as a
        fallback when `headers` cannot be parsed
    :return: one-line description, or `None` if `raw_typst` is not a
        `#styled-table(...)` call (other `{=typst}` content, e.g.
        `#references(...)`, is passed through untouched instead)
    """
    if not raw_typst.strip().startswith("#styled-table("):
        return None
    match = _STYLED_TABLE_HEADERS_RE.search(raw_typst)
    if not match:
        return f"Table from the '{slide_title}' slide"
    headers = [h.strip().strip('"') for h in match.group(1).split(",")]
    headers = [h for h in headers if h]
    description = (
        f"Table of {', '.join(headers)}"
        if headers
        else (f"Table from the '{slide_title}' slide")
    )
    return description


def _render_image_placeholder(
    match: "re.Match[str]", output_dir: str
) -> Tuple[str, str, str]:
    """
    Deterministically render a Markdown image match (see `_IMAGE_RE`) as a
    Typst `#figure(...)` call.

    :param match: `_IMAGE_RE` match
    :param output_dir: directory the chapter file is written to (image
        paths are resolved relative to it)
    :return: (Typst `#figure(...)` snippet, full Typst label e.g.
        "fig:richardfeynman", one-line description used as both the caption
        and the manifest entry handed to the LLM)
    """
    path = match.group("path")
    caption = match.group("caption")
    alt = match.group("alt")
    if caption:
        description = caption.strip()
    elif alt and alt.strip():
        # Fall back to the Markdown alt text (`![alt](path)`), if any,
        # before the filename-derived fallback: it's the author's own
        # words, not a guess from the file's basename.
        description = alt.strip()
    else:
        description = _humanize_figure_name(path)
    label = _slugify_figure_name(path)
    full_label = f"fig:{label}"
    rel_path = os.path.relpath(path, start=output_dir)
    lines = [
        "#figure(",
        f'  image("{rel_path}", width: 80%),',
        f"  caption: [{description}],",
        '  kind: "figure",',
        "  supplement: [Fig.],",
        "  placement: auto,",
        f") <{full_label}>",
    ]
    return "\n".join(lines), full_label, description


def _render_diagram_placeholder(
    lang: str, code: str, *, label: str, description: str
) -> str:
    """
    Render a diagram-source fence (see `_DIAGRAM_FENCE_RE`) as a bare
    (uncommented) fence, followed by `label=`/`caption=` metadata lines.

    `render_images.py` (run as a separate, optional step via
    `run_typst.py -a render_images`, *after* this script) is the one that
    comments the code out, renders it, and inserts the
    `#figure(image(...))` call — it expects to find the fence exactly as
    written in the `.smd` source (not pre-commented or pre-numbered),
    optionally followed by this `label=`/`caption=` metadata, which it
    copies onto the `#figure(...)` call it generates (see its docstring).
    Attaching them here — instead of leaving the eventual figure
    uncaptioned and unlabeled — is what lets it be referenced and
    explained like any other visual.

    :param lang: fence language (`graphviz`, `mermaid`, or `tikz`)
    :param code: diagram source code
    :param label: full Typst label to attach to the rendered figure (e.g.
        "fig:aiandeconomics12")
    :param description: one-line caption for the rendered figure
    :return: the fence, followed by its `label=`/`caption=` metadata
    """
    lines = [
        f"```{lang}",
        code.strip(chr(10)),
        "```",
        f"label={label}",
        f"caption={description}",
    ]
    return "\n".join(lines)


def _wrap_table_placeholder(
    raw_typst: str, *, label: str, description: str
) -> str:
    """
    Wrap a raw `#styled-table(...)` call (from a `{=typst}` fence) in a
    `#figure(...)` so it gets a caption, a label, and can be cross-
    referenced like any other visual.

    :param raw_typst: raw Typst source, starting with `#styled-table(`
    :param label: full Typst label to attach (e.g. "tab:aiandeconomics12")
    :param description: one-line caption
    :return: a `#figure(...)` snippet wrapping `raw_typst`
    """
    # Inside a function-call argument list we are already in code mode, so
    # the leading `#` that makes `styled-table(...)` a statement at the top
    # level of markup must be dropped here.
    call = raw_typst.strip()
    if call.startswith("#"):
        call = call[1:]
    indented_call = "\n".join(f"  {l}" if l else "" for l in call.splitlines())
    lines = [
        "#figure(",
        f"{indented_call},",
        f"  caption: [{description}],",
        '  kind: "table",',
        "  supplement: [Table.],",
        "  placement: auto,",
        f") <{label}>",
    ]
    return "\n".join(lines)


def _build_manifest_block(manifest: Dict[str, Tuple[str, str]]) -> str:
    """
    Build the `Figure manifest:` block appended to the LLM's user message
    (see "Placeholder Tokens" in `prompt.generate_typst_book_chapter_slide.md`).

    :param manifest: token -> (full Typst label, one-line description)
    :return: manifest block text, or "" if `manifest` is empty
    """
    if not manifest:
        return ""
    lines = ["", "Figure manifest:"]
    for token, (label, description) in manifest.items():
        lines.append(f'{token} -> label: {label}, shows: "{description}"')
    return "\n".join(lines)


def _process_slide_body(
    body: str,
    *,
    output_dir: str,
    slide_title: str,
    label_counter: Dict[str, int],
    system_prompt: str,
    model: str,
    llm_backend: str,
) -> Tuple[str, Dict[str, str]]:
    """
    Convert one slide's raw `.smd` body into Typst, in a single LLM call.

    Figures, diagrams, and `{=typst}` raw blocks are pulled out into
    `@@FIGURE_N@@` placeholder tokens and rendered deterministically (see
    `prompt.generate_typst_book_chapter_slide.md` for the placeholder
    contract); only the remaining prose, if any, is sent to the LLM, along
    with a "figure manifest" (label + one-line description per token) so
    the LLM can still write a real cross-reference and explanation for
    each one, without ever seeing its raw markup. A slide body that is
    nothing but a figure/table skips the LLM call entirely. The whole body
    (already stripped of any `::: columns` layout by
    `_strip_column_markup()`) is sent as one call, so the model can weave
    content that used to sit in separate columns (e.g. a `@Pros@`/`@Cons@`
    pair) into one continuous passage.

    :param body: raw slide body text
    :param output_dir: directory the chapter file is written to
    :param slide_title: title of this slide, used to derive a
        caption/description for diagrams and tables (which, unlike
        images, have no source filename to work from)
    :param label_counter: mutable `slug -> count` map, shared across the
        whole document, used to keep diagram/table labels unique (see
        `_next_visual_label()`)
    :param system_prompt: system prompt for the LLM (slide-body prompt)
    :param model: LLM model to use, or "" to use the backend's default
    :param llm_backend: which LLM backend to use, one of `_LLM_BACKENDS`
    :return: (Typst snippet for this slide's body, manifest: full Typst
        label -> one-line description, for every visual found in this
        slide, used by `_ensure_visual_references()` as a document-wide
        safety net)
    """
    placeholders: Dict[str, str] = {}
    manifest: Dict[str, Tuple[str, str]] = {}
    token_idx = [0]

    def _next_token() -> str:
        token_idx[0] += 1
        return f"@@FIGURE_{token_idx[0]}@@"

    def _repl_diagram(m: "re.Match[str]") -> str:
        token = _next_token()
        lang, code = m.group(1), m.group(2)
        label = _next_visual_label("fig", slide_title, label_counter)
        description = _describe_diagram(lang, code, slide_title)
        placeholders[token] = _render_diagram_placeholder(
            lang, code, label=label, description=description
        )
        manifest[token] = (label, description)
        return token

    def _repl_typst_raw(m: "re.Match[str]") -> str:
        token = _next_token()
        raw = m.group(1).strip("\n")
        description = _describe_table(raw, slide_title)
        if description is None:
            # Not a `#styled-table(...)` call (e.g. `#references(...)`):
            # pass it through untouched, as before, with no manifest entry.
            placeholders[token] = raw
        else:
            label = _next_visual_label("tab", slide_title, label_counter)
            placeholders[token] = _wrap_table_placeholder(
                raw, label=label, description=description
            )
            manifest[token] = (label, description)
        return token

    def _repl_image(m: "re.Match[str]") -> str:
        token = _next_token()
        snippet, label, description = _render_image_placeholder(m, output_dir)
        placeholders[token] = snippet
        manifest[token] = (label, description)
        return token

    body = _DIAGRAM_FENCE_RE.sub(_repl_diagram, body)
    body = _TYPST_RAW_FENCE_RE.sub(_repl_typst_raw, body)
    body = _IMAGE_RE.sub(_repl_image, body)
    remaining = body
    for token in placeholders:
        remaining = remaining.replace(token, "")
    if remaining.strip():
        llm_input = body + _build_manifest_block(manifest)
        raw_text = csccouti.call_llm_cached(
            llm_input, system_prompt, model, llm_backend
        )
        text = _strip_code_fence(raw_text)
    else:
        # The slide body is nothing but a figure/table: no prose to
        # convert, so skip the LLM call entirely.
        text = body
    for token, replacement in placeholders.items():
        text = text.replace(token, replacement)
    full_manifest = {
        label: description for label, description in manifest.values()
    }
    return text, full_manifest


def _generate_typst_slide(
    slide_lines: List[str],
    *,
    output_dir: str,
    label_counter: Dict[str, int],
    visual_manifest: Dict[str, str],
    system_prompt: str,
    model: str,
    llm_backend: str,
    source_file: str,
    line_number: int,
) -> str:
    """
    Convert one `* Slide Title` block (with its body) into Typst.

    Any `::: columns` layout in the source is stripped before the body
    reaches the LLM (see `_strip_column_markup()`) and never reconstructed
    in the output: the slide is always converted as one flowing passage,
    regardless of how the source slide was laid out.

    :param slide_lines: lines of the slide, starting with `* Title`
    :param output_dir: directory the chapter file is written to
    :param label_counter: mutable `slug -> count` map, shared across the
        whole document (see `_next_visual_label()`)
    :param visual_manifest: mutable `full Typst label -> one-line
        description` map, shared across the whole document, updated
        in-place with every visual found in this slide; consumed at the
        end by `_ensure_visual_references()`
    :param system_prompt: system prompt for the LLM (slide-body prompt)
    :param model: LLM model to use, or "" to use the backend's default
    :param llm_backend: which LLM backend to use, one of `_LLM_BACKENDS`
    :param source_file: path to the `.smd` source, for the `// From:`
        provenance comment
    :param line_number: 1-based line number of the `* Title` line
    :return: Typst snippet for this slide
    """
    title = slide_lines[0][1:].strip()
    body = "\n".join(slide_lines[1:])
    body = _strip_column_markup(body)
    body_out, slide_manifest = _process_slide_body(
        body,
        output_dir=output_dir,
        slide_title=title,
        label_counter=label_counter,
        system_prompt=system_prompt,
        model=model,
        llm_backend=llm_backend,
    )
    visual_manifest.update(slide_manifest)
    parts = [
        f"// From: {source_file}:{line_number} '{slide_lines[0]}'",
        f"// Slide: {title}",
        f"#strong[{title}]",
        "",
        body_out,
    ]
    return "\n".join(parts)


def _generate_typst_header(
    level: int, title: str, source_file: str, line_number: int
) -> str:
    """
    Convert one `#`/`##`/`###`-prefixed heading line into Typst.

    A level-1 heading becomes a bold paragraph, not a `==` section: the
    chapter-level `#chapter(...)` call (using the lesson's metadata title,
    which is normally the same topic worded differently) has already been
    emitted once, at the top of the document, so a level-1 heading in the
    body is redundant with it rather than introducing a new chapter.

    :param level: heading level (1 for `#`, 2 for `##`, etc.)
    :param title: heading text, without the leading `#`s
    :param source_file: path to the `.smd` source, for the `// From:`
        provenance comment
    :param line_number: 1-based line number of the heading line
    :return: Typst snippet for this heading
    """
    marker = "#" * level
    heading = f"#strong[{title}]" if level == 1 else f"{'=' * level} {title}"
    parts = [
        f"// From: {source_file}:{line_number} '{marker} {title}'",
        f"// Slide: {title}",
        heading,
    ]
    return "\n".join(parts)


def _build_typst_document_header(course_title: str, chapter_title: str) -> str:
    """
    Build the fixed Typst document boilerplate: imports, metadata, and the
    single `#chapter(...)` call.

    :param course_title: course title (e.g., "MSML610: Advanced Machine
        Learning")
    :param chapter_title: chapter title (e.g., "L01.2: AI and Machine
        Learning"), used verbatim as the purple chapter bar's label
    :return: Typst document header
    """
    lines = [
        "// Import AIMA style formatting and macros.",
        '#import "/helpers_root/dev_scripts_helpers/typst/aima_style.typ": (',
        "  aima-style, algorithm, chapter, glossary, styled-table,",
        ")",
        "// Import the custom citation/bibliography system.",
        '#import "/helpers_root/dev_scripts_helpers/typst/'
        'umd_references.typ": cite, references',
        "",
        "// Document metadata",
        "#set document(",
        f'  title: "{chapter_title}",',
        f'  author: "{course_title}",',
        ")",
        "",
        "// Apply the AIMA document template (page/text/heading set + show rules).",
        "#show: aima-style",
        "",
        f'#chapter("{chapter_title}")',
    ]
    return "\n".join(lines)


def _generate_typst_chapter_per_slide(
    input_file: str,
    output_file: str,
    model: str,
    llm_backend: str,
    course_title: str,
    chapter_title: str,
) -> None:
    """
    Generate a `typst_aima` book chapter one slide (or column panel) at a
    time and write it to disk (see "Per-slide Typst generation" above).

    :param input_file: path to the input markdown slides file
    :param output_file: path to write the generated `.typ` chapter to
    :param model: LLM model to use, or "" to use the backend's default
    :param llm_backend: which LLM backend to use, one of `_LLM_BACKENDS`
    :param course_title: course title, used in the document metadata
    :param chapter_title: chapter title, used in `#chapter(...)` and the
        document metadata
    """
    hdbg.dassert_file_exists(input_file)
    content = _INLINE_TYPST_RAW_RE.sub(r"\1", hio.from_file(input_file))
    lines = content.splitlines()
    items = list(hmaslite.iterate_slide_lines(lines))
    system_prompt = _get_system_prompt_slide()
    output_dir = os.path.dirname(output_file) or "."
    # `label_counter` keeps diagram/table labels unique across the whole
    # document (see `_next_visual_label()`); `visual_manifest` accumulates
    # every visual's label -> description across the whole document, and is
    # consumed at the end by `_ensure_visual_references()`.
    label_counter: Dict[str, int] = {}
    visual_manifest: Dict[str, str] = {}
    parts = [_build_typst_document_header(course_title, chapter_title)]
    for item in tqdm(items, desc="Generating slides", unit="slide"):
        if item["type"] == "header":
            first_line = item["content"][0]
            level = len(first_line) - len(first_line.lstrip("#"))
            title = first_line.lstrip("#").strip()
            parts.append(
                _generate_typst_header(
                    level, title, input_file, item["line_number"]
                )
            )
        elif item["type"] == "slide":
            parts.append(
                _generate_typst_slide(
                    item["content"],
                    output_dir=output_dir,
                    label_counter=label_counter,
                    visual_manifest=visual_manifest,
                    system_prompt=system_prompt,
                    model=model,
                    llm_backend=llm_backend,
                    source_file=input_file,
                    line_number=item["line_number"],
                )
            )
        # "comment"/"preamble" items are metadata/frontmatter, already
        # captured by `_extract_course_and_title()`, and not part of the
        # chapter body.
    text = "\n\n".join(parts)
    # Safety net: guarantee every visual is referenced and explained, in
    # case the LLM didn't comply with "Figure, Table, and Diagram
    # Identification Process" (see prompt.generate_book_chapter_common.md).
    text = _ensure_visual_references(text, visual_manifest)
    text = _fix_stray_markdown_bold(text, output_file=output_file)
    text = _insert_provenance_tag(text, "typst_aima")
    hio.to_file(output_file, text)
    _LOG.info("Wrote book chapter to: %s", output_file)


# #############################################################################
# Book chapter generation
# #############################################################################


def _generate_book_chapter(
    input_file: str,
    output_file: str,
    mode: str,
    model: str,
    llm_backend: str,
    lesson: str,
) -> None:
    """
    Generate a book chapter from lecture slides and write it to disk.

    :param input_file: path to the input markdown slides file
    :param output_file: path to write the generated chapter to
    :param mode: generation mode, one of `_MODE_TO_EXTENSION`
    :param model: LLM model to use, or "" to use the backend's default
    :param llm_backend: which LLM backend to use, one of `_LLM_BACKENDS`
    :param lesson: lesson number (e.g., "10.2"), used to derive the Typst
        chapter number
    """
    hdbg.dassert_file_exists(input_file)
    hdbg.dassert_in(mode, _MODE_TO_EXTENSION)
    content = hio.from_file(input_file)
    course_title, chapter_title = _extract_course_and_title(input_file, content)
    if mode == "typst_aima":
        # Per-slide generation path (see "Per-slide Typst generation"
        # above): structure is deterministic, the LLM only converts prose.
        _generate_typst_chapter_per_slide(
            input_file,
            output_file,
            model,
            llm_backend,
            course_title,
            chapter_title,
        )
        return
    system_prompt = _get_system_prompt(mode)
    user_prompt = _build_user_prompt(
        input_file, content, mode, course_title, chapter_title, lesson
    )
    raw_text = _call_llm(user_prompt, system_prompt, model, llm_backend)
    text = _strip_code_fence(raw_text)
    text = _insert_provenance_tag(text, mode)
    hio.to_file(output_file, text)
    _LOG.info("Wrote book chapter to: %s", output_file)


def _lint_typst_file(typst_file: str, *, dry_run: bool) -> None:
    """
    Lint a Typst file in place with `typstyle`, if it is on the PATH.

    :param typst_file: path to the `.typ` file to lint
    :param dry_run: print the command without executing it
    """
    if shutil.which("typstyle") is None:
        _LOG.warning(
            "'typstyle' not found on PATH, skipping lint of '%s'", typst_file
        )
        return
    cmd = f"typstyle --inplace --wrap-text -l 80 {typst_file}"
    hsystem.system(cmd, print_command=True, dry_run=dry_run)


def _lint_with_lint_text(output_file: str, *, dry_run: bool) -> None:
    """
    Lint a Markdown or LaTeX file in place with `lint_text.py`.

    :param output_file: path to the file to lint (its type is inferred
        from its extension)
    :param dry_run: print the command without executing it
    """
    cmd = f"lint_text.py -i {output_file} -o {output_file}"
    hsystem.system(cmd, print_command=True, dry_run=dry_run)


def _get_pdf_file(out_dir: str, basename: str) -> str:
    """
    Compute the path a compiled book chapter's PDF is written to.

    :param out_dir: directory holding the generated chapter
    :param basename: chapter file base name, without extension
    :return: path to the chapter's PDF
    """
    return os.path.join(out_dir, f"{basename}.pdf")


def _render_book_chapter(
    output_file: str,
    out_dir: str,
    basename: str,
    mode: str,
    script_dir: str,
    *,
    no_abort_on_warnings: bool,
    dry_run: bool,
) -> None:
    """
    Compile the generated book chapter to PDF, without opening it.

    - `typst_aima`: delegates to `run_typst.py` (skipping its own
      `open_pdf` action, run separately as this script's `open_pdf`
      action), which compiles the file inside a Docker container and
      asserts on `typst compile` warnings
    - `md`: converts to PDF via pandoc
    - `springer_latex`: not supported (there's no standalone-chapter PDF
      target; it's meant to be compiled as part of the whole book)

    :param output_file: path to the generated chapter file
    :param out_dir: directory holding the generated chapter (and where the
        PDF is written)
    :param basename: chapter file base name, without extension
    :param mode: generation mode, one of `_MODE_TO_EXTENSION`
    :param script_dir: directory of this script (for pandoc header files)
    :param no_abort_on_warnings: don't assert if `typst compile` emits
        warnings (`typst_aima` only, forwarded to `run_typst.py`)
    :param dry_run: print the commands without executing them
    """
    pdf_file = _get_pdf_file(out_dir, basename)
    if mode == "typst_aima":
        run_typst_exec = hgit.find_file("run_typst.py")
        # Explicitly request `render_images` (off by default in
        # `run_typst.py`): it turns the diagram-source placeholders left by
        # `_render_diagram_placeholder()` into real `#figure(image(...))
        # <label>` calls. Without it, any `graphviz`/`mermaid`/`tikz`
        # diagram in the chapter never gets a real Typst label, so a
        # `@fig:...` cross-reference to it fails to compile with "label
        # does not exist".
        cmd = (
            f"{run_typst_exec} --input {output_file} --output {pdf_file} "
            "--action render_images --skip_action open_pdf"
        )
        if no_abort_on_warnings:
            cmd += " --no_abort_on_warnings"
        hsystem.system(cmd, print_command=True, dry_run=dry_run)
    elif mode == "md":
        csccouti.convert_markdown_to_pdf(
            output_file, pdf_file, script_dir, dry_run=dry_run
        )
    else:
        _LOG.warning(
            "PDF preview is not supported for --mode springer_latex. You "
            "need to compile the book (chapter file: '%s')",
            output_file,
        )


def _open_book_chapter_pdf(
    output_file: str, out_dir: str, basename: str, mode: str, *, dry_run: bool
) -> None:
    """
    Open the compiled book chapter PDF in Skim.

    Assumes the PDF was already produced by a prior `render_pdf` action;
    does not compile it.

    :param output_file: path to the generated chapter file
    :param out_dir: directory holding the compiled PDF
    :param basename: chapter file base name, without extension
    :param mode: generation mode, one of `_MODE_TO_EXTENSION`
    :param dry_run: print the commands without executing them
    """
    if mode == "typst_aima":
        pdf_file = _get_pdf_file(out_dir, basename)
        run_typst_exec = hgit.find_file("run_typst.py")
        cmd = (
            f"{run_typst_exec} --input {output_file} --output {pdf_file} "
            "--only_action open_pdf"
        )
        hsystem.system(cmd, print_command=True, dry_run=dry_run)
    elif mode == "md":
        pdf_file = _get_pdf_file(out_dir, basename)
        # TODO(gp): This should be a function checking on mac
        cmd = f"open -a /Applications/Skim.app {pdf_file}"
        hsystem.system(cmd, print_command=True, dry_run=dry_run)
    else:
        _LOG.warning(
            "PDF preview is not supported for --mode springer_latex. You "
            "need to compile the book (chapter file: '%s')",
            output_file,
        )


# #############################################################################
# CLI
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    # TODO(gp2): This should be factor out (maybe also dry run and no_incremental)?
    parser.add_argument(
        "input",
        type=str,
        help="Lecture specification: 'data605/08.1', 'msml610/08.1', "
        "or file path 'msml610/lectures_source/Lesson10.2-Name.smd'",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(_MODE_TO_EXTENSION.keys()),
        required=True,
        help="Output format to generate: 'springer_latex' for a Springer "
        "SNmono LaTeX chapter, 'typst_aima' for a Typst/AIMA-style chapter, "
        "'md' for a plain Markdown chapter",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to the output book chapter file (default: derived from "
        "the input and --mode, as '<dir>/book/<basename>.<ext>')",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print the commands that would be executed without running them",
    )
    parser.add_argument(
        "--no_incremental",
        action="store_true",
        help=(
            "Force regeneration of intermediate files even if they already "
            "exist (by default, steps are skipped if their output already "
            "exists)"
        ),
    )
    parser.add_argument(
        "--llm_backend",
        type=str,
        choices=_LLM_BACKENDS,
        default="hllm",
        help=(
            "LLM backend to use for book chapter generation: 'hllm' "
            "(default) uses `helpers.hllm`, 'hllm_cli_lib' uses "
            "`helpers.hllm_cli` with the `llm` Python library (text-only), "
            "'hllm_cli_exec' uses `helpers.hllm_cli` shelling out to "
            "simonw's `llm` CLI executable (text-only)"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="LLM model to use (e.g., 'gpt-4o', 'claude-opus-4'); empty "
        "string (default) uses the --llm_backend's default model",
    )
    parser.add_argument(
        "--no_abort_on_warnings",
        action="store_true",
        help="Don't assert if `typst compile` emits warnings during the "
        "'render_pdf' action (--mode typst_aima only; forwarded to "
        "run_typst.py)",
    )
    hselacti.add_action_arg(parser, _VALID_ACTIONS, _DEFAULT_ACTIONS)
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Suppress verbose HTTP request logging from the LLM client libraries.
    # shown instead while the chapter is generated.
    import helpers.hllm_cli as hllmcli

    hllmcli.shutup_llm_logging()
    # Print the LLM backend and the (resolved) model that will be used.
    _LOG.info(
        "llm_backend=%s model=%s",
        args.llm_backend,
        _get_model_id(args.model, args.llm_backend),
    )
    # Select the post-generation actions (git_add / lint / render_pdf /
    # open_pdf) and print the full run plan upfront, before any (costly)
    # work starts, following the same idiom as e.g. `run_typst.py`.
    actions = hselacti.select_actions(args, _VALID_ACTIONS, _DEFAULT_ACTIONS)
    _LOG.info(
        "\n%s",
        hselacti.actions_to_string(actions, _VALID_ACTIONS, add_frame=True),
    )
    # Parse and validate arguments.
    dir_arg, lesson_arg = csccouti.parse_lesson_spec(args.input)
    csccouti.validate_dir_lesson_args(dir_arg, lesson_arg)
    # Get source name and compute the output path.
    src_name = csccouti.get_source_name(dir_arg, lesson_arg)
    input_file = f"{dir_arg}/lectures_source/{src_name}"
    extension = _MODE_TO_EXTENSION[args.mode]
    if args.output:
        # Use the user-provided output path, derived `out_dir`/`basename`.
        output_file = args.output
        out_dir = os.path.dirname(output_file) or "."
        basename = os.path.splitext(os.path.basename(output_file))[0]
    else:
        out_dir = f"{dir_arg}/book"
        basename = os.path.splitext(src_name)[0]
        output_file = f"{out_dir}/{basename}.{extension}"
    do_incremental = not args.no_incremental
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Step 1: Create the output directory.
    csccouti.ensure_dir_exists(out_dir)
    # Step 2: Generate the book chapter.
    if do_incremental and os.path.exists(output_file):
        _LOG.warning("Step 2: Skipping, '%s' already exists", output_file)
    else:
        _LOG.info(
            "\n%s",
            hprint.frame(f"Step 2: Generating '{args.mode}' book chapter"),
        )
        if args.dry_run:
            _LOG.warning(
                "As per user request, not generating book chapter for '%s'",
                input_file,
            )
        else:
            _generate_book_chapter(
                input_file,
                output_file,
                args.mode,
                args.model,
                args.llm_backend,
                lesson_arg,
            )
    # Steps 3-6: git_add / lint / render / open_pdf (`actions` selected and
    # printed up front, see above).
    while actions:
        action = actions[0]
        to_execute, actions = hselacti.mark_action(action, actions)
        if not to_execute:
            continue
        if action == "git_add":
            _LOG.info("\n%s", hprint.frame("Action: Adding book chapter to git"))
            csccouti.git_add_with_retry(output_file, dry_run=args.dry_run)
        elif action == "lint":
            _LOG.info(
                "\n%s", hprint.frame(f"Action: Linting '{args.mode}' file")
            )
            if args.mode == "typst_aima":
                # TODO(gp): Move this inside lint_text.py to support also
                # typst.
                _lint_typst_file(output_file, dry_run=args.dry_run)
            else:
                _lint_with_lint_text(output_file, dry_run=args.dry_run)
        elif action == "render_pdf":
            _LOG.info("\n%s", hprint.frame("Action: Rendering to PDF"))
            _render_book_chapter(
                output_file,
                out_dir,
                basename,
                args.mode,
                script_dir,
                no_abort_on_warnings=args.no_abort_on_warnings,
                dry_run=args.dry_run,
            )
        elif action == "open_pdf":
            _LOG.info("\n%s", hprint.frame("Action: Opening PDF"))
            _open_book_chapter_pdf(
                output_file,
                out_dir,
                basename,
                args.mode,
                dry_run=args.dry_run,
            )
        else:
            raise ValueError(f"Invalid action='{action}'")
    hdbg.dassert_eq(
        len(actions or []), 0, "There are unprocessed actions: %s", str(actions)
    )
    _LOG.info("Book chapter generated: %s", output_file)


if __name__ == "__main__":
    _main(_parse())
