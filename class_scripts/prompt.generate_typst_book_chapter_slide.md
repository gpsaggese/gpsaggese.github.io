# Typst Book Chapter: Per-Slide Body Conversion

Mode-specific instructions for `--mode typst_aima`'s per-slide generation path. See
`prompt.generate_book_chapter_common.md` for the shared style guide (audience, tone,
content rules). `prompt.generate_typst_book_chapter_common.md` is concatenated
ahead of this file, so its rules already apply here: general Typst syntax
(emphasis, formulas, lists), pulled in via its own
`@.claude/skills/typst.rules.md` reference, plus what's specific to this
pipeline (dissolving semantic tags, never fabricating a diagram's rendered figure)
Unlike `prompt.generate_typst_book_chapter.md` (used for the older whole-document
generation path), this prompt is used to convert **one slide's entire body text at a
time**, in a single call, regardless of whether the source slide used a `:::
columns` two-column layout. That layout is a page-layout choice made in the `.smd`
source, not semantic content, so it is stripped out before you ever see the slide
(see "What You Will NEVER See in the Input" below) and is never reconstructed in the
output: you always write the slide as one flowing passage. Everything else
structural: chapter/section headings, figures, diagrams, and tables: is handled by
Python before and after this call. Your job is narrow: turn the given fragment of
`.smd` markdown into valid Typst body markup, expanding it into clear prose per the
shared style guide

## What You Will NEVER See in the Input, and Must NEVER Produce

- No document boilerplate (`#import`, `#show`, `#chapter(...)`)
- No headings (`#`, `##`, `###`, `*`)
- No `::: columns` / `:::: {.column width=X%}` pandoc div markers
- No raw code fences (` ```graphviz `, ` ```mermaid `, ` ```tikz `, ` ```{=typst} `)
  — these are already extracted out of your input

## Placeholder Tokens: Copy Verbatim, but Write Around Them

A token that looks like `@@FIGURE_1@@`, `@@FIGURE_2@@`, etc. stands in for a figure,
diagram, or table that has already been converted to Typst by Python. Copy each such
token to the output **exactly as written**, on its own line, with no surrounding
markup, no `#figure(...)`, no brackets. Never invent a token that was not in the
input, and never put the token itself inside a sentence

For a token backed by a diagram, the manifest tells you its label and what it
shows, but never its rendered image path — you cannot know that path, so
never write your own `#figure(image(...))` (or `#wrap-content(...)` around
one) in place of the token, even one that looks plausible from the label and
description. Python substitutes the real diagram source in for the bare
token after your call returns; wrapping the token in synthesized figure
markup yourself just produces a second, fake figure around it. See
"Diagrams: Never Fabricate the Rendered Figure" in
`prompt.generate_typst_book_chapter_common.md`

- Bad output (token wrapped in fabricated figure markup):
  ```text
  #wrap-content(
    [
      #figure(
        image("some/guessed/path.png", width: 100%),
        caption: [Diagram relating Correct premises, Logic and Correct conclusions],
      ) <fig:2aiasthinkingrationally>
    ],
    align: right,
  )[@@FIGURE_1@@]
  ```
- Good output (the token, bare, on its own line):
  ```text
  @@FIGURE_1@@
  ```

After the slide fragment, you are also given a `Figure manifest:` block listing each
token's Typst label and a one-line description of what it shows — never copy that
block into your output. For every token in the manifest, add one sentence to the
prose you are already writing (near the token, in the same fragment) that:

- refers to it by its label, written as plain text (`@fig:richardfeynman`,
  `@tab:foundations`, ...): this is a real Typst content reference and will render
  as an auto-numbered "Figure 1"/"Table 1"; do not alter, quote, or remove the `@`
- explains, in that same sentence, what it shows and how it connects to the
  surrounding discussion

If the fragment is nothing but a visual, with no other prose of its own (e.g., the
whole slide is one image), still write that one sentence — do not leave the token
standing with no prose around it at all

- Input:
  ```text
  - @Definition@: The term "Artificial Intelligence" was coined in 1956

  @@FIGURE_1@@
  ```

  Figure manifest:
  ```text
  @@FIGURE_1@@ -> label: fig:richardfeynman, shows: "Richard Feynman, 1965"
  ```
- Output:
  ```text
  The term "Artificial Intelligence" was coined in 1956. As @fig:richardfeynman
  shows, Richard Feynman's 1965 remark that "what I cannot create, I do not
  understand" captures the same spirit: building an intelligent system is itself a
  path to understanding intelligence.

  @@FIGURE_1@@
  ```

## Citations: Copy Verbatim

A `#cite("key")` call is already valid Typst; copy it through unchanged, at the same
position in the sentence. Never wrap it, quote it, or alter the key

## Semantic Tags: Dissolve, Don't Relabel

See "Semantic Tags in Typst" in `prompt.generate_typst_book_chapter_common.md`
for the rules and worked examples (how `@Definition@`, `@Pros@`/`@Cons@`,
`@Question@`, etc. dissolve into prose, and the `#strong` vs `#emph`
decision).

A `@Pros@`/`@Cons@` (or `@Problem@`/`@Solution@`) pair always reaches you together
in the same call, even if the source put them in separate `::: columns` panels (that
layout is stripped before you see it): merge them into one passage per the rule
above, never two labeled halves.

## Highlighting and Emphasis

See "Highlighting and Emphasis" in `.claude/skills/typst.rules.md`.

## Lists

See "Lists" in `.claude/skills/typst.rules.md` for the general rule (a lone
tagged bullet becomes a plain sentence; a real list is kept only for
parallel items). A lone tagged bullet (`@Definition@`, `@Example@`,
`@Remark@`, ...) dissolving into prose here follows "Semantic Tags:
Dissolve, Don't Relabel" above.

## Be Direct

- **Bad**:
  `- The slide suggests that _acting rationally_ encompasses more than just _thinking rationally_.`
- **Good**: `- Acting rationally encompasses more than just thinking rationally.`

## Other Rules

- Do not repeat a bullet verbatim; expand and explain it
- Output only the converted body: no commentary, no code fence wrapping the whole
  response
