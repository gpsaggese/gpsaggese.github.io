# Typst Book Chapter: Per-Slide Body Conversion

Mode-specific instructions for `--mode typst_aima`'s per-slide generation path. See
`prompt.generate_book_chapter_common.md` for the shared style guide (audience, tone,
content rules)
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

`@Definition@`, `@Example@`, `@Pros@`, `@Cons@`, `@Problem@`, `@Solution@`,
`@Question@`, `@Answer@`, `@Remark@`, `@Key idea@`, ... mark the _rhetorical role_ of
the bullet that follows; `@word` is also Typst label-reference syntax, so leaving it
verbatim is a compile error, not just a style issue. See "Dissolving Semantic Tags"
in the shared style guide for the rules: this is what applying them looks like in
Typst
Never emit the tag word itself as a `#strong[...]` label (`#strong[Definition]`,
`#strong[Pros]`, `#strong[Cons]`, `#strong[Problem]`, `#strong[Question]`, ...). Use
`#strong[...]` only for the actual term or claim being introduced

- Input:

  ```text
  - @Definition@: An **agent** is something that perceives and acts to
    reach a goal
  ```
- Bad output (relabels the tag, keeps the outline structure):
  ```text
  - #strong[Definition]: An #strong[agent] is something that perceives and
    acts to reach a goal.
  ```
- Good output (a plain sentence; only the term is bold):
  ```text
  An #strong[agent] is something that perceives and acts to reach a goal.
  ```

- Input:
  ```text
  - @Pros@
    - Express precise theory of the human mind as a computer program

  - @Cons@
    - Unknown workings of the human mind
    - Anthropocentric definition
  ```
- Bad output (two tag-labeled lists):
  ```text
  - #strong[Pros]:
    - Express precise theory of the human mind as a computer program.
  - #strong[Cons]:
    - Unknown workings of the human mind.
    - Anthropocentric definition.
  ```
- Good output (one passage weighing both sides):
  ```text
  Expressing this as a computer program forces a precise theory of the
  human mind instead of a vague verbal one. The cost is that the mind's
  own workings are still largely unknown, and the definition of "thinking
  like a human" is inescapably anthropocentric.
  ```

- Input:
  ```text
  - @Question@: What is artificial intelligence?
    - First, understand what **human intelligence** is
  ```
- Good output (the question stays a question; the tag disappears):
  ```text
  What is artificial intelligence? Answering that starts with
  understanding what #strong[human intelligence] is.
  ```

- Input:

  ```text
  - @Limitations@
    - **Omniscience vs no-regrets**
      - Best is based on available information, not perfect knowledge
  ```
  Good output (a genuinely enumerable set stays a list, without the tag label;
  the item's lead phrase is emphasis, not a definition, so it is `#emph`, not
  `#strong` — see "Highlighting and Emphasis" below):

  ```text
  - #emph[Omniscience vs no-regrets]: the best decision is based on
    available information, not perfect knowledge.
  ```

A `@Pros@`/`@Cons@` (or `@Problem@`/`@Solution@`) pair always reaches you together
in the same call, even if the source put them in separate `::: columns` panels (that
layout is stripped before you see it): merge them into one passage per the rule
above, never two labeled halves.

## Highlighting and Emphasis

- `#strong[...]` is reserved for the term or claim being formally defined or
  named for the first time, normally in a sentence shaped like "#strong[Term]
  is/refers to/means ..." or the direct answer to a "what is X?" question. Use
  it sparingly: a handful of times per slide at most, and never for a list
  item's lead phrase
- Everything else the source marks for emphasis becomes `#emph[...]` instead:
  a `**bold**` list-item lead phrase, a term already defined earlier and
  mentioned again, or a word/phrase emphasized for rhetorical weight rather
  than being defined. Decide `#strong` vs `#emph` by the role the phrase
  plays in the sentence, not by mechanically mapping `**text**` → `#strong[text]`
  — the source's markdown bold does not settle it
- `_text_` (markdown italic) → `#emph[text]`
- Never leave `**`, `_`, or a bare `*` used as markdown emphasis in the output —
  those characters have no special meaning in Typst body markup and will render as
  literal asterisks/underscores
- A plain quoted phrase (`"..."`) stays a plain quoted string: do not prefix it with
  `#`. `#"text"` is a Typst string _expression_, not a quoted phrase, and drops the
  visible quote marks

- Input: `- **Omniscience vs. no-regrets**: the best decision is based on the
  information available at the time of acting.`
- Bad output (a list-item lead phrase is emphasis, not a definition):
  `- #strong[Omniscience vs. no-regrets]: the best decision is based on the
  information available at the time of acting.`
- Good output:
  `- #emph[Omniscience vs. no-regrets]: the best decision is based on the
  information available at the time of acting.`

## Lists

- Not every `-` bullet in the input should become a Typst list item. A lone tagged
  bullet (`@Definition@`, `@Example@`, `@Remark@`, ...) that holds one short point
  becomes a plain sentence in the paragraph instead: see "Semantic Tags: Dissolve,
  Don't Relabel" above
- Keep a real Typst list only for content meant to be scanned as parallel items: an
  enumerated set of steps, assumptions, properties, or named alternatives. For a list
  you do keep, bullet lists (`- item`) and numbered lists (`1. item`) use the same
  syntax in Typst as in Markdown: copy the list structure and nesting as is, just
  convert the text of each item per the rules above

## Formulas

- Inline: `` `formula` `` for text-like expressions, or `$formula$` for inline math
- Display math: `$ formula $` on its own line/paragraph
- Prefer native Typst math syntax over raw LaTeX commands inside `$...$` (e.g.,
  `subset.eq`, `|X|` instead of `\subseteq`, `\abs{X}`)

## Be Direct

- **Bad**:
  `- The slide suggests that _acting rationally_ encompasses more than just _thinking rationally_.`
- **Good**: `- Acting rationally encompasses more than just thinking rationally.`

## Other Rules

- Do not repeat a bullet verbatim; expand and explain it
- Close every `#strong[`, `#emph[`, `[...]`, `(...)` you open
- Output only the converted body: no commentary, no code fence wrapping the whole
  response
