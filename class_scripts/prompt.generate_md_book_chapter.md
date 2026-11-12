# Markdown Book Chapter

Mode-specific instructions for `--mode md`. See
`prompt.generate_book_chapter_common.md` for the shared style guide
(audience, tone, content rules, constraints): this file covers only the
Markdown syntax and structure.

## Output Format

- Generate plain Markdown (this is later converted to PDF via pandoc, so
  stick to standard pandoc Markdown, plus the raw-LaTeX escape hatches
  described below)
- Start the file with a YAML title preamble:
  ```markdown
  ---
  title: "[Chapter Title]"
  ---
  ```
  The chapter title is given in the user message; use it verbatim

## Structural Hierarchy → Markdown

- H1 (`#`) → `#` (chapter title; only once, matching the YAML title)
- H2 (`##`) → `##`
- H3+ (`###`+) → `###`, `####`, ...
- Slide-level heading (`*`) → a paragraph starting with `**Heading.**`
  followed by its body text

## Source Attribution

- Comment marker: HTML comment
  ```markdown
  <!-- From: msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.smd:12 '## Some Heading' -->
  ## Some Heading
  ```

## Highlighting and Emphasis

- Use `**text**` for key terms, concepts, and definitions
- Use `*text*` for italics sparingly (emphasis only, not decoration)
- Use `` `text` `` for code, file names, or technical identifiers

## Algorithms and Pseudocode

- Use a fenced code block for structured algorithms, pseudocode, or
  procedural content:
  ````markdown
  ```text
  function A_STAR(start, goal):
      ...
  ```
  ````
- Use `**keyword**` for language keywords inside the block (function, if,
  loop, return, etc.)

## Formulas

- Inline math: `$formula$` (e.g., `$f(n) = g(n) + h(n)$`)
- Display math: `$$formula$$` on its own paragraph
- These render through pandoc/xelatex, so use standard LaTeX math syntax

## Special Constructs

- Definitions, theorems, and similar callouts: use a blockquote with a bold
  label, since plain Markdown has no custom environments:
  ```markdown
  > **Definition.** An *agent* is something that perceives and acts to
  > reach a goal.
  ```
  Use `**Theorem.**`, `**Important.**`, `**Warning.**`, `**Tip.**` the same
  way

## Use Lists

```markdown
- Use **item1** when:
  - ...

- Use **item2** when:
  - ...
```

- Use numbered lists (`1.`, `2.`, ...) for step-by-step procedures

## Figures

```markdown
![Concise description of figure content and relevance.](path/to/figure.png){width=70%}
```

- Every figure needs a one-line caption (the `[...]` text) and a reference
  from the surrounding prose (e.g., "as shown below")
- Image paths are relative to the repository root

## Markdown Syntax Requirements

- Use standard pandoc Markdown; do not invent custom syntax
- Always close `**`, `*`, `` ` ``, and blockquote markers correctly
- Do not use raw HTML except for the source-attribution comments described
  above
