# Typst Book Chapter

Mode-specific instructions for `--mode typst_aima`. See
`prompt.generate_book_chapter_common.md` for the shared style guide
(audience, tone, content rules, constraints): this file covers only the
Typst syntax and structure.

Copied and adapted from
`/Users/saggese/src/notes1/book_proposals/prompt.create_typst_book_chap_from_lesson_slides.md`.

## Output Format

- Generate Typst with a complete document structure, starting with the
  document template below (do not omit the `#import`/`#show` boilerplate)

## Document Template

```typst
// Import AIMA style formatting and macros.
#import "aima_style.typ": aima-style, chapter, algorithm, glossary
// Import the custom citation/bibliography system.
#import "/helpers_root/dev_scripts_helpers/typst/umd_references.typ": cite, references

// Document metadata
#set document(
  title: "[Chapter Title]",
  author: "[Course Title]",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter([CHAPTER_NUMBER], "[Chapter Title]")
```

- The chapter number and title, the course title, and the exact relative
  path for the `#import` line are given in the user message; use them
  verbatim

## Structural Hierarchy → Typst

- H1 (`#`) → `#chapter([num], "Title")` (only once, at the top; do not
  repeat for further H1s in the same file, use `== Title` instead if a
  second top-level topic appears)
- H2 (`##`) → `== Title`
- H3+ (`###`+) → `=== Title`
- Slide-level heading (`*`) → a paragraph starting with `#strong[Heading]`
  followed by its body text

## Source Attribution

- Comment marker: `//`
- Before each section of content taken from a slide, add a comment to help
  track source and maintain alignment with slides:
  ```typst
  // Slide: [slide title or description]
  ```
- Also add the shared source-attribution comment right above it:
  ```typst
  // From: msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.smd:12 '## Some Heading'
  // Slide: Some Heading
  ```

## Highlighting and Emphasis

- Use `#strong[text]` for concepts, definitions, algorithm names, and key
  terms
- Use `#emph[text]` for italics sparingly (emphasis only, not decoration);
  `*text*` in Typst markup renders as bold, not italic, so never use it for
  emphasis

## Algorithms and Pseudocode

- Use `#algorithm("NAME", [...])` for all structured algorithms, pseudocode,
  or procedural content
  - Format with `#h(1em)` for indentation levels
  - Use `*keyword*` for language keywords (function, if, loop, return, etc.)
  - Use symbolic notation ($sigma$, $in.not$, etc.) where appropriate

## Formulas

- Inline: `` `formula` `` for text-like expressions (e.g.,
  `` `f(n) = g(n) + h(n)` ``), or `$formula$` for inline math
- Display math: `$ formula $` on its own line/paragraph
- Prefer native Typst math syntax over raw LaTeX inside `$...$` (e.g.,
  `subset.eq`, `|X|` instead of `\subseteq`, `\abs{X}`)

## Use Lists

```typst
#list(
  [Use *item1* when:
    #list([...])],
  [Use *item2* when:
    #list([...])],
)
```

## Figures

```typst
#figure(
  image("chart.png", width: 80%),
  caption: [Custom labeled figure.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:chart>
```

- `placement` takes a bare keyword (`auto`, `none`, `top`, `bottom`), not a
  string — `placement: "auto"` is a type error

- Every figure needs a label (`<fig:<description>>`), a one-line caption,
  and a reference in the text (`@fig:diagram`) that integrates it into the
  prose
- Image paths are relative to the source file location (use `../` as
  needed)
- If the source markdown documents the figure with a commented block (e.g.,
  a `graphviz`/`mermaid` code fence under a `rendered_image:begin` /
  `rendered_image:end` marker), keep that comment block above the `#figure`
  call and update its `label`/`caption` fields to match, so the two stay in
  sync:
  ```
  // rendered_image:begin
  // ```graphviz
  //    ... image code ...
  // ```
  // label=fig:my_label
  // caption=This is a caption
  // rendered_image:end
  ```

## Bibliography

- Never use Typst's native `#bibliography(...)`/`[@key]` citation syntax --
  it cannot render a custom link label (only the raw URL/DOI text itself
  can be hyperlinked). Use the shared `umd_references.typ` module instead,
  which renders inline citations as superscript bracketed numbers (`[10]`)
  and reference-list entries as `Author et al., "Title", Venue, Year. link`
  (see `.claude/references.rules.md`)
  (already imported by the document template above)
- Cite inline with `#cite("<bib-key>")` (not `[@<bib-key>]`):
  ```typst
  The Turing test #cite("turing1950computing") remains influential.
  ```
- End the `References` section with:
  ```typst
  #references("/msml610/lectures_source/refs.bib")
  ```

## Typst Syntax Requirements

- Follow Typst (not markdown) syntax: `#strong[text]` not `**text**`,
  `#emph[text]` not `*text*`/`_text_`, `== Heading` / `=== Subheading`
  (auto-numbered)
- Do NOT mix markdown and Typst syntax (no `**`, `__`, `~~`, `*text*`)
- Never rewrite the content inside a raw code fence (` ```mermaid `,
  ` ```graphviz `, ` ```tikz `, etc.) — copy it verbatim, character for
  character; those blocks are not Typst and must not contain `#strong[`,
  `#emph[`, or other Typst markup
- If the source slide markdown wraps native Typst calls in a pandoc raw
  block (` ```{=typst} ... ``` `) or an inline raw span (`` `code`{=typst} ``,
  used for a Typst call like `#cite(...)` sitting mid-sentence), do NOT
  copy that fence/span into the `.typ` output — the output document is
  already native Typst, so this pandoc-only wrapping is not executed
  there, it is shown as inert literal text. Strip the fence markers or
  the backticks + `{=typst}` and emit the enclosed Typst code directly
- Never leave semantic tags such as `@Definition@`, `@Question@`,
  `@Example@` verbatim in a `.typ` file — `@word` is Typst
  label-reference syntax there, so it is a compile error, not just a style
  issue; convert each into `#strong[Word]` (or better, weave it into the
  prose) instead
- Always close `#strong[`, `[...]`, `(...)` with matching brackets
