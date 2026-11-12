# Typst Book Chapter

Mode-specific instructions for `--mode typst_aima`'s whole-document generation path.
See `prompt.generate_book_chapter_common.md` for the shared style guide (audience,
tone, content rules, constraints). `prompt.generate_typst_book_chapter_common.md` is
concatenated ahead of this file, so its rules already apply here: general Typst
syntax, pulled in via its own `@.claude/skills/typst.rules.md` reference, plus what's
specific to this pipeline (dissolving semantic tags, never fabricating a diagram's
rendered figure). This file additionally inlines the document skeleton to follow,
since only the whole-document path needs it (the per-slide path builds the document
boilerplate in Python instead):

`@.claude/templates/typst.template.typ`

This file covers only what none of the above cover: structure and syntax specific to
generating the whole document in one call.

## Output Format

- Generate Typst with a complete document structure, following the template above (do
  not omit the `#import`/`#show` boilerplate)
- The chapter number and title, the course title, and the exact relative path for the
  `#import` line are given in the user message; use them verbatim in place of the
  template's placeholder values

## Source Attribution

- Comment marker: `//`
- Before each section of content taken from a slide, add a comment to help track
  source and maintain alignment with slides:
  ```typst
  // Slide: [slide title or description]
  ```
- Also add the shared source-attribution comment right above it:
  ```typst
  // From: msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.smd:12 '## Some Heading'
  // Slide: Some Heading
  ```

## Figures

- If the source markdown documents a figure with a commented block (e.g., a
  `graphviz`/`mermaid` code fence under a `rendered_image:begin` /
  `rendered_image:end` marker), keep that comment block above the `#figure` call and
  update its `label`/`caption` fields to match, so the two stay in sync:
  ```
  // rendered_image:begin
  // ```graphviz
  //    ... image code ...
  // ```
  // label=fig:my_label
  // caption=This is a caption
  // rendered_image:end
  ```
  (see "Figures: Required Elements" and "Every Visual Pairs With Its Text" in
  `.claude/skills/typst.rules.md` for the `#figure`/`#wrap-content` rules themselves)

## Typst Syntax Requirements

- Never rewrite the content inside a raw code fence (` ```mermaid `, ` ```graphviz `,
  ` ```tikz `, etc.) — copy it verbatim, character for character; those blocks are
  not Typst and must not contain `#strong[`, `#emph[`, or other Typst markup. Never
  replace the fence with a `#figure(image(...))` call either, however confident you
  are about its eventual caption or path — see "Diagrams: Never Fabricate the
  Rendered Figure" in `prompt.generate_typst_book_chapter_common.md`
- If the source slide markdown wraps native Typst calls in a pandoc raw block (`
  ```{=typst} ... ``` `) or an inline raw span (`` `code`{=typst} ``, used for a
  Typst call like `#cite(...)` sitting mid-sentence), do NOT copy that fence/span
  into the `.typ` output — the output document is already native Typst, so this
  pandoc-only wrapping is not executed there, it is shown as inert literal text.
  Strip the fence markers or the backticks + `{=typst}` and emit the enclosed Typst
  code directly
