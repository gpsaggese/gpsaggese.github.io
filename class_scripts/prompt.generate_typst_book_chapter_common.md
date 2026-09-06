# Typst Book Chapter: Shared Rules

Typst-specific rules shared by both `typst_aima` generation paths:
- `prompt.generate_typst_book_chapter.md` (whole-document generation)
- `prompt.generate_typst_book_chapter_slide.md` (per-slide body generation)

For general Typst syntax that applies to any `.typ` file (document
boilerplate, structural hierarchy, `#strong`/`#emph`, formulas, lists,
figures, `wrap-content`/sizing, bibliography), see the rules below:

`@.claude/skills/typst.rules.md`

This file covers only what's specific to converting *this
repo's slide markdown* into Typst: dissolving `@Tag@` semantic annotations,
and never fabricating a diagram's rendered figure. Keep this file free of
anything specific to only one of the two generation paths above: placeholder
tokens and the input/output framing of a single slide fragment belong in
`prompt.generate_typst_book_chapter_slide.md` instead

## Semantic Tags in Typst

- `@Definition@`, `@Example@`, `@Pros@`, `@Cons@`, `@Problem@`, `@Solution@`,
  `@Question@`, `@Answer@`, `@Remark@`, `@Key idea@`, ... mark the _rhetorical role_
  of the bullet that follows; `@word` is also Typst label-reference syntax, so
  leaving it verbatim is a compile error, not just a style issue. See "Dissolving
  Semantic Tags" in `prompt.generate_book_chapter_common.md` for the rules: this is
  what applying them looks like in Typst
- Never emit the tag word itself as a `#strong[...]` label (`#strong[Definition]`,
  `#strong[Pros]`, `#strong[Cons]`, `#strong[Problem]`, `#strong[Question]`, ...).
  Use `#strong[...]` only for the actual term or claim being introduced

- Input:

  ```text
  - @Definition@: An **agent** is something that perceives and acts to reach a goal
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
  `#strong` — see "Highlighting and Emphasis" in `.claude/skills/typst.rules.md`):

  ```text
  - #emph[Omniscience vs no-regrets]: the best decision is based on
    available information, not perfect knowledge.
  ```

## Diagrams: Never Fabricate the Rendered Figure

- A `graphviz`/`mermaid`/`tikz`/... diagram is never turned into its final
  `#figure(image(...))` call by you
- That is done later, deterministically, by Python (`_render_diagram_placeholder()`
  in `gen_book_chapter.py`, then `render_images.py`), which is the only thing that
  knows the real output path and actually produces the PNG. You never see that path,
  so you cannot write it: any `image(...)` call you invent for a diagram points at a
  file that will never exist, and the chapter fails to compile with "file not found"

- This holds in both generation paths, only the shape of what reaches you differs:
  - Whole-document path: you receive the raw fence (` ```graphviz `,
    ` ```mermaid `, ` ```tikz `, ...) inline in the source — copy it through
    completely unchanged, character for character, at the same position; do not add a
    caption, label, `#figure(...)`, or `#wrap-content(...)` around it, even if you
    can see how it should be captioned. See "Typst Syntax Requirements" in
    `prompt.generate_typst_book_chapter.md`
  - Per-slide path: the fence has already been replaced with an opaque `@@FIGURE_N@@`
    token before you ever see the input — treat it as meaningless text with no
    diagram behind it. See "Placeholder Tokens" in
    `prompt.generate_typst_book_chapter_slide.md`

- Input (whole-document path):
  ````text
  ```graphviz
  digraph laws_of_thought {
      premises [label="Correct\npremises"];
      logic [label="Logic"];
      conclusion [label="Correct\nconclusions"];
      premises -> logic -> conclusion;
  }
  ```
  ````
- Bad output (fabricates a path, caption, and layout no one gave you):
  ```text
  #wrap-content(
    [
      #figure(
        image("Lesson01.2-AI_and_Machine_Learning.typ.figs/Lesson01.2-AI_and_Machine_Learning.1.png", width: 100%),
        caption: [Diagram relating Correct premises, Logic and Correct conclusions],
        kind: "figure",
        supplement: [Fig.],
        placement: auto,
      ) <fig:2aiasthinkingrationally>
    ],
    align: right,
    columns: (1fr, 20%),
  )[...]
  ```
- Good output (the fence, untouched, exactly where it was):
  ````text
  ```graphviz
  digraph laws_of_thought {
      premises [label="Correct\npremises"];
      logic [label="Logic"];
      conclusion [label="Correct\nconclusions"];
      premises -> logic -> conclusion;
  }
  ```
  ````
