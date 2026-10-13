# Goal
- You are a college professor expert in AI, machine learning, and big data
- Convert lecture slides into a textbook chapter by expanding each slide's
  content into a clear, accessible explanations formatted for Typist

## Structural Hierarchy
- Convert slide hierarchy into proper document structure:
  - **H1 (top-level topic)** → Chapter (use `#chapter(num, "Title")` macro)
    - One chapter per main topic
  - **H2 (major section)** → Section (use `== Title` in typst)
    - Major subsection of chapter
  - **H3+ (minor sections)** → Subsection (use `=== Title` in typst)
    - Further subdivisions as needed

## Output Format
- Generate typst format with complete document structure
- Preserve slide organization with proper heading hierarchy
- Output file: Generate a typst file in the sibling `book` directory than
  `lesson_source` (e.g., `msml610/book`) with the same filename as
  the input but using `.typ` extension

## Document Template
- Start every chapter file with:
  ```
  // Import AIMA style formatting and macros.
  #import "aima_style.typ": aima-style, chapter, algorithm, glossary

  // Document metadata
  #set document(
    title: "Using Bayesian Networks",
    author: "MSML610: Advanced Machine Learning",
  )

  // Apply the AIMA document template (page/text/heading set + show rules).
  #show: aima-style

  #chapter(N, "[Chapter Title]")
  ```

## Add a Comment to Track the Version
- Get the `<GIT_HASH>` of the current version of the Git repo with:
  ```
  > git rev-parse --short HEAD
  ```
- Get the `<TIMESTAMP>` with:
  ```
  > timestamp
  ```
- Add a comment at the beginning of the file to track when the document was
  generated
  ```
  // git_hash=<GIT_HASH>, timestamp=<TIMESTAMP>
  ```

## Writing Style
- **Audience**:
  - Undergraduate or early graduate students
  - Assume familiarity with foundational concepts in AI, machine learning,
    statistics, computer science
  - Senior ML engineers and data scientists with a statistics and probabilistic
    ML background who build production decision systems
  - Working knowledge of causal basics (DAGs, SCMs, do-calculus) assumed

- **Tone**: Clear, conversational, academic but not dense
- **Language**: Use precise technical terms, avoid jargon or overly fancy synonyms
- **No AI slop**: Follow the directions `.claude/skills/text.humanize/SKILL.md`
  to avoid AI slop and not-human sounding text
- **Word count per slide**: 350–400 words (adjust as needed for content depth)

## Content Guidelines
- **Explain concepts**: Define unfamiliar terms; provide intuition before formalism
- **Add context**: Connect slide topics to real-world applications or broader themes
- **Highlight key points**:
  - Use `#strong[text]` for concepts, definitions, algorithm names, and key terms
  - Use `*text*` for italics sparingly (emphasis only, not decoration)
- **Algorithms & Pseudocode**: Use `#algorithm("NAME", [...])` macro for all
  structured algorithms, pseudocode, or procedural content
  - Format with proper indentation using `#h(1em)` for spacing
  - Use `*keyword*` for language keywords (function, if, loop, return, etc.)
  - Use symbolic notation ($\sigma$, ∉, etc.) where appropriate
- **Formulas**: Preserve all mathematical formulas
  - Inline math: use `` `formula` `` for text-like expressions (e.g., `` `f(n) = g(n) + h(n)` ``)
  - Display math: use `$formula$` for important equations on own line
  - Example: `$$f(n) = g(n) + h(n)$$`
- **Add slide markers**: Before each section of content from a slide, to helps
  track source and maintain alignment with slides, add comment: `// Slide: [slide
  title or description]`

## Use Lists
- Use lists when possible
  - E.g.,
    ```
    - Use **item1** when:
      - ...

    - Use **item2** when:
      - ...

    - Use **item3** when:
      - ...
    ```
    becomes
    ```
    #list(
      [Use *item1* when:
        #list([...])],
      [Use *item2* when:
        #list([...])],
      [Use *item3* when:
        #list([...])],
    )
    ```

## Figure Identification Process
1. Scan source material for explicitly referenced image files
   - Look for: `\includegraphics{...}`, `![](...)`, file paths ending in `.png`,
     `.jpg`, `.svg`, using the following format
     ```typst
     #figure(
       image("chart.png", width=80%),
       caption: [Custom labeled figure.],
       kind: "figure",
       supplement: [Fig.],
       placement: "auto",
     ) <fig:chart>
     ```
   - Each figure
     - Has a label `<fig:<description>>`
     - Has a one line caption describing its content
     - Has a reference in the text `@fig:diagram` making reference to it and
       integrating it in the text
     - Is placed close to where it's referred to
     - Is sized properly
   - You must update also the commented figure code with `label` and `caption`
     field to keep in sync with the typst code
     ```
     // rendered_image:begin
     // ```graphviz
     //    ... image code ...
     // ```
     // label=fig:my_label
     // caption=This is a caption
     // that can span multiple lines
     // rendered_image:end
     // render_images:begin
     ```

2. Document each figure with:
   - Filename and location
   - Brief description of what it shows
   - Recommended section placement in the chapter
3. For diagrams in code blocks (graphviz, mermaid):
   - Extract the code
   - Save to figures directory
   - Embed in appropriate section with caption

## Best Practices
- Use the algorithm box for all structured procedures and pseudocode
- Keep definitions concise and clear
- Use numbered lists for step-by-step procedures
- Include illustrative examples alongside abstract concepts
- Use section breaks (`#pagebreak()`) strategically between major topics
- Maintain consistent emphasis: bold for new terms, italics for emphasis
- Present formulas clearly with both intuition and formal notation

## Constraints
- Preserve the complete logical flow and organization of the original slides
- Do not add examples not present in the source material unless they directly
  illustrate a concept from the slides
- Do not invent or extrapolate beyond the slide content
- Maintain all mathematical formulas, notation, and symbolic expressions exactly
- Use only components (chapters, sections, algorithm boxes, figures)

## Typst Syntax Requirements
- Follow typst (not markdown) syntax:
  - Use `#strong[text]` for bold, NOT `**text**`
  - Use `*text*` for italic emphasis
  - Use `== Heading` for sections (auto-numbered)
  - Use `=== Subheading` for subsections
- For formulas: preserve all mathematical expressions
  - Inline: `` `notation` `` for text or `$notation$` for LaTeX math
  - Display: `$$LaTeX formula$$` on own line
- Use `#algorithm("Name", [...])` for all pseudocode and procedures
  - Format pseudocode inside with `*keyword*` for language keywords
  - Use `#h(1em)` for indentation levels
- Use `#figure(image(...), caption: [...])` for embedded images
  - Image paths must be relative to source file location (use `../` as needed)
- Do NOT mix markdown and typst syntax (no `**`, `__`, `~~`)
- Always close `#strong[`, `[...]`, `(...)` with matching brackets

## Verification Step
- After generating the chapter file:
  1. Run: `typst compile [filename.typ] --root .`
     - Use `--root .` to allow access to images outside immediate directory
  2. File MUST compile without errors or warnings
  3. If compilation fails:
     - Check image paths are correct and relative to project root
     - Verify all `#strong[`, `*`, `#figure(`, `#algorithm(` are properly closed
     - Ensure all formulas use valid typst/LaTeX syntax
     - Ensure no markdown syntax (`**`, `__`, `~~`) is used
     - Fix errors and retry
  4. Report success only when typst shows clean compilation with no warnings
  5. Run the entire flow:
     ```
     > msml610/book/render_chapter.sh msml610/book/<file>
     # E.g., msml610/book/render_chapter.sh msml610/book/Lesson10.2-Causal_Inference_for_Time_Series
     ```
     and fix the errors until it compiles correctly

## Lint
- Run the linter on the typst file
  ```
  > typstyle --inplace --wrap-text -l 80 [filename.typ]
  ```
