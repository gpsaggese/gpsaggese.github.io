# Goal
- Consider a `<FILE>` with the skeleton of slide deck using:
  - `#` H1
  - `##` H2 headers
  - `* <slides>`

- See conventions about slides in `.claude/skills/slides.rules.md`

- The template looks like:
  ```
  # <H1 title>

  ## <H2 title>

  * <slide title>
  // From: <file>:<line num> '<title of slide>'

  * <slide title>
  // From:

  ## <H2 title>
  ```
  - `<file>` is a repo-root-relative path to the exact source `.txt` / `.smd`
    file, `<line num>` its line, and `'<title of slide>'` that line's full
    text single-quoted verbatim, leading marker included (`* `)
  - If no source slide can be identified, leave a bare `// From:` (nothing
    after the colon) instead of guessing
  - See `.claude/skills/slides.rules.md` ("Source Attribution") for the full
    convention

- E.g.,
  ```
  # Representation as a Foundation for Decisions

  * Why Knowledge Representation Matters
  // From: msml610/lectures_source/Lesson03.1-Knowledge_representation.txt:42 '* Why Knowledge Representation Matters'
  ```

# Workflow
- Find the slides referred from the `// From: ...` clauses
- Inline the slides in that spot
- Keep the `// From: ...` clause in place like:
  ```
  * <slide title>
  // From: <file>:<line num> '<title of slide>'

  <slide content>
  ```

## Example
- E.g., this snippet
  ```
  * Why Knowledge Representation Matters
  // From: msml610/lectures_source/Lesson03.1-Knowledge_representation.txt:42 '* Why Knowledge Representation Matters'

  * <Next slide>
  // From: <file>:<line num> '<title of slide>'
  ```
  becomes
  ```
  * Why Knowledge Representation Matters
  // From: msml610/lectures_source/Lesson03.1-Knowledge_representation.txt:42 '* Why Knowledge Representation Matters'

  - @Motivation@: **Why KR is essential to AI**

    1) **Learning alone is not enough**: machines need to _reason_ about the world
       - Example: a medical AI trained on patient data can predict diseases
       - But to _explain_ a diagnosis to a doctor, it needs structured knowledge

    2) **Bridges perception and reasoning**
       - Sensors provide raw data (perception)
       - KR turns data into actionable knowledge (reasoning)

    3) **Enables explainability**
       - Users can understand _why_ a system made a decision
       - Critical for high-stakes domains (healthcare, law, autonomous systems)

    4) **Enables white-box learning**
       - Robots plan actions using abstract symbolic knowledge
       - Conversational agents reason about intent and context

  * <Next slide>
  // From: <file>:<line num> '<title of slide>'
  ```

## Errors
- Report an error if:
  - a file `<file>` doesn't exist
  - `<file>` exists, but the title `<slide title>` doesn't exist
- In case there is a missing slide title, leave a placeholder with a bare
  `// From:` (nothing after the colon)
  ```
  * <slide title>
  // From:
  ```

## Save Result
- Save the output in the same file as the input, since the work is done in place
