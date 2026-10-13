# Goal
- Create the slides of one lesson from a chapter TOC of a book map
- For lessons from an ad hoc list of topics (no book map) use
  `book_scripts/prompt.create_lesson_slides_from_topic.md`

# Parameters
- The relevant variables are:
  ```
  DIR="/Users/saggese/src/umd_classes1"
  BOOK="book_Modern_AI_for_Finance"
  CHAP="## 01, Why Finance Is Different"
  ```
  - `CHAP` is an H2 chapter of `<DIR>/<BOOK>/book_map.md`
  - The topics come from `### Topics` of that chapter

- Output file is
  `<DIR>/<BOOK>/lectures_source/Lesson<CHAP_NUM>.<ID>_<TITLE>.smd`
  - `<CHAP_NUM>`: chapter number from `CHAP`, two digits
  - `<ID>`: lesson index inside the chapter, two digits (`01` for the first
    lesson, unless the user says otherwise)
  - `<TITLE>`: chapter title from `CHAP` with words joined by `_`
  - E.g., `## 01: Why Decisions, Not Predictions` becomes
    `book_springer/lectures_source/Lesson01.01_Why_Decisions_Not_Predictions.smd`
  - If the file already exists, overwrite it

# Workflow

## Step 1: Understand the Plan
- The format of a book map is in `.claude/skills/book_map.rules.md`
- Read carefully the entire table of content `<DIR>/<BOOK>/book_map.md`
- Read the target chapter `<CHAP>` that you need to convert from TOC into
  slides

## Step 2: Collect the Related Material
- Read
  - `### Goals` and `### Topics` of `<CHAP>`
  - The slide material pointed by `### Lesson Materials`, if present
  - The papers and books pointed by `### Papers` and `### Books`
  - The pointers in `<DIR>/<BOOK>/resources.md`, if present
- Read the lessons already generated in `<DIR>/<BOOK>/lectures_source/Lesson*`
  and make sure that
  - There is a single logic flow with the rest of the lessons
  - There is no repetition with respect to the rest of the lessons
    - A lesson can point back to another lesson (see
      `.claude/skills/slides.rules.md`, "Do Not Cross-Reference Other
      Lessons")

## Step 3: Create a Skeleton of Slides
- Create a list of slide titles only, about 30-35 slides covering `### Topics`
- Show the list to the user and wait for approval
  - Skip the approval only if the user asks for it explicitly
- If you have doubts or something is not clear, stop and ask questions before
  proceeding

## Step 4: Create Slides
- Execute `.claude/skills/slides.write/SKILL.md` with `<NUM_SLIDES>` = 30-35
  using the approved skeleton and the material read in Step 2

- Follow all formatting rules, templates, and structural guidelines in
  `.claude/skills/slides.rules.md`
  - Prose and markdown conventions are in `.claude/skills/text.rules.md` and
    `.claude/skills/markdown.rules.md`

- If you use slides from `### Lesson Materials` or material from `### Papers`
  and `### Books`, add a `// From: ...` comment right after `* <slide title>`
  - Follow "Source Attribution" in `.claude/skills/slides.rules.md`
  - E.g.,
    ```
    // From: msml610/lectures_source/Lesson08.1-Causal_AI_intro.smd:318 '* Where ML Falls Short'
    ```

- Do not run the linter on the slides

## Step 5: Save Results
- Save the slides in the output file described in `# Parameters`

// Examples of parameters, ignore
// DIR="/Users/saggese/src/umd_classes1"
//
// BOOK="book_AI_Software_Engineering"
// CHAP="## 01: LLM-Assisted Code Generation"
// CHAP="## 02: AI Pair Programming Workflows"
// CHAP="## 03: Prompt Engineering for Code"
// CHAP="## 04: Retrieval-Augmented Generation for Codebases"
// CHAP="## 14: Human-AI Collaboration Patterns"
//
// BOOK="book_Decentralized_AI"
// CHAP="## 01: Blockchain Consensus Foundations"
