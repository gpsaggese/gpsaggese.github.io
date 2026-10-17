# Goal
- Create the slides of one lesson from a chapter TOC of a book map
- For lessons from an ad hoc list of topics (no book map) use
  `book_scripts/prompt.create_lesson_slides_from_topic.md`
- The steps shared with that prompt are in
  `book_scripts/prompt.create_lesson_slides.common.md`

# Parameters
- The relevant variables are:
  ```
  DIR="/Users/saggese/src/umd_classes1"
  BOOK="book_Modern_AI_for_Finance"
  CHAP="## 04: Data Engineering"
  ```
  - `CHAP` is an H2 chapter of `<DIR>/<BOOK>/book_map.md`
  - The topics come from `### Topics` of that chapter

- Derive the variables of the output file from `CHAP`
  - `<CHAP_NUM>`: chapter number from `CHAP`, two digits
  - `<ID>`: `01` for the first lesson, unless the user says otherwise
  - `<TITLE>`: chapter title from `CHAP` with words joined by `_`
  - E.g., `## 01: Why Decisions, Not Predictions` becomes
    `book_springer/lectures_source/Lesson01.01_Why_Decisions_Not_Predictions.smd`

<!--
// Examples of parameters (to ignore)
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
-->

# Workflow

## Understand the Plan
- The format of a book map is in `.claude/skills/book_map.rules.md`
- Read carefully the entire table of content `<DIR>/<BOOK>/book_map.md`
- Read the target chapter `<CHAP>` that you need to convert from TOC into
  slides

## Collect the Related Material
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

## Create the Lesson
- Execute `book_scripts/prompt.create_lesson_slides.common.md`
  - The topics are `### Topics` of `<CHAP>`
  - The material read is `### Lesson Materials`, `### Papers`, `### Books`,
    and `<DIR>/<BOOK>/resources.md`
