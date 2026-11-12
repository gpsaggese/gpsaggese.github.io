# Goal
- Create the slides of one lesson from an ad hoc list of topics
- Use it when the chapter is not in a book map yet, or when `BOOK` is a course
  (e.g., `msml610`)
- For lessons from a chapter TOC of a book map use
  `book_scripts/prompt.create_lesson_slides_from_chap_toc.md`
- The steps shared with that prompt are in
  `book_scripts/prompt.create_lesson_slides.common.md`

# Parameters
- The relevant variables are:
  ```
  DIR="/Users/saggese/src/umd_classes1"
  BOOK="msml610"
  CHAP_NUM="09"
  ID="09"
  TITLE="Classical_Search_Algorithms"
  ```
  - The output file is built from them as described in
    `book_scripts/prompt.create_lesson_slides.common.md`

<!--
// Example of parameters (to ignore)
// BOOK="msml610"
// CHAP_NUM="09"
// ID="09"
// TITLE="Classical_Search_Algorithms"
// Topics:
// - defining a search problem via states, initial state, actions, transition
//   model, goal test, and path cost; distinguishing the state space from the
//   search tree/graph
// - uninformed search
// - informed search
// - A* search (and its property)
// - Adversarial search and game trees
// - minimax
// - alphabeta pruning
// - depth limited search
// - Use tic-tac-toe and connect 4 as examples
// Align the content with
// `msml610/lectures_source/Lesson09.8-MonteCarlo_Tree_Search.smd`
-->

## Topics
- List the topics to cover, one per bullet
  - E.g., defining a search problem via states, initial state, actions,
    transition model, goal test, and path cost
  - E.g., uninformed search
- (Optional) Align the content with `<FILE>`

# Workflow

## Understand the Plan
- Read the topics in `## Topics`
- Read the file given in `Align the content with`, if present

## Collect the Related Material
- Read `.claude/skills/slides.rules.md` to understand how slides are organized
- Read the lessons already generated in `<DIR>/<BOOK>/lectures_source/*.smd`
  and make sure that
  - There is no redundancy with the existing material, especially with the
    file given in `Align the content with`
  - A lesson can point back to another lesson (see
    `.claude/skills/slides.rules.md`, "Do Not Cross-Reference Other
    Lessons")

## Create the Lesson
- Execute `book_scripts/prompt.create_lesson_slides.common.md`
  - The topics are `## Topics`
  - The material read is the existing lessons and the file given in
    `Align the content with`
