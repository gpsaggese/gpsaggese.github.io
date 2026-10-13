# Goal
- Create the slides of one lesson from an ad hoc list of topics
- Use it when the chapter is not in a book map yet, or when `BOOK` is a course
  (e.g., `msml610`)
- For lessons from a chapter TOC of a book map use
  `book_scripts/prompt.create_lesson_slides_from_chap_toc.md`

# Parameters
- The relevant variables are:
  ```
  DIR="/Users/saggese/src/umd_classes1"
  BOOK="msml610"
  CHAP_NUM="09"
  ID="09"
  TITLE="Classical_Search_Algorithms"
  ```

- Output file is
  `<DIR>/<BOOK>/lectures_source/Lesson<CHAP_NUM>.<ID>_<TITLE>.smd`
  - `<CHAP_NUM>`: chapter number, two digits
  - `<ID>`: lesson index inside the chapter, two digits (`01` for the first
    lesson)
  - `<TITLE>`: lesson title with words joined by `_`
  - If the file already exists, overwrite it

## Topics
- List the topics to cover, one per bullet
  - E.g., defining a search problem via states, initial state, actions,
    transition model, goal test, and path cost
  - E.g., uninformed search
- (Optional) Align the content with `<FILE>`

# Workflow

## Step 1: Understand the Plan
- Read the topics in `## Topics`
- Read the file given in `Align the content with`, if present

## Step 2: Collect the Related Material
- Read `.claude/skills/slides.rules.md` to understand how slides are organized
- Read the lessons already generated in `<DIR>/<BOOK>/lectures_source/*.smd`
  and make sure that
  - There is no redundancy with the existing material, especially with the
    file given in `Align the content with`
  - A lesson can point back to another lesson (see
    `.claude/skills/slides.rules.md`, "Do Not Cross-Reference Other
    Lessons")

## Step 3: Create a Skeleton of Slides
- Create a list of slide titles only, about 30-35 slides covering `## Topics`
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

- If you reuse slides from an existing lecture file, add a `// From: ...`
  comment right after `* <slide title>`
  - Follow "Source Attribution" in `.claude/skills/slides.rules.md`
  - E.g.,
    ```
    // From: msml610/lectures_source/Lesson08.1-Causal_AI_intro.smd:318 '* Where ML Falls Short'
    ```

- Do not run the linter on the slides

## Step 5: Save Results
- Save the slides in the output file described in `# Parameters`

// Example of parameters, ignore
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
