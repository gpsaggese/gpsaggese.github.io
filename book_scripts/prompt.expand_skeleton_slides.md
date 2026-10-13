- You are a college professor expert in AI, machine learning, and big data

# Goal
- The relevant variables are
  ```
  DIR="/Users/saggese/src/umd_classes1"
  BOOK="book_AI_Software_Engineering"
  CHAP="## Lesson05.1_Tools.smd"
  ```

- Read carefully the entire table of content `$DIR/$BOOK/map.md`

## Write Slides
- Execute `$DIR/helpers_root/.claude/skills/slides.write/SKILL.md` for the
  material in the skeleton slides in `$DIR/$BOOK/CHAP` using the available
  references for each slide
  ```
  * Why AI Agents Need Explicit Rules
  // From: helpers_root/CLAUDE.md
  // From: helpers_root/.claude/skills/coding.rules.md
  ```
  - These reference whole background files rather than a specific slide, so
    they omit the `<line num> '<title>'` part; see
    `.claude/skills/slides.rules.md` ("Source Attribution") for the full
    `// From: <file>:<line num> '<title>'` convention

## Style rules

- Follow the general prose and markdown conventions in `.claude/skills/text.rules.md`
  and `.claude/skills/markdown.rules.md`
- Follow the slide-specific conventions in `.claude/skills/slides.rules.md`
  (cross-referencing, titles, tags, table row order, LaTeX macros, columns
  layout, ...)

## Ask questions
- If you have doubts, do not execute the task but add questions
