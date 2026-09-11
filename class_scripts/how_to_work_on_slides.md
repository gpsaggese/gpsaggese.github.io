# How to Work on Slides

- This document explains how a lecture moves from source to a released slide PDF, a
  book chapter, lecture commentary, and the published class links
  - Source: `msml610/lectures_source/LessonNN.M-Title.smd`, e.g
    `Lesson01.1-Intro.smd`
- Before editing a `.smd` file, read:
  - `.claude/skills/slides.rules.md`: slide content conventions
  - `.claude/skills/book.rules.md`: for how a book is organized
  - `.claude/skills/typst.rules.md`: book chapter conventions

## Command Summary

| Command                     | Purpose                                                                                   |
| :-------------------------- | :---------------------------------------------------------------------------------------- |
| `gen_slides.py`             | Render one lecture's `.smd` source to a slide PDF                                         |
| `notes_to_pdf.py`           | Convert a `.smd`/text file to PDF, HTML, or slides |
| `lint_text.py`              | Lint and format a `.smd` file                                                             |
| `gen_book_chapter.py`       | Generate a Typst, LaTeX, or Markdown book chapter from a lecture                          |
| `run_typst.py`              | Compile a `.typ` file to PDF, and optionally compress it                                  |
| `compress_pdf.py`           | Compress a PDF with Ghostscript                                                           |
| `gen_lecture_commentary.py` | Generate per-slide LLM commentary for a lecture                                           |
| `for_loop_lessons.py`       | Run one action across many lectures                                                       |
| `publish_class_links.py`    | Generate the HTML page linking each lecture's artifacts                                   |

// TODO(ai_gp): add render_...

## File Layout

- `lectures_source/LessonNN.M-Title.smd`: slide source
- `lectures_pdf.tmp/`: staging output of `gen_slides.py`, uncompressed, overwritten
  on every run
- `lectures_pdf/`: released slide PDFs, populated by `gen_slides.py --action release`
- `book/`: generated book chapter per lecture (`.typ`, compiled `.pdf`, and for
  `--mode md` an `.md` file), produced by `gen_book_chapter.py`
- `lectures_commentary/`: per-slide LLM commentary (`.book_chapter.md`,
  `.book_chapter.pdf`, `.book_chapter.html`) plus extracted slide images, produced by
  `gen_lecture_commentary.py`
- `all_tocs.md`: consolidated syllabus, produced by
  `for_loop_lessons.py --action generate_toc`

## Workflow

// TODO(ai_gp): Do not use numbering
### 1. Iterate on the Slide Content

- In brief the flow is
  ```
  > LID=02.5; vi msml610/lectures_source/Lesson$LID*.smd msml610/book/Lesson$LID*.typ
  > LID=02.5; gen_slides.py -i msml610/$LID --daemon
  > LID=02.5; render_book_chapter.py -i msml610/$LID --daemon
  ```

- Edit the source:

  ```bash
  > vi msml610/lectures_source/Lesson03.3-Non_classical_logics.smd
  ```

- Render it to check the output:

  ```bash
  > gen_slides.py -i msml610/lectures_source/Lesson03.3-Non_classical_logics.smd
  ```

  - The PDF is written to `lectures_pdf.tmp/`, not `lectures_pdf/`
- For continuous iteration, run in daemon mode: it watches the source file, rebuilds
  on every save, opens the PDF once, and only refreshes it after that

  ```bash
  > gen_slides.py -i msml610/01.1 --daemon
  ```

  - Daemon mode supports one lecture at a time

### 2. Review and Polish the Slides

- Run these skills in order on the source file (`$SMD_FILE`):
  1. `claude> /slides.review $SMD_FILE`: restructure the slides and fix the
     high-importance issues found
  2. `claude> /slides.add_visuals $SMD_FILE`: propose and add diagrams
  3. `claude> /slides.add_references $SMD_FILE`: add references to papers and books
  4. `claude> /slides.lint $SMD_FILE`: apply the formatting conventions incrementally
  5. `claude> /slides.fix_formatting $SMD_FILE`: fix tags, bold, LaTeX, punctuation,
    unicode
- After each step, confirm the file still renders, and fix any problem before moving
  to the next step:

  ```bash
  > gen_slides.py -i $SMD_FILE
  ```

- Lint the file directly when only formatting needs a pass:

  ```bash
  > lint_text.py -i msml610/lectures_source/Lesson02.1*.smd
  ```

- Use these skills only when the situation calls for them, they are not part of the
  default flow above:
  - `claude> /slides.criticize $SMD_FILE`: get a written critique without applying
    changes
  - `claude> /slides.fix_rendered_pdf $SMD_FILE`: fix a PDF rendering defect
  - `claude> /slides.fix_errors $SMD_FILE`: fix factual errors, keep the structure
  - `claude> /slides.reduce_text $SMD_FILE`: shorten the text, keep the structure
  - `claude> /slides.add_tutorial_links $SMD_FILE`: link to the matching tutorial
    notebook section

### 3. Release the Slide PDF

- Copy the staged PDF from `lectures_pdf.tmp/` to `lectures_pdf/` and compress it:

  ```bash
  > gen_slides.py -i $SMD_FILE --action release
  ```

### 4. Generate the Book Chapter

- Generate the chapter, `$TYP_FILE` lands in `msml610/book/`:

  ```bash
  > gen_book_chapter.py \
    -i $SMD_FILE \
    --mode typst_aima \
    --llm_backend hllm_cli_exec \
    --model openrouter/anthropic/claude-opus-4.6 \
    --no_incremental
  ```

- Humanize the generated text:

  ```bash
  claude> /text.humanize $TYP_FILE
  ```

- Fix heading levels and text tags to match the `.smd` structure:

  ```bash
  claude> /book.fix_headings $TYP_FILE
  claude> /book.improve_text_tags $TYP_FILE
  ```

### Render the Book Chapter
// TODO(ai_gp): Improve
```
  > LID=02.5; render_book_chapter.py -i msml610/$LID --daemon

- Compile and check the PDF, fix any compile problem, and rerun:

  ```bash
  > run_typst.py --input $TYP_FILE
  ```

- Compress the compiled PDF once it looks correct:

  ```bash
  > run_typst.py --input $TYP_FILE --action compress_pdf
  ```

### 5. Generate the Lecture Commentary

- Generate the per-slide commentary for one lecture, output goes to
  `msml610/lectures_commentary/`:

  ```bash
  > gen_lecture_commentary.py msml610/01.1 --image_type jpg
  ```

- After editing a lecture whose commentary already exists, update it instead of
  regenerating it from scratch:

  ```bash
  claude> Execute class_scripts/prompt.update_lecture_commentary.md on msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd
  ```

### 6. Run an Action Across Many Lectures

- Generate the commentary for a range of lectures:

  ```bash
  > for_loop_lessons.py \
    --class msml610 \
    --action generate_lecture_commentary \
    --lectures "01.1-02"
  ```

- Generate the consolidated syllabus:

  ```bash
  > for_loop_lessons.py --class msml610 --action generate_toc
  ```

  - Output: `msml610/all_tocs.md`, headers from every matched lecture up to level 5,
    e.g

    ```markdown
    # Lesson01.1-Intro.smd

    ## Main Topic
    ### Subtopic 1
    #### Sub-subtopic
    ### Subtopic 2
    ```

- Restrict `--lectures` to a subset:
  - Single pattern: `"01*"`
  - Colon-separated union: `"01*:02*:03.1"`
  - Inclusive range: `"01.1-03.2"`

### 7. Publish the Generated Links

- Regenerate the class links page for every course, `msml610` included:

  ```bash
  > website/update_class_links.sh
  ```

  - Writes `website/docs/class_links/msml610.links.html`, linking each lecture's
    slide PDF, commentary, and recap
- Regenerate a single course's page directly when only that one changed:

  ```bash
  > publish_class_links.py \
    --dir msml610 \
    --out_file website/docs/class_links/msml610.links.html \
    --do_not_fail_on_warnings \
    --use_master
  ```

## Full Checklist Template

- For a complete pass on one lecture (review, render, book chapter, compile), start
  from the checklist template at `msml610/book/prompt.slides_and_book_flow.md` and
  follow steps 1-4 above in order
