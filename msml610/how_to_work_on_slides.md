vi msml610/lectures_source/Lesson03.3-Non_classical_logics.smd msml610/book/Lesson03.3-Non_classical_logics.typ

- Create `tasks.md` from template `msml610/prompt.slides_and_book_flow.md`
  ````
  SMD_FILE=msml610/lectures_source/Lesson03.3*.smd

  - Read the conventions for
    - books: `.claude/skills/book.rules.md`
    - slides: `.claude/skills/slides.rules.md`
    - typst code: `.claude/skills/typst.rules.md`

  ### [ ] Review and Improve slides

  - Run the skill
    ```
    /slides.review $SMD_FILE
    ```
    - Implement the restructuring of the slides and fix the high importance issues
      reported by the skill

  - Make sure the SMD_FILE renders correctly:
    ```
    > gen_slides.py -i $SMD_FILE
    ```
    - If not fix the problems

  - Add visuals and references to the $SMD_FILE
    ```
    /slides.add_visuals $SMD_FILE
    /slides.add_references $SMD_FILE
    ```

  - Make sure the SMD_FILE renders correctly:
    ```
    > gen_slides.py -i $SMD_FILE
    ```
    - If not fix the problems


  ### [ ] Generate the book chapter

  - Generate the book chapter for $SMD_FILE
    ```
    > gen_book_chapter.py -i $SMD_FILE --mode typst_aima --llm_backend hllm_cli_exec --model openrouter/anthropic/claude-opus-4.6 --no_incremental
    ```
    which generates a file $TYP_FILE in msml610/lectures_pdf/...typ

  - Humanize
    ```
    /text.humanize $TYP_FILE
    ```

  - Make sure that the generated typst code compiles
    ```
    > run_typst.py --input $TYP_FILE
    ```
  ````

# Workflow in short

/slides.review            03.1   .    .   .   .   .
/slides.add_visuals       .
/slides.add_references    .
/slides.lint              .
/slides.fix_rendered_pdf  .

Edit slides               03.1

Not needed
/slides.fix_errors
/slides.reduce_text
/slides.fix_formatting
/slides.add_tutorial_links

lint_text.py              .
> lint_text.py -i msml610/lectures_source/Lesson02.1*.smd

gen_slides.py             .
> gen_slides.py -i msml610/01.3
> grep "^* " msml610/lectures_source/*.smd | wc -l

tutorials

gen_book_chapter          .
gen_book_chapter.py -i msml610/01.2 --mode typst_aima --llm_backend hllm_cli_exec --model openrouter/anthropic/claude-opus-4.6 --no_incremental

run_typst.py              .
> run_typst.py --input msml610/book/Lesson01.2-AI_and_Machine_Learning.typ

> /text.humanize          .
> /book.fix_headings
> /book.improve_text_tags

Edit book chapter

run_typst.py --compress_pdf .
> compress_pdf.py --input msml610/book/Lesson01.3*.pdf

# Workflows

## Overview

- Extract headers and create a comprehensive syllabus from all lecture materials
  using the `for_loop_lessons.py` orchestration script

## Slides

### Iterate on the Slides

- Generate slides when editing the source
  ```bash
  > gen_slides.py -i msml610/lectures_source/Lesson01.1-Intro.smd
  > gen_slides.py -i msml610/01.1 --daemon
  > gen_slides.py -i msml610/01.1 --daemon
  ```

- The file is generated in `lectures_pdf.tmp`

## Check Slides

```
claude> /slides.criticize msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd
```

## Slides Commentary

### Generate for One Lecture
- Generate one lecture
  ```
  > gen_lecture_commentary.py msml610/01.1 --image_type jpg
  ```

### Generate for All Lectures

- Generate all the lectures
  ```
  > for_loop_lessons.py --class data605 --action generate_lecture_commentary --lectures "01.1-02"

  # Check out.
  > publish_class_links.py --dir msml610 --out_file ./links.html --do_not_fail_on_warnings --use_master
  > open book_springer/lecture_commentary/Lesson04.1_Knowledge_Representation.book_chapter.html
  ```

### Publish the lecture commentary on the website

website/update_class_links.sh

## Update Slides Commentary

```
claude> Execute class_scripts/prompt.update_lecture_commentary.md on msml610/lectures_source/Lesson01.2-AI_and_Machine_Learning.smd
```

## Course Syllabus

### Generate Complete Course Syllabus

- Extract all lecture headers and create a consolidated syllabus:

  ```bash
  > cd /Users/saggese/src/umd_classes1
  > for_loop_lessons.py --class msml610 --action generate_toc
  ```

This generates:
- **Output file**: `msml610/all_tocs.md`
- **Content**: All lecture headers organized hierarchically (up to 5 levels deep)
- **Format**: Markdown with lecture structure preserved

### Generate Syllabus for Specific Lectures

- Extract headers from a subset of lectures using pattern matching:
  ```bash
  # Single lecture pattern
  > for_loop_lessons.py --class msml610 --lectures "01*" --action generate_toc

  # Multiple lecture patterns (colon-separated)
  > for_loop_lessons.py --class msml610 --lectures "01*:02*:03.1" --action generate_toc

  # Continuous range (inclusive)
  > for_loop_lessons.py --class msml610 --lectures "01.1-03.2" --action generate_toc
  ```

### Output Format

- The syllabus markdown file contains structured headers with proper indentation:
  ```markdown
  # Lesson01.1-Intro.smd

  ## Main Topic
  ### Subtopic 1
  #### Sub-subtopic
  ### Subtopic 2

  # Lesson01.2-Topic.smd

  ## Another Main Topic
  ...
  ```

- This provides a complete overview of the course curriculum and lecture structure
