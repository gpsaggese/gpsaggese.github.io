- Definitions and the format of a book map is described in
  `.claude/skills/book_map.rules.md`

# All prompts

// TODO(ai_gp): Add a table for each prompt and a short comment

```
> ls -1 book_scripts/*prompt*
book_scripts/prompt.create_lesson_slides.common.md
book_scripts/prompt.create_lesson_slides_from_chap_toc.md
book_scripts/prompt.create_lesson_slides_from_topic.md
book_scripts/prompt.create_latex_book_chap_from_lesson_slides.md
book_scripts/prompt.create_typst_book_chap_from_lesson_slides.md
book_scripts/prompt.expand_skeleton_slides.md
book_scripts/prompt.extract_skeleton_slides_from_lesson_slides.md
book_scripts/prompt.find_references_to_chap_toc.md
book_scripts/prompt.inline_lesson_slides_from_skeleton_slides.md
book_scripts/prompt.update_book_chap_from_lesson_slides.md
```

# Prompts

// TODO(ai_gp): For each prompt use the format below
// ## <Description>
// **Prompt**: 
// ```
// claude> Execute book_scripts/prompt.create_lesson_slides_from_chap_toc.md
// ```
// **Input**: What is the input
// **Output**: What is genereated / updated
// **Workflow**: the workflow in terms of what is done
// **Examples**: examples, uses, etc

## Create `lesson_slides` from a chapter TOC
- Reads th chapter TOC, `### Topics`, and `### Lesson Materials` from
  `book_map.md`
- Collect the material, then runs the common steps below
  ```
  claude> Execute book_scripts/prompt.create_lesson_slides_from_chap_toc.md
  ```
- E.g., for the agentic AI book, there is a `chap_toc` and we need to create
  `lesson_slides`

## Create `lesson_slides` from a topic list (no book map yet)
- Same goal and workflow as the prompt above, but topics are given as an ad-hoc list
  inline in the prompt instead of being read from a chapter's `### Topics` in
  `book_map.md`
- Useful when the book map does not have that chapter yet, or for a course
  such as `msml610`
  ```
  claude> Execute book_scripts/prompt.create_lesson_slides_from_topic.md
  ```

## Common steps to create `lesson_slides`
- Steps shared by the two prompts above, executed after the material is read
  - Create a skeleton of titles
  - Ask the user for approval
  - Call `/slides.write` for 30-35 slides
  - Create a notebook for the figures that need data or code, saved with
    `hmatplo.save_fig()`
  - Save a new `lectures_source/Lesson<CHAP_NUM>.<ID>_<Title>.smd`
  ```
  claude> Execute book_scripts/prompt.create_lesson_slides.common.md
  ```

## Find material
- Complete the pointers and add references to `resources.md`
  ```
  claude> Execute book_scripts/prompt.find_references_to_chap_toc.md
  ```

## Create `skeleton_slides` from `book_map` and `slides_tutorials`
- Read the TOC for a chapter in the `book_map.md`
- Find the material already available for it
  ```
  claude> Execute book_scripts/prompt.extract_skeleton_slides_from_lesson_slides.md
  ```

## Inline slides in a skeleton slide
- Mechanically copies existing slide content into a skeleton, following each slide's
  `// From: <file>:<line num> '<title>'` pointer
  ```
  claude> book_scripts/prompt.inline_lesson_slides_from_skeleton_slides.md
  ```
  - E.g.,
    ```
    claude> Execute /Users/saggese/src/notes1/book_scripts/prompt.inline_lesson_slides_from_skeleton_slides.md on book_springer/lectures_source/Lesson05.1_Probabilistic_ML.txt
    ```
- This is also implemented as Python script (since it's a mechanical operation)
  ```
  > class_scripts/inline_content_in_skeleton_slides.py
  > vi $(grep -riL "Score:" *.md | while IFS= read -r f; do stat -f "%m%t%N" "$f"; done | sort -n | head -10 | cut -f2- | tr '\n' '\0' | xargs -0)
  ```

- Check that the result is correct with:
  ```
  > gen_slides.py book_springer/05.1
  ```

## Expand a skeleton whose slides point to background files
- Same shape as "Inline slides in a skeleton slide" above, but each skeleton
  slide's `// From: <file>` pointer names a whole background file (e.g. a
  rules doc) rather than a specific source slide/line, so the content must be
  generated from that background reading instead of copied verbatim
  ```
  claude> Execute book_scripts/prompt.expand_skeleton_slides.md
  ```

// TODO(gp): Likely a near-duplicate of "Inline slides in a skeleton slide"
// above (same skeleton -> full-slides direction, different `// From:`
// semantics); consider merging as a documented variant of that prompt.

## Generate lectures commentary
- To generate a commentary in `$DIR/lectures_commentary`
  ```
  > gen_lecture_commentary.py book_springer/04.1 --image_type jpg

  > for_loop_lessons.py --class data605 --action generate_lecture_commentary --lectures "01.1-02"

  > publish_class_links.py --dir book_springer --out_file ./links.html --do_not_fail_on_warnings --use_master

  > open book_springer/lecture_commentary/Lesson04.1_Knowledge_Representation.book_chapter.html
  ```

- Check
  ```
  > ls -1 book_springer/lectures_commentary/Lesson*.{pdf,md,html}
  book_springer/lectures_commentary/Lesson02.1_From_Data_Science_To_Decision_Science.book_chapter.html
  book_springer/lectures_commentary/Lesson02.1_From_Data_Science_To_Decision_Science.book_chapter.md
  book_springer/lectures_commentary/Lesson02.1_From_Data_Science_To_Decision_Science.book_chapter.pdf
  book_springer/lectures_commentary/Lesson04.1_Knowledge_Representation.book_chapter.md
  book_springer/lectures_commentary/Lesson05.1_Probabilistic_ML.book_chapter.md
  ```

## Convert slides to typst or Latex text for a book chapter
  `book_scripts/prompt.create_typst_book_chap_from_lesson_slides.md`
  `book_scripts/prompt.create_latex_book_chap_from_lesson_slides.md`

- Execute
  ```
  claude> Execute book_scripts/prompt.create_latex_book_chap_from_lesson_slides.md
  ```

  ```
  > ls -1 book_springer/book/Lesson*.tex
  book_springer/book/Lesson02.01_From_Data_Science_To_Decision_Science.tex
  book_springer/book/Lesson02.02_Integrating_Causality_And_Probability_in_ML.tex
  book_springer/book/Lesson04.01_Knowledge_Representation.tex
  ```

## Update the a Typst book chapter when the corresponding `lesson_slides` are updated
- `book_scripts/prompt.update_book_chap_from_lesson_slides.md`
- `book_scripts/prompt.springer.add_references.md`

// TODO(gp): This prompt currently only follows the Typst variant of the
// "Convert slides to typst or Latex" prompt above; update it to branch on
// the chapter file's extension (`.typ` -> typst prompt, `.tex` -> latex
// prompt) now that a LaTeX variant also exists.

## Review / Update Slides and Generate book

// From msml610/book_map.md

msml610/book/prompt.slides_and_book_flow.md

```
claude> Execute msml610/book/prompt.slides_and_book_flow.md msml610/lectures_source/Lesson05.2-Overfitting.smd
```

- [x] 5.1
- [x] 5.2
- [x] 5.3

- [x] 6.1
- [x] 6.2

- [ ] 7.1
- [ ] 7.2

LID=06.2; gen_slides.py -i msml610/$LID --daemon
LID=06.2; render_book_chapter.py -i msml610/$LID --daemon

/slides.review            05.1
/slides.add_visuals       .
/slides.add_references    .
/slides.lint              .
/slides.fix_rendered_pdf  .

Edit slides               .

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
> /book_chapter.fix_headings      .
> /book_chapter.improve_text_tags .

Edit book chapter

run_typst.py --compress_pdf .
> compress_pdf.py --input msml610/book/Lesson01.3*.pdf

