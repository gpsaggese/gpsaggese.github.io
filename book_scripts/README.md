- Definitions and the format of a book map is described in
  `.claude/skills/book_map.rules.md`

# All prompts

// TODO(ai_gp): Add book_scripts/ to the path

| **Prompt** | **Comment** |
| :--------- | :---------- |
| `prompt.create_lesson_slides.common.md` | Steps shared by the two lesson prompts |
| `prompt.create_lesson_slides_from_chap_toc.md` | Lesson slides from a chapter TOC |
| `prompt.create_lesson_slides_from_topic.md` | Lesson slides from an ad-hoc topic list |
| `prompt.create_latex_book_chap_from_lesson_slides.md` | LaTeX book chapter from lesson slides |
| `prompt.create_typst_book_chap_from_lesson_slides.md` | Typst book chapter from lesson slides |
| `prompt.expand_skeleton_slides.md` | Skeleton slides expanded from background files |
| `prompt.extract_skeleton_slides_from_lesson_slides.md` | Skeleton slides and TOC from lesson slides |
| `prompt.find_references_to_chap_toc.md` | References for a chapter in `map.md` |
| `prompt.inline_lesson_slides_from_skeleton_slides.md` | Slide content copied into a skeleton |
| `prompt.update_book_chap_from_lesson_slides.md` | Typst chapter updated after its slides change |
| `msml610/book/prompt.slides_and_book_flow.md` | Slides reviewed and book chapter generated |

# Prompts

## Create `lesson_slides` from a chapter TOC

- **Prompt**:
  ```text
  claude> Execute book_scripts/prompt.create_lesson_slides_from_chap_toc.md
  ```
- **Input**:
  - Chapter `CHAP` (H2) in `<DIR>/<BOOK>/book_map.md`
  - Its `### Goals`, `### Topics`, and `### Lesson Materials`
  - Its `### Papers` and `### Books`
  - `<DIR>/<BOOK>/resources.md`, if present
- **Output**: `<DIR>/<BOOK>/lectures_source/Lesson<CHAP_NUM>.<ID>_<TITLE>.smd`
- **Workflow**:
  - Read the book map and the chapter `CHAP`
  - Read the material in `### Lesson Materials`, `### Papers`, and `### Books`
  - Read the existing lessons in `<BOOK>/lectures_source/` to avoid repetition
  - Execute `book_scripts/prompt.create_lesson_slides.common.md`
- **Examples**:
  - For the agentic AI book, a chapter TOC exists, so its `lesson_slides` are
    created with this prompt
  - E.g., `CHAP="## 14, GitHub Projects and Workflows"` in
    `book_AI_Software_Engineering`

## Create `lesson_slides` from a topic list (no book map yet)

- **Prompt**:
  ```text
  claude> Execute book_scripts/prompt.create_lesson_slides_from_topic.md
  ```
- **Input**:
  - Ad-hoc list of topics in the `## Topics` section of the prompt
  - Optional file to align the content with
- **Output**: `<DIR>/<BOOK>/lectures_source/Lesson<CHAP_NUM>.<ID>_<TITLE>.smd`
- **Workflow**:
  - Read the topics and the optional alignment file
  - Read the existing lessons in `<DIR>/<BOOK>/lectures_source/`
  - Execute `book_scripts/prompt.create_lesson_slides.common.md`
- **Examples**:
  - Useful when the book map does not have the chapter yet
  - Useful for a course such as `msml610`
  - E.g., `BOOK="msml610"`, `CHAP_NUM="09"`, `TITLE="Classical_Search_Algorithms"`

## Common steps to create `lesson_slides`

- **Prompt**:
  ```text
  claude> Execute book_scripts/prompt.create_lesson_slides.common.md
  ```
- **Input**:
  - Topics and material read by the calling prompt
  - Variables `<DIR>`, `<BOOK>`, `<CHAP_NUM>`, `<ID>`, and `<TITLE>`
- **Output**:
  - Slides: `<DIR>/<BOOK>/lectures_source/Lesson<CHAP_NUM>.<ID>_<TITLE>.smd`
  - Notebook and figures: `<DIR>/<BOOK>/tutorials/`, if a figure needs data or code
- **Workflow**:
  - Create a skeleton of 30-35 slide titles
  - Get user approval of the skeleton
  - Call `/slides.write` for 30-35 slides
  - Create a notebook for the figures that need data or code
    - Save each figure with `hmatplo.save_fig()`
  - Save the slides as `Lesson<CHAP_NUM>.<ID>_<TITLE>.smd`
- **Examples**:
  - Not run alone: the two lesson prompts above call it

## Find material

- **Prompt**:
  ```text
  claude> Execute book_scripts/prompt.find_references_to_chap_toc.md
  ```
- **Input**: Chapter `CHAP` (H2) in `<DIR>/<BOOK>/map.md`
- **Output**: Updated `### Local Material`, `### Books`, `### Papers`, and
  `### Internet resources` in `map.md`
- **Workflow**:
  - Read the chapter `CHAP` and find related books, papers, and web resources
  - Prefer recent sources and free-access versions, e.g., arXiv
  - Add a working URL for each reference when possible
  - Format books and papers following `.claude/skills/references.rules.md`
- **Examples**:
  - E.g., `CHAP="## 02: AI Pair Programming Workflows"` in
    `book_AI_Software_Engineering`

## Create `skeleton_slides` from `book_map` and `slides_tutorials`

- **Prompt**:
  ```text
  claude> Execute book_scripts/prompt.extract_skeleton_slides_from_lesson_slides.md
  ```
- **Input**:
  - Chapter `CHAP` in `<DIR>/<BOOK>/map.md`, with its `### Goals` and `### Topics`
  - Lesson slides listed in `### Lesson Materials` of `CHAP`
- **Output**:
  - Skeleton slides in `<BOOK>/lectures_source/Lesson0<num>.01_<name>.txt`
  - Updated `### Topics` and `### Goals` of `CHAP`
- **Workflow**:
  - Extract the TOC of each lesson file with `extract_toc_from_text.py`
  - Write the skeleton with `#` H1, `##` H2, and one `* <slide title>` per slide
  - Add a `// From:` line after each slide with its source file, line, and title
  - Update `### Topics` with the H1 and H2 headers
  - Update `### Goals` with 3 short bullets, each under 100 characters
- **Examples**:
  - E.g., `CHAP="## 08: Causal Data Pipelines"` in `book_springer`
  - The skeleton has around 30-35 slides

## Inline slides in a skeleton slide

- **Prompt**:
  ```text
  claude> Execute book_scripts/prompt.inline_lesson_slides_from_skeleton_slides.md
  ```
- **Input**: Skeleton file `<FILE>` whose `// From: <file>:<line num> '<title>'`
  lines point to source slides
- **Output**: The same `<FILE>`, with the source slide content inlined under each
  slide
- **Workflow**:
  - Find the source slide for each `// From:` line
  - Copy its content under the slide title
  - Keep the `// From:` line in place
  - Leave a bare `// From:` and report an error if the title is missing
- **Examples**:
  - E.g.,
    ```text
    claude> Execute /Users/saggese/src/notes1/book_scripts/prompt.inline_lesson_slides_from_skeleton_slides.md on book_springer/lectures_source/Lesson05.1_Probabilistic_ML.txt
    ```
  - Mechanical copy in a Python script, since it is a mechanical operation:
    ```bash
    > class_scripts/inline_content_in_skeleton_slides.py
    > vi $(grep -riL "Score:" *.md | while IFS= read -r f; do stat -f "%m%t%N" "$f"; done | sort -n | head -10 | cut -f2- | tr '\n' '\0' | xargs -0)
    ```
  - Check that the result is correct:
    ```bash
    > gen_slides.py book_springer/05.1
    ```

## Expand a skeleton whose slides point to background files

- **Prompt**:
  ```text
  claude> Execute book_scripts/prompt.expand_skeleton_slides.md
  ```
- **Input**: Skeleton file in `$DIR/$BOOK/CHAP`, whose `// From:` names a whole
  background file (e.g., a rules doc)
- **Output**: Slides for each skeleton section, generated from the background files
- **Workflow**:
  - Read each background file named by a `// From:` line
  - Write the slides from that reading instead of copying verbatim
  - Follow `.claude/skills/text.rules.md`, `.claude/skills/markdown.rules.md`, and
    `.claude/skills/slides.rules.md`
  - Ask questions instead of writing the slides if there are doubts
- **Examples**:
  - Same shape as "Inline slides in a skeleton slide" above, but the `// From:`
    names a whole file, not one slide
  - E.g., `// From: helpers_root/.claude/skills/coding.rules.md`

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

## Convert Slides to a Typst Book Chapter

- **Prompt**:
  ```text
  claude> Execute book_scripts/prompt.create_typst_book_chap_from_lesson_slides.md
  ```
- **Input**: Lesson slides in `<BOOK>/lectures_source/`, e.g.,
  `msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.txt`
- **Output**: Typst file `<BOOK>/book/<FILE>.typ`, with the same name as the slides
- **Workflow**:
  - Map each slide level to the document structure:
    - H1 to a chapter with `#chapter`
    - H2 to a section with `==`
    - H3 and deeper to a subsection with `===`
  - Expand each slide into prose for undergraduate and early graduate students
  - Start the file with the `aima-style` template
  - Add `// git_hash=<GIT_HASH>, timestamp=<TIMESTAMP>` as the first line
- **Examples**:
  - E.g., `msml610/book/Lesson10.2-Causal_Inference_for_Time_Series.typ`
  - Keep it in sync with `book_scripts/prompt.update_book_chap_from_lesson_slides.md`

## Convert Slides to a LaTeX Book Chapter

- **Prompt**:
  ```text
  claude> Execute book_scripts/prompt.create_latex_book_chap_from_lesson_slides.md
  ```
- **Input**: Lesson slides in `<BOOK>/lectures_source/`
- **Output**: LaTeX file `<BOOK>/book/<FILE>.tex`, with the same name as the slides
- **Workflow**:
  - Map each slide level to the document structure:
    - H1 to a chapter with `\chapter`
    - H2 to a section with `\section`
    - H3 and deeper to a subsection with `\subsection`
  - Expand each slide into prose for a textbook chapter
  - Use the Springer SNmono `\documentclass` to match `book_springer/book`
- **Examples**:
  - Check the output:
    ```bash
    > ls -1 book_springer/book/Lesson*.tex
    book_springer/book/Lesson02.01_From_Data_Science_To_Decision_Science.tex
    book_springer/book/Lesson02.02_Integrating_Causality_And_Probability_in_ML.tex
    book_springer/book/Lesson04.01_Knowledge_Representation.tex
    ```

## Update a Typst Book Chapter When the Corresponding `lesson_slides` Are Updated

- **Prompt**:
  ```text
  claude> Execute book_scripts/prompt.update_book_chap_from_lesson_slides.md
  ```
- **Input**:
  - File name `<FILE>`, e.g., `Lesson10.2-Causal_Inference_for_Time_Series`
  - Typst chapter `msml610/book/<FILE>.typ`
  - Lesson slides `msml610/lectures_source/<FILE>.txt`
- **Output**: Updated `msml610/book/<FILE>.typ`
- **Workflow**:
  - Read `<GIT_HASH>` from the first line of the `.typ` file
  - Save the slide changes since that commit in `changes.txt` with `git diff`
  - Apply `changes.txt` to the `.typ` file following
    `book_scripts/prompt.create_typst_book_chap_from_lesson_slides.md`
- **Examples**:
  - E.g.,
    ```bash
    > git diff 382e0cdf HEAD -- msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.txt >changes.txt
    ```
  - Related prompt: `book_scripts/prompt.springer.add_references.md`

// TODO(gp): This prompt currently only follows the Typst variant of the
// "Convert Slides to a Typst Book Chapter" prompt above; update it to branch on
// the chapter file's extension (`.typ` -> typst prompt, `.tex` -> latex
// prompt) now that a LaTeX variant also exists.

## Review / Update Slides and Generate book

- **Prompt**:
  ```text
  claude> Execute msml610/book/prompt.slides_and_book_flow.md msml610/lectures_source/Lesson05.2-Overfitting.smd
  ```
- **Input**: Lesson slides `<SMD_FILE>`, e.g.,
  `msml610/lectures_source/Lesson05.2-Overfitting.smd`
- **Output**:
  - Reviewed slides `<SMD_FILE>`
  - Book chapter `msml610/book/<NAME>.typ`
- **Workflow**:
  - Create a plan file to track the steps
  - Read the conventions for books, slides, and Typst code
  - Review and improve the slides with `/slides.review`
  - Render the slides with `gen_slides.py -i <SMD_FILE>` and fix any problem
  - Add visuals and references with `/slides.add_visuals` and
    `/slides.add_references`
  - Generate the book chapter with `gen_book_chapter.py`
  - Humanize the chapter with `/text.humanize`
  - Check that the Typst code compiles with `run_typst.py`
- **Examples**:
  - The lesson checklist below records the commands run for each lesson

### Checklist

// From msml610/book_map.md
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
