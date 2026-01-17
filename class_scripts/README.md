# classes2

Course management tools and utilities for generating lecture materials.

## Structure of the Dir

No subdirectories.

## Description of Files

- `common_utils.py`
  - Shared utility functions for argument validation, file finding, directory management, and PDF page counting

## Description of Executables

### `count_book_pages.py`

#### What It Does

- Counts pages in all PDF files in the `{DIR}/book/` directory
- Uses macOS `mdls` command to extract PDF metadata
- Displays page counts for each book PDF file

#### Examples

- Count pages in all book PDFs for data605:
  ```bash
  > ./count_book_pages.py data605
  ```

- Count pages in all book PDFs for msml610:
  ```bash
  > ./count_book_pages.py msml610
  ```

### `count_lecture_pages.py`

#### What It Does

- Counts pages in all PDF files in the `{DIR}/lectures/` directory
- Uses macOS `mdls` command to extract PDF metadata
- Displays page counts for each lecture PDF file

#### Examples

- Count pages in all lecture PDFs for data605:
  ```bash
  > ./count_lecture_pages.py data605
  ```

- Count pages in all lecture PDFs for msml610:
  ```bash
  > ./count_lecture_pages.py msml610
  ```

### `count_words.py`

#### What It Does

- Counts words in all files in the `{DIR}/lectures_script/` directory
- Displays word counts for each lecture script file
- Helps track lecture length and content volume

#### Examples

- Count words in all lecture scripts for data605:
  ```bash
  > ./count_words.py data605
  ```

- Count words in all lecture scripts for msml610:
  ```bash
  > ./count_words.py msml610
  ```

### `gen_book_chapter.py`

#### What It Does

- Generates a book chapter from lecture source material
- Performs multiple steps: PDF generation, chapter creation, pandoc conversion
- Opens the final PDF in Skim viewer

#### Examples

- Generate book chapter for data605 lesson 01.1:
  ```bash
  > ./gen_book_chapter.py data605 01.1
  ```

- Generate book chapter for msml610 lesson 02.3:
  ```bash
  > ./gen_book_chapter.py msml610 02.3
  ```

### `gen_lecture_script.py`

#### What It Does

- Generates a complete lecture script from slides using LLM
- Creates intro and outro sections automatically
- Combines all sections and lints the final output

#### Examples

- Generate lecture script for data605 lesson 01.1:
  ```bash
  > ./gen_lecture_script.py data605 01.1
  ```

- Generate lecture script with extra options:
  ```bash
  > ./gen_lecture_script.py msml610 02.3 --force
  ```

### `gen_quizzes.py`

#### What It Does

- Generates questions from lecture content using LLM via llm_cli.py
- Two modes:
  - Multiple choice quizzes: 20 questions with 5 answers each
    - Saved to: `{DIR}/lectures_quizzes/<lesson>.quizzes.md`
  - Discussion/review questions: 3-6 open-ended questions
    - Saved to: `{DIR}/lectures_recap/<lesson>.recap.md`
- Automatically formats output using lint_txt.py with prettier (use `--no_lint` to skip)

#### Examples

- Generate multiple choice quizzes for data605 lesson 01.1:
  ```bash
  > ./gen_quizzes.py --for_class_quizzes data605 01.1
  ```

- Generate discussion/review questions for msml610 lesson 02.3:
  ```bash
  > ./gen_quizzes.py --for_class_recap msml610 02.3
  ```

- Generate quizzes without linting:
  ```bash
  > ./gen_quizzes.py --for_class_recap data605 01.2 --no_lint
  ```

- Generate quizzes with extra LLM options:
  ```bash
  > ./gen_quizzes.py --for_class_quizzes data605 01.1 --model gpt-4
  ```

### `gen_slides.py`

#### What It Does

- Generates lecture slide PDFs from source files
- Uses notes_to_pdf.py to convert markdown to PDF
- Accepts additional options to pass through to notes_to_pdf.py

#### Examples

- Generate slides for data605 lesson 01.1:
  ```bash
  > ./gen_slides.py data605 01.1
  ```

- Generate slides with extra options:
  ```bash
  > ./gen_slides.py msml610 02.3 --theme dark
  ```

### `get_lecture_file.py`

#### What It Does

- Finds and prints the path to a lecture source file
- Searches for files matching `{DIR}/lectures_source/Lesson{LESSON}*`
- Validates that exactly one matching file exists

#### Examples

- Find lecture file for data605 lesson 01.1:
  ```bash
  > ./get_lecture_file.py data605 01.1
  ```

- Find lecture file for msml610 lesson 02.3:
  ```bash
  > ./get_lecture_file.py msml610 02.3
  ```

### `slide_check.py`

#### What It Does

- Checks and fixes text in lecture slides using LLM
- Uses process_slides.py with text_check_fix action
- Corrects spelling, grammar, and formatting issues

#### Examples

- Check and fix slides for data605 lesson 01.1:
  ```bash
  > ./slide_check.py data605 01.1
  ```

- Check slides with extra options:
  ```bash
  > ./slide_check.py msml610 02.3 --dry-run
  ```

### `slide_improve.py`

#### What It Does

- Improves lecture slides using LLM suggestions
- Uses process_slides.py with slide_improve action
- Enhances clarity, structure, and pedagogical effectiveness

#### Examples

- Improve slides for data605 lesson 01.1:
  ```bash
  > ./slide_improve.py data605 01.1
  ```

- Improve slides with extra options:
  ```bash
  > ./slide_improve.py msml610 02.3 --max-suggestions 5
  ```

### `slide_reduce.py`

#### What It Does

- Reduces and simplifies lecture slides using LLM
- Uses process_slides.py with slide_reduce action
- Removes redundancy and condenses content

#### Examples

- Reduce slides for data605 lesson 01.1:
  ```bash
  > ./slide_reduce.py data605 01.1
  ```

- Reduce slides with extra options:
  ```bash
  > ./slide_reduce.py msml610 02.3 --target-length 50
  ```
