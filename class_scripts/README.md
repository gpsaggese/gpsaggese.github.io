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

### `process_lessons.py`

#### What It Does

- Main script for generating PDF slides and reading scripts from lecture source
  files
- Supports multiple actions including PDF generation script generation and LLM
  transformations
- Can process single or multiple lectures using pattern matching
- Provides dry-run mode to preview commands without execution

#### Examples

- Generate PDF slides for a specific lecture:
  ```bash
  > process_lessons.py --lectures 01.1 --class data605 --action generate_pdf
  ```

- Generate reading scripts for multiple lectures:
  ```bash
  > process_lessons.py --lectures 01*:02* --class data605 --action generate_script
  ```

- Generate both PDFs and scripts:
  ```bash
  > process_lessons.py --lectures 01* --class msml610 --action generate_pdf --action generate_script
  ```

- Generate all slides for multiple lessons:
  ```bash
  > process_lessons.py --lectures 0*:1* --class data605 --action generate_pdf
  ```

# DATA605 / To reorg

## Check correctness of all the slides

- Check one lecture (simple method - recommended)
  ```bash
  > slide_check.sh 01.2
  ```
  This checks Lesson 01.2 for formatting and content issues.

- Check multiple lectures
  ```bash
  > slide_check.sh 01.*
  ```
  This checks all lessons in section 01.

- Check several lessons with limits (for testing)
  ```bash
  > process_lessons.py --lectures 01.1* --class data605 --action check_slide --limit 0:2
  ```
  This checks only the first 2 slides of matching lessons.

- Check one lecture from inside the container (advanced)
  ```bash
  > SRC_NAME=$(ls $DIR/lectures_source/Lesson02*); echo $SRC_NAME
  > DST_NAME=process_slides.txt
  docker> process_slides.py --in_file $SRC_NAME --action text_check --out_file $DST_NAME --use_llm_transform --limit 0:10
  > vimdiff $SRC_NAME process_slides.txt
  ```
  This runs the check inside Docker and compares the output with vimdiff.

## Improve slides

- Improve a specific lecture using LLM
  ```bash
  > llm_transform.py -i data605/lectures_source/Lesson07.2-Data_Wrangling.txt -p slide_improve -v DEBUG
  ```
  This uses AI to improve the content and formatting of the slides.

## Reduce all slides

- Reduce one lecture (simple method - recommended)
  ```bash
  > slide_reduce.sh 01.1
  ```
  This reduces the content of Lesson 01.1 to make it more concise.

- Reduce multiple lectures
  ```bash
  > slide_reduce.sh 01.1*
  ```
  This reduces all lessons matching the pattern (e.g., 01.1, 01.2, 01.3).

- Reduce from inside the container (advanced)
  ```bash
  > SRC_NAME=$(ls $DIR/lectures_source/Lesson04.2*); echo $SRC_NAME
  > process_slides.py --in_file $SRC_NAME --action slide_reduce --out_file $SRC_NAME --use_llm_transform --limit 0:10
  ```
  This reduces only the first 10 slides using the container environment.

## Generate the PDF for all the slides

- Generate PDFs for multiple lessons
  ```bash
  > process_lessons.py --lectures 0*:1* --class data605 --action generate_pdf
  ```
  This generates PDF files for all lessons starting with 0 or 1 (e.g., 01.1, 01.2, 10.1, etc.).

## Generate the lecture script

- Generate script for one lecture (simple method - recommended)
  ```bash
  > gen_data605_script.sh 04.3
  ```
  This generates a lecture script for Lesson 04.3.

- Generate the intro for a lecture
  ```bash
  > TAG=08.3; llm_cli.py -i data605/lectures_script/Lesson${TAG}*.script.txt -p "You are a college professor and you need to do an introduction in 50 word the content of the slides starting with In this lesson" -o -
  ```
  This creates a 50-word introduction for Lesson 08.3.

- Generate the outro for a lecture
  ```bash
  > TAG=08.3; llm_cli.py -i data605/lectures_script/Lesson${TAG}*.script.txt -p "You are a college professor and you need to summarize what was discussed in less than 50 word in the slides like In this lesson we have discussed" -o -
  ```
  This creates a 50-word summary/conclusion for Lesson 08.3.

- Generate script from inside a container (advanced)
  ```bash
  > i docker_bash --base-image=623860924167.dkr.ecr.eu-north-1.amazonaws.com/cmamp --skip-pull

  docker> sudo /bin/bash -c "(source /venv/bin/activate; pip install --upgrade openai)"

  docker> generate_slide_script.py \
    --in_file data605/lectures_source/Lesson01-Intro.txt \
    --out_file data605/lectures_source/Lesson01-Intro.script.txt \
    --slides_per_group 3 \
    --limit 1:5
  ```
  This generates a script for slides 1-5, grouping 3 slides at a time.

## Count pages

- Count pages for all PDFs (simple method - recommended)
  ```bash
  > count_pages.sh
  ```
  This displays the page count for all lecture PDFs.

- Count pages and copy to clipboard
  ```bash
  > count_pages.sh | pbcopy
  ```
  This counts pages and copies the results to your clipboard.

- Example output:
  ```
  data605/lectures/Lesson01.1-Intro.pdf   10
  data605/lectures/Lesson01.2-Big_Data.pdf        17
  data605/lectures/Lesson01.3-Is_Data_Science_Just_Hype.pdf       14
  ```

- Manual count with find (alternative method)
  ```bash
  > find data605/lectures/Lesson0*.pdf -type f -name "*.pdf" -print -exec mdls -name kMDItemNumberOfPages {} \;
  ```

- Manual count with formatting (advanced)
  ```bash
  > find data605/lectures/Lesson0*.pdf -type f -name "*.pdf" -print0 | while IFS= read -r -d '' file; do     pages=$(mdls -name kMDItemNumberOfPages "$file" | awk -F'= ' '{print $2}');     echo -e "${file}\t${pages}"; done | tee tmp.txt
  ```
  This creates a formatted table and saves it to tmp.txt.

## Count words

- Count words in all scripts (simple method - recommended)
  ```bash
  > ./count_words.sh
  ```
  This counts the number of words in each lecture script file.

- Manual word count (alternative method)
  ```bash
  > dir="data605/lectures_script/"; for f in "$dir"/*; do [ -f "$f" ] && printf "%s\t%s\n" "$(basename "$f")" "$(wc -w < "$f")"; done
  ```
  This loops through all script files and counts words for each.

## Review scripts

- Generate and edit script for a specific lesson
  ```bash
  > TAG=10.1; gen_data605.sh $TAG; vi $(ls data605/lectures_script/*${TAG}*)
  ```
  This generates the script for Lesson 10.1 and opens it in vi for editing.

- Open PDF and edit script side-by-side
  ```bash
  > TAG=08.3; open data605/lectures/Lesson${TAG}*.pdf; vi $(ls data605/lectures_script/*${TAG}*)
  ```
  This opens the PDF slides and the script file for Lesson 08.3 for comparison/editing.

## Format figures in slides

- Format figures in a specific lecture
  ```bash
  > FILE=data605/lectures_source/Lesson02-Git_Data_Pipelines.txt
  > process_slides.py --in_file $FILE --action slide_format_figures --out_file $FILE --use_llm_transform
  ```
  This reformats figure references in the slides using LLM assistance.

# MSML610 / To Reorg

## Quick commands

- Navigate to project root
  ```bash
  > cd $GIT_ROOT
  ```

- Generate slides for DATA605 lesson
  ```bash
  > gen_data605.sh 01
  ```
  This generates all materials for DATA605 Lesson 01.

- Generate slides for MSML610 lesson
  ```bash
  > gen_msml610.sh 02
  ```
  This generates all materials for MSML610 Lesson 02.

## Process MSML610 slides

- Format figures and check slides
  ```bash
  > FILE=msml610/lectures_source/Lesson05*
  > process_slides.py --in_file $FILE --action slide_format_figures --out_file $FILE --use_llm_transform
  > process_slides.py --in_file $FILE --action slide_check --out_file ${FILE}.check --use_llm_transform --limit None:10
  ```
  This formats figures in Lesson 05 and checks the first 10 slides.

## Sync and open lectures

- Sync lectures from remote server and open
  ```bash
  > rsync -avz -e "ssh -i ~/.ssh/ck/saggese-cryptomatic.pem" saggese@$DEV1:/data/saggese/src/umd_classes1/msml610/lectures/ msml610/lectures/; open msml610/lectures/*07.1* -a "skim"
  ```
  This downloads Lesson 07.1 from the remote server and opens it in Skim.

## Convert markdown to PDF

- Convert markdown notes to PDF slides
  ```bash
  > notes_to_pdf.py --input data605/lectures_md/final_enhanced_markdown_lecture_2.txt --output tmp.pdf --type slides --skip_action cleanup_after --debug_on_error --toc_type navigation --filter_by_slides 1:4
  ```
  This converts markdown lecture notes to PDF format (slides 1-4 only).

## Run the tutorials

- Start Jupyter in Docker
  ```bash
  > cd msml610/tutorials
  > i docker_jupyter --skip-pull --stage local --version 1.0.0
  ```
  This starts a Jupyter Lab server in a Docker container.

- Open a specific notebook
  ```bash
  > open -a "Chrome" http://127.0.0.1:5011/lab/tree/notebooks/Bayesian_Coin.ipynb
  ```
  This opens the Bayesian Coin tutorial in Chrome.

## Fix slides with LLM

- Fix slides using a prompt template
  ```bash
  > FILE=data605/lectures_source/Lesson09.2-Spark_Primitives.txt
  > llm_cli.py --input $FILE -pf "fix_slides.prompt.md" -o improved.md --model "gpt-4o" -b
  ```
  This uses GPT-4o to fix and improve the slides based on the prompt template.

## Generate book

- Generate a book chapter for a specific lesson
  ```bash
  > gen_book_chapter.sh data605 03.2
  ```
  This generates a book chapter PDF for DATA605 Lesson 03.2.

- Concatenate all chapters into a single book
  ```bash
  > concatenate_pdfs.py --input_files 'data605/book/Lesson*.pdf' --output_file data605/book/book.pdf
  ```
  This combines all lesson PDFs into a single book.pdf file.
