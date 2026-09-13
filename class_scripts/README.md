# Summary

A suite of command-line tools and scripts for managing university courses, generating
lecture materials (slides, scripts, quizzes), and improving slide quality through
automated LLM-powered transformations

## Structure of the Dir

- The current course/book directories are:
  - `data605`
  - `msml610`
  - `book_springer`
  - `book.Agentic_AI`
  - `book.AI_for_coding`
  - `book.AI_for_data_science`
  - `book.Causal_Probabilistic_ML`
  - `book.Decentralized_AI`
  - `book.Modern_AI_for_Finance`

- Each course/book directory that the scripts read from and write to (e.g.,
  `data605/`, `msml610/`) contains a subset of the following dirs:
  - `lectures_source/`: directory containing `Lesson*.smd` files
  - `lectures_pdf.tmp/`: directory storing temporary PDF slides
  - `lectures_pdf/`: directory storing generated PDF slides
    - E.g., `gen_slides.py`, `for_loop_lessons.py --action generate_pdf`
  - `lectures_tex/`: directory for generated `.tex` slide sources
    - E.g., `for_loop_lessons.py --action generate_tex`
  - `lectures_video_script/`: directory for generated video script files
  - `lectures_quizzes/`: directory for multiple choice quiz files
  - `lectures_recap/`: directory for discussion and recap question files
  - `lectures_commentary/`: directory with md, pdf, and html files commenting the
    corresponding slides
  - `book/`: directory for the book chapters in PDFs
    - E.g., `gen_lecture_commentary.py`
  - `book_source/`: directory for the book source as `.typ` or `.tex`
  - `tutorial/`: Dir with the Jupyter notebook tutorials for each lecture
  - `test/`: unit tests for the scripts in this directory
    - One `test_<module>.py` per source module

// TODO(ai_gp2): Make sure all dirs have this layout

## Description of Files
- The scripts implement the functionalities from the following class:
  - **Analysis**: Counting and analysis scripts
  - **Generation**: Generation of other artifacts from the scripts
  - **Quality**: Slide quality/improvement scripts
  - **Processing**: Image and PDF processing scripts
  - **Test**: Test utilities
  - **Utility**: Utility scripts
  - **Orchestration**: Orchestration scripts

- The scripts under `class_scripts` are

| Goal          | Script                              | Function                                                                     |
| :------------ | :---------------------------------- | :---------------------------------------------------------------------------- |
| Analysis      | `count_lecture_slides.py`           | Count slides, headers, lines, words, and characters in source files          |
| Analysis      | `count_pdf_pages.py`                | Count pages in PDF files (lecture slides or book chapters)                   |
| Analysis      | `count_words.py`                    | Count words in lecture scripts                                               |
| Generation    | `create_book_toc_from_slides.py`    | Extract table of contents from lecture slides                                |
| Generation    | `gen_lecture_commentary.py`         | Generate book chapters from lecture source material                          |
| Generation    | `gen_lecture_video_script.py`       | Generate lecture scripts from slides with intro/outro sections               |
| Generation    | `gen_quizzes.py`                    | Generate quizzes (20 MC) or discussion questions (3-6) from lectures         |
| Generation    | `gen_slides.py`                     | Generate lecture slide PDFs from source files                                |
| Generation    | `generate_class_images.py`          | Generate images using DALL-E from text prompts                               |
| Generation    | `publish_class_links.py`            | Generate an HTML page linking to each lesson's slides, commentary, and recap |
| Quality       | `fix_bold_in_slides.sh`             | Replace `**Tag**` bold labels with `@Tag@` for canonical slide tags          |
| Quality       | `slide_check.py`                    | Check and fix text in lecture slides (spelling, grammar)                     |
| Quality       | `slide_improve.py`                  | Improve slides using LLM suggestions                                         |
| Quality       | `slide_reduce.py`                   | Reduce and simplify slides using LLM                                         |
| Processing    | `extract_png_from_pdf.py`           | Extract PDF pages as PNG images with customizable DPI                        |
| Processing    | `process_slides.py`                 | Process slides with LLM transformations (reduce, check, improve)             |
| Test          | `gen_slides_test_utils.py`          | Helper functions for slide generation testing and validation                 |
| Utility       | `common_utils.py`                   | Argument validation, file finding, directory management                      |
| Utility       | `get_lecture_file.py`               | Find and print path to lecture source file                                   |
| Utility       | `slides_utils.py`                   | Extract and process slide content                                            |
| Orchestration | `for_loop_lessons.py`               | Main orchestrator for processing a set of lectures                           |
| Orchestration | `for_loop_slides.py`                | Transform slides of a lectures using LLM                                     |

- The scripts under `helpers_root/dev_scripts_helpers/` used by the scripts in
  `class_scripts` are:

| Script                | Description                                                            |
| :-------------------- | :--------------------------------------------------------------------- |
| `concatenate_pdfs.py` | Combines multiple PDF files into one (creates full book from chapters) |
| `lint_text.py`         | Lints and formats text using prettier; used for quiz output            |
| `notes_to_pdf.py`     | Converts markdown to PDF (slides, documents); used by gen_slides.py    |
| `to_github.py`        | Converts a local file path to its GitHub URL; used by publish_class_links.py |

## Script Dependency Hierarchy

- `for_loop_lessons.py` (Main orchestrator)
  - action=`generate_pdf` -> `notes_to_pdf.py`
  - action=`generate_tex` -> `notes_to_pdf.py`
  - action=`generate_script` -> `gen_lecture_video_script.py`
  - action=`reduce_slide` -> `process_slides.py`
  - action=`check_slide` -> `process_slides.py`
  - action=`improve_slide` (not yet implemented)
  - action=`generate_lecture_commentary` -> `gen_lecture_commentary.py`
  - action=`generate_class_quizzes` -> `gen_quizzes.py`
  - action=`generate_class_recap` -> `gen_quizzes.py`
  - action=`generate_toc` -> `extract_toc_from_txt.py`

- **Wrapper/Convenience Scripts**
  - `gen_slides.py`
    - `notes_to_pdf.py`
  - `gen_lecture_video_script.py`
    - `llm_cli.py`
    - `lint_text.py`
  - `gen_lecture_commentary.py`
    - `notes_to_pdf.py`
  - `slide_check.py`
    - `process_slides.py`
  - `slide_improve.py`
    - `process_slides.py`
  - `slide_reduce.py`
    - `process_slides.py`
  - `gen_quizzes.py`
    - `llm_cli.py`
    - `lint_text.py`
  - `publish_class_links.py`
    - `to_github.py`

- **Standalone Analysis/Utility Scripts** (no dependencies on other scripts)
  - `count_pdf_pages.py`
  - `count_lecture_slides.py`
  - `count_words.py`
  - `get_lecture_file.py`
  - `extract_png_from_pdf.py`
  - `generate_class_images.py`
  - `fix_bold_in_slides.sh`

# Counting and Analysis Scripts

## `count_lecture_slides.py`

### What It Does

- Counts slides, headers (at 3 levels), lines, words, and characters in lecture
  source files in `{DIR}/lectures_source/` directory
- Displays results in a formatted table supporting markdown (default), TSV, and
  CSV output formats

### Examples

- Count lecture slides with default markdown output:
  ```bash
  > count_lecture_slides.py msml610
  | File                                             |   Slides |   H1 |   H2 |   H3 |   Lines |   Words |   Chars |
  |--------------------------------------------------|----------|------|------|------|---------|---------|---------|
  | Lesson01.1-Intro.txt                             |        9 |    0 |    0 |    0 |     201 |     723 |    5903 |
  | Lesson01.2-Big_Data.txt                          |       16 |    0 |    0 |    0 |     309 |    1282 |    8845 |
  | Lesson01.3-Is_Data_Science_Just_Hype.txt         |       13 |    0 |    0 |    0 |     185 |     671 |    5274 |
  | Lesson02.1-Git.txt                               |       14 |    0 |    0 |    0 |     364 |    1265 |    9457 |
  ```

- Count with TSV format for easy spreadsheet import:
  ```bash
  > count_lecture_slides.py msml610 --format tsv
  File	  Slides	  H1	  H2	  H3	  Lines	  Words	  Chars
  Lesson01.1-Intro.txt	       9	   0	   0	   0	    201	    723	   5903
  Lesson01.2-Big_Data.txt	      16	   0	   0	   0	    309	   1282	   8845
  ```

- Count with CSV format:
  ```bash
  > count_lecture_slides.py msml610 --format csv
  File,Slides,H1,H2,H3,Lines,Words,Chars
  Lesson01.1-Intro.txt,9,0,0,0,201,723,5903
  Lesson01.2-Big_Data.txt,16,0,0,0,309,1282,8845
  ```

## `count_pdf_pages.py`

### What It Does

- Counts pages in PDF files using macOS `mdls` command to extract PDF metadata
- `--dir DIR`: counts pages in all PDF files matching `Lesson*.pdf` in `DIR`
  (e.g., the `{DIR}/lectures/` dir with slides generated by `gen_slides.py`,
  or the `{DIR}/book/` dir with book chapter PDFs generated by
  `gen_lecture_commentary.py`)
- `-i, --input FILE`: counts pages in the single PDF file `FILE`

### Examples

- Count pages for all lecture PDFs in a course:
  ```bash
  > count_pdf_pages.py --dir data605/lectures
  > count_pdf_pages.py --dir msml610/lectures
  Lesson01.1-Intro.pdf	8
  Lesson01.2-Big_Data.pdf	9
  Lesson01.3-Is_Data_Science_Just_Hype.pdf	10
  Lesson01.4-Data_Models.pdf	12
  Lesson02.1-Git.pdf	15
  ```

- Count pages for all book chapter PDFs in a course:
  ```bash
  > count_pdf_pages.py --dir msml610/book
  Lesson01.1-Intro.book_chapter.pdf	12
  Lesson01.2-Big_Data.book_chapter.pdf	18
  Lesson01.3-Is_Data_Science_Just_Hype.book_chapter.pdf	16
  Lesson01.4-Data_Models.book_chapter.pdf	14
  ```

- Count pages for a single PDF file:
  ```bash
  > count_pdf_pages.py --input data605/lectures/Lesson01.1-Intro.pdf
  data605/lectures/Lesson01.1-Intro.pdf	8
  ```

## `count_words.py`

### What It Does

- Counts words in all files in the `{DIR}/lectures_video_script/` directory to help
  track lecture length and content volume

### Examples

- Count words in lecture scripts:
  ```bash
  > count_words.py data605
  > count_words.py msml610
  Lesson01.1-Intro.script.txt	1346
  Lesson01.2-Big_Data.script.txt	2088
  Lesson01.3-Is_Data_Science_Just_Hype.script.txt	1339
  Lesson01.4-Data_Models.script.txt	1957
  Lesson02.1-Git.script.txt	2811
  ```

# Generation Scripts

## `gen_slides.py`

### What It Does

- Generates lecture slide PDFs from source files using `notes_to_pdf.py` to
  convert markdown to PDF, writing output to `<dir>/lectures/`
  - `-i`/`--input`: either `<dir>/<lesson>` (e.g., `data605/08.1`) or a direct
    path to a `Lesson*.txt` file
  - `--daemon` watches the input file and regenerates the PDF on change;
  - `--slides_engine {beamer,typst}` selects the rendering engine
  - `--notes_to_pdf_args` passes a verbatim options string through to
    `notes_to_pdf.py`

### Examples

- Generate slides with default settings:
  ```bash
  > gen_slides.py -i data605/01.1
  Running command: notes_to_pdf.py --input=data605/lectures_source/Lesson01.1-Intro.txt --output=data605/lectures/Lesson01.1-Intro.pdf --type=slides --toc_type=navigation --debug_on_error --skip_action=cleanup_before --skip_action=cleanup_after
  [notes_to_pdf output...]
  ```

- Generate slides directly from a source file path:
  ```bash
  > gen_slides.py -i msml610/lectures_source/Lesson10.2-Causal_Discovery.txt
  ```

- Regenerate the PDF automatically whenever the source file changes:
  ```bash
  > gen_slides.py -i data605/01.1 --daemon
  ```

- Render with the Typst engine and pass an extra option through to
  `notes_to_pdf.py`:
  ```bash
  > gen_slides.py -i msml610/02.3 --slides_engine typst --notes_to_pdf_args="--filter_by_slides 1:5"
  ```

## `gen_lecture_video_script.py`

### What It Does

- Generates complete lecture scripts from slides using LLM
- Automatically creates intro and outro sections, combines all sections, and
  lints the final output

### Examples

- Generate script for the video lecture
  ```bash
  > gen_lecture_video_script.py data605/01.1
  Generating lecture script for Lesson01.1
  Reading slides from: data605/lectures_source/Lesson01.1-Intro.txt
  Generating intro section...
  Generating script sections...
  Generating outro section...
  Combining and linting output...
  Script saved to: data605/lectures_video_script/Lesson01.1-Intro.script.txt
  Total words: 1346
  ```

- Generate script with a larger slide grouping:
  ```bash
  > gen_lecture_video_script.py msml610/02.3 --slides_per_group 5
  ```

- Process only a range of slides:
  ```bash
  > gen_lecture_video_script.py data605/01.1 --limit "10:20"
  ```

## `gen_lecture_commentary.py`

### What It Does

- Generates a commented PDF ("lecture commentary") from a lecture source file in
  one step: converts the lecture source to a temporary PDF with `notes_to_pdf.py`,
  extracts slide images from the PDF and generates per-slide LLM commentary,
  renders the result to PDF with `pandoc`, and opens it in Skim
- Extracts title from markdown file (e.g., from `\text{\blue{Lesson 2.1: Git}}`)
  and adds YAML preamble for pandoc metadata
- Validates that the number of slides in markdown matches the number of PNG
  files (expects `num_slides + 1 = num_pngs` to account for title slide)
- Properly aligns title slide (first PNG) with content slides (remaining PNGs)
  to ensure header, slide image, and commentary are synchronized
- First slide (PNG 1) is treated as title slide with only the image (no title or
  commentary)
- Content slides (PNG 2+) are paired with corresponding markdown slides, with
  centered headers formatted as "idx / tot: title" and LLM-based commentary
- Formats output with `prettier` for consistent markdown formatting

### Examples

- Generate commentary for a single lesson:
  ```bash
  > gen_lecture_commentary.py data605/01.1
  > gen_lecture_commentary.py msml610/02.3
  ```
  Output: `<class>/book/Lesson##.#-Topic.book_chapter.{txt,pdf}`

## `gen_quizzes.py`

### What It Does

- Generates questions from lecture content using LLM
- Supports two modes:
  - _Multiple choice quizzes_: 20 questions with 5 answers each ->
    `{DIR}/lectures_quizzes/<lesson>.quizzes.md`
  - _Discussion/review questions_: 3-6 open-ended questions ->
    `{DIR}/lectures_recap/<lesson>.recap.md`
- Automatically formats output using `lint_text.py` with prettier (use
  `--no_lint` to skip)

### Examples

- Generate multiple choice quiz:
  ```bash
  > gen_quizzes.py --for_class_quizzes -i data605/01.1
  Reading lecture from: data605/lectures_source/Lesson01.1-Intro.txt
  Generating 20 multiple choice questions...
  Formatting with prettier...
  Quiz saved to: data605/lectures_quizzes/Lesson01.1-Intro.quizzes.md
  Generated 20 questions with 5 options each
  ```

- Generate discussion questions:
  ```bash
  > gen_quizzes.py --for_class_recap -i msml610/02.3
  Reading lecture from: msml610/lectures_source/Lesson02.3-...txt
  Generating 5 discussion questions...
  Recap saved to: msml610/lectures_recap/Lesson02.3-...recap.md
  ```

- Generate without linting:
  ```bash
  > gen_quizzes.py --for_class_recap -i data605/01.2 --no_lint
  ```

- Generate with specific model:
  ```bash
  > gen_quizzes.py --for_class_quizzes -i data605/01.1 --model gpt-4
  ```

## `create_book_toc_from_slides.py`

### What It Does

- Extracts and combines table of contents from lecture slides into a structured
  document
- Supports two modes:
  - batch processing to separate file
  - in-place insertion
- Parses lesson files from `### Lessons` sections in markdown files from
  `book_map.md`
- Creates combined TOC with chapter organization and formatted lesson sections
  - `--max_level` caps the header depth extracted
  - `--max_number` caps the number of chapters (H2 headers) included (0 = all,
    the default)

### Examples

- Generate combined TOC to separate file:
  ```bash
  > create_book_toc_from_slides.py --input=book_map.md --output=book_toc.md --max_level=2
  Found 8 chapters
  Processing lectures: 100%|████████| 24/24
  Wrote output to 'book_toc.md'
  ```

- Insert TOC directly into a markdown file (in-place mode):
  ```bash
  > create_book_toc_from_slides.py --input=book.Causal_Probabilistic_ML/book_map.md --in_place --max_level=3
  Inserted TOC into 'book.Causal_Probabilistic_ML/book_map.md'
  ```

- Book map format with `### Lessons` section:
  ```markdown
  ## 1: Chapter Title
  
  ### Lessons
  - `msml610/lectures_source/Lesson08.1-Causal_AI_intro.txt`
  - `msml610/lectures_source/Lesson08.2-Causal_Models.txt`
  ```

- After running with `--in_place`, inserts:
  ```markdown
  ### Current TOC
  
  // msml610/lectures_source/Lesson08.1-Causal_AI_intro.txt
  
  - Topic 1 (5)
    - Subtopic A (2)
  - Topic 2 (8)
  
  // msml610/lectures_source/Lesson08.2-Causal_Models.txt
  
  - Introduction (3)
  ```

## `publish_class_links.py`

### What It Does

- Scans `{DIR}/lectures_source/` for lesson files and generates an HTML page
  linking each lesson to its generated artifacts:
  - Slides PDF: `{DIR}/lectures_pdf/{LESSON}.pdf`
  - Lecture commentary: `{DIR}/lectures_commentary/{LESSON}.book_chapter.html`
    and `.book_chapter.pdf`
  - Lesson recap: `{DIR}/lectures_recap/{LESSON}.recap.md`
- Each existing artifact is linked to its GitHub URL (via `to_github.py`)
  instead of a local path, so the page works when shared outside a local
  checkout
  - `--use_master` links to the `master` branch instead of the current branch
- By default stops with an error as soon as an artifact is missing
  - `--do_not_fail_on_warnings` logs missing artifacts as warnings instead and
    lists the lesson without the missing link

### Examples

- Generate the links page for a class, failing on the first missing artifact:
  ```bash
  > publish_class_links.py --dir data605 --out_file data605/class_links.html
  ```

- Generate the links page while tolerating missing artifacts, linking to
  `master`:
  ```bash
  > publish_class_links.py --dir msml610 --out_file /tmp/links.html --do_not_fail_on_warnings --use_master
  ```

## `process_slides.py`

### What It Does

- Extracts individual slides from a markdown file and processes each with an
  LLM prompt selected by `--action`
  - E.g., `slide_reduce`, `slide_improve`, `text_check`, `text_check_fix`, from
    `llm_prompts.py`
  - `--in_file`, `--action`, and `--out_file` are required
  - `--use_llm_transform` actually invokes the LLM (otherwise slides are passed
    through unchanged, useful for dry-testing the extraction/reassembly logic)
  - `--no_abort_on_error` continues processing remaining slides if one LLM call
    fails
  - `--limit` restricts processing to a slide range (e.g., `1:5`)

### Examples

- Process slides with LLM transformation:
  ```bash
  > process_slides.py --in_file lecture.txt --action slide_reduce --out_file output.txt --use_llm_transform
  Processing input file: lecture.txt
  Found 15 slides
  Processing with action: slide_reduce (threads: 4)
  Slide 1/15: Processing... [2.5s]
  Slide 2/15: Processing... [2.3s]
  ...
  Completed: 15/15 slides
  Output saved to: output.txt
  Processing time: 45s
  ```

- Check slide quality and generate a report:
  ```bash
  > process_slides.py --in_file lecture.txt --action text_check --out_file check_report.txt --use_llm_transform
  ```

- Process only a specific slide range:
  ```bash
  > process_slides.py --in_file lecture.txt --action slide_reduce --out_file output.txt --use_llm_transform --limit "1:5"
  ```

- Continue processing remaining slides even if one fails:
  ```bash
  > process_slides.py --in_file lecture.txt --action slide_reduce --out_file output.txt --use_llm_transform --no_abort_on_error
  ```

# Slide Improvement Scripts

## `slide_check.py`

### What It Does

- Checks and fixes text (spelling, grammar, formatting) in lecture slides,
  in place, by shelling out to `process_slides.py --action text_check_fix
  --use_llm_transform`
- `--process_slides_args` passes a verbatim options string through to
  `process_slides.py` (e.g., `--limit`)

### Examples

- Check and fix slides for a lesson (overwrites the source file):
  ```bash
  > slide_check.py -i data605/01.1
  ```

- Check and fix only a slide range:
  ```bash
  > slide_check.py -i msml610/02.3 --process_slides_args="--limit 1:5"
  ```

## `slide_improve.py`

### What It Does

- Improves lecture slides in place using LLM suggestions, by shelling out to
  `process_slides.py --action slide_improve --use_llm_transform`
- `--process_slides_args` passes a verbatim options string through to
  `process_slides.py`

### Examples

- Improve slides for a lesson (overwrites the source file):
  ```bash
  > slide_improve.py -i data605/01.1
  Improving slides in: data605/lectures_source/Lesson01.1-Intro.txt
  Processing 9 slides with LLM improvement suggestions...
  Slide 1: Suggested clearer explanation of key concepts
  Slide 2: Suggested adding examples for better understanding
  Slide 3: Suggested reorganizing content for better flow
  ...
  Suggestions saved to: data605/lectures_source/Lesson01.1-Intro.improved.txt
  Total suggestions: 7
  ```

- Improve only a slide range:
  ```bash
  > slide_improve.py -i msml610/02.3 --process_slides_args="--limit 1:5"
  ```

## `slide_reduce.py`

### What It Does

- Reduces and simplifies lecture slides in place using LLM, by shelling out
  to `process_slides.py --action slide_reduce --use_llm_transform`
- `--process_slides_args` passes a verbatim options string through to
  `process_slides.py`

### Examples

- Reduce slide content for a lesson (overwrites the source file):
  ```bash
  > slide_reduce.py -i data605/01.1
  Reducing slides in: data605/lectures_source/Lesson01.1-Intro.txt
  Processing 9 slides with LLM reduction...
  Slide 1: Reduced from 150 words to 95 words (37% reduction)
  Slide 2: Reduced from 120 words to 78 words (35% reduction)
  Slide 3: No reduction needed (already concise)
  ...
  Results saved to: data605/lectures_source/Lesson01.1-Intro.reduced.txt
  Total reduction: 35% (avg)
  ```

- Reduce only a slide range:
  ```bash
  > slide_reduce.py -i msml610/02.3 --process_slides_args="--limit 1:5"
  ```

## `fix_bold_in_slides.sh`

### What It Does

- Replaces `**Tag**` bold labels with `@Tag@` for the canonical set of
  slide/lecture tags (see "### Tags" in `.claude/skills/slides.rules.md`)
- Edits files in place using `perl -i -pe`
- Requires at least one file argument

### Examples

- Fix bold tags in a single lecture file:
  ```bash
  > fix_bold_in_slides.sh data605/lectures_source/Lesson01.1-Intro.txt
  ```

- Fix bold tags across multiple lecture files:
  ```bash
  > fix_bold_in_slides.sh data605/lectures_source/Lesson*.txt
  ```

# Image and PDF Processing Scripts

## `extract_png_from_pdf.py`

### What It Does

- Extracts each page of a PDF file as a separate PNG image
- Numbers output files sequentially (slides001.png, slides002.png, etc.)
- Supports customizable DPI for image quality control
- Creates output directory automatically with optional from-scratch mode

### Examples

- Extract all pages from a PDF with default settings:
  ```bash
  > extract_png_from_pdf.py --input_file data605/lectures/Lesson01.1-Intro.pdf --output_dir output
  Processing PDF file: data605/lectures/Lesson01.1-Intro.pdf
  Output directory: output
  Converting PDF to images with DPI=300
  Found 8 pages in PDF
  Extracting pages: 100%|████████| 8/8
  Successfully extracted 8 PNG images to output
  ```
  Output files created: `output/slides001.png`, `output/slides002.png`, ..., `output/slides008.png`

- Extract with higher DPI for better image quality:
  ```bash
  > extract_png_from_pdf.py --input_file lecture.pdf --output_dir slides --dpi 300
  ```

- Create output directory from scratch:
  ```bash
  > extract_png_from_pdf.py --input_file presentation.pdf --output_dir ./images/ --from_scratch
  ```

## `generate_class_images.py`

### What It Does

- Generates multiple images using OpenAI's DALL-E 3 API from text prompts
- Supports both standard and HD quality image generation in 1024x1024 resolution
- Includes special workload mode for generating predefined image sets for course
  materials

### Examples

- Generate 5 HD quality images from a prompt:
  ```bash
  > generate_class_images.py "A sunset over mountains" --dst_dir ./images
  Generating 5 images with prompt: 'A sunset over mountains'
  Resolution: 1024x1024, Quality: hd
  Generating image 1/5
  Downloading image to ./images/image_01_hd.png
  Saved image to: ./images/image_01_hd.png
  Generating image 2/5
  Downloading image to ./images/image_02_hd.png
  Saved image to: ./images/image_02_hd.png
  ...
  Image generation complete. Images saved to: ./images
  ```

- Generate standard quality images with custom count:
  ```bash
  > generate_class_images.py "A cat wearing a hat" --dst_dir ./images --count 3 --low_res
  ```

- Generate images for the predefined MSML610 workload (note: the code checks
  for the literal string `"MSLM610"`):
  ```bash
  > generate_class_images.py --dst_dir ./course_images --workload MSLM610
  ```

# Test Utilities

## `gen_slides_test_utils.py`

### What It Does

- Provides shared helper functions for slide generation testing
- Discovers available lessons, drives slide generation for a given lesson, and
  validates the resulting output
- Not an executable; imported by test modules (e.g., `test/test_gen_slides.py`)
  rather than run directly from the command line

# Utility Scripts

## `get_lecture_file.py`

### What It Does

- Finds and prints the path to a lecture source file matching
  `{DIR}/lectures_source/Lesson{LESSON}*`
- Validates that exactly one matching file exists

### Examples

- Find lecture file:
  ```bash
  > get_lecture_file.py data605/01.1
  > get_lecture_file.py msml610/02.3
  Lecture file: data605/lectures_source/Lesson01.1-Intro.txt
  ```

# Orchestration Scripts

## `for_loop_slides.py`

### What It Does

- Transforms lecture slides using LLM with specified rules, reading slides
  from lesson files and writing the transformed slides back to the file or
  output
- Extracts slides from lesson files while preserving file structure
- Applies LLM transformations using specified rule prompts
- Processes slides in configurable batches for efficiency
- Supports multiple batch modes: individual, shared_prompt, combined
- Integrates with `hllm_cli._process_batches` for batch processing

**Command Line Arguments:**

- `--input_file`: Input lesson file containing slides (required)
- `--output_file`: Output file for transformed slides (optional)
- `--prompt_file` or `--prompt`: Rule prompt to guide transformation
- `--model`: LLM model to use (default: "claude-opus-4")
- `--batch_size`: Number of slides per batch (default: 1)
- `--batch_mode`: Batch processing mode - "individual", "shared_prompt", or
  "combined" (default: "individual")
- `-v/--log_level`: Set logging verbosity (DEBUG, INFO, WARNING, ERROR)

### Examples

- Transform slides using a prompt file:
  ```bash
  > for_loop_slides.py --input_file data605/lectures_source/Lesson01.1-Intro.txt \
      --prompt_file my_rules.prompt.md --output_file output.txt
  ```

- Transform slides in batches with custom model:
  ```bash
  > for_loop_slides.py --input_file msml610/lectures_source/Lesson02.1-*.txt \
      --prompt "Simplify the content" --model claude-opus-4 --batch_size 5
  ```

- Transform with combined batch mode:
  ```bash
  > for_loop_slides.py --input_file lecture.txt --prompt_file rules.txt \
      --batch_mode combined --batch_size 3
  ```

## `for_loop_lessons.py`

### What It Does

- Orchestrates the generation of multiple outputs from lecture source files
  for educational materials; the main entry point for processing lecture
  content into various formats
- Converts lecture text source files to PDF slides using `notes_to_pdf.py`
- Generates reading scripts from lecture materials with transition text
- Applies LLM-based transformations for slide reduction and quality checking
- Generates book chapters from lecture content
- Supports batch processing of multiple lectures using pattern matching
- Provides slide range limiting for focused processing
- Includes dry-run mode for previewing commands

**Available Actions:**

- `generate_pdf`: Generate presentation slide PDFs from text source files,
  written to `<class>/lectures/`
- `generate_tex`: Generate `.tex` slide sources (no PDF), written to
  `<class>/lectures_tex/`
- `generate_script`: Generate instructor reading scripts with commentary
- `reduce_slide`: Apply LLM transformation to reduce slide content
- `check_slide`: Apply LLM validation to check slide quality
- `improve_slide`: Not yet implemented — aborts with an error
- `generate_lecture_commentary`: Generate book chapter PDF (via
  `gen_lecture_commentary.py`) from lecture content, written to
  `<class>/book/`
- `generate_class_quizzes`: Generate multiple choice quizzes from lecture
  content using LLM
- `generate_class_recap`: Generate open-ended discussion/review questions from
  lecture content using LLM
- `generate_toc`: Extract table of contents (headers) from all lectures and
  create a consolidated course syllabus

**Lecture Pattern Examples:**

- Single lecture: `01.1`
- Wildcard pattern: `01*`
- Multiple patterns: `01*:02*:03.1` (separated by colons)
- Continuous range: `01.1-03.2` (inclusive)

**Command Line Arguments:**

- `--lectures`: Lecture(s) to process (optional; if omitted, processes all
  lessons in `--class`)
  - Single pattern: '01.1' or '01\*'
  - Union of patterns (colon-separated): '01*:02*:03.1'
  - Continuous range (hyphen-separated): '01.1-03.2' (inclusive)
  - Note: Range and union syntax cannot be mixed
- `--class`: Class directory name (required, e.g., data605, msml610)
- `--action`: Actions to perform (default: generate_pdf)
  - Can specify multiple: `--action generate_pdf --action generate_script`
- `--limit`: Optional slide range to process (e.g., '1:3')
  - Only works when processing a single lecture file
- `--cmd_opts`: Extra options string passed through verbatim to the invoked
  commands (e.g., `gen_lecture_commentary.py`, `notes_to_pdf.py`,
  `process_slides.py`), to control their behavior from inside the loop
  - E.g., `--cmd_opts="--no_incremental --open_pdf"`
- `--dry_run`: Print commands without executing them
- `-v/--log_level`: Set logging verbosity (DEBUG, INFO, WARNING, ERROR)

### Examples

- Generate PDF for single lecture:
  ```bash
  > for_loop_lessons.py --lectures 01.1 --class data605 --action generate_pdf
  ```

- Generate scripts for multiple lectures:
  ```bash
  > for_loop_lessons.py --lectures 01*:02* --class data605 --action generate_script
  ```

- Multiple actions on same lectures:
  ```bash
  > for_loop_lessons.py --lectures 01* --class msml610 --action generate_pdf --action generate_script
  ```

- Partial slide processing:
  ```bash
  > for_loop_lessons.py --lectures 01.1 --limit 1:3 --class data605 --action generate_pdf
  ```

- Process a continuous range of lessons:
  ```bash
  > for_loop_lessons.py --lectures "01.1-03.2" --class data605 --action generate_pdf
  ```

- Reduce slide content using LLM for a single lecture:
  ```bash
  > for_loop_lessons.py --lectures "01.1" --class data605 --action reduce_slide
  ```

- Generate multiple choice quizzes from lecture content:
  ```bash
  > for_loop_lessons.py --lectures "01.1" --class data605 --action generate_class_quizzes
  ```

- Process with verbose logging for debugging:
  ```bash
  > for_loop_lessons.py --lectures "01.1" --class data605 --action generate_pdf -v DEBUG
  ```

### Workflow

1. Parse lecture patterns or ranges from command line arguments (e.g., '01*',
   '01.1', '01*:03\*', '01.1-03.2')
2. Find matching lecture source files in `<class>/lectures_source/` directory
3. For each matching file, execute specified actions in sequence
4. Output generated files to appropriate directories:
   - PDF slides -> `<class>/lectures/`
   - TeX sources -> `<class>/lectures_tex/`
   - Scripts -> `<class>/lectures_video_script/`
   - Book chapters -> `<class>/book/`
   - Multiple choice quizzes -> `<class>/lectures_quizzes/`
   - Discussion/recap questions -> `<class>/lectures_recap/`

# Common Workflows

## Generating Course Materials

### Generate PDF Slides for All Lessons in a Course

- Generates PDF files for all lessons starting with 0 or 1 (e.g., 01.1, 01.2,
  10.1, etc.) in `data605/lectures/`:
  ```bash
  > for_loop_lessons.py --lectures "0*:1*" --class data605 --action generate_pdf
  ```

### Generate Both PDF Slides and Reading Scripts

- Generates PDFs in `lectures/` and scripts in `lectures_video_script/`:
  ```bash
  > for_loop_lessons.py --lectures 01* --class msml610 --action generate_pdf --action generate_script
  ```

### Generate PDF and Book Chapter for a Single Lesson

- Creates slide PDF and corresponding book chapter with pandoc conversion:
  ```bash
  > for_loop_lessons.py --lectures 01.1 --class data605 --action generate_pdf --action generate_lecture_commentary
  ```

## Lecture Commentary Generation

Generate book chapter PDFs from lecture source files (slides + commentary text). Book
chapters include slide images with LLM-generated commentary on each slide, formatted
for reading and study.

### Generate Commentary for Single Lesson

- Generate a book chapter for one specific lesson:

```bash
> gen_lecture_commentary.py data605/01.1
> gen_lecture_commentary.py msml610/02.3
```

- This script:
  1. Generates PDF from lecture source using `notes_to_pdf.py`
  2. Extracts PNG images from the PDF
  3. Generates markdown book chapter with LLM commentary on each slide
  4. Converts markdown to PDF using pandoc
  5. Opens the resulting PDF in Skim

- Output: `<class>/book/Lesson##.#-Topic.book_chapter.{txt,pdf}`

### Generate Commentary for All Lessons in a Course

- Generate book chapters for all lessons in a course:

```bash
> for_loop_lessons.py --class data605 --action generate_lecture_commentary
> for_loop_lessons.py --class msml610 --action generate_lecture_commentary
```

- This processes all lecture source files and generates:
  - Markdown book chapter: `<class>/book/Lesson##.#-Topic.book_chapter.md`
  - PDF book chapter: `<class>/book/Lesson##.#-Topic.book_chapter.pdf`

### Generate Commentary for All Lessons, Controlling `gen_lecture_commentary.py` Behavior

- Use `--cmd_opts` to pass extra flags through to the underlying
  `gen_lecture_commentary.py` invocation for every lesson in the loop, e.g.,
  force regeneration and open each PDF as it's produced:
  ```bash
  > for_loop_lessons.py --class data605 \
      --action generate_lecture_commentary \
      --cmd_opts="--no_incremental --image_type jpg"
  ```

### Generate Commentary for Pattern-Matched Lessons

- Generate book chapters for specific lessons using patterns or ranges:
  ```bash
  # Pattern: all lessons starting with 01
  > for_loop_lessons.py --class data605 --lectures "01*" --action generate_lecture_commentary

  # Multiple patterns (colon-separated)
  > for_loop_lessons.py --class msml610 --lectures "01*:02*:03.1" --action generate_lecture_commentary

  # Continuous range (inclusive)
  > for_loop_lessons.py --class data605 --lectures "01.1-03.2" --action generate_lecture_commentary
  ```

## Assessment Generation

### Generate Multiple Choice Quizzes

- Creates 20-question quizzes saved to `lectures_quizzes/<lesson>.quizzes.md`:
  ```bash
  > for_loop_lessons.py --lectures 01* --class data605 --action generate_class_quizzes
  ```

- Alternatively, use the direct script:
  ```bash
  > gen_quizzes.py --for_class_quizzes -i data605/01.1
  ```

### Generate Discussion/review Questions

- Creates 3-6 open-ended discussion questions saved to
  `lectures_recap/<lesson>.recap.md`:
  ```bash
  > for_loop_lessons.py --lectures 01* --class data605 --action generate_class_recap
  ```

- Alternatively, use the direct script:
  ```bash
  > gen_quizzes.py --for_class_recap -i data605/01.1
  ```

## Slide Quality Improvement

### Check and Fix Spelling/grammar in Slides

- Use for_loop_lessons.py:
  ```bash
  > for_loop_lessons.py --lectures 01.1 --class data605 --action check_slide
  ```

- **Advanced usage** (check one lecture from inside the container):
  ```bash
  > SRC_NAME=$(ls $DIR/lectures_source/Lesson02*); echo $SRC_NAME
  > DST_NAME=process_slides.txt
  > i docker_bash
  docker> process_slides.py --in_file $SRC_NAME --action text_check --out_file $DST_NAME --use_llm_transform --limit 0:10
  > vimdiff $SRC_NAME process_slides.txt
  ```

- Alternatively, use the direct script:
  ```bash
  > slide_check.py -i data605/01.1
  ```

### Improve Slide Clarity and Structure

- `for_loop_lessons.py --action improve_slide` is **not yet implemented** and
  aborts with an error; use the direct script instead:
  ```bash
  > slide_improve.py -i data605/01.1
  ```

- Or use `llm_transform.py` directly:
  ```bash
  > llm_transform.py -i data605/lectures_source/Lesson07.2-Data_Wrangling.txt -p slide_improve -v DEBUG
  ```

### Reduce Slide Length and Remove Redundancy

- Use for_loop_lessons.py:
  ```bash
  > for_loop_lessons.py --lectures 01.1 --class data605 --action reduce_slide
  ```

- **Advanced usage** (reduce from inside the container):
  ```bash
  > SRC_NAME=$(ls $DIR/lectures_source/Lesson04.2*); echo $SRC_NAME
  > docker> process_slides.py --in_file $SRC_NAME --action slide_reduce --out_file $SRC_NAME --use_llm_transform --limit 0:10
  ```

- Alternatively, use the direct script:
  ```bash
  > slide_reduce.py -i data605/01.1
  ```

### Fix Slides with Custom LLM Prompt

- Uses GPT-4o to fix and improve slides based on a prompt template:
  ```bash
  > FILE=data605/lectures_source/Lesson09.2-Spark_Primitives.txt
  > llm_cli.py --input $FILE -pf "fix_slides.prompt.md" -o improved.md --model "gpt-4o" -b
  ```

## Lecture Script Generation

### Generate Complete Lecture Script with Intro/outro

- Use the direct script:
  ```bash
  > gen_lecture_video_script.py data605/01.1
  ```

- Or use `for_loop_lessons.py`:
  ```bash
  > for_loop_lessons.py --lectures 01.1 --class data605 --action generate_script
  ```

### Generate Just the Intro for a Lecture

- Creates a 50-word introduction for Lesson 08.3:
  ```bash
  > TAG=08.3; llm_cli.py -i data605/lectures_video_script/Lesson${TAG}*.script.txt -p "You are a college professor and you need to do an introduction in 50 word the content of the slides starting with In this lesson" -o -
  ```

### Generate Just the Outro/summary for a Lecture

- Creates a 50-word summary/conclusion for Lesson 08.3:
  ```bash
  > TAG=08.3; llm_cli.py -i data605/lectures_video_script/Lesson${TAG}*.script.txt -p "You are a college professor and you need to summarize what was discussed in less than 50 word in the slides like In this lesson we have discussed" -o -
  ```

### Generate Scripts From Inside a Container (advanced)

- Generates a script for slides 1-5, grouping 3 slides at a time:
  ```bash
  > i docker_bash --base-image=623860924167.dkr.ecr.eu-north-1.amazonaws.com/cmamp --skip-pull
  docker> sudo /bin/bash -c "(source /venv/bin/activate; pip install --upgrade openai)"
  docker> gen_lecture_video_script.py data605/01 --slides_per_group 3 --limit 1:5
  ```

## Format Conversion

### Convert Markdown Notes to PDF Slides

- Converts markdown lecture notes to PDF format (slides 1-4 only):
  ```bash
  > notes_to_pdf.py --input data605/lectures_md/final_enhanced_markdown_lecture_2.txt --output tmp.pdf --type slides --skip_action cleanup_after --debug_on_error --toc_type navigation --filter_by_slides 1:4
  ```

## Course Syllabus and Structure

### Generate Complete Course Syllabus

- Extracts all lecture headers and creates a consolidated syllabus:
  ```bash
  > for_loop_lessons.py --class data605 --action generate_toc
  > for_loop_lessons.py --class msml610 --action generate_toc
  ```
- Output: `<class>/all_tocs.md` containing all lecture headers organized hierarchically

### Generate Syllabus for Specific Lectures

- Extract headers from pattern-matched lectures:
  ```bash
  > for_loop_lessons.py --class data605 --lectures "01*" --action generate_toc
  > for_loop_lessons.py --class data605 --lectures "01*:02*:03.1" --action generate_toc
  > for_loop_lessons.py --class data605 --lectures "01.1-03.2" --action generate_toc
  ```

## Analysis and Reporting

### Count Pages in All Book PDFs
```bash
> count_pdf_pages.py --dir data605/book
```

### Count Pages in All Lecture PDFs
```bash
> count_pdf_pages.py --dir data605/lectures
```

### Count Words in All Lecture Scripts

- Helps track lecture length and content volume:
  ```bash
  > count_words.py data605
  ```

## Partial Processing

### Generate Specific Slides From a Lecture (slides 1-3 Only)

- Only applies to `generate_pdf` action when a single lecture file matches:
  ```bash
  > for_loop_lessons.py --lectures 01.1 --limit 1:3 --class data605 --action generate_pdf
  ```

### Preview Commands Without Executing (dry-run)

- Prints all commands that would be executed without running them:
  ```bash
  > for_loop_lessons.py --lectures 01* --class data605 --action generate_pdf --dry_run
  ```
