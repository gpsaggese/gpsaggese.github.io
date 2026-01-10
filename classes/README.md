# Classes Directory

This directory contains tools for processing and generating lecture materials for
university courses (`data605` and `msml610`).

## Description of Files

- `fix_slides.prompt.md`
  - LLM prompt for improving slides with better structure examples and grammar
- `process_lessons.md`
  - Documentation for process_lessons.py including usage examples and architecture
- `suggest_improvements.prompt.md`
  - LLM prompt for suggesting which slides to remove or merge

## Description of Executables

### `count_pages.sh`

#### What It Does

- Counts the number of pages in PDF lecture files for data605 class
- Iterates through Lesson*.pdf files and uses mdls to extract page counts
- Outputs filename and page count in tab-separated format

#### Examples

- Count pages in all data605 lecture PDFs:
  ```bash
  > count_pages.sh
  ```

- Copy page counts to clipboard:
  ```bash
  > count_pages.sh | pbcopy
  ```

### `count_words.sh`

#### What It Does

- Counts words in lecture script files located in data605/lectures_script/
- Uses wc to count words in each file
- Outputs filename and word count for each script file

#### Examples

- Count words in all lecture scripts:
  ```bash
  > count_words.sh
  ```

### `gen_data605_script.sh`

#### What It Does

- Generates a complete lecture script with LLM-generated intro and outro for
  data605 class
- Takes lesson number as input and finds matching source file
- Calls generate_slide_script.py then uses LLM to create intro/outro sections
- Applies text linting and formatting to the final output

#### Examples

- Generate script for lesson 1:
  ```bash
  > gen_data605_script.sh 1
  ```

- Generate script for lesson 4.3:
  ```bash
  > gen_data605_script.sh 04.3
  ```

- Generate script for lesson 5 with custom options:
  ```bash
  > gen_data605_script.sh 5 --custom-option
  ```

### `gen_data605.sh`

#### What It Does

- Generates PDF slides from text source files for data605 class
- Finds the lesson source file based on lesson number argument
- Calls notes_to_pdf.py to convert text to PDF slides with navigation

#### Examples

- Generate PDF for lesson 1:
  ```bash
  > gen_data605.sh 1
  ```

- Generate PDF for lesson 1 with specific tag:
  ```bash
  > TAG=01; gen_data605.sh $TAG
  ```

- Generate PDF for lesson 3 with additional options:
  ```bash
  > gen_data605.sh 3 --skip_action open
  ```

### `gen_msml610.sh`

#### What It Does

- Generates PDF slides from text source files for msml610 class
- Finds lesson source files based on lesson number pattern
- Converts text files to PDF slides using notes_to_pdf.py

#### Examples

- Generate PDF for lesson 2:
  ```bash
  > gen_msml610.sh 02
  ```

- Generate PDF with custom options:
  ```bash
  > gen_msml610.sh 2 --skip_action cleanup_after
  ```

### `get_data605.sh`

#### What It Does

- Finds and displays the path of a specific lesson file in data605/lectures_source/
- Validates that exactly one matching file exists
- Simple helper script for locating lesson source files

#### Examples

- Find lesson 1 source file:
  ```bash
  > get_data605.sh 1
  ```

- Find lesson 3.2 source file:
  ```bash
  > get_data605.sh 3.2
  ```

### `get_msml610.sh`

#### What It Does

- Finds and displays the path of a specific lesson file in msml610/lectures_source/
- Validates that exactly one matching file exists
- Simple helper script for locating lesson source files

#### Examples

- Find lesson 1 source file:
  ```bash
  > get_msml610.sh 1
  ```

- Find lesson 4 source file:
  ```bash
  > get_msml610.sh 4
  ```

### `slide_check.sh`

#### What It Does

- Checks slides for errors and improvements using LLM transformation
- Takes lesson number as input and finds matching data605 source file
- Generates a separate check report file with suggestions and issues

#### Examples

- Check slides for lesson 1.2:
  ```bash
  > slide_check.sh 01.2
  ```

- Check slides with additional options:
  ```bash
  > slide_check.sh 2 --verbose
  ```

### `slide_compress.sh`

#### What It Does

- Compresses large images in lecture materials using ImageMagick
- Finds the 70 largest image files in data605/lectures_source/images/
- Resizes images to max width of 800px while maintaining aspect ratio

#### Examples

- Compress all large lecture images:
  ```bash
  > slide_compress.sh
  ```

### `slide_improve.sh`

#### What It Does

- Improves slide content using LLM transformation and modifies file in place
- Takes lesson number and finds matching data605 source file
- Uses process_slides.py with slide_improve action to enhance content

#### Examples

- Improve slides for lesson 1:
  ```bash
  > slide_improve.sh 1
  ```

- Improve slides with custom options:
  ```bash
  > slide_improve.sh 3 --dry_run
  ```

### `slide_reduce.sh`

#### What It Does

- Reduces and simplifies slide content using LLM transformation
- Takes lesson number and modifies source file in place
- Uses process_slides.py with slide_reduce action to condense content

#### Examples

- Reduce slides for lesson 1.1 with wildcard:
  ```bash
  > slide_reduce.sh 01.1*
  ```

- Reduce slides for lesson 4:
  ```bash
  > slide_reduce.sh 4 --verbose
  ```

### `generate_class_images.py`

#### What It Does

- Generates multiple images using OpenAI's DALL-E API from text prompts
- Supports both standard and HD quality modes
- Can generate images for predefined workloads like MSLM610 with specific word
  sets

#### Examples

- Generate 5 standard quality images from a prompt:
  ```bash
  > generate_class_images.py "A sunset over mountains" --dst_dir ./images --low_res
  ```

- Generate HD quality images:
  ```bash
  > generate_class_images.py "A cat wearing a hat" --dst_dir ./images
  ```

- Generate custom number of images:
  ```bash
  > generate_class_images.py "Abstract art" --dst_dir ./images --count 3
  ```

- Generate images for MSLM610 workload:
  ```bash
  > generate_class_images.py --workload MSLM610 --dst_dir ./msml610_images
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
