# Python Scripts for Course Management

## Overview

The scripts are organized into several categories:

### Utility Modules

- **common_utils.py** - Shared utility functions including:
  - Argument validation and file finding
  - Directory management
  - PDF operations (page counting using mdls)

### File Finding Scripts

- **get_lecture_file.py** - Find and print the path to a lecture file
  - Replaces: `get_data605.sh` and `get_msml610.sh`
  - Usage: `get_lecture_file.py <DIR> <LESSON>`

### Counting/Statistics Scripts

- **count_words.py** - Count words in lecture script files
  - Replaces: `count_words.sh`
  - Usage: `count_words.py <DIR>`
- **count_book_pages.py** - Count pages in book PDF files
  - Replaces: `count_book_pages.sh`
  - Usage: `count_book_pages.py <DIR>`
- **count_lecture_pages.py** - Count pages in lecture PDF files
  - Replaces: `count_lecture_pages.sh`
  - Usage: `count_lecture_pages.py <DIR>`

### Slide Processing Scripts

- **slide_check.py** - Check and fix text in lecture slides
  - Replaces: `slide_check.sh`
  - Usage: `slide_check.py <DIR> <LESSON> [extra_opts...]`
- **slide_improve.py** - Improve lecture slides using LLM
  - Replaces: `slide_improve.sh`
  - Usage: `slide_improve.py <DIR> <LESSON> [extra_opts...]`
- **slide_reduce.py** - Reduce lecture slides using LLM
  - Replaces: `slide_reduce.sh`
  - Usage: `slide_reduce.py <DIR> <LESSON> [extra_opts...]`

### Document Generation Scripts

- **gen_slides.py** - Generate lecture slides PDF
  - Replaces: `gen_slides.sh`
  - Usage: `gen_slides.py <DIR> <LESSON> [extra_opts...]`
- **gen_quizzes.py** - Generate quizzes for a lecture using LLM
  - Replaces: `gen_quizzes.sh`
  - Usage: `gen_quizzes.py <DIR> <LESSON> [extra_opts...]`
- **gen_book_chapter.py** - Generate a book chapter from lecture source
  - Replaces: `gen_book_chapter.sh`
  - Usage: `gen_book_chapter.py <DIR> <LESSON>`
- **gen_lecture_script.py** - Generate lecture script from slides
  - Replaces: `gen_data605_script.sh`
  - Usage: `gen_lecture_script.py <DIR> <LESSON> [extra_opts...]`

## Examples

```bash
# Find a lecture file
./classes2/get_lecture_file.py data605 01.1

# Generate slides
./classes2/gen_slides.py data605 01.1

# Generate quizzes
./classes2/gen_quizzes.py msml610 02.3

# Count pages in book PDFs
./classes2/count_book_pages.py data605

# Check and fix slide text
./classes2/slide_check.py data605 01.1
```
