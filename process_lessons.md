# process_lessons.py

## Goal

This script generates PDF slides and/or reading scripts for lecture materials

## Usage Examples

- Generate PDF slides for a specific lecture
  ```bash
  > process_lessons.py --lectures 01.1 --class data605 --target pdf
  ```

- Generate reading scripts for multiple lectures
  ```bash
  > process_lessons.py --lectures 01*:02* --class data605 --target script
  ```

- Generate both PDFs and scripts
  ```bash
  > process_lessons.py --lectures 01* --class msml610 --target pdf,script
  ```

- Generate specific slides from a lecture
  ```bash
  > process_lessons.py --lectures 01.1 --limit 1:3 --class data605 --target pdf
  ```

- Process all lectures in a class
  ```bash
  > process_lessons.py --lectures "0*" --class data605 --target pdf,script
  ```

## Command Line Arguments

- `--lectures`: Lecture pattern(s) to process. Can be:
  - Single lecture: `01.1`
  - Wildcard pattern: `01*`
  - Multiple patterns: `01*:02*:03.1` (separated by colons)
- `--class`: Class directory name (`data605` or `msml610`)
- `--target`: Output types to generate:
  - `pdf`: Generate PDF slides
  - `script`: Generate reading scripts
  - `pdf,script`: Generate both (comma-separated)
- `--limit`: Slide range to process (e.g., `1:3`). Only valid when a single
  lecture file matches the pattern.
- `--log_level`: Logging verbosity (optional)

## Architecture

### Data Flow

```
- Command Line Arguments
- Parse patterns, targets, and options
- Find matching lecture files
- For each file:
  - Process PDF target (if specified)
    - notes_to_pdf.py → lectures/*.pdf
  - Process script target (if specified)
    - generate_slide_script.py → lectures_script/*.script.txt
    - perl (remove prefixes) → lectures_script/*.script.txt
    - lint_txt.py → lectures_script/*.script.txt
```

### Directory Structure

```
{class_dir}/
  lectures_source/     # Input: Lesson*.txt files
  lectures/            # Output: Generated PDF files
  lectures_script/     # Output: Generated script files
```
