# process_lessons.py

## Goal

This script generates PDF slides and/or reading scripts for lecture materials, and
can process slides using LLM transformations

## Usage Examples

- Generate PDF slides for a specific lecture
  ```bash
  > process_lessons.py --lectures 01.1 --class data605 --action pdf
  ```

- Generate reading scripts for multiple lectures
  ```bash
  > process_lessons.py --lectures 01*:02* --class data605 --action script
  ```

- Generate both PDFs and scripts
  ```bash
  > process_lessons.py --lectures 01* --class msml610 --action pdf --action script
  ```

- Generate using default actions (PDF only)
  ```bash
  > process_lessons.py --lectures 01* --class msml610
  ```

- Generate all available actions
  ```bash
  > process_lessons.py --lectures 01* --class data605 --all
  ```

- Skip specific actions
  ```bash
  > process_lessons.py --lectures 01* --class data605 --skip_action script
  ```

- Reduce slides using LLM transformation (modifies in place)
  ```bash
  > process_lessons.py --lectures 01.1 --class data605 --action slide_reduce
  ```

- Check slides using LLM transformation (creates separate report)
  ```bash
  > process_lessons.py --lectures 01.1 --class data605 --action slide_check
  ```

- Generate specific slides from a lecture
  ```bash
  > process_lessons.py --lectures 01.1 --limit 1:3 --class data605 --action pdf
  ```

- Process all lectures in a class
  ```bash
  > process_lessons.py --lectures "0*" --class data605 --action pdf --action script
  ```

## Command Line Arguments

- `--lectures`: Lecture pattern(s) to process. Can be:
  - Single lecture: `01.1`
  - Wildcard pattern: `01*`
  - Multiple patterns: `01*:02*:03.1` (separated by colons)
- `--class`: Class directory name (`data605` or `msml610`)
- `--action`: Actions to execute. Can be specified multiple times:
  - `pdf`: Generate PDF slides
  - `script`: Generate reading scripts
  - `slide_reduce`: Reduce slides using LLM transformation (modifies source in
    place)
  - `slide_check`: Check slides using LLM transformation (creates separate
    report file)
  - Default: `pdf` (if no action specified)
- `--skip_action`: Actions to skip (mutually exclusive with `--action`)
- `--all`: Execute all available actions (mutually exclusive with `--action`)
- `--limit`: Slide range to process (e.g., `1:3`). Only valid when a single
  lecture file matches the pattern. Only applies to `pdf` action.
- `--dry_run`: Print commands without executing them
- `--log_level`: Logging verbosity (optional)

## Architecture

### Data Flow

```
- Command Line Arguments
- Parse patterns, actions, and options
- Select actions to execute (based on --action, --skip_action, --all, or defaults)
- Find matching lecture files
- For each file:
  - Process PDF action (if selected)
    - notes_to_pdf.py → lectures/*.pdf
  - Process script action (if selected)
    - generate_slide_script.py → lectures_script/*.script.txt
    - perl (remove prefixes) → lectures_script/*.script.txt
    - lint_txt.py → lectures_script/*.script.txt
  - Process slide_reduce action (if selected)
    - process_slides.py --action slide_reduce --use_llm_transform → modifies source in place
  - Process slide_check action (if selected)
    - process_slides.py --action slide_check --use_llm_transform → creates *.slide_check.txt
```

### Directory Structure

```
{class_dir}/
  lectures_source/     # Input: Lesson*.txt files
  lectures/            # Output: Generated PDF files
  lectures_script/     # Output: Generated script files
```
