# Pitchbook Plugin

This plugin provides tools for extracting, parsing, and processing people data
from PitchBook MHTML exports and generating marketing outreach messages using
LLMs.

## Overall flow

- Search people on PitchBook with https://my.pitchbook.com/as-criteria/PERSON/PEOPLE/search
- Switch to max number of results per page
- Save "Webpage, Single File" to generate MHTML files, one per page
- Run `parse_mhtml.py` to parse the MHTML files (single file or batch process directory)
- Merge the tables in a single CSV
- Save the result into a Gsheet

### Single File Mode

```bash
> parse_mhtml.py --input_file '/Users/saggese/Desktop/People Results - People Screener_1.mhtml' --mode grids --table_index 0 --output_file output/out1.csv
> parse_mhtml.py --input_file '/Users/saggese/Desktop/People Results - People Screener_2.mhtml' --mode grids --table_index 0 --output_file output/out2.csv
> parse_mhtml.py --input_file '/Users/saggese/Desktop/People Results - People Screener_3.mhtml' --mode grids --table_index 0 --output_file output/out3.csv
> merge_mhtml_csv_files.py --input_file output/out1.csv --input_file output/out2.csv --input_file output/out3.csv --output_file output/merged.csv
> csvformat -T output/merged.csv | pbcopy
```

### Batch Directory Mode

```bash
> parse_mhtml.py --input_dir /Users/saggese/Desktop/mhtml_files --mode grids --output_dir output
> merge_mhtml_csv_files.py --input_dir output --output_file output/merged.csv
> csvformat -T output/merged.csv | pbcopy
```

## Structure of the Dir

- No subdirectories

## Description of Files

- `connect_chrome.py`
  - Automated pagination script that connects to Chrome via CDP, clicks through
    pages, and saves each as MHTML
- `merge_mhtml_csv_files.py`
  - Merge multiple MHTML-derived CSV files with separate name and data sections
    into standardized format
- `parse_mhtml.py`
  - Parse MHTML files and extract HTML content, traditional tables, or div-based
    data grids
- `process_pb.ipynb`
  - Jupyter notebook for interactive processing and analysis of PitchBook data
    with LLM integration
- `process_pitchbook.py`
  - Core library for loading, sanitizing, processing PitchBook CSV data and
    generating personalized outreach messages

## Description of Executables

### `connect_chrome.py`

#### What It Does

- Automates pagination through PitchBook search results by connecting to Chrome
  via CDP
- Sets page size to 50 results per page for optimal data extraction
- Clicks on page 1, then iterates through all pages by clicking Next button
- Saves each page as MHTML file with sequential numbering (out001.mhtml,
  out002.mhtml, etc.)
- Requires output directory to be specified via `--output_dir` parameter
- Uses randomized delays (4-6 seconds) between pages to mimic human behavior
- Displays progress bar showing percentage complete and pages processed
- Reports total number of pages processed

#### Examples

- **Start Chrome with remote debugging enabled (run first in separate
  terminal)**

  ```bash
  > /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
    --remote-debugging-port=9222 \
    --user-data-dir=/tmp/pw-chrome \
    --no-first-run
  ```

- **Navigate to PitchBook search results page manually in Chrome, then run
  automation**

  ```bash
  > python connect_chrome.py --output_dir ./output
  ```

- **Process all pages and save to custom directory**

  ```bash
  > python connect_chrome.py --output_dir /tmp/pitchbook_results
  ```

- **Process pages with debug logging enabled**
  ```bash
  > python connect_chrome.py --output_dir ./output -v DEBUG
  ```

### `parse_mhtml.py`

#### What It Does

- Extracts HTML content from MHTML files saved from web browsers
- Supports multiple extraction modes: DOM inspection, traditional HTML tables,
  and modern div-based grids
- Processes single files or batch processes entire directories of MHTML files
- Saves extracted data to CSV files with sequential numbering (out001.csv,
  out002.csv, etc.)

#### Examples

- **Print DOM structure for inspection**

  ```bash
  > python parse_mhtml.py --input_file file.mhtml
  ```

- **Extract all HTML tables from single file to directory**

  ```bash
  > python parse_mhtml.py --input_file file.mhtml --mode tables --output_dir ./output
  ```

- **Extract div-based grids from single file to directory**

  ```bash
  > python parse_mhtml.py --input_file file.mhtml --mode grids --output_dir ./output
  ```

- **Extract specific grid by index to single file**

  ```bash
  > python parse_mhtml.py --input_file file.mhtml --mode grids --table_index 0 --output_file ./output.csv
  ```

- **Batch process all MHTML files in directory (recommended)**
  ```bash
  > python parse_mhtml.py --input_dir ./mhtml_files --mode grids --output_dir ./output
  ```

### `merge_mhtml_csv_files.py`

#### What It Does

- Merges multiple CSV files with non-standard format (header + names section +
  CSV data section)
- Normalizes variable-length CSV rows into standardized 20-column PitchBook
  format
- Handles missing fields intelligently based on heuristics about field count and
  content
- Supports both individual file specification and directory-based batch
  processing

#### Examples

- **Merge single file to standardized format**

  ```bash
  > python merge_mhtml_csv_files.py --input_file output/out1.csv --output_file merged.csv
  ```

- **Merge multiple files into one CSV**

  ```bash
  > python merge_mhtml_csv_files.py --input_file output/out1.csv --input_file output/out2.csv --input_file output/out3.csv --output_file merged.csv
  ```

- **Merge all CSV files from a directory**

  ```bash
  > python merge_mhtml_csv_files.py --input_dir output --output_file merged.csv
  ```

- **Merge with debug logging**
  ```bash
  > python merge_mhtml_csv_files.py --input_file output/out1.csv --output_file merged.csv -v DEBUG
  ```

### `parse_mhtml.sh`

#### What It Does

- Batch processing script that orchestrates the complete PitchBook data
  extraction workflow
- Parses multiple MHTML files, extracts grids, merges results, and copies to
  clipboard
- Chains together parse_mhtml.py and merge_mhtml_csv_files.py commands

#### Examples

- **Run the complete extraction workflow**
  ```bash
  > bash parse_mhtml.sh
  ```

##

> /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
>   --remote-debugging-port=9222 \
>   --user-data-dir=/tmp/pw-chrome \
>   --no-first-run
