---
title: "Ripgrep in 30 mins"
authors:
  - gpsaggese
date: 2026-02-10
description:
categories:
  - Python
---

TL;DR `ripgrep` is a blazingly fast search tool that combines the usability and
speed.

<!-- more -->

## Introduction
- In this tutorial, we'll explore why `ripgrep` is my search tool and how to use
  it effectively

## Why Ripgrep?
- `ripgrep` stands out for several compelling reasons:
  - **Speed**: Written in Rust with highly optimized algorithms, `ripgrep` is
    often 5-10x faster than alternatives
  - **Smart defaults**: Automatically respects `.gitignore`, skips binary files,
    and excludes hidden directories
  - **Parallel execution**: Searches multiple files simultaneously using all
    available CPU cores
  - **Memory efficient**: Uses memory-mapped files and streaming search to
    handle large files gracefully
  - **Cross-platform**: Works seamlessly on Linux, macOS, and Windows
  - **Rich features**: Supports regex, multiline search, file type filtering,
    and context display

- Here's a speed comparison on a typical codebase:

  | Tool      | Time (seconds) | Notes                         |
  | :-------- | :------------- | :---------------------------- |
  | `grep -r` | 4.2            | Basic recursive search        |
  | `ack`     | 3.5            | Perl-based search             |
  | `rg`      | 0.4            | Ripgrep with default settings |

## Installation
- Installing `ripgrep` is straightforward across all platforms

- On macOS using Homebrew:
  ```bash
  > brew install ripgrep
  ```

- On Ubuntu/Debian:
  ```bash
  > apt-get install ripgrep
  ```

- Verify the installation:
  ```bash
  > rg --version
  ```

## Basic Usage
- The simplest `ripgrep` command searches for a pattern in the current
  directory:
  ```bash
  > rg "pattern"
  ```
  - This recursively searches all files, respecting `.gitignore` and skipping
    binary files automatically

- Search with case-insensitive matching:
  ```bash
  > rg -i "pattern"
  ```

- Search in a specific directory:
  ```bash
  > rg "pattern" /path/to/directory
  ```

- Search only in files matching a glob pattern:
  ```bash
  > rg "pattern" -g "*.py"
  ```

- Show line numbers (enabled by default when output is to terminal):
  ```bash
  > rg -n "pattern"
  ```

## Understanding the Output
- `ripgrep` output is designed for clarity:
  ```bash
  > rg "def calculate"
  ```
  Output:
  ```text
  src/utils.py
  23:def calculate_total(items):
  45:def calculate_average(values):

  src/models.py
  12:    def calculate_score(self):
  ```
  The format is:
  - Filename (colored for visibility)
  - Line number followed by colon
  - The matching line with pattern highlighted

## Advanced Features

### File Type Filtering
- `ripgrep` understands common file types:
  ```bash
  > rg "pattern" -t py
  ```
  This searches only Python files

- List all available types:
  ```bash
  > rg --type-list
  ```

- Exclude specific file types:
  ```bash
  > rg "pattern" -T js
  ```

### Context Lines
- Show lines before and after matches:
  ```bash
  > rg "pattern" -C 3
  ```

- Show only lines before:
  ```bash
  > rg "pattern" -B 2
  ```

- Show only lines after:
  ```bash
  > rg "pattern" -A 2
  ```

### Search Only Filenames
- List files containing matches without showing the matches:
  ```bash
  > rg "pattern" -l
  ```

- List files not containing matches:
  ```bash
  > rg "pattern" --files-without-match
  ```

### Multiline Search
- Search across multiple lines:
  ```bash
  > rg -U "pattern1.*pattern2"
  ```
  The `-U` flag enables multiline mode where `.` matches newlines

### Word Boundaries
- Match whole words only:
  ```bash
  > rg -w "word"
  ```
  This prevents matching "word" inside "password" or "wording"

## Practical Examples

### Find All TODO Comments
- Find all TODO comments:
  ```bash
  > rg "TODO|FIXME|XXX" -t py
  ```

### Search for Function Definitions
- Search for function definitions:
  ```bash
  > rg "^def \w+\(" -t py
  ```

### Find All Imports of a Module
- Find all imports of a module:
  ```bash
  > rg "^import pandas|^from pandas" -t py
  ```

### Search in Git History
- Combine with `git` to search across all branches:
  ```bash
  > git grep "pattern" $(git rev-list --all)
  ```

- But for current working tree, `ripgrep` is faster:
  ```bash
  > rg "pattern"
  ```

### Find Large Files with Pattern
- Find large files with pattern:
  ```bash
  > rg "pattern" --stats
  ```
  The `--stats` flag shows per-file statistics including file sizes searched

## Configuration and Customization
- Create a configuration file at `~/.ripgreprc`:
  ```bash
  # Always show line numbers
  --line-number

  # Always show context
  --context=2

  # Custom colors
  --colors=match:fg:red
  --colors=match:style:bold
  ```

- Enable the config file:
  ```bash
  > export RIPGREP_CONFIG_PATH=~/.ripgreprc
  ```

## Tips and Tricks

### Ignore Additional Patterns
- Create a `.rgignore` file in your project root:
  ```text
  # Ignore build artifacts
  build/
  dist/
  *.pyc

  # Ignore large data files
  data/
  *.csv
  ```

### Search Hidden Files
- By default, `ripgrep` skips hidden files
- Include them:
  ```bash
  > rg "pattern" --hidden
  ```

### Search All Files Including Ignored
- Override `.gitignore` and search everything:
  ```bash
  > rg "pattern" --no-ignore
  ```

### Replace Text Across Files
- While `ripgrep` doesn't replace text, combine it with `sed`:
  ```bash
  > rg "old_pattern" -l | xargs sed -i 's/old_pattern/new_pattern/g'
  ```

### Count Matches
- Count occurrences of a pattern:
  ```bash
  > rg "pattern" -c
  ```

- Show total count across all files:
  ```bash
  > rg "pattern" -c | awk -F: '{sum+=$2} END {print sum}'
  ```

## Common Gotchas

### Pattern Escaping
- Special regex characters need escaping:
  ```bash
  > rg "function\(\)"  # Match "function()"
  > rg '\$variable'     # Match "$variable"
  ```

### Binary Files
- `ripgrep` skips binary files by default
- To search them:
  ```bash
  > rg "pattern" -a
  ```

### Symbolic Links
- By default, `ripgrep` doesn't follow symbolic links
- To follow them:
  ```bash
  > rg "pattern" -L
  ```

## Integration with Development Tools

### Vim Integration
- Add to `.vimrc`:
  ```vim
  set grepprg=rg\ --vimgrep
  set grepformat=%f:%l:%c:%m
  ```

- Use with:
  ```vim
  :grep pattern
  :copen
  ```

### VS Code Integration
- VS Code uses `ripgrep` by default for file searching
- Configure search exclusions in `settings.json`:
  ```json
  {
    "search.exclude": {
      "**/node_modules": true,
      "**/dist": true
    }
  }
  ```

### Command Line Aliases
- Add useful aliases to your shell configuration:
  ```bash
  # Search Python files
  alias rgpy='rg -t py'

  # Search with context
  alias rgc='rg -C 3'

  # Search and open in vim
  rgv() { vim $(rg -l "$1"); }
  ```

## Comparison with Other Tools
Ripgrep vs grep:

| Feature                    | ripgrep | grep    |
| :------------------------- | :------ | :------ |
| Speed                      | +++     | +       |
| Respects .gitignore        | Yes     | No      |
| Parallel search            | Yes     | Limited |
| Automatic binary exclusion | Yes     | No      |
| Regex flavor               | Rust    | POSIX   |

Ripgrep vs The Silver Searcher:

| Feature            | ripgrep | ag      |
| :----------------- | :------ | :------ |
| Speed              | +++     | ++      |
| Memory usage       | Lower   | Higher  |
| Unicode support    | Better  | Good    |
| Active development | Yes     | Limited |
