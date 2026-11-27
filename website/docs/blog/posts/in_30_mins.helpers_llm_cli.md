---
title: "LLM CLI in 30 mins"
draft: false
authors:
    - gpsaggese
date: 2026-05-08
description: Transform text files with LLMs from the command line, integrate AI into your editing workflow
categories:
    - Developer Tools
    - LLM
---

TL;DR: `llm_cli` is a lightweight CLI tool that applies LLM transformations to
text and code files. Use it to refactor code, improve docs, apply linting
rules, or run custom prompts on file chunks, all from shell without leaving
editor.

<!-- more -->

## Introduction

- Text transformation is a common dev task: refactoring code, improving docs,
  fixing style issues, applying rules to slide decks, generating summaries
- `llm_cli` automates this: pipe text through LLM directly from shell

- `llm_cli` solves several problems:
    - Apply LLM transformations without leaving terminal
    - Use system prompts, rules, or skills to guide LLM behavior
    - Process specific file chunks without touching the rest
    - Chain transformations in shell pipelines
    - Auto-lint output with `Prettier`

- Unlike generic LLM wrappers, `llm_cli` integrates with Claude Code skills and
  rules: apply same transformations from command line as from IDE

## When to Use It

- Use `llm_cli` when you need to:
    - Refactor multiple files with same rules
    - Apply a skill (e.g., "fix documentation" or "improve code style") to a file
    - Extract a file section, transform with LLM, and reassemble original
    - Integrate LLM transformations into shell scripts or `Makefiles`
    - Test a prompt or rule before applying via Claude Code

- Similar tools are 
    - `llm` CLI (Simon Willison)
    - ChatGPT
    - one-off Python scripts
  but none combine LLM + file handling + rule integration as cleanly

## Prerequisites

- Python 3.11 or later
- Access to LLM API (OpenAI, Anthropic, or OpenRouter)
- `llm` Python library installed (auto-included in helpers)
- `.claude/skills/` dir with skill definitions (optional, but useful)

## Installation and Setup

- `llm_cli` comes with helpers library. Verify it's available:

    ```bash
    > llm_cli.py --help
    usage: llm_cli.py [-h] [--input INPUT] [--input_text INPUT_TEXT] ...
    ```

- Configure LLM API key:

    ```bash
    > export OPENAI_API_KEY="your-key-here"
    ```

- Or for Claude via Anthropic:

    ```bash
    > export ANTHROPIC_API_KEY="your-key-here"
    ```

- Check that LLM is working:

    ```bash
    > llm_cli.py --input_text "Say hello" --output -
    Hello! How can I help you today?
    ```

## Core Concepts

- `llm_cli` operates in these stages:
    1. **Read input**: From file, stdin, or directly as text
    2. **Extract chunk (optional)**: Use `--select` to process only part of file
    3. **Choose a prompt**: Inline prompt, file, rule from `.claude/skills/`, or full skill
    4. **Transform**: Send text through LLM
    5. **Optionally lint**: Auto-format output (e.g., with `Prettier`)
    6. **Write output**: To file, stdout, or back to input file

- Each stage optional depending on use case. Simplest usage: input and output

### Key Options

- `--input FILE` / `-i FILE`: Input file path. Use `-` for stdin
- `--input_text TEXT`: Input text from command line
- `--output FILE` / `-o FILE`: Output file path. Use `-` for stdout
- `--system_prompt TEXT` / `-p TEXT`: Prompt text to guide the LLM
- `--system_prompt_file FILE` / `-pf FILE`: Read prompt from file
- `--rule SPEC`: Extract a rule from `.claude/skills/topic.rules.md`
- `--skill NAME`: Use a skill's full `SKILL.md` file as prompt
- `--select SPEC`: Process only lines matching a selection spec
- `--lint`: Auto-format output with `Prettier`
- `--model MODEL`: Which LLM to use (default: gpt-4)
- `--modify_in_place` / `-m`: Edit file in place instead of creating new one

## Hands-On Examples

### Example 1: Basic Text Transformation

- Start with simplest case: transform input text and print result

- Create a sample file:

```bash
> cat > input.txt << 'EOF'
The quick brown fox jumps over the lazy dog.
It was a dark and stormy night.
The hero entered the room with caution.
EOF
```

- Transform with a simple prompt:

```bash
> llm_cli.py -i input.txt -o - --system_prompt "Rewrite this in one sentence"
The quick brown fox leaped over a lazy dog during a dark, stormy night as the cautious hero entered the room.
```

- `-o -` prints to stdout instead of writing to file

### Example 2: Edit a File in Place

- Process a file and save result back to itself:

```bash
> llm_cli.py -i input.txt --system_prompt "Make this more formal" -m
```

- Check result:

```bash
> cat input.txt
A swift, auburn canine traversed an obstacle formed by a sluggish animal.
It was an exceptionally dark evening accompanied by severe meteorological conditions.
The protagonist proceeded cautiously into the chamber.
```

- `-m` flag modifies file in place without separate output file

### Example 3: Apply a Skill From Claude Code

- If you have a skill in `.claude/skills/`, apply it directly:

```bash
> llm_cli.py -i code.py --skill coding.fix_docstring -m
```

- This applies entire `coding.fix_docstring` skill to your file
- Skills more powerful than inline prompts: contain detailed instructions and examples

### Example 4: Transform Only Part of a File

- Extract a chunk, transform it, reassemble. Useful when only specific lines need changes

- Create a sample file with markers:

```bash
> cat > slides.txt << 'EOF'
## Slide 1: Introduction

This is a basic intro slide.
It needs better content.

## Slide 2: Main Topic

The main point is important.
But unclear.

## Slide 3: Conclusion

Wrap up the presentation.
Make it memorable.
EOF
```

- Transform only Slide 2 using line numbers:

```bash
> llm_cli.py -i slides.txt --select 6:8 --system_prompt "Improve clarity" -m
```

- `--select 6:8` processes only lines 6-8, leaving rest untouched

### Example 5: Apply a Rule with Auto-Linting

- Rules are snippets from a skill file. Extract one and apply with auto-formatting:

```bash
> llm_cli.py -i README.md --rule '.claude/skills/markdown.rules.md:42:# Fix Bold Labels' --lint -m
```

- Rule specified as `file:line_number:rule_name`. `--lint` runs `Prettier` on output for consistent formatting

## Tips and Gotchas

### Tip 1: Use Pipes for Chaining

- `llm_cli` integrates with Unix pipes: transform output from one tool into input for another

```bash
> cat raw_notes.txt | llm_cli.py -i - -o - --system_prompt "Summarize in 3 bullet points"
```

### Tip 2: Estimate Output Size for Large Files

- By default, `llm_cli` shows a progress bar but doesn't know output size
- Help it:

```bash
> llm_cli.py -i large_file.py --system_prompt "Add type hints" --expected_num_chars 50000
```

- Or let it auto-estimate:

```bash
> llm_cli.py -i large_file.py --system_prompt "Add type hints" --progress_bar
```

### Tip 3: Use Dry-Run to Preview

- Before modifying files, dry-run to see what would happen:

```bash
> llm_cli.py -i important_file.py --system_prompt "Refactor" --dry_run
```

- Shows LLM params without calling API or modifying files

### Gotcha 1: Stdin Requires Output Specification

- When reading from stdin with `-i -`, must specify an output:

```bash
> echo "text" | llm_cli.py -i - -o output.txt  # OK
> echo "text" | llm_cli.py -i -                 # ERROR
```

- Use `-o -` to print to stdout:

```bash
> echo "text" | llm_cli.py -i - -o -
```

### Gotcha 2: Only One Prompt Option at a Time

- Use `-p` (inline), `-pf` (from file), `--rule` (from rules), or `--skill` (full skill), not multiple:

```bash
> llm_cli.py -i file.txt -p "Fix it" --rule '.claude/skills/my.rules.md:10:Rule'  # ERROR
```

### Gotcha 3: Linting Only Works with Markdown

- `--lint` flag currently formats output as Markdown with `Prettier`. If working with code files, linting won't apply:

```bash
> llm_cli.py -i code.py --system_prompt "Add comments" --lint  # Linting won't affect Python
```

## Next Steps

- Read full docs in `dev_scripts_helpers/llms/README.md`
- Explore existing skills in `.claude/skills/` to see available transformations
- Create custom rule for repetitive tasks (e.g., "Fix grammar in slide decks")
- Integrate `llm_cli` into `Makefile` or shell script for batch processing
- Experiment with different models using `--model openrouter/anthropic/claude-opus-4.6`

## Advanced: Combining with Other Tools

- Use `llm_cli` alongside other helpers:
    - Refactor code and run tests:

        ```bash
        > llm_cli.py -i module.py --system_prompt "Refactor for readability" -m && python -m pytest module_test.py
        ```

    - Fix a specific function in a file:

        ```bash
        > llm_cli.py -i file.py --select "def my_func" --skill coding.fix_docstring -m
        ```

    - Process multiple files in a loop:

        ```bash
        > for file in *.md; do llm_cli.py -i "$file" --skill markdown.fix_bullet_points -m; done
        ```

- `llm_cli` bridges terminal and LLM capabilities. Use it whenever you write
  manual prompts to fix or improve text: that's a sign the transformation should
  be automated
