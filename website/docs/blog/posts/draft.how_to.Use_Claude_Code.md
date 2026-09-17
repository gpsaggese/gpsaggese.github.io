---
title: "How to Use Claude Code"
draft: true
authors:
  - gpsaggese
date: 2026-06-11
description: Complete guide on using Claude Code for coding, including best
  practices for prompting, tool usage, model capabilities, and getting started
  on dev servers
categories:
  - AI Tools
  - Software Engineering
  - Developer Tools
---

TL;DR: Learn how to use Claude Code effectively with best practices for
prompting, tool usage, model capabilities, and API integration.

<!-- more -->

## Overview

- Claude Code is a CLI tool that lets you work with Claude directly inside your
  project directory

- It acts as a coding assistant that can plan and execute tasks, generate code,
  run commands, test, debug, and iterate on your code end-to-end

- The following are some useful resources to get you started
  - [Quickstart](https://code.claude.com/docs/en/quickstart)
  - [Common workflows](https://code.claude.com/docs/en/common-workflows)
  - [Best practices for agentic coding](https://www.anthropic.com/engineering/claude-code-best-practices)
  - [Claude Code in Action](https://anthropic.skilljar.com/claude-code-in-action)

# Using Claude Code

Claude Code is a coding-focused interface for the Claude AI assistant. It helps
you write, understand, and debug code more efficiently.

## 1. Opening Claude Code

- Run `claude` in your Git client

- Initially you should give permissions to `claude` for any action

- It's not a good idea to use `claude --dangerously-skip-permissions` unless you
  know what you are doing

<!-- TODO(ai_gp): Run claude --help and comment on the most interesting options-->


## 2. Creating and Editing Code

You can use Claude Code in two ways:

1. Paste or write code directly
   - Paste existing code into the editor.
   - Ask Claude something like:
     - "Explain what this function does."
     - "Refactor this into smaller functions."
     - "Convert this from JavaScript to Python."

2. Start from a natural-language request
   - Type a prompt such as:
     - "Create a Python script that reads a CSV and prints summary stats."
     - "Generate a React component for a login form with validation."

3. I like to use a `instr.md` file with the instructions to `claude` so that
   - E.g., see the template `.claude/templates/ai.instruction_template.md`
   ```bash
   > cp .claude/templates/ai.instruction_template.md instr.md
   # Customize instr.md with detailed instructions
   claude> execute instr.md
   ```

Claude will:

- Propose code in the chat
- Often insert or update code directly in the editor

Save or copy the generated code into your own environment to run it.

## 3. Running and Testing Code

Depending on the integration, Claude Code may:

- Simulate running code (describe what would happen, spot logical issues)
- Provide test cases and examples
- Help you write unit tests

Ask things like:

- "Write unit tests for this function using pytest."
- "What edge cases might break this code?"
- "Can you step through this algorithm with an example input?"

Then paste results or error messages back into the chat so Claude can help
debug.

## 4. Debugging with Claude Code

When you get an error in your local environment:

1. Copy the error message and the relevant code snippet.
2. Paste them into Claude Code.
3. Ask:
   - "Here is the error I am seeing. Can you help me fix it?"
   - "Why does this raise a TypeError? Suggest a fix."

Claude will:

- Explain the error
- Suggest changes
- Sometimes give rewritten code blocks

Review changes carefully and test them yourself.

## 5. Working with Larger Projects

For multi-file projects:

- Use any available file browser or project view
- Open individual files in the editor.
- Ask questions like:
  - "Give me a high-level overview of this repository."
  - "Where is the main entry point for this app?"
  - "Find where the user authentication logic is implemented."

Claude can:

- Summarize architecture
- Trace how functions call each other
- Suggest refactors at the project level

## 6. Prompting Tips

To get better results:

1. Be specific
   - Instead of "Fix this", say:
     - "Reduce the time complexity."
     - "Make this function pure and side-effect free."
     - "Rewrite this using async/await."

2. Set constraints
   - "Use only standard library."
   - "Target Python 3.10."
   - "Avoid external dependencies."

3. Iterate
   - Start with a rough version.
   - Ask Claude to optimize, clean up, or document it:
     - "Add comments explaining each step."
     - "Improve variable names and structure."

## 7. Starting a Session on Dev Servers

### Start a Claude Session in Your Project Dir

```bash
> heanhs@dev2:~/src/csfy1$ claude
```

### Log in to Your Claude Account

- Choose Option 2 for pay-per-use API usage

  ![](/docs/ai_coding/ai.claude_code.how_to_guide_figs/image3.png)

- Open the provided link in your browser, choose the Causify.AI organization,
  authorize Claude Code to create a key, and paste the token back into the
  terminal

  ![](/docs/ai_coding/ai.claude_code.how_to_guide_figs/image2.png)

### Using Claude Code

- Once everything is set up, you can begin using Claude Code in your project

- To exit, just type `/exit`

  ![](/docs/ai_coding/ai.claude_code.how_to_guide_figs/image1.png)

## Developer Guide

### Prompting Best Practices

- Use clear and explicit instructions
  - Do not rely on the model to infer from vague prompts
  - Think of Claude as a brilliant but new employee
  - Be specific about output and constraints
  - Provide instructions as sequential steps

- Use examples
  - Aka few-shot prompting
  - Create examples that are relevant and diverse
  - Wrap examples in `<examples>` tags
  - You can ask Claude to evaluate examples and provide additional ones

- Use XML tags
  - XML tags help Claude not get confused with instructions, context, input
  - E.g.,
    ```
    <documents>
    <document index=1>
    ...
    ```

- Give Claude a role
  - Use the system prompt to focus Claude's behavior and tone
    ```
    You are a helpful coding assistant specializing in Python
    ```

- When there are large docs
  - Put longform data at the top
  - Put the query at the end
  - Structure document with XML tags
  - Ask Claude to quote relevant parts of the documents first before carrying out
    its tasks

- Control the format of responses
  - Tell Claude what to do and what not to do
  - Use XML format indicators
  - Match the prompt style to the desired output

### Tool Usage

- Leverage Claude's tools for file operations, command execution, and web search

### Optimize Parallel Tool Calling

- Run multiple speculative searches during research
- Read several files at once to build context faster

## Model Capabilities

### Extended Thinking

### Adaptive Thinking

### Effort

### Fast Mode

### Structured Outputs

### Citations

### Streaming Messages

### Batch Processing

### PDF Support

### Search Results

### Multilingual Support

### Embeddings

### Vision

## Planning Mode and Other Modes

- Planning mode

> claude --permission-mode plan

Create a plan without coding

- Cheap mode

> claude --model haiku

> claude --output-format text -p "Fix update_md.py -i docs/datapull/all.add_new_data_source.how_to_guide.md -a summarize -a apply_style" --dangerously-skip-permissions

# Workflows

This document consolidates all AI development workflows, coding conventions, and
tools used in the Causify/helpers ecosystem.

## Overview

This documentation provides comprehensive guidelines for:

- Writing Python code following Causify conventions
- Creating unit tests
- Formatting notes and documentation
- Using AI-powered review and transformation tools
- Integrating with ChatGPT API
- Understanding the helpers repository architecture

## Interesting Files

- `.claude/templates/code_template.py` shows our coding style
- `.claude/templates/unit_test_template.py` shows how our unit tests look like

- `CLAUDE.md`: Project architecture overview and development conventions for
  Claude Code working with the `helpers` repository
- `.claude/templates/ai.instruction_template.md`: Workflow template for creating
  Python scripts with tests, documentation, planning steps, and AI todos
- `.claude/skills/coding.rules.md`: Python coding standards including
  hdbg assertions, hsystem usage, logging patterns, and script templates
- `docs/ai_prompts/testing.format.md`: Unit testing conventions
  including test structure, naming patterns, and golden file testing

- `docs/ai_prompts/blog.format.md`: Markdown formatting guidelines for
  writing blog posts with proper structure, code blocks, and metadata

- `docs/ai_coding/ai.md_instructions.md`: Style guide for writing structured
  bullet-point notes optimized for clarity and AI/human readability

## AI Development Workflow Template

When creating a Python script:

1. **Write a Python script** following the instructions in
   `.claude/skills/coding.rules.md`

2. **Generate unit tests** for the code following the instructions in
   `docs/ai_prompts/testing.format.md`

3. **Generate a short description** of how to use the script in a file close to
   the script with extension `.md`
   - Explain the goal of the script
   - Report some examples of how to use the tool
   - Describe the architecture

## AI Review and Transform Tools

### Operations Overview

There are several operations we want to perform using LLMs:

- Apply a transformation to a chunk of text (e.g., create a unit test)
- Create comments and lints in the form of a `cfile` (e.g., lint or AI review
  based on certain criteria)
- Apply modifications from a `cfile` to a set of files (e.g., from linter and AI
  review)
- Add TODOs from a `cfile` to Python or markdown files
- Apply a set of transformations to an entire Python file (e.g.,
  styling/formatting code)
- Rewrite an entire markdown to fix English mistakes without changing its
  structure

**Important:** Always commit your code before applying automatic transforms, in
the same way that we run the `linter` on a clean tree. This way, modifying a
file is a separate commit and it's easy to review.

### Use Templates

We use templates for code and documentation to show and describe how a document
or code should look like:

- `all.how_to_guide_template_doc.md` shows how a Diataxis how-to guide should be
  structured and look like

The same templates have multiple applications:

- **For Humans:**
  - Understand how to write documentation and code
  - As boilerplate (e.g., "copy the template and customize it to achieve a
    certain goal")
- **For LLMs:**
  - As reference style to apply transforms
  - To report violations of coding styles
  - As boilerplate (e.g., "explain this piece of code using this template")

### Available Tools

- `llm_transform.py`
- `transform_text.py`: Some transformations don't need LLMs and are implemented
  as code.
- `ai_review.py`: The rules for AI are saved in
  `./docs/code_guidelines/all.coding_style_guidelines.reference.md`. This file
  has a special structure:
  - First level represents the target language (e.g., `General`, `Python`)
  - Second level represents a rule topic (e.g., `Imports`, `Functions`)
  - Third level represents instructions for an LLM vs Linter
- `inject_todos.py`: Injects TODOs from a `cfile` into source files.
- `apply_todos.py`: Automatically applies TODOs from a `cfile` using an LLM.

## Memory

<!-- From https://code.claude.com/docs/en/memory -->

To make Claude Code stop remembering information between sessions:

- Run /memory and turn Auto Memory off, or set:
  ```
  {
    "autoMemoryEnabled": false
  }
  ```

- Disable it for a session:
  ```
  CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 claude
  ```

- To remove existing memories, delete files under:
  ```
  ~/.claude/projects/<project>/memory/
  ```

- To disable persistent instructions from CLAUDE.md, use:
  ```
  > claude --no-memory
  ```

- Use `/memory` at any time to check whether memory is enabled for the current project

## Print Thinking / Verbosity

- Global configuration (settings file)
  - Open the settings file used by Claude Code.
    `/Users/saggese/.claude/settings.json` or the project‑local
    `.claude/settings.json`
  - Add/modify the relevant flags:
    ```
    {
      "displayThinking": false,      // hide all thinking blocks
      "verbosity": "minimal",       // suppress most internal messages
      "showToolUse": false          // optional – hide tool‑use logs
    }
    ```
  - Save and restart the CLI / reload the workspace.
  - No thinking lines appear in the terminal or UI.

- Run Claude Code with a flag that disables thinking for that invocation only:
   ```
   claude-code --no-thinking          # or: --verbosity minimal
   ```
- You can see the exact flag name with claude-code --help.

## API Reference

## MCP

## Resources

## Release Notes
