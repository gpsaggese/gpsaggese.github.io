# Tutorial Checklist

## Goal

Complete this onboarding to understand the tutorial codebase structure, code
style, and quality standards. 

## Quality Principles

Before starting, understand these non-negotiable principles:

- **Avoid AI slop at any cost**
  - Humans can instantly detect low-quality AI-slop
    - Once detected, everything else sounds equally horrible
  - A tutorial cannot be "throw a prompt into Claude Code and copy-paste the
    result"
    - It's ok to use AI, but the output needs to be higher than using
      only AI or using only human
  - Your goal is to read information and synthesize that for somebody else
    to reduce the time it takes for them to learn by 10x
    - If it takes 5 mins to generate and 60 mins to generate

- **Use a professional tone**
  - Write as if explaining to a peer, not a student or a buddy

- **Focus only on examples without repeating content**
  - Everything should be stated only once and in the right place
  - Eliminate redundancy between sections

- **Maintain high quality standards**
  - Every sentence should serve a purpose
  - Each example should illustrate a concept clearly

## Tutorial Checklist

- Create an issue called "Clean up tutorial XYZ" with the following action items

### 1. Understand the Core Documentation

- [ ] Read `class_project/project_template/docker_scripts.README.md`
  - Is there anything unclear, incorrect, or that can be improved?

### 2. Study Code Examples

- [ ] Carefully examine all code in `class_project/project_template`
  - Make sure you understand every file and module perfectly
  - Note the structure, naming conventions, and patterns used

### 3. Learn the Claude Skills

- [ ] Become familiar with the Claude Skills in `helpers_root/.claude/skills`
  - Skills are living documentation of how we maintain and create code

- [ ] Explore the available skills using these commands:
  ```bash
  > mdm skill describe coding
  > mdm skill describe notebooks
  > mdm skill describe testing
  > mdm skill describe tool_X_in_60_mins
  ```

- [ ] Review key skills relevant to tutorials:
  - `coding.format`: Python code conventions
  - `testing.format`: Unit test patterns
  - `notebook.format`: Jupyter notebook structure
  - `markdown.format`: Documentation style
  - E.g.,
    ```bash
    > mdm skill edit coding.rules
    ```

- [ ] Is there anything unclear or that could be improved in the skills?

### 4. Study Reference Tutorials

- [ ] Read 2-3 tutorials
  - Tutorials that are closer to the standards are:
    - Autogen
  - Internalize its structure, style, and approach
  - Note patterns, structure, and what makes them effective
  - Identify what works well and what could be improved

### 5. Verify Code Quality

- [ ] Review the test structure, specifically
  `class_project/project_template/test/test_docker_all.py`
  - Understand what tests each tutorial should include
  - Ensure your tutorial will have comprehensive test coverage
