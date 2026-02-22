---
title: "Learn X in 60 Minutes"
authors:
  - gpsaggese
date: 2026-02-07
categories:
  - Software Engineering
---

TL;DR: This post describes how the "Learn X in 60 Minutes" series of tutorial is
structured.

<!-- more -->

- **Goal**: give everything needed for someone to become familiar with a big
  data / AI / LLM / data science technology in 60 minutes

## What Are the Goals for Each Tutorial

Each tutorial aims to provide:

- **Conceptual understanding**: Clear explanations of what the technology is and
  when to use it
- **Practical application**: A complete example showing real-world usage and
  working code examples that run immediately
- **Reproducibility**: Guaranteed to work through automated testing and all
  dependencies and setup handled via Docker

## Technologies Covered

Examples of technologies included in this tutorial series:

- **Big Data**: Spark, Dask, Hadoop
- **Databases**: PostgreSQL, MongoDB, Redis
- **Workflow Orchestration**: Airflow, Prefect
- **Probabilistic Programming**: PyMC, Stan
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Deep Learning**: PyTorch, TensorFlow, JAX
- **LLMs & AI**: LangChain, OpenAI API, Anthropic API
- **DevOps**: Docker, Docker Compose, Git

## Tutorial Structure

Each 60-minute tutorial follows this time breakdown for a reader (and thus a
writer):

1. **Setup (5 min)**: Clone repo, start Docker container, verify environment
2. **Introduction (10 min)**: Read overview markdown, understand use cases
3. **API Exploration (20 min)**: Work through `XYZ.API.ipynb` notebook
4. **Complete Example (25 min)**: Work through `XYZ.example.ipynb` notebook

## Invariants

All tutorials maintain these standards:

- **Code repository**: All code is on GitHub in a format common to all tutorials
- **Dependency management**: All packages are handled through Docker in a
  standard approach (e.g., `docker_build`, `docker_bash`)
- **Consistent structure**: The format of the tutorial follows the same
  structure across all topics
- **Centralized location**: All tutorial material is in a directory in the
  [`tutorials`](https://github.com/gpsaggese/umd_classes/tree/master/tutorials)
  repo and in the [`//helpers`](https://github.com/causify-ai/helpers) sub-repo

## Content

- Each tutorial about `XYZ` contains:
  1. A Docker container with everything needed to build and run using the
     Causify `dev-system` format or a simple set of bash scripts
  2. A markdown `XYZ.API.md` about the technology described:
     - The native API
     - The software layer written by us on top of the native API
  3. A Jupyter notebook `XYZ.API.ipynb` with an example of using the native /
     our APIs
  4. A markdown `XYZ.example.md` with a full example of an application using the
     API
  5. A Jupyter notebook `XYZ.example.ipynb` with a full example
  6. A file `XYZ_utils.py` with Python utility functions

### README.md

- Each project contains a `README.md` summarizing its status and a changelog

### Docker Container

- The Docker container should:
  - Contain everything so that one is ready to run tutorials and develop with
    that technology
  - Often installing and getting a package to work (e.g., PyMC) takes a long
    time

- The Docker structure and approach should follow the template
  `class_project/project_template/`
  - Feel free to copy that directory as a base for 

### Jupyter Notebooks

- Each Jupyter notebook should:
  - Run end-to-end after a restart
    - It's super frustrating when a tutorial doesn't work because the version of
      the library is not compatible with the code anymore
    - This is enforced by the unit test through `pytest`, in this way we are
      guaranteed that it works
  - Be self-contained and linear
    - Each example is explained thoroughly without having to jump from tutorial
      to tutorial
    - Each cell and its output is commented and explained
  - Take less than few minutes to execute end-to-end

### Markdown

- Markdown documents should cover information about:
  - What the technology / Python package to library is
  - What problem it solves
  - What are the alternatives, both open source and commercial with comments
    about advantages and disadvantages
  - A description of the native API of the technology
  - A description of the Docker container
  - Visual aids with `mermaid`, `graphviz`, `tikz` (e.g., flow diagrams, data
    transformation steps, and plots) to enhance understanding of how the library
    and the example works
  - References to books and in-depth tutorial that we have run and we think are
    awesome
  - All sources should be referred and acknowledged

- This is the same approach we use in
  - [DATA605](https://github.com/gpsaggese/umd_classes/blob/master/data605/tutorials)
    - E.g.,
      - Git
      - Docker
      - Docker compose
      - Postgres
      - MongoDB
      - Airflow
      - Dask
      - Spark
  - [MSML610](https://github.com/gpsaggese/umd_classes/blob/master/msml610/tutorials/notebooks)
    even if not all these tutorials don't use the Causify dev system, but some
    simpler bash scripts.

# `project.API.ipynb`

- It should contain a walkthrough of the library / package with an example
  - In practice it merges the XYZ.API.md and the XYZ.API.ipynb into a single
    notebook
  - There should be a little markdown to explain the library / theory and then an
    example to show how things work
  - The API ipynb should use simple / synthetic examples since it needs to run quickly

- Most code should be moved to a `*_utils.py` file

# Tools of the trade
- Format a markdown file
  ```bash
  > lint_txt.py -i ...
  ```

- Clean up the Python code using Claude Code
  ```bash
  cc> Execute docs/ai_prompts/coding.lint.md on tutorials/Autogen/autogen_utils.py
  ```

- Format the blog
  ```bash
  cc> Execute docs/ai_prompts/blog.format_rules.md on website/docs/blog/posts/all.learn_Autogen_in_60_minutes.how_to_guide.md
  ```

- Align the Docker system with
  ```bash
  cc> Execute docs/ai_prompts/docker.align_with_template.md on tutorials/Autogen
  ```

- Rendering the blogs locally
  ```bash
  > website/test.sh
  ```

# Checklist

- [ ] Is our Docker approach followed?
  - If not follow `class_project/project_template/README.md`
- [ ] Is all the possible code in the notebook moved to a `*_utils.py` file?
- [ ] Is `project.API.ipynb` in the right format?
- [ ] Do the notebooks run end-to-end?
- [ ] Are all the notebooks paired using Jupytext?
- [ ] Did you run `linters2/lint_branch.sh`?

# References

- A description of the Docker build system is 
  `class_project/project_template/README.md`

- class_project/FAQ.md
- class_project/README.md
