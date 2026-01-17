

<!-- toc -->

- [Tutorials "Learn X in 60 minutes"](#tutorials-learn-x-in-60-minutes)
  * [What are the goals for each tutorial](#what-are-the-goals-for-each-tutorial)

<!-- tocstop -->

# "Learn X in 60 Minutes" Tutorials

- **Goal**: give everything needed for someone to become familiar with a big data
  / AI / LLM / data science technology in 60 minutes

- Each tutorial conceptually corresponds to:
  - A blog entry
  - A project of one of the classes (e.g., DATA605, MSML610)

## What are the goals for each tutorial

Each tutorial aims to provide:
- **Hands-on experience**: Working code examples that run immediately
- **Conceptual understanding**: Clear explanations of what the technology is and
  when to use it
- **Practical application**: A complete example showing real-world usage
- **Self-sufficiency**: All dependencies and setup handled via Docker
- **Reproducibility**: Guaranteed to work through automated testing

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
- **Dependency management**: All packages are handled through Docker in our
  standard approach
- **Consistent structure**: The format of the tutorial follows the same structure
  across all topics
- **Centralized location**: All tutorial material is in a directory in the
  [`//tutorials`](https://github.com/causify-ai/tutorials) repo and in the
  [`//helpers`](https://github.com/causify-ai/helpers) sub-repo

## Content

- Each tutorial contains:
  1) A Docker container with everything needed to build and run using the Causify
     `dev-system` format
  2) A markdown `XYZ.API.md` about:
     - The native API
     - The software layer written by us on top of the native API
  3) A Jupyter notebook `XYZ.API.ipynb` with an example of using the native / our
     APIs
  4) A markdown `XYZ.example.md` with a full example of an application using the
     API
  5) A Jupyter notebook `XYZ.example.ipynb` with a full example
  6) A file `XYZ_utils.py` with utility functions

### README
- Each project contains a `readme.md` summarizing its status

### Docker container
- The Docker container should:
  - Contain everything so that one is ready to run tutorials and develop with
    that technology
  - Often installing and getting a package to work (e.g., PyMC) takes a long time

### Jupyter notebooks
- Each Jupyter notebook should:
   - Run end-to-end after a restart
     - This is enforced by the unit test through `pytest`
     - In this way we are guaranteed that it works
     - It's super frustrating when a tutorial doesn't work because the version of
       the library is not compatible with the code anymore
   - Be self-contained and linear
     - Each example is explained thoroughly without having to jump from tutorial
       to tutorial
     - Each cell and its output is commented and explained
  - Take less than few minutes to execute end-to-end

### Markdown
- Markdown documents should cover information about:
  - What the Python package / library is
  - What problem it solves
  - What are the alternatives, both open source and commercial with comments about
    advantages and disadvantages
  - A description of the native API, i.e., the package
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
  - [MSML610](//github.com/gpsaggese/umd_classes/blob/master/msml610/tutorials/notebooks)
  even if not all these tutorials don't use the Causify dev system, but some
  simpler bash scripts.
