# Class Project Guidelines "Learn X in 60 minutes"

- The goal of the class project (e.g., for `DATA605`, `MSML610`) is to learn a
  cutting-edge modern big data technology and write a (small) example of a
  system using it
- The result of each project is a "tutorial" that can teach a curious computer
  scientist a new technology in 60 minutes

- Each class project is similar in spirit to the tutorials for various
  technologies we have looked at and studied in classes (e.g., `Git`, `Docker`,
  `SQL`, `Mongo`, `Airflow`, `Dask`)
- Through the class projects you will learn how a specific tool fits your data
  science, data engineering, machine learning workflows

## Choosing a Project

### Project Types

There are three types of projects:

- **"Build X using Y"**: Build a system or application using a specific
  technology — individual or teams of at most 3 students. The best projects
  become blog entries and potential publications.
- **Implement examples for the lectures**: Implement tutorial examples that
  illustrate lecture concepts — individual projects only. The best projects are
  used as tutorials in future classes.
- **Research projects**: Explore an open research question — teams of at most 3
  students. The best projects may lead to a blog post or publication, but these
  are the most challenging.

### Selection Rules

- The project is done by a single student or a group of students
  - Students should not have exactly the same project
  - Groups are made of at most 3 students (`<= 3`)
  - All team members receive the same score, so choose partners carefully
  - Students working on different projects can discuss and help each other

- Each student or group picks one project from the sign-up sheet shared during
  class
  - Each project has a description in the corresponding directory
  - You can pick two projects as a backup; if there is a conflict, projects are
    assigned at random

- The goal of the project is to get your hands dirty and figure things out
  - Often solving problems is about trying different approaches until one works
  - Make sure you understand the tool and what your code is doing
  - `Google` and `ChatGPT` are your friends, but don't abuse them: copy-pasting
    code is not recommended and won't benefit the learning outcomes

- We expect a project to take 6-8 full days to complete (e.g., 40 hours)

- Your project should align with your learning goals and interests
- Project selection must be finalized within 1 or 2 weeks to allow sufficient
  time for planning and execution

- Your grade will be based on:
  - **Project complexity**: depth and sophistication of the technology used
  - **Effort and understanding**: demonstrated grasp of the tool and quality of
    the tutorial
  - **Adherence to guidelines**: correct structure, naming, and PR workflow

### Paid Cloud Services

- If you choose to use a paid service (e.g., an Amazon service), you are
  responsible for the costs incurred
  - You are expected to use the services efficiently to keep them within free
    tier usage
  - To save costs/improve usage, you should make sure that the services are
    turned off/shutdown when not being used

### Project Timeline

- The project schedule is described in the "Class assignment" column in the
  class schedule

- In practice, the process is:
  - Pick 2 projects
  - Finalize the project and make sure everything is clear
  - ... Work, work, work by yourself at home ...
  - ... Work, work, work with us in class lab ...
  - First checkpoint / PR of the project
  - ... Work, work, work by yourself at home ...
  - ... Work, work, work with us in class lab ...
  - Final submission

## Pre-requisites

- Watch, star, and fork the repos
  - [`umd_classes`](https://github.com/gpsaggese/umd_classes)
  - [`helpers`](https://github.com/causify-ai/helpers)

- Install `Docker` on your computer
  - You can use `Docker` natively on `Mac` and `Linux`
  - Use `VMware` in `Windows` or dual-boot
    - If you have problems installing it on your laptop, it is recommended to
      use one computer from `UMD` laboratories

- Check your `GitHub` issue on https://github.com/gpsaggese/umd_classes/issues
  - Make sure you are assigned to it
- Only `Python` should be used together with the needed configs for the specific
  tools

- Unless specified by project description, everything needs to run locally
  without using cloud resources
  - E.g., it's not ok to use an `AWS` DB instance, you want to install
    `Postgres` in your container for any database requirements

### Learn or Refresh Basic Data Science Tools

- If you did not take DATA605 or you are not familiar with the basic technology
  we rely on (e.g., `Git`, `Docker`, `Jupyter`, `Python`, `bash`), take time to
  get familiar with them through the DATA605 tutorials

## Contribution to the Repo

- You will work in the same way open-source developers contribute to a project
- Each project will need to be organized like a proper open source project,
  including filing issues, opening PRs, checking in the code in
  [https://github.com/gpsaggese/umd_classes/tree/master](https://github.com/gpsaggese/umd_classes/tree/master)

- Set up your working environment by following the instructions in the
  [document](https://github.com/causify-ai/helpers/blob/master/docs/onboarding/intern.set_up_development_on_laptop.how_to_guide.md)

- Each step of the project is delivered by committing code to the dir
  corresponding to your project and doing a `GitHub` Pull Request (PR)
  - You should commit regularly and not just once at the end
  - We will specifically do reviews of intermediate results of the project and
    give you some feedback on what to improve (adopting `Agile` methodology)

- **Project Tag Naming Convention**
  - Your project tag should follow this format:
    `Spring{year}_{project_title_without_spaces}`
    - Example: if your project title is **"Redis cache to fetch user
      profiles"** for Spring 2025, your project tag will be:
      **`Spring2025_Redis_cache_to_fetch_user_profiles`**

- **Create a GitHub Issue**
  - [ ] Create a **`GitHub` issue** with your **project tag** as the title
    - Example: `Spring2025_Redis_cache_to_fetch_user_profiles`
  - [ ] Copy/paste the project description and add a link to the `Google Doc`
        with the details
  - [ ] Assign the issue to yourself. This issue will be used for
        project-related discussions

- **Create a Git Branch Named After the Issue**
  - [ ] Name your `Git` branch as follows:
        `TutorTask{issue_number}_{project_tag}`
    - Example: If your issue number is **#645**, your branch name should be:
      **`TutorTask645_Spring2025_Redis_cache_to_fetch_user_profiles`**

- **Steps to create the branch:**

  ```bash
  > cd $HOME/src
  > git clone --recursive git@github.com:gpsaggese/umd_classes.git umd_classes1
  > cd $HOME/src/umd_classes1
  > git checkout master
  > git checkout -b TutorTask645_Spring2025_Redis_cache_to_fetch_user_profiles
  ```

- **Add Files Only in Your Project Directory**
  - Add your project files under the following directory:
    `{GIT_ROOT}/class_project/{COURSE_CODE}/{TERM}{YEAR}/projects/{branch_name}`
    - Example for DATA605 Spring 2025:
      `~/src/umd_classes1/class_project/DATA605/Spring2025/projects/TutorTask645_Spring2025_Redis_cache_to_fetch_user_profiles`
  - Copy the project template to your directory:
    ```bash
    > cp -r ~/src/umd_classes1/class_project/project_template/ \
        ~/src/umd_classes1/class_project/DATA605/Spring2025/projects/TutorTask645_Spring2025_Redis_cache_to_fetch_user_profiles
    ```
  - Start working on the files

- **Create a Pull Request (PR)**:
  - [ ] Always create a **Pull Request (PR)** from your branch
  - [ ] Name the PR the same as your project branch, and reference the issue
        number your branch is based on
  - [ ] Add your TAs and `@gpsaggese` as reviewers
  - [ ] Assign the PR to yourself
  - [ ] Do **not** push directly to the `master` branch. Only push commits to
        **your project branch**

- **Naming for Consecutive Updates**
  - When making progress, use incremental branch names by appending `_1`, `_2`
    to your branch name, etc.
    - Example:
      - `TutorTask645_Spring2025_Redis_cache_to_fetch_user_profiles_1`
      - `TutorTask645_Spring2025_Redis_cache_to_fetch_user_profiles_2`

## Quick Start

- After creating your branch and project directory (see "Contribution to the
  Repo" above), change directory to your project folder:
  ```bash
  > cd ~/src/umd_classes1/class_project/DATA605/Spring2025/projects/{branch_name}
  ```

- Copy the project template files into your directory (if not already done):
  ```bash
  > cp -r ~/src/umd_classes1/class_project/project_template/* .
  ```

- Customize the content of the files `Dockerfile`, `requirements.txt`

- Build the Docker image:
  ```bash
  > ./docker_build.sh
  ```

- Launch Jupyter to work on the notebooks:
  ```bash
  > ./docker_jupyter.sh
  ```

- To open a bash shell inside the container for debugging:
  ```bash
  > ./docker_bash.sh
  ```

- To remove the container and free resources:
  ```bash
  > ./docker_clean.sh
  ```

- Complete the notebooks in this order:
  1. **`{project}.API.ipynb`**: explore the tool's native API and core features
  2. **`{project}.example.ipynb`**: Build and demonstrate your end-to-end
     application

## Configuring Your System

### `docker_template`

- There are simple scripts (`docker_build.sh`, `docker_bash.sh`,
  `docker_jupyter.sh`) to help you build the container, launch it, and debug it

- In this approach each directory is self-contained and nothing is shared among
  directories and projects
  - The only common part is that there are scripts with a shared interface that
    makes it easy to understand how to run the basic functionalities
  - This is the approach we use for the `data605/tutorials`

- To use this approach:
  ```bash
  > cp -r class_project/docker_template ...
  ```
  - Then customize the `Dockerfile`, expose other ports, or add
    project-specific dependencies as needed

- Examples are in `class_project/docker_template` and
  `class_project/docker_template_example`

## Working on the Project

### Tutorial Goal

For your course project, you are not just building something — you are also
teaching others how to use a Big Data, AI, LLM, or data science technology. The
deliverable is a hands-on, beginner-friendly tutorial that teaches the
technology in 60 minutes.

Each 60-minute tutorial follows this time breakdown for a reader:

1. **Setup (5 min)**: Clone repo, start Docker container, verify environment
2. **Introduction (10 min)**: Read overview markdown, understand use cases
3. **API Exploration (20 min)**: Work through `{project}.API.ipynb` notebook
4. **Complete Example (25 min)**: Work through `{project}.example.ipynb`
   notebook

Each tutorial aims to provide:

- **Conceptual understanding**: Clear explanations of what the technology is and
  when to use it
- **Practical application**: A complete example showing real-world usage and
  working code examples that run immediately
- **Reproducibility**: Guaranteed to work through automated testing with all
  dependencies and setup handled via Docker

### Invariants

All tutorials must maintain these standards:

- **Code repository**: All code is on GitHub in a format common to all tutorials
- **Dependency management**: All packages are handled through Docker in a
  standard approach (e.g., `docker_build`, `docker_bash`)
- **Consistent structure**: The format of the tutorial follows the same
  structure across all topics
- **Centralized location**: All tutorial material is in a directory in the
  [`tutorials`](https://github.com/gpsaggese/umd_classes/tree/master/tutorials)
  repo and in the [`//helpers`](https://github.com/causify-ai/helpers) sub-repo

### Understanding the Deliverables

- Use the example tutorials (e.g., `tutorials/autogen`, `tutorials/tensorflow`)
  and `class_project/project_template` to understand the deliverables and
  coding style. They consist of:

- **Utils Module** (`{project}_utils.py`):
  - Contains helper functions, reusable logic, and wrappers around the tool
  - Keep the notebooks focused on documentation and outputs; place all logic
    inside this module
- **API Notebook** (`{project}.API.ipynb`):
  - Explores the tool's native API: core classes, functions, and configuration
  - Describes the lightweight wrapper layer you have written on top
  - Contains a walkthrough of the library/package with examples
  - Uses simple/synthetic examples since it needs to run quickly
  - Most code should be moved to a `*_utils.py` file
- **Example Notebook** (`{project}.example.ipynb`):
  - Demonstrates an end-to-end application using your wrapper layer
  - Calls functions from `{project}_utils.py` to keep cells concise

In general:

- **For API notebook**: describe the tool's architecture, key abstractions, and
  how your wrapper simplifies it
- **For example notebook**: demonstrate the tool according to the specifications
  in your project description

### Docker Container

- The Docker container should:
  - Contain everything so that one is ready to run tutorials and develop with
    that technology
  - Often installing and getting a package to work (e.g., PyMC) takes a long
    time

- The Docker structure and approach should follow the template
  [`class_project/project_template/`](https://github.com/gpsaggese/umd_classes/tree/master/class_project/project_template)

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
  - Take less than a few minutes to execute end-to-end

### Markdown

- Markdown documents should cover information about:
  - What the technology/Python package or library is
  - What problem it solves
  - What are the alternatives, both open source and commercial with comments
    about advantages and disadvantages
  - A description of the native API of the technology
  - A description of the Docker container
  - Visual aids with `mermaid`, `graphviz`, `tikz` (e.g., flow diagrams, data
    transformation steps, and plots) to enhance understanding of how the library
    and the example works
  - References to books and in-depth tutorials that we have run and we think are
    awesome
  - All sources should be referred and acknowledged

- This is the same approach used in:
  - [DATA605](https://github.com/gpsaggese/umd_classes/blob/master/data605/tutorials)
    - E.g., Git, Docker, Docker Compose, Postgres, MongoDB, Airflow, Dask, Spark
  - [MSML610](https://github.com/gpsaggese/umd_classes/blob/master/msml610/tutorials/notebooks)

### Tools of the Trade

- Format a markdown file:
  ```bash
  > lint_txt.py -i ...
  ```

- Clean up the Python code using Claude Code:
  ```bash
  cc> Execute docs/ai_prompts/coding.lint.md on tutorials/Autogen/autogen_utils.py
  ```

- Format the blog:
  ```bash
  cc> Execute docs/ai_prompts/blog.format_rules.md on website/docs/blog/posts/all.learn_Autogen_in_60_minutes.how_to_guide.md
  ```

- Align the Docker system with:
  ```bash
  cc> Execute docs/ai_prompts/docker.align_with_template.md on tutorials/Autogen
  ```

- Render the blogs locally:
  ```bash
  > website/test.sh
  ```

## Submission

Your submission must include the following files:

- `{project}.API.ipynb`:
  - A `Jupyter` notebook exploring the tool's native API: core classes,
    functions, and configuration
  - Describes the lightweight wrapper layer you have written on top
  - Contains a walkthrough of the library/package with examples
  - Uses simple/synthetic examples so it runs quickly

- `{project}.example.ipynb`:
  - A `Jupyter` notebook demonstrating an end-to-end application
  - Calls functions from `{project}_utils.py` to keep cells concise

- `{project}_utils.py`:
  - A `Python` module containing reusable utility functions and wrappers around
    the package
  - The notebooks should invoke logic from this file instead of embedding
    complex code inline

### Folder Structure

```text
COURSE_CODE/
└── Term20xx/
    └── projects/
        └── TutorTaskXX_Name_of_issue/
            ├── {project}_utils.py       # reusable helper functions
            ├── {project}.API.ipynb      # tool's native API exploration
            ├── {project}.example.ipynb  # end-to-end application demo
            ├── Dockerfile
            ├── docker_build.sh          # build the Docker image
            ├── docker_bash.sh           # open a shell in the container
            ├── docker_jupyter.sh        # launch Jupyter inside the container
            ├── docker_clean.sh          # remove the container and image
            ├── requirements.txt
            └── README.md
```

### Video Recording Guidelines

The final project requires students to submit a video recording of their
project. The goal is to learn how to present your work in a professional manner
(which will be extremely important in your career).

- Video duration
  - Minimum: 10 minutes
  - Maximum: 20 minutes

- Submission method
  - Students should upload the video in this
    [Google Drive dir](https://drive.google.com/drive/folders/1QLtgPCAS0mqE9cr1hE3UVoIbzakCNtaC)

- Required video structure
  - Step 1: Introduction
    - Name, UID, tool and difficulty, project title
  - Step 2: Showcase all files in the PR and confirm naming conventions
  - Step 3: Execute the Docker image and show the successful execution message
    - If there was a problem with Docker explain what it was and how you worked
      around it
  - Step 4: Open Jupyter Notebook
    - Steps 1-4 should take approximately 1-2 minutes
  - Step 5: Full project walkthrough
    - Run every required code cell
    - Provide clear verbal explanation of what each cell does
    - Demonstrate functionality and correctness of the tool
    - Students should spend the majority of time here
  - Step 6: Discuss results
    - Interpret the outputs
    - Explain what the results mean
    - Describe how the chosen tool helped address the problem statement
  - Step 7: Documentation review
    - Show how the documentation is organized
    - Explain how a non-technical reader can understand the project using the
      documentation
    - Highlight completeness and clarity

## Examples of a Class Project

The layout of each project should follow the examples in:

- [`tutorials/autogen`](https://github.com/gpsaggese/umd_classes/tree/master/tutorials/autogen)
  — the closest reference for the expected structure (`docker_build.sh`,
  `autogen.API.ipynb`, `autogen.example.ipynb`, `autogen_utils.py`)
- [`tutorials/tensorflow`](https://github.com/gpsaggese/umd_classes/tree/master/tutorials/tensorflow)
  — another self-contained example following the same template
- [`class_project/project_template`](https://github.com/gpsaggese/umd_classes/tree/master/class_project/project_template)
  — the canonical starting point with blank `template.API.ipynb` and
  `template.example.ipynb` notebooks

## Review Best Projects

- Review some of the best projects from previous years:
  - [`tutorial_langchain`](https://github.com/causify-ai/tutorials/tree/master/tutorial_langchain)
  - [`tutorial_langgraph`](https://github.com/causify-ai/tutorials/tree/master/tutorial_langgraph)
  - [`tutorial_openai`](https://github.com/causify-ai/tutorials/tree/master/tutorial_openai)
  - [`tutorial_prophet`](https://github.com/causify-ai/tutorials/tree/master/tutorial_prophet)

- Peruse/draw inspiration from lots of projects from your colleagues in the
  sister class DATA605:
  - Projects:
    [https://github.com/gpsaggese/umd_classes/tree/master/class_project/DATA605](https://github.com/gpsaggese/umd_classes/tree/master/class_project/DATA605)
  - Videos:
    [https://drive.google.com/drive/folders/1QLtgPCAS0mqE9cr1hE3UVoIbzakCNtaC](https://drive.google.com/drive/folders/1QLtgPCAS0mqE9cr1hE3UVoIbzakCNtaC)

- If you see any problem in the tutorials (code or video), please send an email
  to me with your TAs in cc. To make this more interesting, there might be (or
  maybe not!) some mistakes on purpose. See if you find them!

## Grading Rubric

- All deliverables delivered (10 points)
  - Are all required components submitted, including code, documentation, and
    any other specified materials
  - Does everything follow the project's standard structure and formatting
    guidelines
  - Is the submission complete enough for someone else to evaluate or run
    without requesting extra files or explanations

- Working Docker (5 points)
  - Does the Docker container build without errors, following the provided
    instructions
  - Does the project run successfully inside the container and behave as
    expected
  - Are all necessary environment configurations, ports, and dependencies
    correctly set

- Documentation quality (5 points)
  - Does the documentation clearly explain how to set up, run, and understand
    the project
  - Is it well-written, free from major grammar or formatting issues, and easy
    to follow
  - Does it include all required sections such as installation steps, usage
    examples, API descriptions, or architectural decisions
  - Does it adhere to the formatting and organizational guidelines we provided

- Actual project complexity (5 points)
  - How much depth and effort does the project demonstrate beyond the initial
    suggested scope
  - Did the student add significant features, handle edge cases, or show
    creative problem-solving
  - Is the architecture or technical implementation non-trivial or particularly
    well-considered

- Code quality (5 points)
  - Is the code clean, modular, and easy to understand
  - Are comments and docstrings used appropriately to explain non-obvious logic
  - Is the overall structure of the codebase sensible and maintainable
  - Does it follow consistent style conventions (e.g., PEP 8 or another
    relevant standard)

- PR quality (5 points)
  - Was the pull request (PR) well-organized, with meaningful commit messages
    and a clear description
  - Did the student follow good GitHub practices, such as linking issues,
    keeping PRs scoped, and avoiding unnecessary noise
  - Was the PR free of obvious clutter like unrelated test code or temporary
    debug statements

- Depth and understanding (5 points)
  - Does the student show a solid understanding of the tools and techniques they
    used
  - Do the design decisions reflect thoughtful trade-offs and justification,
    rather than blindly following tutorials
  - Is there evidence that the student could explain and defend their
    implementation in a review

- Late submission (-5 points)
  - Was the submission turned in after the deadline without an approved
    extension

- Incomplete work (-5 points)
  - Are there major parts of the project that are missing or obviously broken
  - Does the submission fail to meet critical functional or structural
    expectations
  - Are there signs that the project was rushed or left unfinished or AI
    generated

### Submission Checklist

- [ ] Is our Docker approach followed?
- [ ] Is all the possible code in the notebook moved to a `*_utils.py` file?
- [ ] Is `project.API.ipynb` in the right format?
- [ ] Do the notebooks run end-to-end?
- [ ] Are all the notebooks paired using Jupytext?
- [ ] Did you run `linters2/lint_branch.sh`?
