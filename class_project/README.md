<!-- toc -->

<<<<<<< Updated upstream
- [Summary](#summary)
- [Getting Started (Quick Checklist)](#getting-started-quick-checklist)
=======
>>>>>>> Stashed changes
- [Class Project Guidelines](#class-project-guidelines)
  * [Quick Start](#quick-start)
  * [Choosing a Project](#choosing-a-project)
  * [Pre-Requisites](#pre-requisites)
    + [Contribution to the Repo](#contribution-to-the-repo)
  * [Configuring Your System](#configuring-your-system)
<<<<<<< Updated upstream
    + [`docker_template`](#docker_template)
=======
    + [Docker Template Setup — Recommended for Students](#docker-template-setup--recommended-for-students)
>>>>>>> Stashed changes
  * [Working on the Project](#working-on-the-project)
    + [Project Goal](#project-goal)
    + [Understanding the Deliverables](#understanding-the-deliverables)
  * [Submission](#submission)
    + [Difference Between `{project}.API.*` and `{project}.example.*`](#difference-between-projectapi-and-projectexample)
    + [Folder Structure](#folder-structure)
    + [Submission Guidelines](#submission-guidelines)
  * [Examples of a Class Project](#examples-of-a-class-project)

<!-- tocstop -->

# Summary

This guide explains how to complete the class project for courses such as
`DATA605` and `MSML610`. You will pick a big data, AI, or data science
technology from the sign-up sheet, build a small working system with it,
write a hands-on tutorial, and submit everything via a GitHub Pull Request.
The goal is practical, documented, reproducible code — not just a report.

## Getting Started (Quick Checklist)

Follow these steps in order:

1. **Pick a project** — choose from the sign-up sheet
   ([Choosing a Project](#choosing-a-project))
2. **Fork and clone the repos** — `umd_classes` and `helpers`
   ([Pre-Requisites](#pre-requisites))
3. **Set up Docker** — install Docker and use the `docker_template` workflow
   ([Configuring Your System](#configuring-your-system))
4. **Create a GitHub issue** — title it with your project tag
   ([Contribution to the Repo](#contribution-to-the-repo))
5. **Create your branch** — name it `TutorTask{N}_{project_tag}`
   ([Contribution to the Repo](#contribution-to-the-repo))
6. **Build the deliverables** — `{project}.API.*`, `{project}.example.*`, and
   `{project}_utils.py` ([Understanding the Deliverables](#understanding-the-deliverables))
7. **Open a Pull Request** — add reviewers and iterate
   ([Submission Guidelines](#submission-guidelines))

# Class Project Guidelines

- The goal of the class project (e.g., for `DATA605`, `MSML610`) is to learn a
  cutting-edge modern big data technology and write a (small) example of a
  system using it
- Each class project is similar in spirit to the tutorials for various
  technologies we have looked at and studied in classes (e.g., `Git`, `Docker`,
  `SQL`, `Mongo`, `Airflow`, `Dask`)
- Through the class projects you will learn how a specific tool fits your data
  science, data engineering, machine learning workflows

## Quick Start

- Change directory to your project folder:
  ```bash
  > cd ~/src/umd_classes1/class_project/COURSECODE/Term20xx/projects/{branch_name}
  ```

- Copy the project template files into your directory:
  ```bash
  > cp -r ~/src/umd_classes1/class_project/project_template/* .
  ```

- Build the Docker image:
  ```bash
  > ./docker_build.sh
  ```

- Launch Jupyter to work on the notebooks:
  ```bash
  > ./docker_jupyter.sh
  ```

- Open the notebooks in this order:
  1. **`{project}.API.ipynb`**: Start here to explore the tool's native API and
     core features
  2. **`{project}.example.ipynb`**: Proceed to this notebook to build and
     demonstrate your end-to-end application

## Choosing a Project

- The project is done by a single student or a group of students
  - Students should not have exactly the same project
  - Groups are made of less than 3 students
  - Students working on different projects can discuss and help each other (they
    will do that even if we say not to)

- Each student or group should pick one project from the sign up sheet shared
  around the mid of course
  - Each project has a description in the corresponding directory

- The goal of the project is to get your hands dirty and figure things out
  - Often solving problems is about trying different approaches until one
    approach works out
  - Make sure you code by understanding the tool and what your code is doing
    with it
  - `Google` and `ChatGPT` are your friends, but don't abuse them: copy-pasting
    code is not recommended and won't benefit the learning outcomes

- The projects are designed in a way that once you understand the underlying
  technology
  - We expect a project to takes 6-8 full days to complete (e.g., 40 hours)

- It is highly recommended to choose a project from the sign up sheet
  - If you really need to propose a new idea or suggest modifications, please
    contact us: we will review but we won't guarantee we can accommodate all
    requests
- Your project should align with your learning goals and interests, offering an
  opportunity to explore various technologies and strengthen your resume
- If selecting a project from the sign-up sheet, ensure you fill out the
  corresponding details promptly. For modifications, email us with the necessary
  information, and we will update the sign-up sheet and Google Doc accordingly
- **Project selection must be finalized within 1 or 2 weeks** to allow
  sufficient time for planning and execution
- The project duration is approximately **4 to 6 weeks**, making timely
  selection crucial
- Your grade will be based on:
  - **Project complexity**: depth and sophistication of the technology
    used
  - **Effort and understanding**: demonstrated grasp of the tool and
    quality of the tutorial
  - **Adherence to guidelines**: correct structure, naming, and PR
    workflow

**NOTE**:

- If you choose to use a paid service (e.g., an Amazon service), you are
  responsible for the costs incurred
  - You are expected to use the services efficiently to keep them within free
    tier usage
  - To save costs/improve usage, you should make sure that the services are
    turned off/shutdown when not being used

## Pre-Requisites

- Watch, star, and fork the repos
  - [`umd_classes`](https://github.com/gpsaggese/umd_classes)
  - [`helpers`](https://github.com/causify-ai/helpers)

- Install `Docker` on your computer
  - You can use `Docker` natively on `Mac` and `Linux`
  - Use `VMware` in `Windows` or dual-boot
    - If you have problems installing it on your laptop, it is recommended to use
      one computer from `UMD` laboratories

- Check your `GitHub` issue on https://github.com/gpsaggese/umd_classes/issues
  - Make sure you are assigned to it
- Only `Python` should be used together with the needed configs for the specific
  tools
  - You can always communicate with the tech using `Python` libraries or `HTTP`
    APIs

- Unless specified by project description, everything needs to run locally
  without using cloud resources
  - E.g., it's not ok to use an `AWS` DB instance, you want to install
    `Postgres` in your container for any database requirements

### Contribution to the Repo

- You will work in the same way open-source developers (and specifically
  developers on Causify.AI) contribute to a project

- Each project will need to be organized like a proper open source project,
  including filing issues, opening PRs, checking in the code in
  [https://github.com/gpsaggese/umd_classes/tree/master](https://github.com/gpsaggese/umd_classes/tree/master)

- Set up your working environment by following the instructions in the
  [document](https://github.com/causify-ai/helpers/blob/master/docs/onboarding/intern.set_up_development_on_laptop.how_to_guide.md)

- Each step of the project is delivered by committing code to the dir
  corresponding to your project and doing a `GitHub` Pull Request (PR)
  - You should commit regularly and not just once at the end
  - We will specifically do a reviews of intermediate results of the project and
    give you some feedback on what to improve (adopting `Agile` methodology)

- **Project Tag Naming Convention**
  - Your project tag should follows this format:
    `Spring{year}_{project_title_without_spaces}`
    - Example: if your project title is **"Redis cache to fetch user
      profiles"**, your project branch will be:
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
    - Example: If you cloned the repo on your laptop for `DATA605`, your
      directory should be:
      `~/src/umd_classes1/class_project/DATA605/Spring2025/projects/TutorTask645_Spring2025_Redis_cache_to_fetch_user_profiles`
  - Copy the project template to your directory:
    ```bash
    > cp -r ~/src/umd_classes1/class_project/project_template/ ~/src/umd_classes1/class_project/COURSECODE/Term20xx/projects/{branch_name}
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

## Configuring Your System

<<<<<<< Updated upstream
Before starting implementation, set up the `docker_template` workflow below.
Finalize your setup before proceeding with development.
=======
Before starting implementation, set up the `Docker`-based workflow from the
project template.
>>>>>>> Stashed changes

### Docker Template Setup — Recommended for Students

- Each project gets its own directory with a set of standard scripts
  (`docker_build.sh`, `docker_bash.sh`, `docker_jupyter.sh`) to build the
  container, launch it, and debug it
  - This is the same approach used for the tutorials in this repo (e.g.,
    `tutorials/autogen`, `tutorials/tensorflow`)

- The template lives at `class_project/project_template`

<<<<<<< Updated upstream
- Examples are:
  - `class_project/docker_template`
  - `class_project/docker_template_example`

- To use this approach:
  ```bash
  > cp -r class_project/docker_template ...
  ```
  - Then you customize the `Dockerfile`, expose other ports, or add
    project-specific dependencies as needed
=======
- To start a new project, copy the template:
  ```bash
  > cp -r class_project/project_template ~/src/umd_classes1/class_project/COURSECODE/Term20xx/projects/{branch_name}
  ```
  - Then customize the `Dockerfile`, expose additional ports, or add
    project-specific dependencies in `requirements.txt` as needed
>>>>>>> Stashed changes

- Pros
  - Very simple to use: just copy and modify
  - Mirrors the structure of every tutorial in the repo

- Cons
  - Each directory is self-contained; improvements to scripts in one project do
    not propagate to others

## Working on the Project

### Project Goal

- For your course project, you're not just building something cool, but you're
  also teaching others how to use a Big Data, AI, LLM, or data science tech
- As a project report, you'll create a tutorial that's hands-on and
  beginner-friendly
  - Think of it as your chance to help a classmate get started with the same
    tech
  - The goal of this tutorial is to help pickup a new technology in 60 Minutes
  - That should make sure the tutorial is not lengthy and covers all the
    important aspects a developer should know before starting building with that
    technology

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
- **Example Notebook** (`{project}.example.ipynb`):
  - Demonstrates an end-to-end application using your wrapper layer
  - Calls functions from `{project}_utils.py` to keep cells concise

For more guidance on this structure and the rationale behind it, see
[How to write the
Tutorial](https://github.com/gpsaggese/umd_classes/blob/master/website/docs/blog/posts/all.learn_X_in_60_minutes.how_to_guide.md)

In general:

- **For API notebook**: describe the tool's architecture, key abstractions, and
  how your wrapper simplifies it
- **For example notebook**: demonstrate the tool according to the specifications
  in your project description

## Submission

Your submission must include the following files:

<<<<<<< Updated upstream
**Important**: "package" here refers to the tool's internal interface—not an
external data-provider package. Please keep the focus on the tool itself.

- `XYZ.API.ipynb`:
  - A `Jupyter` notebook demonstrating usage of the native package and your
=======
- `XYZ.API.ipynb`:
  - A `Jupyter` notebook demonstrating usage of the tool's native API and your
>>>>>>> Stashed changes
    wrapper layer, with clean, minimal cells
  - Document the native programming interface (classes, functions,
    configuration objects) of your chosen tool or library
  - Describe the lightweight wrapper layer you have written on top

- `XYZ.example.ipynb`:
  - A `Jupyter` notebook demonstrating end-to-end functionality
  - Shows a complete application that uses your wrapper layer

- `XYZ_utils.py`:
  - A `Python` module containing reusable utility functions and wrappers around
    the tool
  - The notebooks should invoke logic from this file instead of embedding
    complex code inline

### Difference Between `{project}.API.*` and `{project}.example.*`

<<<<<<< Updated upstream
- **`{project}.API.*`**: stable contract-only layer. Holds dataclasses, enums,
  and abstract service interfaces so anyone can integrate without pulling in
  your runtime code
=======
- **`{project}.API.*`**: explores the tool's native API, key abstractions, and
  the lightweight wrapper layer you build on top
>>>>>>> Stashed changes

  ```python
  from dataclasses import dataclass
  from typing import Protocol

  @dataclass
  class User:
      id: int
      email: str

  class AuthService(Protocol):
      """Authenticate users without revealing storage details."""
      def register(self, user: User, password: str) -> None: ...
      def login(self, email: str, password: str) -> str: ...  # returns JWT
  ```

- **`{project}.example.*`**: runnable reference implementation that satisfies
  the package with real storage, I/O, and third-party calls

  ```python
  import sqlite3
  import bcrypt
  import jwt
  from project_package.auth import User, AuthService

  class SqliteAuthService(AuthService):
      _DB = "users.db"

      def register(self, user: User, password: str) -> None:
          hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
          with sqlite3.connect(self._DB) as conn:
              conn.execute(
                  "INSERT INTO users(id, email, password) VALUES (?,?,?)",
                  (user.id, user.email, hashed),
              )

      def login(self, email: str, password: str) -> str:
          with sqlite3.connect(self._DB) as conn:
              row = conn.execute("SELECT password FROM users WHERE email=?", (email,)).fetchone()
          if not row or not bcrypt.checkpw(password.encode(), row[0]):
              raise PermissionError("invalid credentials")
          return jwt.encode({"sub": email}, "supersecret", algorithm="HS256")
  ```

### Folder Structure
```text
COURSE_CODE/
└── Term20xx/
    └── projects/
        └── TutorTaskXX_Name_of_issue/
            ├── {project}_utils.py
            ├── {project}.API.ipynb
            ├── {project}.example.ipynb
            ├── Dockerfile
            └── README.md
```

### Submission Guidelines

- Each markdown file should explain the intent and design decisions:
  - Avoid copy-pasting code cells or raw outputs from the notebooks
  - Instead, use the markdown to communicate the reasoning behind your choices

- Each notebook should:
  - Be self-contained and executable from top to bottom via "Restart and Run
    All"
  - Use functions from `XYZ_utils.py` to keep the cells concise and maintainable
  - Demonstrate functionality clearly and logically with clean, commented
    outputs

- `Docker` setup:
  - Include clear instructions on how to build and run your `Docker` container
  - Mention expected terminal outputs when running scripts (e.g., starting
    `Jupyter`, mounting volumes, etc.) E.g.,

  Your README should contain sections like:

      ### To Build the Image

      ```bash
      bash docker_build.sh
      ```

      ### To Run the Container

      ```bash
      bash docker_bash.sh
      ```

- Visual documentation:
  - Include diagrams and flowcharts when relevant (e.g., using `mermaid`) E.g.,

  ```mermaid
  flowchart TD
    A[Start] --> B{Decision}
    B -- Yes --> C[Process 1]
    B -- No  --> D[Process 2]
    C --> E[End]
    D --> E
  ```
  - Provide schema descriptions if your project uses a database or structured
    data E.g.,

  ```mermaid
  erDiagram
    USERS {
        INT id PK
        VARCHAR name
        VARCHAR email
        TIMESTAMP created_at
    }
    ORDERS {
        INT id PK
        INT user_id FK
        DECIMAL total
        TIMESTAMP placed_at
    }
    USERS ||--o{ ORDERS : places
  ```

- **Projects that do not run end-to-end or lack proper documentation will be
  considered incomplete**
  - In case of issues, they will be flagged through `GitHub` issues, and you
    will be expected to resolve them in a timely manner

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
