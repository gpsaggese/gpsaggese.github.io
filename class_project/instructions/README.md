<!-- toc -->

- [Class Project Guidelines](#class-project-guidelines)
  * [Choosing a project](#choosing-a-project)
  * [Pre-requisites](#pre-requisites)
    + [Contribution to the repo](#contribution-to-the-repo)
  * [Configuring your system](#configuring-your-system)
    + [Simple Docker Setup (`data605_style`) — Recommended for Students](#simple-docker-setup-data605_style--recommended-for-students)
    + [Want to Learn More?](#want-to-learn-more)
  * [Working on the project](#working-on-the-project)
    + [Project Goal](#project-goal)
    + [Understanding the deliverables](#understanding-the-deliverables)
  * [Submission](#submission)
    + [Difference between `{project}.API.*` and `{project}.example.*`](#difference-between-projectapi-and-projectexample)
    + [Folder Structure](#folder-structure)
    + [Submission Guidelines](#submission-guidelines)
  * [Examples of a class project](#examples-of-a-class-project)

<!-- tocstop -->

# Class Project Guidelines

- The goal of the class project (e.g., for DATA605, MSML610) is to learn a
  cutting-edge modern big data technology and write a (small) example of a system
  using it
- Each class project is similar in spirit to the tutorials for various
  technologies we have looked at and studied in classes (e.g., Git, Docker, SQL,
  Mongo, Airflow, Dask)
- Through the class projects you will learn how a tool fits your data science,
  data engineering, machine learning workflows.

## Choosing a project

- The project is individual or group
  - Students can discuss and help each other (they will do that even if we say
    not to)
  - Students should not have exactly the same project
  - Groups are less than 3

- Each student should pick one project from the sign up sheet
  - The difficulty of the project does affect the final grade, but we want to
    give a way for everyone to select a project based on their level of computer
    literacy
  - Each project has a description in the corresponding directory
    - MSML610 Fall 2025:
      - [List of projects](https://docs.google.com/spreadsheets/d/1H_Ev1psuPpUrrRcmBrBb2chfurSo5rPcAdd6i2SIUTQ/edit?gid=0#gid=0)
      - [Description of
        projects](https://github.com/gpsaggese/umd_classes/tree/master/class_project/MSML610/Fall2025/project_descriptions)
    - DATA605 Spring 2025:
      - [Description of projects](https://github.com/gpsaggese/umd_classes/blob/master/class_project/DATA605/Spring2025/project_description.md)

- You need to fill out the [sign up form](https://docs.google.com/forms/d/1TPCt7UFnTOEICltrPU3sIu9RoCbILR32zHbZNzRi9jw/edit)
  - Once done, we will add you to the repo so that you can start working

- The goal is to get your hands dirty and figure things out
  - Often working is all about trying different approaches until one works out
  - Make sure you code by understanding the tool and what your code is doing
    with it
  - Google and ChatGPT are your friends, but don't abuse them: copy-pasting code
    is not recommended and won't benefit the learning outcomes
- The projects are designed in a way that once you understand the underlying
  technology:
  - Easy Project: Takes 2-3 days to complete
  - Medium Difficulty Project: Takes 4-5 to complete
  - Difficult Project: Takes 6-8 to complete.

- It is highly recommended to choose a project from the sign up sheet
  - If you really need to propose a new idea or suggest modifications, please
    contact us: we will review but we won't guarantee we can accommodate all
    requests
- Your project should align with your learning goals and interests, offering a
  great opportunity to explore various technologies and strengthen your resume.
- If selecting a project from the sign-up sheet, ensure you fill out the
  corresponding details promptly. For modifications, email us with the necessary
  information, and we will update the sign-up sheet and Google Doc accordingly.
- **Project selection must be finalized within 1 or 2 weeks** to allow sufficient
  time for planning and execution.
- The project duration is approximately **4 to 6 weeks**, making timely selection
  crucial.
- Your grade will be based on **project complexity, effort, understanding, and
  adherence to guidelines**.

**NOTE**:

- If you choose to use a paid service, you are responsible for the costs
  incurred. In any case, you are expected to use the services efficiently to
  keep them within free tier usage
- To save costs/improve usage, you should make sure that the services are turned
  off/shutdown when not being used.

## Pre-requisites

- Watch, star, and fork the Causify.AI repos
  - [umd_classes](https://github.com/gpsaggese/umd_classes)
  - [tutorials](https://github.com/causify-ai/tutorials)
  - [helpers](https://github.com/causify-ai/helpers)

- Install Docker on your computer
  - You can use Docker natively on Mac and Linux
  - Use VMware in Windows or dual-boot
    - If you have problems installing it on your laptop, it recommended to use
      one computer from UMD, or perhaps a device of one your friends
- After signing up for a project accept the invitation to collaborate sent to
  the email that you used to register your GitHub account, or check
  [here](https://github.com/gpsaggese/umd_classes/invitations)
- Check your GitHub issue on https://github.com/gpsaggese/umd_classes/issues
  - Make sure you are assigned to it
- Only Python should be used together with the needed configs for the specific
  tools
  - You can always communicate with the tech using Python libraries or HTTP APIs

- Unless specified by project description, everything needs to run locally
  without using cloud resources.
  - E.g., it's not ok to use an AWS DB instance, you want to install Postgres in
    your container for any database requirements

### Contribution to the repo

- You will work in the same way open-source developers (and specifically
  developers on Causify.AI) contribute to a project

- Each project will need to be organized like a proper open source project,
  including filing issues, opening PRs, checking in the code in
  [https://github.com/gpsaggese/umd_classes/tree/master](https://github.com/gpsaggese/umd_classes/tree/master)

- Set up your working environment by following the instructions in the
  [document](https://github.com/causify-ai/helpers/blob/master/docs/onboarding/intern.set_up_development_on_laptop.how_to_guide.md)

- Each step of the project is delivered by committing code to the dir
  corresponding to your project and doing a GitHub Pull Request (PR)
  - You should commit regularly and not just once at the end
  - We will specifically do a reviews of intermediate results of the project and
    give you some feedback on what to improve (adopting Agile methodology)

- **Project Tag Naming Convention**
  - Your project tag should follows this format:
    `Spring{year}_{project_title_without_spaces}`
    - Example: if your project title is **"Redis cache to fetch user
      profiles"**, your project branch will be:
      **`Spring2025_Redis_cache_to_fetch_user_profiles`**

- **Create a GitHub Issue**
  - [ ] Create a **GitHub issue** with your **project tag** as the title.
    - Example: `Spring2025_Redis_cache_to_fetch_user_profiles`
  - [ ] Copy/paste the project description and add a link to the Google Doc with the details.
  - [ ] Assign the issue to yourself. This issue will be used for project-related discussions.

- **Create a Git Branch Named After the Issue**
  - [ ] Name your Git branch as follows: `TutorTask{issue_number}_{project_tag}`
    - Example: If your issue number is **#645**, your branch name should be: **`TutorTask645_Spring2025_Redis_cache_to_fetch_user_profiles`**

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
    - Example: If you cloned the repo on your laptop for DATA605, your directory should be:
      `~/src/umd_classes1/class_project/DATA605/Spring2025/projects/TutorTask645_Spring2025_Redis_cache_to_fetch_user_profiles`
  - Copy the template files to the project directory:
    ```bash
    > cp -r ~/src/umd_classes1/class_project/instructions/tutorial_template/ ~/src/umd_classes1/class_project/COURSECODE/Term20xx/projects/{branch_name}
    > cd ~/src/umd_classes1/class_project/COURSECODE/Term20xx/projects/{branch_name}
    ```
  - Start working on the files

- **Create a Pull Request (PR)**:
  - [ ] Always create a **Pull Request (PR)** from your branch.
  - [ ] Name the PR the same as your project branch, and reference the issue number your branch is based on.
  - [ ] Add your TAs and `@gpsaggese` as reviewers.
  - [ ] Assign the PR to yourself.
  - [ ] Do **not** push directly to the `master` branch. Only push commits to **your project branch**.

- **Naming for Consecutive Updates**
  - When making progress, use incremental branch names by appending `_1`, `_2`
    to your branch name, etc.
    - Example:
      - `TutorTask645_Spring2025_Redis_cache_to_fetch_user_profiles_1`
      - `TutorTask645_Spring2025_Redis_cache_to_fetch_user_profiles_2`

## Configuring your system

Before starting implementation, you need to choose **one** of the two supported
Docker-based workflows. Finalize your setup choice before proceeding with
development.

### Simple Docker Setup (`simple`) — Recommended for Students

- A minimal and straightforward setup, modeled after what we use in class
  tutorials.
- The environment comes with Python, Jupyter, and commonly-used packages already
  installed.
- Simple scripts (`docker_build.sh`, `docker_bash.sh`, `docker_jupyter.sh`) help
  you build the container, launch it, and start working immediately.
- Ideal for students who:
  - Are new to Docker or want to avoid setup overhead
  - Need a reliable, pre-built environment to focus on the tutorial and project
    code

- You may still customize the Dockerfile, expose other ports, or add
  project-specific dependencies as needed.

## Working on the project

### Project Goal

- For your course project, you're not just building something cool, but you're
  also teaching others how to use a Big Data, AI, LLM, or data science tech
- As a project report, you'll create a tutorial that's hands-on and
  beginner-friendly
  - Think of it as your chance to help a classmate get started with the same
    tech
  - The goal of this tutorial is to help pickup a new technology in 60 Minutes!
  - That should make sure the tutorial is not lengthy and covers all the
    important aspects a developer should know before starting building with that
    technology.

### Understanding the deliverables

- Use the project template files in `instructions/tutorial_template` to
  understand the deliverables and the coding style. They consist of:

- **Utils Module**:
  - This file is meant to contain helper functions, reusable logic, and API
    wrappers.
  - Keep the notebooks focused on documentation and outputs. Place any logic or
    workflow functions inside this module.
- **Scripts/Notebooks**:
  - You will work on one API file and one Example (Your project) file.
  - We encourage you to use Python files (Utils module) and call the code from
    notebooks.
- **Markdowns**:
  - One markdown file linked to each python script, i.e, API and example.

For more guidance on this structure and the rationale behind it, see
[How to write the
Tutorial](https://github.com/causify-ai/tutorials/blob/master/docs/all.learn_X_in_60_minutes.how_to_guide.md)

In general

- For API: you are expected to describe the API, its architecture, etc.
- For Example: You are expected to use the project tool according to the
  specifications mentioned in the project description

## Submission

Your submission must include the following files:

**Important**: "API" here refers to the tool's internal interface—not an
external data‑provider API. Please keep the focus on the tool itself.

1. `XYZ.API.md`:
   - Document the native programming interface (classes, functions,
     configuration objects) of your chosen tool or library.
   - Describe the lightweight wrapper layer you have written on top of this
     native API.

2. `XYZ.API.ipynb`:
   - A Jupyter notebook demonstrating usage of the native API and your wrapper
     layer, with clean, minimal cells

3. `XYZ.example.md`:
   - A markdown file presenting a complete example of an application that uses
     your API layer

4. `XYZ.example.ipynb`:
   - A Jupyter notebook corresponding to the example above, demonstrating
     end-to-end functionality

5. `XYZ_utils.py`:
   - A Python module containing reusable utility functions and wrappers around
     the API
   - The notebooks should invoke logic from this file instead of embedding
     complex code inline

### Difference between `{project}.API.*` and `{project}.example.*`

- **`{project}.API.*`**: stable contract‑only layer. Holds dataclasses, enums,
  and abstract service interfaces so anyone can integrate without pulling in
  your runtime code.

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
  the API with real storage, I/O, and third‑party calls.

  ```python
  import sqlite3
  import bcrypt
  import jwt
  from project.API.auth import User, AuthService

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
```
COURSE_CODE/
└── Term20xx/
    └── projects/
        └── TutorTaskXX_Name_of_issue/
            ├── utils_data_io.py
            ├── utils_post_processing.py
            ├── API.ipynb
            ├── API.md
            ├── example.ipynb
            ├── example.md
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

- Docker setup:
  - Include clear instructions on how to build and run your Docker container
  - Mention expected terminal outputs when running scripts (e.g., starting
    Jupyter, mounting volumes, etc.) E.g.,

  ```md
  ### To Build the Image

  ''' <- triple backticks here bash docker_build.sh '''

  ### To Run the Container

  ''' bash docker_bash.sh '''
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
  - In case of issues, they will be flagged through GitHub issues, and you will
    be expected to resolve them in a timely manner

## Examples of a class project

The layout of each project should follow the examples in

- Example for
  [langchain tutorial](https://github.com/causify-ai/tutorials/tree/master/tutorial_langchain)
- Examples for
  [neo4j](https://github.com/causify-ai/tutorials/tree/master/tutorial_neo4j)
- Example for
  [open_ai tutorial](https://github.com/causify-ai/tutorials/tree/master/tutorial_openai)
- Example for
  [github tutorial (class_style)](https://github.com/causify-ai/tutorials/tree/master/tutorial_github_data605_style)
- Example for
  [github tutorial (causify_style)](https://github.com/causify-ai/tutorials/tree/master/tutorial_github_causify_style)

> Note that the tutorials from DATA605 class are built using a simpler approach
> for Docker and bash (e.g., `bash` scripts instead of Python code)
