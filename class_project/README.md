# Class Project Guidelines

- The goal of the class project (e.g., for `DATA605`, `MSML610`) is to learn a
  cutting-edge modern big data technology and write an example of a system using
  it

- There are two types of projects: research projects and tutorials

## Research projects
- Explore an open research question
- Teams of at most 3 students
- Examples are in
  - `research/ideas/README.md`
  - `research/ideas/*.md`

- Well done projects lead to a blog post or publication, but these are the most
  challenging class projects

## Tutorials "Learn X in 60 mins"

- Learn a technology and build a system using that specific technology
  - The result of each project is a "tutorial" that can teach a curious computer
    scientist a new technology in 60 minutes
- Individual or teams of at most 3 students
- Examples are in :
  - `class_project/project_descriptions/DATA605`
  - `class_project/project_descriptions/MSML610`
  - `class_project/all_projects.md`
- Each tutorial is similar in spirit to the tutorials for various technologies we
  have looked at and studied in classes
  - E.g., `Git`, `Docker`, `SQL`, `Mongo`, `Airflow`, `Dask`

- Well done projects become blog entries and potential publications

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
  - Search engines and AI tools are helpful, but don't abuse them: copy-pasting
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

## Pre-Requisites
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
- Only `Python` should be used, along with necessary configuration files for the
  specific tools

- Unless specified by project description, everything needs to run locally
  without using cloud resources
  - E.g., instead of using an `AWS` RDS instance, install `Postgres` in your
    Docker container for any database requirements

- If you did not take DATA605 or you are not familiar with the basic technology
  we rely on (e.g., `Git`, `Docker`, `Jupyter`, `Python`, `bash`), take time to
  get familiar with them through the DATA605 tutorials

## Contribution to the Repo
- You will work in the same way open-source developers contribute to a project
- Each class project will need to be organized like a proper open source
  project, including filing issues, opening PRs, and checking in code in the
  [umd_classes repository](https://github.com/gpsaggese/umd_classes)

- (Optional) You can use some of the tooling we use for interns and Causify
  - Set up your working environment by following the instructions in the
    [document](https://github.com/causify-ai/helpers/blob/master/docs/onboarding/intern.set_up_development_on_laptop.how_to_guide.md)

- Each step of the project is delivered by committing code to the dir
  corresponding to your project and doing a `GitHub` Pull Request (PR)
  - You should commit regularly and not just once at the end
  - We will specifically do reviews of intermediate results of the project and
    give you some feedback on what to improve (adopting `Agile` methodology)

### Project Tag Naming Convention
- Your project tag should follow this format:
  `{Class}_Spring{year}_{project_title_without_spaces}`
  - Example: if your project title is **"Redis cache to fetch user profiles"**
    for DATA605 Spring 2026, your project tag will be:
    `DATA605_Spring2026_Redis_cache_to_fetch_user_profiles`

### Create a GitHub Issue for Your Project
- Create a **GitHub issue** with your **project tag** as the title
  - Example: `DATA605_Spring2026_Redis_cache_to_fetch_user_profiles`

- Copy/paste the project description and add a link to the document with
  project specs
  - E.g.,
    https://github.com/gpsaggese/gpsaggese.github.io/blob/master/class_project/projects_descriptions/DATA605/ActiveCampaign_Project_Description.md

- Assign the GitHub issue to yourself (if you have permissions) or tag the
  issue with the individuals working on the project
  - E.g., `Author: @gpsaggese`
  - This issue will be used for project-related discussions

### Create a Local Branch to Work in
Create a new branch in your fork with the following naming convention:

- Name your `Git` branch as follows: `UmdTask{issue_number}_{project_tag}`
  - Example: If your issue number is `#645`, your branch name should be:
    `UmdTask645_DATA605_Spring2026_Redis_cache_to_fetch_user_profiles`

- **Steps to create the branch:**
  ```bash
  > cd $HOME/src
  > git clone --recursive git@github.com:gpsaggese/umd_classes.git umd_classes1
  > cd $HOME/src/umd_classes1
  > git checkout master
  > git checkout -b UmdTask645_DATA605_Spring2026_Redis_cache_to_fetch_user_profiles
  ```

### Add Files Only in Your Project Directory
- Add your project files under the following directory:
  `{GIT_ROOT}/class_project/{COURSE_CODE}/{TERM}{YEAR}/projects/{branch_name}`
  - Example for DATA605 Spring 2026:
    `~/src/umd_classes1/class_project/DATA605/Spring2026/projects/UmdTask645_DATA605_Spring2026_Redis_cache_to_fetch_user_profiles`
- **Important**: You should add files only under your project directory!
- Follow the instructions (carefully!) in
  `class_project/project_template/docker_scripts.README.md`
- Start working on the files

### Create a Pull Request (PR)
- Always create a **Pull Request (PR)** from your branch
- Name the PR the same as your project branch and reference the issue number
  your branch is based on
- Add your TAs and `@gpsaggese` as reviewers
- Assign the PR to yourself
- You should **not** be able to push directly to the `master` branch—only push
  commits to **your project branch**

- When making progress during the semester (e.g., when a PR is merged), use
  incremental branch names by appending `_1`, `_2` to your branch name, etc.
  - Example:
    - `UmdTask645_DATA605_Spring2026_Redis_cache_to_fetch_user_profiles_1`
    - `UmdTask645_DATA605_Spring2026_Redis_cache_to_fetch_user_profiles_2`

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

- The layout of each project should follow these reference examples in the
  `umd_classes` repository:
  - **[`tutorials/Autogen`](https://github.com/gpsaggese/umd_classes/tree/master/tutorials/Autogen)**
    - Contains: `docker_build.sh`, `autogen.API.ipynb`, `autogen.example.ipynb`,
      `autogen_utils.py`
    - Study this first for the recommended directory layout

  - **[`tutorials/tensorflow`](https://github.com/gpsaggese/umd_classes/tree/master/tutorials/tensorflow)**

  - **[`class_project/project_template`](https://github.com/gpsaggese/umd_classes/tree/master/class_project/project_template)**
    - The canonical starting point template for new projects
    - Contains blank `template.API.ipynb` and `template.example.ipynb` notebooks
    - Use this as a foundation for your project structure

- Review these exemplary projects from colleagues to understand what excellence
  looks like:
  - [`tutorial_langchain`](https://github.com/gpsaggese/umd_classes/tree/master/tutorials/tutorial_langchain)
  - [`tutorial_langgraph`](https://github.com/gpsaggese/umd_classes/tree/master/tutorials/tutorial_langgraph)

- **From DATA605 class projects:**
  - **Projects:**
    [https://github.com/gpsaggese/umd_classes/tree/master/class_project/DATA605](https://github.com/gpsaggese/umd_classes/tree/master/class_project/DATA605)
  - **Project videos:**
    [Google Drive folder](https://drive.google.com/drive/folders/1QLtgPCAS0mqE9cr1hE3UVoIbzakCNtaC)

- If you see any problem in the tutorials (code or video), please send an email
  to the instructor with your TAs in cc
- To make this more interesting, there might be (or maybe not!) some mistakes on
  purpose in the examples: see if you find them!

## Grading Rubric
- **All deliverables delivered (10 points)**
  - Are all required components submitted, including code, documentation, and
    any other specified materials
  - Does everything follow the project's standard structure and formatting
    guidelines
  - Is the submission complete enough for someone else to evaluate or run
    without requesting extra files or explanations

- **Working Docker (5 points)**
  - Does the Docker container build without errors, following the provided
    instructions
  - Does the project run successfully inside the container and behave as
    expected
  - Are all necessary environment configurations, ports, and dependencies
    correctly set

- **Documentation quality (5 points)**
  - Does the documentation clearly explain how to set up, run, and understand
    the project
  - Is it well-written, free from major grammar or formatting issues, and easy
    to follow
  - Does it include all required sections such as installation steps, usage
    examples, API descriptions, or architectural decisions
  - Does it adhere to the formatting and organizational guidelines we provided

- **Actual project complexity (5 points)**
  - How much depth and effort does the project demonstrate beyond the initial
    suggested scope
  - Did the student add significant features, handle edge cases, or show
    creative problem-solving
  - Is the architecture or technical implementation non-trivial or particularly
    well-considered

- **Code quality (5 points)**
  - Is the code clean, modular, and easy to understand?
  - Are comments and docstrings used appropriately to explain non-obvious logic
  - Is the overall structure of the codebase sensible and maintainable
  - Does it follow consistent style conventions (e.g., PEP 8 or another relevant
    standard)

- **PR quality (5 points)**
  - Was the pull request (PR) well-organized, with meaningful commit messages
    and a clear description
  - Did the student follow good GitHub practices, such as linking issues,
    keeping PRs scoped, and avoiding unnecessary noise
  - Was the PR free of obvious clutter like unrelated test code or temporary
    debug statements

- **Depth and understanding (5 points)**
  - Does the student show a solid understanding of the tools and techniques they
    used
  - Do the design decisions reflect thoughtful trade-offs and justification,
    rather than blindly following tutorials
  - Is there evidence that the student could explain and defend their
    implementation in a review

- **Late submission (-5 points)**
  - Was the submission turned in after the deadline without an approved
    extension

- **Incomplete work (-5 points)**
  - Are there major parts of the project that are missing or obviously broken
  - Does the submission fail to meet critical functional or structural
    expectations
  - Are there signs that the project was rushed or left unfinished
