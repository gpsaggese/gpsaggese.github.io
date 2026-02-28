# DATA605, MSML610 - Class Project FAQs

# Summary

- This document provides frequently asked questions and guidance for completing
  class projects for DATA605 and MSML610
- It covers project selection, setup, development process, video recording
  requirements, and grading rubric

## Read Instructions for the Class Projects

Read the full instructions before proceeding:
[class_project/README.md](https://github.com/gpsaggese/umd_classes/blob/master/class_project/README.md)

## Review Some of the Best Projects From Previous Years

- [`tutorial_langchain`](https://github.com/causify-ai/tutorials/tree/master/tutorial_langchain)
- [`tutorial_langgraph`](https://github.com/causify-ai/tutorials/tree/master/tutorial_langgraph)
- [`tutorial_openai`](https://github.com/causify-ai/tutorials/tree/master/tutorial_openai)
- [`tutorial_prophet`](https://github.com/causify-ai/tutorials/tree/master/tutorial_prophet)

If you see any problem in the tutorials (code or video), pls send an email to me
with your TAs in cc. To make this more interesting, there might be (or maybe
not!) some mistakes on purpose. See if you find them. Happy egg hunt!

## Review Past Projects

Peruse/draw inspiration from lots of projects from your colleagues in the sister
class DATA605

- Projects:
  [https://github.com/gpsaggese/umd_classes/tree/master/class_project/DATA605](https://github.com/gpsaggese/umd_classes/tree/master/class_project/DATA605)

- Videos:
  [https://drive.google.com/drive/folders/1QLtgPCAS0mqE9cr1hE3UVoIbzakCNtaC](https://drive.google.com/drive/folders/1QLtgPCAS0mqE9cr1hE3UVoIbzakCNtaC)

## Pick Your Project

- See [README.md](https://github.com/gpsaggese/umd_classes/blob/master/class_project/README.md)
  for the full project types and selection rules

- **MSML610 sign-up links** (Fall 2025):
  - Project list:
    [spreadsheet](https://docs.google.com/spreadsheets/d/1H_Ev1psuPpUrrRcmBrBb2chfurSo5rPcAdd6i2SIUTQ/edit?gid=0#gid=0)
  - Sign-up form:
    [form](https://docs.google.com/forms/d/1TPCt7UFnTOEICltrPU3sIu9RoCbILR32zHbZNzRi9jw/edit)
  - You can pick two projects as a backup; conflicts are resolved at random

- For group projects, teams are up to 3 members
  - Team projects require a project rated "hard" difficulty
  - All members receive the same score — choose partners carefully

## Learn or Refresh Basic Data Science Tools

If you did not take DATA605 or you are not familiar with the basic technology we
rely on (e.g., `Git`, `Docker`), take time to get familiar with them through the
DATA605 tutorials:

- [Git tutorial](https://github.com/gpsaggese/umd_classes/tree/master/data605/tutorials/tutorial_git)
  ([video](https://drive.google.com/file/d/12TcMNs4vZoZUWq47TUgcqB875GMfqK01/view?usp=drive_link))
- [Jupyter tutorial](https://github.com/gpsaggese/umd_classes/tree/master/data605/tutorials/tutorial_jupyter)
- [GitHub tutorial](https://github.com/gpsaggese/umd_classes/tree/master/data605/tutorials/tutorial_github)
- [Docker tutorial](https://github.com/gpsaggese/umd_classes/tree/master/data605/tutorials/tutorial_docker)
  ([video](https://drive.google.com/file/d/1N9bq5ibN3oKmc0KF07lz9OSa4mbgsrzy/view?usp=drive_link))
- [Docker Compose tutorial](https://github.com/gpsaggese/umd_data605/tree/main/tutorials/tutorial_docker_compose)
  ([video](https://drive.google.com/file/d/1XbBPLwyGZ8-xpjnZ6bZRJNrvd9xhwwF2/view?usp=drive_link))

The project schedule is described in the "Class assignment" column in the
schedule at
https://docs.google.com/document/d/1YXCrqh6KGg3xm4-Lr4QGdBnjeWEfkqz67FHNHB_rdAk/edit?tab=t.0

In practice, the process is:

- Pick 1 or 2 projects
- Finalize the project and make sure everything is clear
- Work, work, work by yourself at home
- Work, work, work with us in class lab
- First checkpoint of the project
- Work, work, work by yourself at home
- Work, work, work with us in class lab
- Final submission

## Guidelines for Video Recording

The final project requires students to submit a video recording of their
project.

The goal is to learn how to present your work in a professional manner (which
will be extremely important in your career).

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

## A Note From Frank Underwood

If you are interested in working in our research team, the best way to be
noticed is with thoughtful questions, showing that you have thought about the
question and read the material.

Like Frank Underwood said: "Don't disappoint me. And more importantly, do not
disappoint me".

# Rubric for Grading

Here is the rubric we used to grade the projects.

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
  - Does it follow consistent style conventions (e.g., PEP 8 or another relevant
    standard)

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
