# Summary

This directory contains course materials for DATA605, including lecture content,
tutorials, book chapters, and supporting infrastructure for building and
deploying course materials

# Directory Structure

## Course Content

- `lectures/`: Final compiled lecture PDFs organized by lesson number
  - Contains all lecture presentations in PDF format
  - Named as `Lesson##.#-Topic.pdf`

- `lectures_source/`: Source text files for lecture content
  - Contains raw text content for lectures
  - Includes images directory with lecture diagrams and figures
  - Used as input for generating lecture PDFs

- `lectures_script/`: Video presentation scripts for lectures
  - Contains script files with speaking notes for each lesson
  - Named as `Lesson##.#-Topic.script.txt`

- `lectures_quizzes.secret/`: Quiz materials for the course
  - Private directory containing quiz content

- `lectures_recap.secret/`: Lecture recap materials
  - Private directory containing recap summaries

## Book and Publishing

- `book/`: Compiled book chapters and full course book
  - Contains individual chapter PDFs
  - Includes full `book.pdf` combining all chapters
  - Contains PNG image directories for each chapter

## Tutorials

- `tutorials/`: Hands-on tutorial materials organized by topic
  - `tutorial_airflow/`: Apache Airflow workflow tutorials
  - `tutorial_dask/`: Dask parallel computing tutorials
  - `tutorial_docker/`: Docker containerization tutorials
  - `tutorial_docker_compose/`: Docker Compose multi-container tutorials
  - `tutorial_git/`: Git version control tutorials
  - `tutorial_github/`: GitHub platform tutorials
  - `tutorial_jupyter/`: Jupyter notebook tutorials
  - `tutorial_mongodb/`: MongoDB database tutorials
  - `tutorial_pandas/`: Pandas data analysis tutorials
  - `tutorial_parquet/`: Parquet file format tutorials
  - `tutorial_postgres/`: PostgreSQL database tutorials
  - `tutorial_spark/`: Apache Spark tutorials

## Infrastructure

- `dev_scripts/`: Development automation scripts
  - Contains Docker management scripts
  - Includes formatting and linting utilities
  - Contains Jupyter and tmux setup scripts

- `docker_common/`: Shared Docker configuration files
  - Contains common Docker setup scripts
  - Includes bashrc and sudoers configuration
  - Contains Jupyter extension installation scripts

- `gp/`: Group project database materials
  - Contains SQL dump file for project database
  - Includes README with group project instructions

## Archives

- `lectures_Spring2023/`: Archived lecture PDFs from Spring 2023 semester
  - Historical reference materials

- `lectures_Spring2025/`: Archived lecture PDFs from Spring 2025 semester
  - Historical reference materials
