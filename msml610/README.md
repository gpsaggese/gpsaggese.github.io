# Summary

This directory contains course materials for MSML610, including lecture content,
tutorials, book chapters, and supporting infrastructure for building and
deploying course materials

# Directory Structure

## Course Content

- `lectures/`: Final compiled lecture PDFs organized by lesson number
  - Contains all lecture presentations in PDF format
  - Named as `Lesson##.#-Topic.pdf`

- `lectures_source/`: Source text files for lecture content
  - Contains raw text content for lectures
  - Includes `figures/` and `figures.hires/` directories with lecture diagrams
  - Includes `covers/` directory with lesson covers
  - Used as input for generating lecture PDFs

- `lectures_tex/`: LaTeX source files for lectures
  - Contains LaTeX markup for lecture presentations
  - Used for generating final PDF lecture materials

## Book and Publishing

- `book/`: Compiled book chapters and course book materials
  - Contains individual chapter PDFs with `.book_chapter.pdf` suffix
  - Contains text versions of chapters (`.book_chapter.md`)
  - Contains PNG image directories for each chapter

- `jupyter_book/`: Jupyter Book configuration and build files
  - Contains the Jupyter Book project setup for the course
  - Used for generating interactive HTML versions of course materials

## Tutorials

- `tutorials/`: Hands-on tutorial materials organized by topic
  - Contains Jupyter notebooks and Python scripts for each lesson
  - Named as `L##_##_topic.ipynb` and `L##_##_topic.py`
  - See `helpers/htutorial.py` at repository root for tutorial utility functions
  - Includes `notebook_template.ipynb` for creating new notebooks

## Infrastructure

- `devops/`: Development operations automation
  - `docker_build/`: Docker image build configuration
  - `docker_run/`: Docker container runtime scripts
  - `compose/`: Docker Compose multi-container configuration
  - `env/`: Environment variable configuration

## Supporting Materials

- `mats/`: Course materials and resources
  - Contains additional course-related documents and references

- `test/`: Test suite and validation scripts
  - Contains test files for course materials and utilities

## Archives

- `lectures_Fall2025/`: Archived lecture PDFs from Fall 2025 semester
  - Historical reference materials
