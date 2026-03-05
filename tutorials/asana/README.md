# Asana Tutorial

- A self-contained tutorial for the Asana REST API and a Python analytics
  wrapper that aggregates user-activity metrics across projects.

## Quick Start

- Change directory to the tutorial folder:
  ```bash
  > cd tutorials/tutorial_asana
  ```

- Build the Docker image:
  ```bash
  > ./docker_build.sh
  ```

- Launch Jupyter Lab:
  ```bash
  > ASANA_ACCESS_TOKEN=<your_pat> ASANA_PROJECT_GID=<project_gid> ./docker_jupyter.sh
  ```

- Open the notebooks in order:
  1. **`asana.API.ipynb`** — Learn the core Asana API concepts (authentication,
     projects, tasks, stories) with minimal runnable examples.
  2. **`asana.example.ipynb`** — End-to-end user-activity analytics: fetch
     tasks and comments, compute per-user statistics, and display results.

- For more information on the Docker build system refer to the
  [Project template README](https://github.com/gpsaggese/umd_classes/blob/master/class_project/project_template/README.md).
