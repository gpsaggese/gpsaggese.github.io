# CausalML Diabetes Study Tutorial

- This folder contains the setup for running CausalML tutorials within a
  containerized environment

## Quick Start

- From the root of the repository, change your directory to the CausalML
  tutorial folder:
  ```bash
  > cd tutorials/CausalML_Diabetes_Study
  ```

- Once the location has been changed to the repo run the command to build the
  image to run dockers:
  ```bash
  > ./docker_build.sh
  ```

- Download the CDC BRFSS Diabetes dataset from:
  https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
  - Place the downloaded file in `data/unprocessed/`
  - Note: the `data/unprocessed/` directory is tracked using `.gitkeep` but
    the dataset itself must be downloaded separately

- Once the docker has been built you can then go ahead and run the container
  and launch jupyter notebook using the created image using the command:
  ```bash
  > ./docker_jupyter.sh
  ```

- Once the `./docker_jupyter.sh` script is running, you can execute the
  tutorials

- For more informations on the Docker build system refer to [Project template
  readme](/class_project/project_template/docker_scripts.README.md)

## Tutorial Notebooks

Work through the following notebooks in order:

- [`CausalML.API.ipynb`](CausalML.API.ipynb): Tutorial notebook demonstrating
  the application of `CausalML` methods for causal effect estimation
  - Master the fundamental commands and basic configurations of the `CausalML`
    framework

- [`CausalML.example.ipynb`](CausalML.example.ipynb): Advanced end-to-end
  causal analysis example
  - Estimates the causal impact of lifestyle interventions (dietary
    modifications, structured exercise) on diabetes-related health outcomes
  - Uses the CDC BRFSS dataset with heterogeneous treatment effect estimation
    via meta-learners
  - Rigorously accounts for confounding variables to ensure credible and
    unbiased effect estimates

- [`utils.py`](utils.py): Utility functions for data loading, preprocessing,
  modeling, and evaluation required by the example notebooks

## Changelog

- 2026-03-18: Initial release
