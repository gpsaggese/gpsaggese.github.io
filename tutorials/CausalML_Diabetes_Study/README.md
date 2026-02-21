<!-- toc -->

- [Summary](#summary)
- [Measuring the Impact of Lifestyle Programs on Diabetes Outcomes](#measuring-the-impact-of-lifestyle-programs-on-diabetes-outcomes)
  * [Overview](#overview)
    + [Project Objective](#project-objective)
  * [Installation and Docker Setup](#installation-and-docker-setup)
    + [1. Build the Image](#1-build-the-image)
    + [2. Download the Dataset](#2-download-the-dataset)
    + [3. Run the Container](#3-run-the-container)
    + [4. Access the Project](#4-access-the-project)
  * [Folder Structure](#folder-structure)
  * [Resources](#resources)

<!-- tocstop -->

# Summary

This project estimates the causal impact of lifestyle interventions on diabetes
outcomes using the `CausalML` library and `CDC BRFSS` dataset. The analysis
demonstrates heterogeneous treatment effect estimation with meta-learners,
provides `Docker`-based reproducibility, and includes comprehensive
documentation for both the API and example implementation.

# Measuring the Impact of Lifestyle Programs on Diabetes Outcomes

## Overview

### Project Objective

To estimate the causal impact of lifestyle interventions such as dietary
modifications and structured exercise programs on diabetes-related health
outcomes (for example, `HbA1c` levels and disease progression), while rigorously
accounting for confounding variables to ensure credible and unbiased effect
estimates

## Installation and Docker Setup

To ensure reproducibility, this project is containerized. Follow these steps to
build and run the analysis

### 1. Build the Image

- Run this command in the project root (where the `Dockerfile` is located):
  ```bash
  > ./docker_build.sh
  ```

### 2. Download the Dataset

- Download the CDC BRFSS Diabetes dataset from:
  https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- Place the downloaded file in:

  data/unprocessed/

> Note: The `data/unprocessed/` directory is tracked using `.gitkeep`
  but the dataset itself must be downloaded separately.

### 3. Run the Container

- Start the `Jupyter` environment with volume mounting (to save your notebook
  changes):
  ```bash
  > ./docker_jupyter.sh
  ```

### 4. Access the Project

- Click the `http://127.0.0.1:8888...` link in your terminal to open
  `JupyterLab`
- Open `CausalML.API.ipynb` to test the tool
- Open `CausalML.example.ipynb` to see the full Diabetes analysis

## Folder Structure

- `data/`: Contains datasets used for analysis  
  - `unprocessed/`: Location where the raw CDC BRFSS dataset should be placed (tracked via `.gitkeep`)  
- `CausalML.API.ipynb`: Jupyter notebook demonstrating the application of `CausalML` methods for causal effect estimation  
- `CausalML.API.md`: Documentation explaining the API interface used in the project  
- `CausalML.API.py`: Script version of the API notebook  
- `CausalML.example.ipynb`: End-to-end notebook performing the diabetes causal analysis  
- `CausalML.example.md`: Documentation corresponding to the example notebook  
- `CausalML.example.py`: Script version of the example notebook  
- `blog_CausalML.md`: Blog-style narrative explanation of the project and results  
- `utils.py`: Utility functions for data loading, preprocessing, modeling, and evaluation  
- `Dockerfile`: Docker configuration defining the reproducible runtime environment  
- `docker_build.sh`: Builds the Docker image  
- `docker_jupyter.sh`: Launches Jupyter inside the Docker container with volume mounting  
- `docker_bash.sh`: Opens an interactive shell inside the Docker container  
- `docker_exec.sh`: Executes commands in a running container  
- `docker_cmd.sh`: Runs a one-off command inside a container  
- `docker_clean.sh`: Removes the Docker image and related artifacts  
- `docker_push.sh`: Pushes the Docker image to a registry  
- `docker_name.sh`: Defines image naming variables used by Docker scripts  
- `docker_build.version.log`: Records version information from the Docker build  
- `run_jupyter.sh`: Starts Jupyter Notebook inside the container  
- `install_jupyter_extensions.sh`: Installs Jupyter extensions inside the container  
- `bashrc`: Shell configuration used inside the container  
- `etc_sudoers`: Sudo configuration file used during image build  
- `pyproject.toml`: Project configuration and dependency definitions (used by `uv`)  
- `uv.lock`: Locked dependency versions for reproducible builds  
- `requirements.txt`: Alternative dependency specification file  
- `version.sh`: Prints version information for Python and installed packages  
- `README.md`: Project overview and setup instructions  

## Resources

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

- [`CausalML` Library Documentation](https://causalml.readthedocs.io/en/latest/)

- [`MSML610` Advanced Machine Learning - Fall 2025](https://github.com/gpsaggese-org/umd_classes)
