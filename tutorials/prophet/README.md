# Prophet Tutorial

- This folder contains the setup for running Prophet time series forecasting
  tutorials within a containerized environment

## Quick Start

- From the root of the repository, change your directory to the Prophet tutorial
  folder:
  ```bash
  > cd tutorials/tutorial_prophet
  ```

- Build the Docker image:
  ```bash
  > ./docker_build.sh
  ```

- Launch Jupyter Lab inside the container:
  ```bash
  > ./docker_jupyter.sh
  ```

- Once `./docker_jupyter.sh` is running, follow this sequence to explore the
  tutorials:
  1. **`prophet.API.ipynb`**: Start here to learn the core Prophet API —
     basic usage, trend types, seasonality, holidays, external regressors,
     and cross-validation with small self-contained examples.
  2. **`prophet.example.ipynb`**: Proceed to this notebook for a complete
     end-to-end forecasting workflow on a synthetic daily time series with
     controlled trend, seasonality, holiday, and autoregressive components.

- For more information on the Docker build system refer to the [Project
  template README](https://github.com/gpsaggese/umd_classes/blob/master/class_project/project_template/README.md)

## Changelog

- 2026-03-01: Initial release
