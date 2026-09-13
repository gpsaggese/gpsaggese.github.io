# Causal Success Tutorial

- This folder contains the setup for running the Causal Analysis of Success
  tutorials within a containerized environment.

## Quick Start

- From the root of the repository, change your directory to the causal_success
  tutorial folder:
  ```bash
  > cd tutorials/causal_success
  ```

- Once the location has been changed to the repo run the command to build the
  image to run dockers:
  ```bash
  > ./docker_build.sh
  ```

- Once the docker has been built you can then go ahead and run the container and
  launch jupyter notebook using the created image using the command:
  ```bash
  > ./docker_jupyter.sh
  ```

- Once the `./docker_jupyter.sh` script is running, follow this sequence to
  explore the tutorials:
  1. **`causal_success.API.ipynb`**: Start here to understand the building
     blocks — the Agent class, simulation engine, inequality metrics, policy
     simulation, and Bayesian inference functions.
  2. **`causal_success.example.ipynb`**: Run the full end-to-end analysis:
     simulation → DML → Causal Forests → policy comparison.

- Both notebooks run end-to-end in about 3 minutes on a laptop. All seeds are
  fixed, so you will get the same results every time.

- For more information on the Docker build system refer to [Project template
  readme](https://github.com/gpsaggese/umd_classes/blob/master/class_project/project_template/docker_scripts.README.md).
