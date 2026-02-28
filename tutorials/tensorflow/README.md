# TensorFlow Tutorial

Learn TensorFlow in 60 minutes: a hands-on introduction to deep learning with
TensorFlow and Keras covering neural networks, model training, and
probabilistic programming.

## Description of Files

- `Dockerfile`
  - Docker image build configuration with Ubuntu, Python, Jupyter, TensorFlow,
    and project dependencies

- `docker_name.sh`
  - Configuration file defining Docker repository and image naming variables

- `docker_build.sh`
  - Shell script for building the Docker container image using BuildKit

- `docker_bash.sh`
  - Shell script for launching an interactive bash shell inside the Docker
    container

- `docker_cmd.sh`
  - Shell script for executing arbitrary commands inside the Docker container

- `docker_exec.sh`
  - Shell script for attaching to an already running Docker container

- `docker_jupyter.sh`
  - Shell script for launching Jupyter Lab inside the Docker container

- `docker_push.sh`
  - Shell script for pushing the Docker image to Docker Hub

- `docker_clean.sh`
  - Shell script for removing Docker images matching the project configuration

- `run_jupyter.sh`
  - Shell script launched inside the container to start Jupyter Lab server

- `install_jupyter_extensions.sh`
  - Shell script for installing and configuring Jupyter notebook extensions

- `version.sh`
  - Shell script reporting Python, pip, Jupyter, and package version
    information

- `bashrc`
  - Bash configuration file enabling `vi` mode for command-line editing

- `etc_sudoers`
  - Sudoers configuration file granting passwordless sudo access

- `requirements.txt`
  - Python package dependencies for TensorFlow, scikit-learn, and data science
    libraries

## Workflows

- All commands should be run from inside the project directory
  ```bash
  > cd tutorials/tutorial_tensorflow
  ```

- To build the container
  ```bash
  > ./docker_build.sh
  ```

- To test the container
  ```bash
  > ./docker_bash.sh ls
  ```

- To start Jupyter
  ```bash
  > ./docker_jupyter.sh
  # Go to localhost:8888
  ```

- To start Jupyter on a specific port with vim support
  ```bash
  > ./docker_jupyter.sh -p 8890 -u
  # Go to localhost:8890
  ```

- To run a command inside the container
  ```bash
  > ./docker_cmd.sh python script.py
  ```

## Description of Executables

### `docker_build.sh`

- Builds Docker container image using Docker BuildKit
- Supports single-architecture builds (default) or multi-architecture builds

- Build container image for current architecture:
  ```bash
  > ./docker_build.sh
  ```

### `docker_bash.sh`

- Launches an interactive bash shell inside the Docker container
- Mounts the current working directory as `/data` inside the container
- Exposes port 8889 for potential services

- Launch bash shell in the container:
  ```bash
  > ./docker_bash.sh
  ```

### `docker_cmd.sh`

- Executes arbitrary commands inside the Docker container
- Mounts current directory as `/data` for accessing project files
- Removes container automatically after command execution

- Run Python script inside container:
  ```bash
  > ./docker_cmd.sh python script.py
  ```

- Run tests inside container:
  ```bash
  > ./docker_cmd.sh pytest tests/
  ```

### `docker_exec.sh`

- Attaches to an already running Docker container with an interactive bash shell
- Finds the container ID automatically based on the image name

- Attach to running container:
  ```bash
  > ./docker_exec.sh
  ```

### `docker_jupyter.sh`

- Launches Jupyter Lab server inside the Docker container
- Supports custom port configuration (default 8888), vim keybindings, and
  custom directory mounting

- Start Jupyter on default port 8888:
  ```bash
  > ./docker_jupyter.sh
  ```

- Start Jupyter on custom port with vim bindings:
  ```bash
  > ./docker_jupyter.sh -p 8889 -u
  ```

### `docker_push.sh`

- Authenticates to Docker registry and pushes the project Docker image

- Push container image to registry:
  ```bash
  > ./docker_push.sh
  ```

### `docker_clean.sh`

- Removes all Docker images matching the project's full image name

- Remove project's Docker images:
  ```bash
  > ./docker_clean.sh
  ```

### `version.sh`

- Reports version information for Python3, pip3, and Jupyter
- Lists all installed Python packages with versions

- Display version information:
  ```bash
  > ./version.sh
  ```
