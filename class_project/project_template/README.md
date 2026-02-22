# Project Template
This directory contains a Docker-based development environment template with
utility scripts, configuration files, and Jupyter notebook templates for
standardizing projects.

## Description of Files
- `bashrc`
  - Bash configuration file enabling `vi` mode for command-line editing

- `copy_docker_files.py`
  - Python script for copying Docker configuration files to destination
    directories

- `docker_build.version.log`
  - Log file containing Python, `pip`, Jupyter, and package version information
    from Docker build

- `docker_cmd.sh`
  - Shell script for executing arbitrary commands inside Docker containers with
    volume mounting

- `docker_jupyter.sh`
  - Shell script for launching Jupyter Lab server inside Docker containers

- `docker_name.sh`
  - Configuration file defining Docker repository and image naming variables

- `Dockerfile`
  - Docker image build configuration with Ubuntu, Python, Jupyter, and project
    dependencies

- `etc_sudoers`
  - Sudoers configuration file granting passwordless sudo access for postgres
    user

- `README.md`
  - Documentation file describing directory contents, files, and executable
    scripts

- `template_utils.py`
  - Python utility functions supporting tutorial notebooks with data processing
    and modeling helpers

- `template.API.ipynb`
  - Jupyter notebook template for API exploration and library usage examples

- `template.example.ipynb`
  - Jupyter notebook template for project examples and demonstrations

- `utils.sh`
  - Bash utility library with reusable functions for Docker operations

## Workflows

- All commands should be run from inside the project directory
  ```bash
  > cd tutorials/FilterPy
  ...
  ```

- To build the container for a project
  ```
  > cd $PROJECT
  # Build the container.
  > docker_build.sh
  # Test the container.
  > docker_bash.sh ls
  ```

- Start Jupyter
  ```bash
  > docker_jupyter.sh
  # Go to localhost:8888
  ```

- Start Jupyter on a specific port with vim support
  ```bash
  > docker_jupyter.sh -p 8890 -u
  # Go to localhost:8890
  ```

## How to customize a project template

- Copy the
  ```
  > cp -r class_project/project_template $TARGET
  ```
- TODO(gp): Complete

## Description of Executables

### `copy_docker_files.py`
- **What It Does**

- Copies Docker configuration and utility files from project_template to a
  destination directory
- Preserves all file permissions and attributes during copying
- Creates destination directory if it doesn't exist

- Copy all Docker files to a target directory:
  ```bash
  > ./copy_docker_files.py --dst_dir /path/to/destination
  ```

- Copy with verbose logging:
  ```bash
  > ./copy_docker_files.py --dst_dir /path/to/destination -v DEBUG
  ```

### `docker_bash.sh`
- **What It Does**
  - Launches an interactive bash shell inside a Docker container
  - Mounts the current working directory as `/data` inside the container
  - Exposes port 8889 for potential services running in the container

- Launch bash shell in the container:
  ```bash
  > ./docker_bash.sh
  ```

### `docker_build.sh`
- **What It Does**
  - Builds Docker container images using Docker BuildKit
  - Supports single-architecture builds (default) or multi-architecture builds
    (`linux/arm64`, `linux/amd64`)
  - Copies project files to temporary build directory and generates build logs

- Build container image for current architecture:
  ```bash
  > ./docker_build.sh
  ```

- Build multi-architecture image (requires setting `DOCKER_BUILD_MULTI_ARCH=1` in
  the script):
  ```bash
  > # Edit docker_build.sh to set DOCKER_BUILD_MULTI_ARCH=1
  > ./docker_build.sh
  ```

### `docker_clean.sh`
- **What It Does**

- Removes all Docker images matching the project's full image name
- Lists images before and after removal for verification
- Uses force removal to ensure cleanup completes

- Remove project's Docker images:
  ```bash
  > ./docker_clean.sh
  ```

### `docker_cmd.sh`
- **What It Does**
  - Executes arbitrary commands inside a Docker container
  - Mounts current directory as `/data` for accessing project files
  - Automatically removes container after command execution completes

- Run Python script inside container:
  ```bash
  > ./docker_cmd.sh python script.py --arg value
  ```

- List files in the container:
  ```bash
  > ./docker_cmd.sh ls -la /data
  ```

- Run tests inside container:
  ```bash
  > ./docker_cmd.sh pytest tests/
  ```

### `docker_exec.sh`
- **What It Does**
  - Attaches to an already running Docker container with an interactive bash
    shell
  - Finds the container ID automatically based on the image name
  - Useful for debugging or inspecting running containers

- Attach to running container:
  ```bash
  > ./docker_exec.sh
  ```

### `docker_jupyter.sh`
- **What It Does**
  - Launches Jupyter Lab server inside a Docker container
  - Supports custom port configuration (default 8888), vim keybindings, and
    custom directory mounting
  - Runs `run_jupyter.sh` script inside the container with specified options

- Start Jupyter on default port 8888:
  ```bash
  > ./docker_jupyter.sh
  ```

- Start Jupyter on custom port with vim bindings:
  ```bash
  > ./docker_jupyter.sh -p 8889 -u
  ```

- Start Jupyter with external directory mounted:
  ```bash
  > ./docker_jupyter.sh -d /path/to/notebooks -p 8889
  ```

- Start Jupyter in verbose mode:
  ```bash
  > ./docker_jupyter.sh -v -p 8890
  ```

### `docker_push.sh`
- **What It Does**
  - Authenticates to Docker registry using credentials from
    `~/.docker/passwd.$REPO_NAME.txt`
  - Pushes the project's Docker image to the remote repository
  - Lists images before pushing for verification

- Push container image to registry:
  ```bash
  > ./docker_push.sh
  ```

### `install_jupyter_extensions.sh`
- **What It Does**
  - Installs Jupyter contrib nbextensions and enables productivity extensions
  - Configures extensions including autosave, code prettify, collapsible
    headings, execute time, and vim bindings
  - Creates Jupyter configuration files and data directories

- Install and configure Jupyter extensions:
  ```bash
  > ./install_jupyter_extensions.sh
  ```

### `run_jupyter.sh`
- **What It Does**
  - Launches Jupyter Lab server with no authentication (token and password
    disabled)
  - Binds to all network interfaces (0.0.0.0) on port 8888
  - Allows root access for container environments

- Start Jupyter Lab server (typically called from docker_jupyter.sh):
  ```bash
  > ./run_jupyter.sh
  ```

### `utils.sh`
- **What It Does**
  - Provides reusable bash functions for Docker operations (not meant to be
    executed directly)
  - Includes functions for building, removing, pushing, pulling images, and
    managing containers
  - Sources configuration from `docker_name.sh` to set repository and image
    names

- This is a utility library sourced by other scripts:
  ```bash
  > source utils.sh
  > build_container_image
  ```

### `version.sh`
- **What It Does**
  - Reports version information for Python3, pip3, and Jupyter
  - Lists all installed Python packages with versions
  - Used during Docker image builds to log environment configuration

- Display version information:
  ```bash
  > ./version.sh
  ```

- Save version information to a log file:
  ```bash
  > ./version.sh 2>&1 | tee version.log
  ```
