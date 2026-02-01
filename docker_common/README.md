# docker_common

This directory contains shared Docker utility scripts and configuration files for
building, running, and managing Docker containers across different projects.

## Structure of the Dir

This directory has no subdirectories - all files are at the root level.

## Description of Files

- `bashrc`
  - Bash configuration file enabling vi mode for command-line editing

- `docker_bash.sh`
  - Launches an interactive bash shell inside a Docker container with volume
    mounting

- `docker_build.sh`
  - Builds Docker container images with support for single or multi-architecture
    builds

- `docker_clean.sh`
  - Removes Docker container images matching the project's image name

- `docker_cmd.sh`
  - Executes arbitrary commands inside a Docker container with volume mounting

- `docker_exec.sh`
  - Attaches to an already running Docker container with an interactive bash
    shell

- `docker_jupyter.sh`
  - Launches Jupyter notebook server inside a Docker container with customizable
    port and directory

- `docker_push.sh`
  - Pushes Docker container images to a remote Docker registry

- `etc_sudoers`
  - Sudoers configuration file granting passwordless sudo access for postgres
    user

- `install_jupyter_extensions.sh`
  - Installs and enables Jupyter notebook extensions including vim bindings and
    productivity tools

- `utils.sh`
  - Bash utility functions for Docker operations including build, push, pull,
    and container management

## Description of Executables

### `docker_bash.sh`

#### What It Does

- Launches an interactive bash shell inside a Docker container
- Mounts the current working directory as `/data` inside the container
- Exposes port 8889 for potential services running in the container

#### Examples

- Launch bash shell in the container:
  ```bash
  > ./docker_bash.sh
  ```

### `docker_build.sh`

#### What It Does

- Builds Docker container images using Docker BuildKit
- Supports single-architecture builds (default) or multi-architecture builds
  (linux/arm64, linux/amd64)
- Copies project files to temporary build directory and generates build logs

#### Examples

- Build container image for current architecture:
  ```bash
  > ./docker_build.sh
  ```

- Build multi-architecture image (requires setting DOCKER_BUILD_MULTI_ARCH=1 in
  the script):
  ```bash
  > # Edit docker_build.sh to set DOCKER_BUILD_MULTI_ARCH=1
  > ./docker_build.sh
  ```

### `docker_clean.sh`

#### What It Does

- Removes all Docker images matching the project's full image name
- Lists images before and after removal for verification
- Uses force removal to ensure cleanup completes

#### Examples

- Remove project's Docker images:
  ```bash
  > ./docker_clean.sh
  ```

### `docker_cmd.sh`

#### What It Does

- Executes arbitrary commands inside a Docker container
- Mounts current directory as `/data` for accessing project files
- Automatically removes container after command execution completes

#### Examples

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

#### What It Does

- Attaches to an already running Docker container with an interactive bash shell
- Finds the container ID automatically based on the image name
- Useful for debugging or inspecting running containers

#### Examples

- Attach to running container:
  ```bash
  > ./docker_exec.sh
  ```

### `docker_jupyter.sh`

#### What It Does

- Launches Jupyter notebook server inside a Docker container
- Supports custom port configuration (default 8888), vim keybindings, and custom
  directory mounting
- Runs `run_jupyter.sh` script inside the container with specified options

#### Examples

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

#### What It Does

- Authenticates to Docker registry using credentials from
  `~/.docker/passwd.$REPO_NAME.txt`
- Pushes the project's Docker image to the remote repository
- Lists images before pushing for verification

#### Examples

- Push container image to registry:
  ```bash
  > ./docker_push.sh
  ```

### `install_jupyter_extensions.sh`

#### What It Does

- Installs Jupyter contrib nbextensions and enables productivity extensions
- Configures extensions including autosave, code prettify, collapsible headings,
  execute time, and vim bindings
- Creates Jupyter configuration files and data directories

#### Examples

- Install and configure Jupyter extensions:
  ```bash
  > ./install_jupyter_extensions.sh
  ```

### `utils.sh`

#### What It Does

- Provides reusable bash functions for Docker operations (not meant to be
  executed directly)
- Includes functions for building, removing, pushing, pulling images, and
  managing containers
- Sources configuration from `docker_name.sh` to set repository and image names

#### Examples

- This is a utility library sourced by other scripts:
  ```bash
  > source utils.sh
  > build_container_image
  ```
