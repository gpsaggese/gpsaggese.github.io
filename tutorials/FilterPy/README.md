# Tutorial Template

This directory provides a template for creating Docker-based tutorial
environments.

## Structure of the Dir

This directory contains no subdirectories.

## Description of Files

- `Dockerfile`
  - Ubuntu 20.04-based image with Python 3, Jupyter, PostgreSQL client, and Jupyter extensions configured
- `bashrc`
  - Shell configuration enabling vi mode for command line editing
- `etc_sudoers`
  - Sudoers configuration granting NOPASSWD sudo access to postgres user
- `docker_build.version.log`
  - Build log capturing installed package versions from container build process

## Description of Executables

### `docker_bash.sh`

#### What It Does

- Launches an interactive Docker container with the tutorial environment
- Mounts the current directory to `/data` in the container
- Exposes ports 8888 for Jupyter and 5432 for PostgreSQL

#### Examples

- Run the tutorial environment interactively:
  ```bash
  > ./docker_bash.sh
  ```

### `docker_build.sh`

#### What It Does

- Builds the Docker image for the tutorial environment
- Uses `gpsaggese/umd_data605_XYZ` as the image name
- Sources common Docker utilities from `docker_common/utils.sh` in the git root

#### Examples

- Build the Docker image:
  ```bash
  > ./docker_build.sh
  ```

### `docker_clean.sh`

#### What It Does

- Removes the Docker image from the local system
- Calls the `remove_container_image` utility function to clean up

#### Examples

- Remove the tutorial Docker image:
  ```bash
  > ./docker_clean.sh
  ```

### `docker_exec.sh`

#### What It Does

- Executes a command in a running container or starts a new shell session
- Uses the `exec_container` utility function for container interaction

#### Examples

- Open a shell in the running container:
  ```bash
  > ./docker_exec.sh
  ```

### `docker_push.sh`

#### What It Does

- Pushes the built Docker image to Docker Hub
- Uploads to the `gpsaggese` repository

#### Examples

- Push the image to Docker Hub:
  ```bash
  > ./docker_push.sh
  ```

### `install_jupyter_extensions.sh`

#### What It Does

- Installs and enables Jupyter notebook extensions for enhanced functionality
- Configures extensions including code prettify, TOC, execute time, spell checker, and vim bindings
- Creates necessary Jupyter data directories and configuration files

#### Examples

- Install Jupyter extensions (typically run during Docker image build):
  ```bash
  > ./install_jupyter_extensions.sh
  ```

### `run_jupyter.sh`

#### What It Does

- Starts Jupyter Notebook server on port 8888
- Configures server to accept connections from any IP without authentication
- Allows root user access (suitable for containerized environments)

#### Examples

- Start Jupyter Notebook server:
  ```bash
  > ./run_jupyter.sh
  ```

### `version.sh`

#### What It Does

- Reports versions of all installed software and packages
- Displays Python 3, pip3, Jupyter, MongoDB, and Python package versions
- Useful for debugging and documenting the environment

#### Examples

- Display all installed package versions:
  ```bash
  > ./version.sh
  ```
