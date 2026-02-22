#!/bin/bash
# """
# This script launches a Docker container with an interactive bash shell for
# development.
# """

# Exit immediately if any command exits with a non-zero status.
set -e

# Print each command to stdout before executing it.
set -x

# Import the utility functions from the project template.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/project_template/utils.sh

# Load Docker configuration variables for this script.
get_docker_vars_script ${BASH_SOURCE[0]}
source $DOCKER_NAME
print_docker_vars

# List the available Docker images matching the expected image name.
run "docker image ls $FULL_IMAGE_NAME"

# Configure and run the Docker container with interactive bash shell.
# - Container is removed automatically on exit (--rm)
# - Interactive mode with TTY allocation (-ti)
# - Port forwarding for Jupyter (8888)
# - Current directory mounted to /data inside container
# - MPS fallback enabled for Apple Silicon
CONTAINER_NAME=${IMAGE_NAME}_bash
PORT=8888
cmd="docker run --rm -ti \
    --name $CONTAINER_NAME \
    -p $PORT:$PORT \
    -e PYTORCH_ENABLE_MPS_FALLBACK=1 \
    -v $(pwd):/data \
    $FULL_IMAGE_NAME"
run $cmd
