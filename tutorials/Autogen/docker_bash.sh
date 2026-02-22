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
# - Port forwarding for Jupyter or other services
# - Current directory mounted to /data inside container
CONTAINER_NAME=${IMAGE_NAME}_bash
PORT=8889
cmd="docker run --rm -ti \
    --name $CONTAINER_NAME \
    -p $PORT:$PORT \
    -v $(pwd):/data \
    -v $GIT_ROOT:/git_root \
    -e PYTHONPATH=/git_root:/git_root/helpers_root \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    $FULL_IMAGE_NAME"
run $cmd
