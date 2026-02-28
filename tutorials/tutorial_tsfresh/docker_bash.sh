#!/bin/bash
# """
# Launch an interactive bash shell inside the tsfresh Docker container.
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

CONTAINER_NAME=${IMAGE_NAME}_bash
PORT=8889
cmd="docker run --rm -ti \
    --name $CONTAINER_NAME \
    -p $PORT:$PORT \
    -v $(pwd):/data \
    -v $GIT_ROOT:/git_root \
    -e PYTHONPATH=/git_root:/git_root/helpers_root \
    $FULL_IMAGE_NAME"
run $cmd
