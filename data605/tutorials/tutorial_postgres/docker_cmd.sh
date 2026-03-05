#!/bin/bash -e
# """
# Execute a command in a Docker container.
#
# This script runs a specified command inside a new Docker container instance.
# The container is removed automatically after the command completes. The
# current directory is mounted to /data inside the container.
# """

# Exit immediately if any command exits with a non-zero status.
set -e
#set -x

# Capture the command to execute from command-line arguments.
CMD="$@"
echo "Executing: '$CMD'"

# Import the utility functions.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/project_template/utils.sh

# Load Docker configuration variables for this script.
get_docker_vars_script ${BASH_SOURCE[0]}
source $DOCKER_NAME
print_docker_vars

# List available Docker images matching the expected image name.
run "docker image ls $FULL_IMAGE_NAME"
#(docker manifest inspect $FULL_IMAGE_NAME | grep arch) || true

# Configure and run the Docker container with interactive bash shell.
# - Container is removed automatically on exit (--rm)
# - Interactive mode with TTY allocation (-ti)
# - Port forwarding for Jupyter and PostgreSQL services
# - Current directory mounted to /data inside container
CONTAINER_NAME=${IMAGE_NAME}_cmd
PORT=8888
cmd="docker run --rm -ti \
    --name $CONTAINER_NAME \
    -p $PORT:$PORT \
    -p 5432:5432 \
    -v $(pwd):/data \
    -v $GIT_ROOT:/git_root \
    -e PYTHONPATH=/git_root:/git_root/helpers_root \
    $FULL_IMAGE_NAME \
    bash -c '$CMD'"
run $cmd
