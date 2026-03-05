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

# Configure and run the Docker container with the specified command.
DOCKER_RUN_OPTS=""
CONTAINER_NAME=$IMAGE_NAME
run "docker run \
    --rm -ti \
    --name $CONTAINER_NAME \
    $DOCKER_RUN_OPTS \
    -v $(pwd):/data \
    $FULL_IMAGE_NAME \
    bash -c '$CMD'"
