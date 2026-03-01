#!/bin/bash
# """
# Build the Docker container image for the tsfresh tutorial.
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

export DOCKER_BUILDKIT=1
export DOCKER_BUILD_MULTI_ARCH=0

# Build the container image (add --no-cache to force a fresh build).
build_container_image
