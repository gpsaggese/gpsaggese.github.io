#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

# Print each command to stdout before executing it.
set -x

# Import the utility functions.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/project_template/utils.sh

# Execute the script setting the vars for this tutorial.
get_docker_vars_script ${BASH_SOURCE[0]}
source $DOCKER_NAME
print_docker_vars

# Build container.
export DOCKER_BUILDKIT=1
#export DOCKER_BUILDKIT=0

#export DOCKER_BUILD_MULTI_ARCH=1
export DOCKER_BUILD_MULTI_ARCH=0
build_container_image
