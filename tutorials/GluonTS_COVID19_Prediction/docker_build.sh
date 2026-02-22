#!/bin/bash
# """
# Build a Docker container image for the project.
#
# This script sets up the build environment with error handling and command
# tracing, loads Docker configuration from docker_name.sh, and builds the
# Docker image using the build_container_image utility function. It supports
# both single-architecture and multi-architecture builds via the
# DOCKER_BUILD_MULTI_ARCH environment variable.
# """

# Exit immediately if any command exits with a non-zero status.
set -e

# Print each command to stdout before executing it.
set -x

# Import the utility functions.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/project_template/utils.sh

# Load Docker configuration variables (REPO_NAME, IMAGE_NAME, FULL_IMAGE_NAME).
get_docker_vars_script ${BASH_SOURCE[0]}
source $DOCKER_NAME
print_docker_vars

# Configure Docker build settings.
# Enable BuildKit for improved build performance and features.
export DOCKER_BUILDKIT=1
#export DOCKER_BUILDKIT=0

# Configure single-architecture build (set to 1 for multi-arch build).
#export DOCKER_BUILD_MULTI_ARCH=1
export DOCKER_BUILD_MULTI_ARCH=0

# Build the container image.
# Uncomment the line below to build without using Docker cache.
#build_container_image --no-cache
build_container_image
