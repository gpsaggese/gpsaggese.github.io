#!/bin/bash
# """
# Push Docker container image to Docker Hub or registry.
#
# This script authenticates with the Docker registry using credentials from
# ~/.docker/passwd.$REPO_NAME.txt and pushes the locally built container
# image to the remote repository.
# """

# Exit immediately if any command exits with a non-zero status.
set -e

# Import the utility functions.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/project_template/utils.sh

# Parse default args (-h, -v) and enable set -x if -v is passed.
parse_default_args "$@"

# Load Docker image naming configuration.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $SCRIPT_DIR/docker_name.sh

# Push the container image to the registry.
push_container_image
