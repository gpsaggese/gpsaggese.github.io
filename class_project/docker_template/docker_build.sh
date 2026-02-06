#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

# Print each command to stdout before executing it.
set -x

GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/docker_common/utils.sh

# Source Docker image naming configuration.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $SCRIPT_DIR/docker_name.sh

# Build container.
#export DOCKER_BUILDKIT=1
export DOCKER_BUILDKIT=0
build_container_image
