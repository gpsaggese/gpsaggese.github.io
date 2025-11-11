#!/bin/bash -e

GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/docker_common/utils.sh

REPO_NAME=gpsaggese
IMAGE_NAME=umd_data605_postgres

# Build container.
export DOCKER_BUILDKIT=1
#export DOCKER_BUILDKIT=0

# Build container for multiple architectures.
#export DOCKER_BUILD_MULTI_ARCH=1
build_container_image
