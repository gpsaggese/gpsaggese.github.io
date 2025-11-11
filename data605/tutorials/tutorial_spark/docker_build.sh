#!/bin/bash -xe

GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/docker_common/utils.sh

REPO_NAME=gpsaggese
IMAGE_NAME=umd_data605_spark

# Build container.
#export DOCKER_BUILDKIT=1
export DOCKER_BUILDKIT=0
build_container_image
