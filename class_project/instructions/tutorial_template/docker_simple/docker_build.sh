#!/bin/bash -e

GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/docker_common/utils.sh

REPO_NAME=umd_msml610
IMAGE_NAME=umd_msml610_image

# Build container.
export DOCKER_BUILDKIT=1
#export DOCKER_BUILDKIT=0
build_container_image
