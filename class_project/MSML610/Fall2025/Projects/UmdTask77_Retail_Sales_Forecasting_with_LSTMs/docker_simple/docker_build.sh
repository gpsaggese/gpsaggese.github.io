#!/bin/bash -e

GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/docker_common/utils.sh

REPO_NAME=msml610
IMAGE_NAME=retail_sales_forecasting_lstms

# Build container.
export DOCKER_BUILDKIT=1
#export DOCKER_BUILDKIT=0
HOST_ARCH=$(uname -m)
if [[ "$HOST_ARCH" == "arm64" || "$HOST_ARCH" == "aarch64" ]]; then
    echo "Detected ARM host ($HOST_ARCH); building linux/amd64 image (compatible with the pinned JAX wheel)."
    export DOCKER_DEFAULT_PLATFORM=linux/amd64
fi
build_container_image
