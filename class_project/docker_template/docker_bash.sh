#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

# Print each command to stdout before executing it.
set -x

# Source Docker image naming configuration.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $SCRIPT_DIR/docker_name.sh

docker image ls $FULL_IMAGE_NAME

CONTAINER_NAME=$IMAGE_NAME
docker run --rm -ti \
    --name $CONTAINER_NAME \
    -p 8888:8888 -p 5432:5432 \
    -v $(pwd):/data \
    $FULL_IMAGE_NAME
