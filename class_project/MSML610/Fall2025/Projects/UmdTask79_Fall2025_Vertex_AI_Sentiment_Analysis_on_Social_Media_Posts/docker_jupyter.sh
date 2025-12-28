#!/bin/bash -xe

# Docker container name configuration
source $(dirname "$0")/docker_name.sh

docker image ls $FULL_IMAGE_NAME

CONTAINER_NAME=$IMAGE_NAME
docker run --rm -ti \
    --name $CONTAINER_NAME \
    -p 8888:8888 \
    -v $(pwd):/data \
    -w /data \
    $FULL_IMAGE_NAME \
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
