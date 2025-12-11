#!/bin/bash
set -e

# Defaults
JUPYTER_HOST_PORT=8888
TARGET_DIR=$(pwd)   # mount the current folder
VERBOSE=0

# Docker image and container names
IMAGE_NAME="umd_msml610_image"   # <- use your locally built image
CONTAINER_NAME="umd_msml610_jupyter"

# Parse optional parameters
while getopts p:d:v flag
do
    case "${flag}" in
        p) JUPYTER_HOST_PORT=${OPTARG};;
        d) TARGET_DIR=${OPTARG};;
        v) VERBOSE=1;;
    esac
done

# Optional verbosity
if [[ $VERBOSE == 1 ]]; then
    set -x
fi

# Docker run options
DOCKER_RUN_OPTS="-p $JUPYTER_HOST_PORT:$JUPYTER_HOST_PORT -v $TARGET_DIR:/data"

# Run Jupyter Notebook inside the container
docker run --rm -ti \
    --name $CONTAINER_NAME \
    $DOCKER_RUN_OPTS \
    $IMAGE_NAME \
    jupyter notebook --ip=0.0.0.0 --port=$JUPYTER_HOST_PORT --no-browser --allow-root
