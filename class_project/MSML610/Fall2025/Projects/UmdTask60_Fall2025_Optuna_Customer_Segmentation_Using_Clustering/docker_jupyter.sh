#!/bin/bash
#
# Execute run_jupyter.sh in the container.
# 
# Usage:
# > docker_jupyter.sh -d /Users/saggese/src/git_gp1/code/book.2018.Martin.Bayesian_Analysis_with_Python.2e -v -u -p 8889
#

set -e
#set -x

# Parse params.
export JUPYTER_HOST_PORT=8888
export JUPYTER_USE_VIM=0
export TARGET_DIR=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# FIX: Only set TARGET_DIR if the path exists, otherwise leave empty
FALLBACK_PATH="$SCRIPT_DIR/../../../MSML610/Fall2025/Projects"
if [ -d "$FALLBACK_PATH" ]; then
    TARGET_DIR="$(realpath "$FALLBACK_PATH")"
else
    TARGET_DIR=""
fi

export VERBOSE=0

OLD_CMD_OPTS=$@
while getopts p:d:uv flag
do
    case "${flag}" in
        p) JUPYTER_HOST_PORT=${OPTARG};;
        u) JUPYTER_USE_VIM=1;;
        d) TARGET_DIR=${OPTARG};;
        # /Users/saggese/src/git_gp1/code/
        v) VERBOSE=1;;
    esac
done

if [[ $VERBOSE == 1 ]]; then
    set -x
fi;

# Define Docker variables directly (replaces docker_name.sh)
export IMAGE_NAME="optuna_jupyter"
export FULL_IMAGE_NAME="msml610_image"
export CONTAINER_NAME=$IMAGE_NAME

# Print Docker configuration
echo "=== Docker Configuration ==="
echo "Image: $FULL_IMAGE_NAME"
echo "Container: $CONTAINER_NAME"
echo "Port: $JUPYTER_HOST_PORT"
if [[ $TARGET_DIR != "" ]]; then
    echo "Target Dir: $TARGET_DIR"
fi
echo "==========================="

# Run the script.
DOCKER_RUN_OPTS="-p $JUPYTER_HOST_PORT:$JUPYTER_HOST_PORT"
if [[ $TARGET_DIR != "" ]]; then
    DOCKER_RUN_OPTS="$DOCKER_RUN_OPTS -v $TARGET_DIR:/data"
fi;

# Check if image exists
echo "Checking Docker image..."
docker image ls $FULL_IMAGE_NAME

echo "Starting Jupyter container..."
docker run \
    --rm -ti \
    --name $CONTAINER_NAME \
    $DOCKER_RUN_OPTS \
    -v $(pwd):/workspace \
    -w /workspace \
    $FULL_IMAGE_NAME \
    bash -c "pip install openpyxl && jupyter lab --ip=0.0.0.0 --port=$JUPYTER_HOST_PORT --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
