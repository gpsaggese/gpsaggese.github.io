#!/bin/bash
#
# Run Streamlit dashboard in the container.
#
# Usage:
# > ./docker_streamlit.sh [-p PORT]
#

set -e

# Parse params.
export STREAMLIT_HOST_PORT=8501
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR"

while getopts p: flag
do
    case "${flag}" in
        p) STREAMLIT_HOST_PORT=${OPTARG};;
    esac
done

# Import the utility functions.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/docker_common/utils.sh

REPO_NAME=umd_msml610
IMAGE_NAME=onnx_timeseries_forecasting
FULL_IMAGE_NAME=$REPO_NAME/$IMAGE_NAME

print_docker_vars

# Run the script.
DOCKER_RUN_OPTS="-p $STREAMLIT_HOST_PORT:$STREAMLIT_HOST_PORT"
if [[ $TARGET_DIR != "" ]]; then
    DOCKER_RUN_OPTS="$DOCKER_RUN_OPTS -v $TARGET_DIR:/data"
fi

CMD="cd /data && streamlit run streamlit_dashboard.py --server.port=$STREAMLIT_HOST_PORT --server.address=0.0.0.0"

# Run container
echo "Starting Streamlit on port $STREAMLIT_HOST_PORT..."
docker run \
    --rm -ti \
    $DOCKER_RUN_OPTS \
    $FULL_IMAGE_NAME \
    bash -c "$CMD"
