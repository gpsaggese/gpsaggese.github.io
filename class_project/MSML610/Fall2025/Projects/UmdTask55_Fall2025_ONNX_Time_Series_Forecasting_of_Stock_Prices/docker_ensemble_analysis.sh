#!/bin/bash
#
# Run only the ensemble analysis step in the container.
# This script assumes that steps 1-4 have already been completed
# (preprocessing and all model training).
#
# Usage:
# > ./docker_ensemble_analysis.sh
#

set -e

# Parse params.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR"

# Import the utility functions.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/docker_common/utils.sh

REPO_NAME=umd_msml610
IMAGE_NAME=onnx_timeseries_forecasting
FULL_IMAGE_NAME=$REPO_NAME/$IMAGE_NAME

print_docker_vars

# Run the script.
DOCKER_RUN_OPTS=""
if [[ $TARGET_DIR != "" ]]; then
    DOCKER_RUN_OPTS="$DOCKER_RUN_OPTS -v $TARGET_DIR:/data"
fi

CMD="cd /data && PYTHONPATH=/data:$PYTHONPATH python3 ensemble/05_ensemble_analysis.py"

# Run container
echo "Starting ensemble analysis..."
echo "Note: This assumes preprocessing and all model training steps are complete."
echo ""

docker run \
    --rm -ti \
    $DOCKER_RUN_OPTS \
    $FULL_IMAGE_NAME \
    bash -c "$CMD"
