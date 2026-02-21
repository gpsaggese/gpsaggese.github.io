#!/bin/bash
#
# Execute run_jupyter.sh in the container.
#
# Usage:
# > ./docker_jupyter.sh
# > ./docker_jupyter.sh -p 8890
#

set -euo pipefail

# Parse params.
export JUPYTER_HOST_PORT=8888
export TARGET_DIR=""
export VERBOSE=0

while getopts p:d:v flag
do
    case "${flag}" in
        p) JUPYTER_HOST_PORT=${OPTARG};;
        d) TARGET_DIR=${OPTARG};;
        v) VERBOSE=1;;
    esac
done

if [[ "$VERBOSE" == 1 ]]; then
    set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/docker_name.sh"

# Run the script.
DOCKER_RUN_OPTS=(-p "$JUPYTER_HOST_PORT:8888")
if [[ -n "$TARGET_DIR" ]]; then
    DOCKER_RUN_OPTS+=(-v "$TARGET_DIR:/extra_data")
fi

ENV_FILE_OPT=()
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    ENV_FILE_OPT=(--env-file "$SCRIPT_DIR/.env")
fi

docker image ls "$FULL_IMAGE_NAME"
docker run --rm -it \
    --name "$CONTAINER_NAME" \
    "${DOCKER_RUN_OPTS[@]}" \
    "${ENV_FILE_OPT[@]}" \
    -e JUPYTER_PORT=8888 \
    -v "$SCRIPT_DIR:/app" \
    -w /app \
    "$FULL_IMAGE_NAME" \
    /bin/bash /app/run_jupyter.sh
