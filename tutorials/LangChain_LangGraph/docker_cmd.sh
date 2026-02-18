#!/bin/bash
#
# Execute a command in the container.
#

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <command> [args...]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/docker_name.sh"

ENV_FILE_OPT=()
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    ENV_FILE_OPT=(--env-file "$SCRIPT_DIR/.env")
fi

docker image ls "$FULL_IMAGE_NAME"
docker run --rm -it \
    --name "$CONTAINER_NAME" \
    "${ENV_FILE_OPT[@]}" \
    -v "$SCRIPT_DIR:/app" \
    -w /app \
    "$FULL_IMAGE_NAME" \
    "$@"
