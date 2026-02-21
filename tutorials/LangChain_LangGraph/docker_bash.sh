#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/docker_name.sh"

ENV_FILE_OPT=()
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    ENV_FILE_OPT=(--env-file "$SCRIPT_DIR/.env")
fi

docker image ls "$FULL_IMAGE_NAME"
docker run --rm -it \
    --name "$CONTAINER_NAME" \
    -p 8888:8888 \
    "${ENV_FILE_OPT[@]}" \
    -v "$SCRIPT_DIR:/app" \
    -w /app \
    "$FULL_IMAGE_NAME" \
    bash
