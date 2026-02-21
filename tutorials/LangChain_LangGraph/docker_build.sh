#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/docker_name.sh"

docker build "$SCRIPT_DIR" \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t "$FULL_IMAGE_NAME"
