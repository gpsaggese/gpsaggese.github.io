#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/docker_name.sh"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker rmi -f "$FULL_IMAGE_NAME" 2>/dev/null || true
