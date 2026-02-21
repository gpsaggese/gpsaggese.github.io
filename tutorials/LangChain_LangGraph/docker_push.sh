#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/docker_name.sh"

docker image inspect "$FULL_IMAGE_NAME" >/dev/null
docker push "$FULL_IMAGE_NAME"
