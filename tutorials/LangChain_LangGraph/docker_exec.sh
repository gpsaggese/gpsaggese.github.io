#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/docker_name.sh"

if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    docker exec -it "$CONTAINER_NAME" bash
    exit 0
fi

if docker compose -f "$SCRIPT_DIR/docker-compose.yml" ps --services --filter status=running | grep -qx "jupyter"; then
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" exec jupyter bash
    exit 0
fi

echo "No running container found for '$CONTAINER_NAME' or compose service 'jupyter'."
echo "Start one with './docker_bash.sh' or 'docker compose up' first."
exit 1
