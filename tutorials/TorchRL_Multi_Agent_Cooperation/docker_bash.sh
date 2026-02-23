#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-torchrl_mac:latest}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-}"
PORT="${PORT:-8888}"

echo "Starting Jupyter Lab container (image: ${IMAGE_TAG})"
echo "Open: http://localhost:${PORT}"

PLATFORM_FLAG=""
if [ -n "${DOCKER_PLATFORM}" ]; then
  PLATFORM_FLAG="--platform=${DOCKER_PLATFORM}"
fi

docker run --rm -it ${PLATFORM_FLAG} \
  -p "${PORT}:8888" \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  --shm-size=1g \
  "${IMAGE_TAG}"

