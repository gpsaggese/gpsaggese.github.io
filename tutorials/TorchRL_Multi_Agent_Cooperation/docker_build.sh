#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-torchrl_mac:latest}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-}"

echo "Building image ${IMAGE_TAG}"
if [ -n "${DOCKER_PLATFORM}" ]; then
  docker build --platform="${DOCKER_PLATFORM}" -t "${IMAGE_TAG}" .
else
  docker build -t "${IMAGE_TAG}" .
fi
echo "Done."
