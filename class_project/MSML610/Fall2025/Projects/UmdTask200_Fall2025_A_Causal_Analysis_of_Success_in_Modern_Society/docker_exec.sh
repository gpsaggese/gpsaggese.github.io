#!/usr/bin/env bash
set -e

IMAGE_NAME="causal_success_analysis"

echo "============================================"
echo "Attaching to running container"
echo "Image: ${IMAGE_NAME}"
echo "============================================"

# Find the first running container from this image
CONTAINER_ID=$(docker ps -q --filter "ancestor=${IMAGE_NAME}")

if [ -z "${CONTAINER_ID}" ]; then
    echo "No running container found for image '${IMAGE_NAME}'."
    echo "Start Jupyter first with: ./docker_jupyter.sh"
    exit 1
fi

echo "Container ID: ${CONTAINER_ID}"
echo "Opening interactive shell..."
echo "============================================"

docker exec -it "${CONTAINER_ID}" bash