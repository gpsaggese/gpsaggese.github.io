#!/usr/bin/env bash
set -e

IMAGE_NAME="causal_success_analysis"

if [ -z "$1" ]; then
    echo "Usage: ./docker_push.sh <docker-hub-username>"
    exit 1
fi

DOCKER_USER="$1"
REMOTE_IMAGE="${DOCKER_USER}/${IMAGE_NAME}:latest"

echo "============================================"
echo "Preparing Docker image for push"
echo "Local image : ${IMAGE_NAME}"
echo "Remote image: ${REMOTE_IMAGE}"
echo "============================================"

echo "Tagging image..."
docker tag "${IMAGE_NAME}" "${REMOTE_IMAGE}"

echo "Pushing image to Docker Hub..."
docker push "${REMOTE_IMAGE}"

echo "============================================"
echo "Push complete!"
echo "Image available at:"
echo "https://hub.docker.com/r/${DOCKER_USER}/${IMAGE_NAME}"
echo "============================================"