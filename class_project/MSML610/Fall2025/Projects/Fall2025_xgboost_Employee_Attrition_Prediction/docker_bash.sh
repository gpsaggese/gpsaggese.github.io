#!/usr/bin/env bash
set -e

IMAGE_NAME="xgboost-docker-image"
CONTAINER_NAME="attrition-ml-bash"

# Run container with current folder mounted into /workspace
docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  -v "$(pwd)":/workspace \
  -p 8888:8888 \
  "${IMAGE_NAME}" \
  /bin/bash
