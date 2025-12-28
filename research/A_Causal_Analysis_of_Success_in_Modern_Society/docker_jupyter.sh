#!/usr/bin/env bash
set -e

IMAGE_NAME="causal_success_analysis"
CONTAINER_NAME="causal_success_analysis_container"
PORT_HOST=8888
PORT_CONTAINER=8888

echo "============================================"
echo "Starting Jupyter Lab"
echo "Image:     ${IMAGE_NAME}"
echo "Container: ${CONTAINER_NAME}"
echo "URL:       http://localhost:${PORT_HOST}"
echo "============================================"
echo "Press Ctrl+C to stop."
echo ""

docker run -it --rm \
  --name "${CONTAINER_NAME}" \
  -p "${PORT_HOST}:${PORT_CONTAINER}" \
  -v "$(pwd):/app" \
  -w /app \
  "${IMAGE_NAME}"