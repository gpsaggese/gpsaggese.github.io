#!/bin/bash
# Run Docker container with bash shell

IMAGE_NAME="iot-anomaly-detection"
IMAGE_TAG="latest"

echo "======================================================================"
echo "Starting Docker container: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "======================================================================"

docker run -it --rm \
    -v "$(pwd):/app" \
    -w /app \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    /bin/bash
