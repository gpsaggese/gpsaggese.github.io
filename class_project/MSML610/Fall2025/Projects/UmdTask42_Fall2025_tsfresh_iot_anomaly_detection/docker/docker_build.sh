#!/bin/bash
# Build Docker image for IoT Anomaly Detection project

IMAGE_NAME="iot-anomaly-detection"
IMAGE_TAG="latest"

echo "======================================================================"
echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "======================================================================"

docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f docker/Dockerfile .

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Docker image built successfully!"
    echo "======================================================================"
    echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "To run the container:"
    echo "  ./docker/docker_bash.sh"
    echo ""
    echo "To run Jupyter:"
    echo "  ./docker/docker_jupyter.sh"
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "✗ Docker build failed!"
    echo "======================================================================"
    exit 1
fi
