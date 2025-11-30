#!/bin/bash
# Run Jupyter notebook server in Docker container

IMAGE_NAME="iot-anomaly-detection"
IMAGE_TAG="latest"
PORT=8888

echo "======================================================================"
echo "Starting Jupyter Notebook server in Docker"
echo "======================================================================"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Port: ${PORT}"
echo "======================================================================"

docker run -it --rm \
    -v "$(pwd):/app" \
    -w /app \
    -p ${PORT}:${PORT} \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    jupyter notebook --ip=0.0.0.0 --port=${PORT} --no-browser --allow-root
