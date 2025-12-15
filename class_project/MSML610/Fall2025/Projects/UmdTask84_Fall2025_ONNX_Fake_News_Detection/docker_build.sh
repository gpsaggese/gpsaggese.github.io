#!/bin/bash

set -e

IMAGE_NAME="onnx_fake_news_detection"

echo "Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" .

echo "Done. Image '${IMAGE_NAME}' built successfully."
