#!/bin/bash


set -e

IMAGE_NAME="onnx_fake_news_detection"

echo "Starting bash inside Docker container from image: ${IMAGE_NAME}"

docker run --rm -it \
  -p 8888:8888 \
  -p 8000:8000 \
  -v "$(pwd)":/app \
  "${IMAGE_NAME}" \
  bash
