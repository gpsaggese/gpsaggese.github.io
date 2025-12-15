#!/bin/bash


set -e

IMAGE_NAME="onnx_fake_news_detection"

echo "Starting Jupyter Notebook from image: ${IMAGE_NAME}"

docker run --rm -it \
  -p 8888:8888 \
  -v "$(pwd)":/app \
  "${IMAGE_NAME}" \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
