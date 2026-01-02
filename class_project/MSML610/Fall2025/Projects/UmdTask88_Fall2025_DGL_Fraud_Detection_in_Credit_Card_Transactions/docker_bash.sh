#!/usr/bin/env bash
set -e

IMAGE_NAME="umdtask88-dgl-fraud"

docker run --rm -it \
  -v "$PWD":/app \
  -w /app \
  -p 8888:8888 \
  "$IMAGE_NAME" \
  /bin/bash
