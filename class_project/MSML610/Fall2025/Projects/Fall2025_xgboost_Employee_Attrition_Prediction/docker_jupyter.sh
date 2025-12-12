#!/usr/bin/env bash
set -e

IMAGE_NAME="xgboost-docker-image"
CONTAINER_NAME="attrition-ml-jupyter"

docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  -v "$(pwd)":/workspace \
  -p 8888:8888 \
  "${IMAGE_NAME}" \
  jupyter-notebook \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.notebook_dir='/workspace'

